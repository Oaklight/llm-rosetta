"""The extension index, ``Requires-Python`` and the ``ext`` gaps (#1704).

``show``, ``update``, ``install <id>``, ``--json``, ``--dry-run`` and the doctor's
child-process import check.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import importlib
import json
import sys
import threading
from pathlib import Path

import pytest
from _ext_wheels import build_wheel

from llm_rosetta import cli
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION, command
from llm_rosetta.extensions import (
    ExtensionInstallError,
    _discovery,
    configure_extensions_dir,
    install_wheel,
)
from llm_rosetta.extensions import _index as index_mod
from llm_rosetta.extensions._index import (
    IndexEntry,
    install_from_index,
    load_index,
    parse_index,
    plan_updates,
    select_entry,
    version_key,
)
from llm_rosetta.extensions._install import (
    inspect_wheel,
    installed_extensions,
    python_requirement_problem,
)
from llm_rosetta.shims import list_shims
from llm_rosetta.shims._loader import _PLUGIN_SOURCES
from llm_rosetta.shims.provider_shim import unregister_shim


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    saved = dict(command._REGISTRY)
    command._REGISTRY.clear()
    monkeypatch.setattr(command, "_entry_points", lambda: [])
    monkeypatch.setattr(_discovery, "_entry_points", lambda group: [])
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
    monkeypatch.delenv(index_mod.INDEX_ENV, raising=False)
    configure_extensions_dir(None)
    before = {shim.name for shim in list_shims()}
    sources = dict(_PLUGIN_SOURCES)
    path = list(sys.path)
    yield
    command._REGISTRY.clear()
    command._REGISTRY.update(saved)
    configure_extensions_dir(None)
    sys.path[:] = path
    for shim in list_shims():
        if shim.name not in before:
            unregister_shim(shim.name)
    _PLUGIN_SOURCES.clear()
    _PLUGIN_SOURCES.update(sources)
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]
    importlib.invalidate_caches()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _wheel(ext_id: str, version: str, **kw) -> bytes:
    return build_wheel(ext_id, version, [f"{ext_id}--chat_completions"], **kw)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _publish(root: Path, items: dict[str, list[tuple[str, bytes, dict]]]) -> Path:
    """Write wheels and an ``index.json`` (relative urls) under *root*."""
    root.mkdir(parents=True, exist_ok=True)
    extensions: dict[str, list[dict]] = {}
    for ext_id, releases in items.items():
        extensions[ext_id] = []
        for version, data, extra in releases:
            name = f"llm_rosetta_ext_{ext_id}-{version}-py3-none-any.whl"
            (root / name).write_bytes(data)
            extensions[ext_id].append(
                {"version": version, "url": name, "sha256": _sha(data), **extra}
            )
    path = root / "index.json"
    path.write_text(
        json.dumps({"index_version": 1, "extensions": extensions}), encoding="utf-8"
    )
    return path


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "exts"
    path.mkdir()
    return path


def _ext(*argv: str) -> int:
    return cli.main(["ext", *argv])


# --------------------------------------------------------------------------- #
# Requires-Python
# --------------------------------------------------------------------------- #


class TestRequiresPython:
    def test_a_satisfied_specifier_has_no_problem(self):
        assert python_requirement_problem(">=3.10", (3, 12, 1)) is None
        assert python_requirement_problem(">=3.10,<4", (3, 10, 0)) is None

    def test_an_unmet_clause_is_named_with_the_running_version(self):
        text = python_requirement_problem(">=3.13", (3, 12, 1))
        assert text and ">=3.13" in text and "3.12.1" in text

    def test_each_clause_must_hold(self):
        assert python_requirement_problem(">=3.10,<3.12", (3, 12, 0))
        assert python_requirement_problem("!=3.12.1", (3, 12, 1))
        assert python_requirement_problem("==3.12", (3, 12, 0)) is None

    def test_a_specifier_it_cannot_read_is_refused_not_ignored(self):
        text = python_requirement_problem("~=3.10", (3, 12, 0))
        assert text and "cannot read" in text

    def test_install_refuses_a_wheel_that_needs_a_newer_python(self, home):
        data = _wheel("acme", "0.1.0", requires_python=">=99")
        with pytest.raises(ExtensionInstallError, match="needs Python >=99"):
            install_wheel(data, dest=home)
        assert installed_extensions(home) == []
        assert not (home / "llm_rosetta_ext" / "acme").exists()

    def test_install_accepts_a_wheel_the_interpreter_satisfies(self, home):
        data = _wheel("acme", "0.1.0", requires_python=">=3.8")
        assert install_wheel(data, dest=home).id == "acme"

    def test_a_wheel_without_the_header_is_not_checked(self, home):
        assert install_wheel(_wheel("acme", "0.1.0"), dest=home).id == "acme"

    def test_inspect_runs_the_same_check_and_writes_nothing(self, home):
        data = _wheel("acme", "0.1.0", requires_python=">=99")
        with pytest.raises(ExtensionInstallError, match="needs Python"):
            inspect_wheel(data)
        assert inspect_wheel(_wheel("acme", "0.1.0")).id == "acme"
        assert list(home.iterdir()) == []


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #

_GOOD = {
    "version": "0.1.0",
    "url": "a.whl",
    "sha256": "a" * 64,
}


def _doc(**entry):
    return {"index_version": 1, "extensions": {"acme": [{**_GOOD, **entry}]}}


class TestParse:
    def test_a_good_index_lists_its_entries(self):
        parsed = parse_index(
            _doc(requires_python=">=3.10", surface=1, ir=1, requires_extensions=["x"])
        )
        entry = parsed["acme"][0]
        assert (entry.version, entry.requires_python, entry.surface) == (
            "0.1.0",
            ">=3.10",
            1,
        )
        assert entry.requires_extensions == ("x",)

    @pytest.mark.parametrize(
        "document",
        [
            [],
            {"index_version": 2, "extensions": {}},
            {"index_version": 1},
            {"index_version": 1, "extensions": {"Bad-Id": []}},
            {"index_version": 1, "extensions": {"acme": {}}},
            _doc(sha256="xyz"),
            _doc(version="1.0"),
            _doc(url=""),
            _doc(surface="1"),
            _doc(surface=True),
            _doc(requires_extensions="x"),
            _doc(requires_python=3),
        ],
    )
    def test_a_malformed_index_is_refused(self, document):
        with pytest.raises(ExtensionInstallError):
            parse_index(document)

    def test_a_relative_url_resolves_against_a_file_or_a_url(self):
        assert parse_index(_doc(), base="/srv/idx/index.json")["acme"][0].url == str(
            Path("/srv/idx/a.whl")
        )
        assert (
            parse_index(_doc(), base="https://h.example/r/index.json")["acme"][0].url
            == "https://h.example/r/a.whl"
        )
        absolute = "https://other.example/a.whl"
        assert (
            parse_index(_doc(url=absolute), base="/srv/idx/index.json")["acme"][0].url
            == absolute
        )

    def test_the_sha256_is_lowercased(self):
        assert parse_index(_doc(sha256="A" * 64))["acme"][0].sha256 == "a" * 64


class TestDisplayFields:
    """What a listing of not-yet-installed extensions shows, and the header."""

    def test_the_listing_fields_are_parsed(self):
        entry = parse_index(
            _doc(
                name="Kilo (Free)",
                maturity="beta",
                tags=["tier:free"],
                shims=["kilo_free--chat_completions"],
                logo="https://h.example/kilo.svg",
                commit="ABCDEF1234567",
            )
        )["acme"][0]
        assert entry.name == "Kilo (Free)" and entry.maturity == "beta"
        assert entry.tags == ("tier:free",)
        assert entry.shims == ("kilo_free--chat_completions",)
        assert entry.logo == "https://h.example/kilo.svg"
        assert entry.commit == "abcdef1234567"

    def test_an_entry_without_them_still_parses(self):
        entry = parse_index(_doc())["acme"][0]
        assert (entry.name, entry.maturity, entry.logo, entry.commit) == (
            None,
            None,
            None,
            None,
        )
        assert entry.tags == () and entry.shims == ()

    @pytest.mark.parametrize(
        "extra",
        [
            {"maturity": "finished"},
            {"tags": "tier:free"},
            {"tags": [1]},
            {"shims": ["", "x"]},
            {"name": ""},
            {"logo": 3},
            {"commit": "not-a-sha"},
        ],
    )
    def test_a_malformed_listing_field_is_refused(self, extra):
        with pytest.raises(ExtensionInstallError):
            parse_index(_doc(**extra))

    def test_the_header_is_exposed_as_attributes(self):
        document = _doc()
        document["source"] = "contrib"
        document["generated_at"] = "2026-11-26T08:00:00Z"
        index = parse_index(document)
        assert (index.source, index.generated_at) == ("contrib", "2026-11-26T08:00:00Z")
        assert "acme" in index and index["acme"][0].version == "0.1.0"

    def test_a_missing_header_is_empty_not_an_error(self):
        index = parse_index(_doc())
        assert (index.source, index.generated_at) == ("", "")

    @pytest.mark.parametrize(
        "header",
        [
            {"source": "Bad Source"},
            {"source": 3},
            {"generated_at": "yesterday"},
            {"generated_at": 20261126},
        ],
    )
    def test_a_malformed_header_is_refused(self, header):
        with pytest.raises(ExtensionInstallError):
            parse_index({**_doc(), **header})


class TestVersions:
    def test_numbers_compare_as_numbers(self):
        assert version_key("0.10.0") > version_key("0.9.0")

    def test_a_pre_release_sorts_below_its_final(self):
        assert version_key("1.0.0-rc.1") < version_key("1.0.0")

    def test_pre_release_identifiers_compare_as_numbers_and_then_text(self):
        assert version_key("1.0.0-rc.2") < version_key("1.0.0-rc.10")
        assert version_key("1.0.0-alpha") < version_key("1.0.0")
        assert version_key("1.0.0-1") < version_key("1.0.0-alpha")
        assert version_key("1.0.0-rc.1") < version_key("1.0.0-rc.1.1")

    def test_a_non_version_is_refused(self):
        with pytest.raises(ValueError):
            version_key("1.0")

    def test_build_metadata_is_refused(self):
        with pytest.raises(ValueError):
            version_key("0.1.0+build.5")
        document = _doc()
        document["extensions"]["acme"][0]["version"] = "0.1.0+build.5"
        with pytest.raises(ExtensionInstallError):
            parse_index(document)


def _entry(version, **kw) -> IndexEntry:
    return IndexEntry("acme", version, f"{version}.whl", "a" * 64, **kw)


class TestSelect:
    def test_the_newest_compatible_version_wins(self):
        index = {"acme": [_entry("0.1.0"), _entry("0.10.0"), _entry("0.9.0")]}
        assert select_entry(index, "acme").version == "0.10.0"

    def test_an_incompatible_newer_version_is_skipped(self):
        index = {
            "acme": [
                _entry("0.1.0"),
                _entry("0.2.0", requires_python=">=99"),
                _entry("0.3.0", surface=SURFACE_VERSION + 1),
                _entry("0.4.0", ir=IR_VERSION + 1),
            ]
        }
        assert select_entry(index, "acme").version == "0.1.0"

    def test_when_nothing_fits_every_reason_is_named(self):
        index = {"acme": [_entry("0.2.0", requires_python=">=99")]}
        with pytest.raises(ExtensionInstallError, match=r"0\.2\.0: needs Python >=99"):
            select_entry(index, "acme")

    def test_an_unknown_id_or_version_is_an_error(self):
        index = {"acme": [_entry("0.1.0")]}
        with pytest.raises(ExtensionInstallError, match="no extension 'zzz'"):
            select_entry(index, "zzz")
        with pytest.raises(ExtensionInstallError, match="no version 9.9.9"):
            select_entry(index, "acme", version="9.9.9")

    def test_a_pinned_version_is_chosen_over_a_newer_one(self):
        index = {"acme": [_entry("0.1.0"), _entry("0.2.0")]}
        assert select_entry(index, "acme", version="0.1.0").version == "0.1.0"


class TestLoad:
    def test_a_local_file_with_relative_wheels(self, tmp_path):
        path = _publish(
            tmp_path / "pub", {"acme": [("0.1.0", _wheel("acme", "0.1.0"), {})]}
        )
        entry = load_index(str(path))["acme"][0]
        assert Path(entry.url).is_file()

    def test_the_environment_names_the_index(self, tmp_path, monkeypatch):
        path = _publish(
            tmp_path / "pub", {"acme": [("0.1.0", _wheel("acme", "0.1.0"), {})]}
        )
        monkeypatch.setenv(index_mod.INDEX_ENV, str(path))
        assert "acme" in load_index()

    def test_an_unreadable_or_non_json_file_is_an_install_error(self, tmp_path):
        with pytest.raises(ExtensionInstallError, match="cannot read the index"):
            load_index(str(tmp_path / "missing.json"))
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(ExtensionInstallError, match="not JSON"):
            load_index(str(bad))

    def test_an_oversized_index_is_refused(self, tmp_path, monkeypatch):
        path = _publish(tmp_path / "pub", {})
        monkeypatch.setattr(index_mod, "MAX_INDEX_BYTES", 5)
        with pytest.raises(ExtensionInstallError, match="byte cap"):
            load_index(str(path))

    def test_plain_http_to_a_non_loopback_host_is_refused(self):
        with pytest.raises(ExtensionInstallError, match="plain http"):
            load_index("http://example.com/index.json")

    def test_an_index_served_over_loopback_http(self, tmp_path):
        path = _publish(
            tmp_path / "pub", {"acme": [("0.1.0", _wheel("acme", "0.1.0"), {})]}
        )

        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

        handler = functools.partial(Quiet, directory=str(path.parent))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/index.json"
            entry = load_index(url)["acme"][0]
            assert entry.url.startswith("http://127.0.0.1:")
            home = tmp_path / "exts"
            home.mkdir()
            done = install_from_index("acme", load_index(url), dest=home)
            assert [item.id for item in done] == ["acme"]
        finally:
            server.shutdown()
            server.server_close()


class TestInstallFromIndex:
    def test_a_wrong_digest_installs_nothing(self, tmp_path, home):
        path = _publish(
            tmp_path / "pub", {"acme": [("0.1.0", _wheel("acme", "0.1.0"), {})]}
        )
        doc = json.loads(path.read_text())
        doc["extensions"]["acme"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(doc))
        with pytest.raises(ExtensionInstallError, match="sha256 mismatch"):
            install_from_index("acme", load_index(str(path)), dest=home)
        assert installed_extensions(home) == []

    def test_a_required_extension_is_installed_first_and_only_when_missing(
        self, tmp_path, home
    ):
        path = _publish(
            tmp_path / "pub",
            {
                "base": [("0.1.0", _wheel("base", "0.1.0"), {})],
                "top": [
                    (
                        "0.1.0",
                        _wheel("top", "0.1.0", requires_extensions=["base"]),
                        {"requires_extensions": ["base"]},
                    )
                ],
            },
        )
        index = load_index(str(path))
        done = install_from_index("top", index, dest=home)
        assert [item.id for item in done] == ["base", "top"]
        again = install_from_index("top", index, dest=home)
        assert [item.id for item in again] == ["top"]

    def test_a_requirement_cycle_is_an_error(self, tmp_path, home):
        path = _publish(
            tmp_path / "pub",
            {
                "a": [("0.1.0", _wheel("a", "0.1.0"), {"requires_extensions": ["b"]})],
                "b": [("0.1.0", _wheel("b", "0.1.0"), {"requires_extensions": ["a"]})],
            },
        )
        with pytest.raises(ExtensionInstallError, match="cycle"):
            install_from_index("a", load_index(str(path)), dest=home)


class TestPlan:
    def test_each_status(self, tmp_path, home):
        old = install_wheel(_wheel("up", "0.1.0"), dest=home)
        cur = install_wheel(_wheel("cur", "0.2.0"), dest=home)
        gone = install_wheel(_wheel("gone", "0.1.0"), dest=home)
        stuck = install_wheel(_wheel("stuck", "0.1.0"), dest=home)
        index = {
            "up": [_entry_for("up", "0.2.0")],
            "cur": [_entry_for("cur", "0.2.0")],
            "stuck": [_entry_for("stuck", "0.3.0", requires_python=">=99")],
        }
        plans = {p.id: p for p in plan_updates(index, [old, cur, gone, stuck])}
        target = plans["up"].target
        assert plans["up"].status == "update"
        assert target is not None and target.version == "0.2.0"
        assert plans["cur"].status == "current"
        assert plans["gone"].status == "not_in_index"
        assert plans["stuck"].status == "no_compatible"


def _entry_for(ext_id, version, **kw) -> IndexEntry:
    return IndexEntry(ext_id, version, f"{version}.whl", "a" * 64, **kw)


def _fetched(*args, **kwargs):
    raise AssertionError("a dry run fetched")


# --------------------------------------------------------------------------- #
# ext install <id>, update, remove, --dry-run
# --------------------------------------------------------------------------- #


@pytest.fixture
def published(tmp_path) -> Path:
    return _publish(
        tmp_path / "pub",
        {
            "acme": [
                ("0.1.0", _wheel("acme", "0.1.0"), {}),
                ("0.2.0", _wheel("acme", "0.2.0"), {}),
            ],
            "other": [("1.0.0", _wheel("other", "1.0.0"), {})],
        },
    )


class TestInstallById:
    def test_the_newest_version_is_installed(self, home, published, capsys):
        assert (
            _ext("install", "acme", "--index", str(published), "--dir", str(home)) == 0
        )
        assert "installed acme 0.2.0" in capsys.readouterr().out

    def test_a_version_can_be_pinned(self, home, published, capsys):
        assert (
            _ext(
                "install",
                "acme",
                "--version",
                "0.1.0",
                "--index",
                str(published),
                "--dir",
                str(home),
            )
            == 0
        )
        assert "installed acme 0.1.0" in capsys.readouterr().out

    def test_the_environment_supplies_the_index(self, home, published, monkeypatch):
        monkeypatch.setenv(index_mod.INDEX_ENV, str(published))
        assert _ext("install", "other", "--dir", str(home)) == 0

    def test_an_unknown_id_exits_1_naming_it(self, home, published, capsys):
        assert (
            _ext("install", "zzz", "--index", str(published), "--dir", str(home)) == 1
        )
        assert "no extension 'zzz'" in capsys.readouterr().err

    def test_a_missing_index_exits_1(self, home, tmp_path, capsys):
        code = _ext(
            "install", "acme", "--index", str(tmp_path / "no.json"), "--dir", str(home)
        )
        assert code == 1
        assert "cannot read the index" in capsys.readouterr().err

    def test_sha256_does_not_apply_to_an_id(self, home, published):
        code = _ext(
            "install",
            "acme",
            "--sha256",
            "a" * 64,
            "--index",
            str(published),
            "--dir",
            str(home),
        )
        assert code == 2

    def test_version_does_not_apply_to_a_file(self, home, tmp_path):
        wheel = tmp_path / "w.whl"
        wheel.write_bytes(_wheel("acme", "0.1.0"))
        code = _ext("install", str(wheel), "--version", "0.1.0", "--dir", str(home))
        assert code == 2

    def test_an_existing_file_named_like_an_id_is_a_file(
        self, home, tmp_path, monkeypatch
    ):
        wheel = tmp_path / "acme"
        wheel.write_bytes(_wheel("acme", "0.1.0"))
        monkeypatch.chdir(tmp_path)
        assert _ext("install", "acme", "--dir", str(home)) == 0


class TestDryRun:
    def test_install_by_id_says_what_it_would_do_and_writes_nothing(
        self, home, published, capsys
    ):
        code = _ext(
            "install",
            "acme",
            "--index",
            str(published),
            "--dir",
            str(home),
            "--dry-run",
        )
        assert code == 0
        assert "would install acme 0.2.0" in capsys.readouterr().out
        assert installed_extensions(home) == []

    def test_install_by_id_lists_a_missing_dependency(self, home, tmp_path, capsys):
        path = _publish(
            tmp_path / "pub",
            {
                "base": [("0.1.0", _wheel("base", "0.1.0"), {})],
                "top": [
                    ("0.1.0", _wheel("top", "0.1.0"), {"requires_extensions": ["base"]})
                ],
            },
        )
        assert (
            _ext(
                "install", "top", "--index", str(path), "--dir", str(home), "--dry-run"
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "would install the required extension base" in out
        assert installed_extensions(home) == []

    def test_install_a_file_validates_it_and_names_the_replaced_version(
        self, home, tmp_path, capsys
    ):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        wheel = tmp_path / "w.whl"
        wheel.write_bytes(_wheel("acme", "0.2.0"))
        assert _ext("install", str(wheel), "--dir", str(home), "--dry-run") == 0
        out = capsys.readouterr().out
        assert "would install acme 0.2.0" in out and "(replaces 0.1.0)" in out
        assert installed_extensions(home)[0].version == "0.1.0"

    def test_install_a_bad_file_still_fails_in_a_dry_run(self, home, tmp_path, capsys):
        wheel = tmp_path / "w.whl"
        wheel.write_bytes(_wheel("acme", "0.1.0", requires_python=">=99"))
        assert _ext("install", str(wheel), "--dir", str(home), "--dry-run") == 1
        assert "needs Python" in capsys.readouterr().err

    def test_install_a_url_checks_the_address_and_fetches_nothing(
        self, home, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            "llm_rosetta.extensions._install.fetch_bytes",
            _fetched,
        )
        assert (
            _ext(
                "install",
                "https://h.example/a.whl",
                "--sha256",
                "a" * 64,
                "--dir",
                str(home),
                "--dry-run",
            )
            == 0
        )
        assert "would download https://h.example/a.whl" in capsys.readouterr().out
        assert (
            _ext(
                "install",
                "http://h.example/a.whl",
                "--sha256",
                "a" * 64,
                "--dir",
                str(home),
                "--dry-run",
            )
            == 1
        )

    def test_remove_names_what_it_would_remove_and_removes_nothing(self, home, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert _ext("remove", "acme", "--dir", str(home), "--dry-run") == 0
        assert "would remove acme 0.1.0" in capsys.readouterr().out
        assert [i.id for i in installed_extensions(home)] == ["acme"]
        assert _ext("remove", "zzz", "--dir", str(home), "--dry-run") == 1

    def test_update_reports_the_move_and_installs_nothing(
        self, home, published, capsys
    ):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert (
            _ext("update", "--index", str(published), "--dir", str(home), "--dry-run")
            == 0
        )
        assert "acme: would update 0.1.0 -> 0.2.0" in capsys.readouterr().out
        assert installed_extensions(home)[0].version == "0.1.0"


class TestUpdate:
    def test_an_older_install_is_updated(self, home, published, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert _ext("update", "--index", str(published), "--dir", str(home)) == 0
        assert "acme: updated 0.1.0 -> 0.2.0" in capsys.readouterr().out
        assert installed_extensions(home)[0].version == "0.2.0"

    def test_a_current_install_is_left_alone(self, home, published, capsys):
        install_wheel(_wheel("acme", "0.2.0"), dest=home)
        assert _ext("update", "--index", str(published), "--dir", str(home)) == 0
        assert "acme: up to date (0.2.0)" in capsys.readouterr().out

    def test_an_extension_outside_the_index_is_reported_not_failed(
        self, home, published, capsys
    ):
        install_wheel(_wheel("local", "0.1.0"), dest=home)
        assert _ext("update", "--index", str(published), "--dir", str(home)) == 0
        assert "local: not in the index (0.1.0)" in capsys.readouterr().out

    def test_one_id_updates_only_that_extension(self, home, published, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        install_wheel(_wheel("other", "0.5.0"), dest=home)
        assert (
            _ext("update", "acme", "--index", str(published), "--dir", str(home)) == 0
        )
        out = capsys.readouterr().out
        assert "acme" in out and "other" not in out

    def test_an_id_that_is_not_installed_exits_1(self, home, published, capsys):
        assert _ext("update", "zzz", "--index", str(published), "--dir", str(home)) == 1
        assert "not installed" in capsys.readouterr().err

    def test_an_unreadable_index_exits_1(self, home, tmp_path, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert (
            _ext("update", "--index", str(tmp_path / "no.json"), "--dir", str(home))
            == 1
        )
        assert "cannot read the index" in capsys.readouterr().err

    def test_a_failed_update_is_reported_and_the_old_version_stays(
        self, home, tmp_path, capsys
    ):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        path = _publish(
            tmp_path / "pub", {"acme": [("0.2.0", _wheel("acme", "0.2.0"), {})]}
        )
        doc = json.loads(path.read_text())
        doc["extensions"]["acme"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(doc))
        assert _ext("update", "--index", str(path), "--dir", str(home)) == 1
        assert "update failed" in capsys.readouterr().err
        assert installed_extensions(home)[0].version == "0.1.0"

    def test_with_nothing_installed_it_says_so(self, home, published, capsys):
        assert _ext("update", "--index", str(published), "--dir", str(home)) == 0
        assert "no extensions installed" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# list / show / doctor: --json
# --------------------------------------------------------------------------- #


class TestJson:
    def test_list_json(self, home, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert _ext("list", "--json", "--dir", str(home)) == 0
        document = json.loads(capsys.readouterr().out)
        assert document["directory"] == str(home)
        row = document["extensions"][0]
        assert (row["id"], row["state"], row["version"]) == ("acme", "loaded", "0.1.0")

    def test_list_json_of_an_empty_directory(self, home, capsys):
        assert _ext("list", "--json", "--dir", str(home)) == 0
        assert json.loads(capsys.readouterr().out)["extensions"] == []

    def test_show_text(self, home, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert _ext("show", "acme", "--dir", str(home)) == 0
        out = capsys.readouterr().out
        assert out.startswith("acme  0.1.0  [loaded]")
        assert "shims:" in out and "acme--chat_completions" in out
        assert f"surface {SURFACE_VERSION}" in out

    def test_show_json(self, home, capsys):
        install_wheel(_wheel("acme", "0.1.0"), dest=home)
        assert _ext("show", "acme", "--json", "--dir", str(home)) == 0
        document = json.loads(capsys.readouterr().out)
        assert document["state"] == "loaded" and document["problems"] == []
        assert document["manifest"]["provides"]["shims"] == ["acme--chat_completions"]
        assert document["manifest"]["requires"]["surface"] == SURFACE_VERSION

    def test_show_names_a_command_summary(self, home, capsys):
        manifest = (
            f"id: acme\\nname: acme\\nversion: 0.1.0\\nmaturity: beta\\nrequires:\\n"
            f"  surface: {SURFACE_VERSION}\\n  ir: {IR_VERSION}\\nprovides:\\n  commands:\\n"
            "    go: { en: Go there. }\\n"
        ).replace("\\n", "\n")
        install_wheel(build_wheel("acme", "0.1.0", [], manifest=manifest), dest=home)
        assert _ext("show", "acme", "--json", "--dir", str(home)) == 0
        document = json.loads(capsys.readouterr().out)
        assert document["manifest"]["provides"]["commands"] == {
            "go": {"en": "Go there."}
        }

    def test_show_of_an_unknown_id_exits_1(self, home, capsys):
        assert _ext("show", "zzz", "--dir", str(home)) == 1
        assert "no extension 'zzz'" in capsys.readouterr().err

    def test_show_reports_a_missing_requirement_as_a_problem(self, home, capsys):
        install_wheel(_wheel("top", "0.1.0", requires_extensions=["base"]), dest=home)
        assert _ext("show", "top", "--json", "--dir", str(home)) == 0
        problems = json.loads(capsys.readouterr().out)["problems"]
        assert any("'base'" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# doctor: the child-process import check
# --------------------------------------------------------------------------- #


def _command_manifest(ext_id="acme", verbs=("go",)) -> str:
    return (
        f"id: {ext_id}\nname: {ext_id}\nversion: 0.1.0\nmaturity: beta\nrequires:\n"
        f"  surface: {SURFACE_VERSION}\n  ir: {IR_VERSION}\nprovides:\n  commands:\n"
        + "".join(f"    {verb}: {{ en: x }}\n" for verb in verbs)
    )


def _install_commands(home: Path, source: str, verbs=("go",)) -> None:
    install_wheel(
        build_wheel(
            "acme",
            "0.1.0",
            [],
            manifest=_command_manifest(verbs=verbs),
            members={"commands.py": source},
        ),
        dest=home,
    )


GOOD = "def go(argv):\n    return 0\nCOMMANDS = {'go': go}\n"


class TestDoctorImport:
    def test_a_good_extension_has_no_problem(self, home, capsys):
        _install_commands(home, GOOD)
        assert _ext("doctor", "--dir", str(home)) == 0
        assert "ok" in capsys.readouterr().out

    def test_json_carries_the_import_result(self, home, capsys):
        _install_commands(home, GOOD)
        assert _ext("doctor", "--json", "--dir", str(home)) == 0
        found = json.loads(capsys.readouterr().out)["extensions"][0]
        assert found["import"]["ok"] is True
        assert found["import"]["isolated"] is True
        assert found["problems"] == []

    def test_an_import_error_is_a_problem_and_exit_1(self, home, capsys):
        _install_commands(home, "raise RuntimeError('boom at import')\n")
        assert _ext("doctor", "--dir", str(home)) == 1
        out = capsys.readouterr().out
        assert "import failed" in out and "boom at import" in out

    def test_an_extension_that_exits_the_interpreter_is_contained(self, home, capsys):
        _install_commands(home, "import sys\nsys.exit(3)\n")
        assert _ext("doctor", "--dir", str(home)) == 1
        assert "import failed" in capsys.readouterr().out

    def test_a_commands_mapping_that_disagrees_with_the_manifest_is_a_problem(
        self, home, capsys
    ):
        _install_commands(home, "COMMANDS = {'other': print}\n")
        assert _ext("doctor", "--dir", str(home)) == 1
        assert "manifest lists ['go']" in capsys.readouterr().out

    def test_a_missing_mapping_is_a_problem(self, home, capsys):
        _install_commands(home, "x = 1\n")
        assert _ext("doctor", "--dir", str(home)) == 1
        assert "no COMMANDS mapping" in capsys.readouterr().out

    def test_no_import_skips_the_check(self, home, capsys):
        _install_commands(home, "raise RuntimeError('boom at import')\n")
        assert _ext("doctor", "--no-import", "--json", "--dir", str(home)) == 0
        found = json.loads(capsys.readouterr().out)["extensions"][0]
        assert found["import"] is None and found["problems"] == []

    def test_a_host_without_a_python_interpreter_checks_in_process(
        self, home, capsys, monkeypatch
    ):
        from llm_rosetta.extensions import _probe

        monkeypatch.setattr(_probe, "_can_spawn", lambda: False)
        _install_commands(home, GOOD)
        assert _ext("doctor", "--json", "--dir", str(home)) == 0
        found = json.loads(capsys.readouterr().out)["extensions"][0]
        assert found["import"]["ok"] is True and found["import"]["isolated"] is False

    def test_in_process_it_still_reports_an_import_error(
        self, home, capsys, monkeypatch
    ):
        from llm_rosetta.extensions import _probe

        monkeypatch.setattr(_probe, "_can_spawn", lambda: False)
        _install_commands(home, "raise RuntimeError('boom at import')\n")
        assert _ext("doctor", "--dir", str(home)) == 1
        assert "boom at import" in capsys.readouterr().out

    def test_a_timeout_is_a_failed_check(self, home, capsys, monkeypatch):
        import subprocess

        from llm_rosetta.extensions import _probe

        def slow(*a, **k):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(_probe.subprocess, "run", slow)
        _install_commands(home, GOOD)
        assert _ext("doctor", "--dir", str(home)) == 1
        assert "did not finish" in capsys.readouterr().out

    def test_doctor_json_lists_retired_entry_points_and_problems(self, home, capsys):
        install_wheel(_wheel("top", "0.1.0", requires_extensions=["base"]), dest=home)
        assert _ext("doctor", "--json", "--dir", str(home)) == 1
        document = json.loads(capsys.readouterr().out)
        assert document["extensions"][0]["problems"]
        assert document["retired"] == []

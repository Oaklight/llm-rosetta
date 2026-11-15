"""Tests for declared shim manifests and per-source selection (#1557).

Three layers are pinned separately: the manifest reader (what a package may
declare), the scan (what a selection actually loads), and the config bridge
(how ``shims.include`` / ``shims.exclude`` reach the loader).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llm_rosetta.shims.provider_shim import _reset_registry
from llm_rosetta.shims.providers import load_providers, load_providers_from_dir
from llm_rosetta_gateway.shim_sources import parse_shims_section
from llm_rosetta_gateway.shims import sources as shim_sources
from llm_rosetta_gateway.shims.manifest import (
    MANIFEST_FILENAME,
    ShimManifestError,
    read_shim_manifest,
)
from llm_rosetta_gateway.shims.sources import (
    ShimSourceError,
    ShimSourceStatus,
    _scan_dir,
    rebuild_shims,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Start and end every test with no source shims registered."""
    _reset_registry()
    yield
    _reset_registry()
    load_providers()


def _leaf(root: Path, slug: str, name: str | None = None) -> Path:
    """Write one chat shim under ``root/<slug>`` and return its directory."""
    name = name or f"{slug}--chat_completions"
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.yaml").write_text(
        f"name: {name}\n"
        "format: chat_completions\n"
        "model_type: llm\n"
        "connection:\n"
        f"  base_url: https://{slug}.example/v1\n"
        "  api_key_env: ACME_API_KEY\n",
        encoding="utf-8",
    )
    return directory


def _manifest(root: Path, body: str) -> None:
    (root / MANIFEST_FILENAME).write_text(body, encoding="utf-8")


def _three_shims(root: Path) -> None:
    for slug in ("a", "b", "c"):
        _leaf(root, slug)


MANIFEST = """version: 1
shims:
  - name: a--chat_completions
    path: a
  - name: b--chat_completions
    path: b
  - name: c--chat_completions
    path: c
"""


class _Capture(logging.Handler):
    """Collect the gateway logger's messages; it does not propagate."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def captured():
    """Collect gateway warnings; the two modules log under two names.

    ``shims/sources.py`` uses its module logger and ``shim_sources.py`` uses the
    package one, so a handler on either alone would miss half the warnings.
    """
    handler = _Capture()
    loggers = [
        shim_sources.logger,
        logging.getLogger("llm-rosetta-gateway"),
    ]
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


class TestManifestReader:
    """What a package is allowed to declare."""

    def test_absent_returns_none(self, tmp_path: Path):
        assert read_shim_manifest(tmp_path) is None

    def test_declared_entries_keep_their_order(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.names == (
            "a--chat_completions",
            "b--chat_completions",
            "c--chat_completions",
        )
        assert [entry.path for entry in manifest.entries] == ["a", "b", "c"]

    def test_an_empty_list_declares_nothing(self, tmp_path: Path):
        """A package may ship a manifest that selects none of its own leaves."""
        _three_shims(tmp_path)
        _manifest(tmp_path, "version: 1\nshims: []\n")

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.names == ()

    @pytest.mark.parametrize(
        "body",
        [
            "version: 2\nshims: []\n",
            "- not a mapping\n",
            "version: 1\nshims:\n  - name: a--chat_completions\n  - name: a--chat_completions\n",
            "version: 1\nshims:\n  - name: x\n    path: ../outside\n",
            "version: 1\nshims:\n  - name: x\n    path: /abs\n",
            "version: 1\nshims:\n  - name: x\n    path: https://example.com/x\n",
            "version: 1\nshims:\n  - name: x\n    path: missing\n",
            "version: 1\nshims:\n  - path: a\n",
            "version: 1\nshims:\n  - name: x\n",
            "version: 1\nshims: not-a-list\n",
        ],
    )
    def test_an_unusable_manifest_is_refused(self, tmp_path: Path, body: str):
        _leaf(tmp_path, "a")
        _manifest(tmp_path, body)

        with pytest.raises(ShimManifestError):
            read_shim_manifest(tmp_path)


class TestScanSelection:
    """What a selection actually loads."""

    def test_a_full_load_follows_the_manifest(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert status.shim_names == (
            "a--chat_completions",
            "b--chat_completions",
            "c--chat_completions",
        )
        assert status.manifest_shims == status.shim_names

    def test_a_directory_outside_the_manifest_is_never_read(self, tmp_path: Path):
        """Authority means authority: an undeclared leaf is not even parsed."""
        _three_shims(tmp_path)
        stray = _leaf(tmp_path, "d")
        (stray / "provider.yaml").write_text("this: is not a shim\n", encoding="utf-8")
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert "d--chat_completions" not in status.shim_names

    def test_include_keeps_only_the_named_shims(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(
            tmp_path,
            kind="path",
            location="src",
            include=("b--chat_completions",),
        )

        assert status.shim_names == ("b--chat_completions",)

    def test_include_accepts_a_glob(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(tmp_path, kind="path", location="src", include=("a*", "c*"))

        assert status.shim_names == ("a--chat_completions", "c--chat_completions")

    def test_exclude_wins_over_include(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(
            tmp_path,
            kind="path",
            location="src",
            include=("a*", "b*"),
            exclude=("b*",),
        )

        assert status.shim_names == ("a--chat_completions",)

    def test_a_manifest_less_source_filters_by_name(self, tmp_path: Path):
        """The legacy layout keeps working, and the selection still applies."""
        _three_shims(tmp_path)

        status = _scan_dir(tmp_path, kind="path", location="src", exclude=("a*", "c*"))

        assert status.shim_names == ("b--chat_completions",)
        assert status.manifest_shims == ()

    def test_an_unmatched_include_is_reported(self, tmp_path: Path, captured):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(
            tmp_path, kind="path", location="src", include=("ghost--*", "a*")
        )

        assert status.unmatched == ("ghost--*",)
        assert status.shim_names == ("a--chat_completions",)
        assert any("ghost--*" in message for message in captured)

    def test_a_broken_manifest_fails_the_source(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(
            tmp_path,
            "version: 1\nshims:\n  - name: a--chat_completions\n    path: nope\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert not status.ok
        assert "shims.yaml" in (status.error or "")

    def test_a_manifest_entry_loading_another_name_is_warned(
        self, tmp_path: Path, captured
    ):
        _leaf(tmp_path, "a", name="renamed--chat_completions")
        _manifest(
            tmp_path, "version: 1\nshims:\n  - name: a--chat_completions\n    path: a\n"
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.shim_names == ("renamed--chat_completions",)
        assert any("declares no shim named" in message for message in captured)

    def test_the_selection_is_echoed_on_the_status(self, tmp_path: Path):
        _three_shims(tmp_path)

        status = _scan_dir(
            tmp_path, kind="path", location="src", include=("a*",), exclude=("b*",)
        )

        payload = status.to_dict()
        assert payload["include"] == ["a*"]
        assert payload["exclude"] == ["b*"]
        assert payload["unmatched"] == []
        assert payload["manifest_shims"] == []


class TestCoreLoaderSelection:
    """The two hooks the gateway uses, exercised without the gateway."""

    def test_only_paths_selects_one_leaf(self, tmp_path: Path):
        _three_shims(tmp_path)

        shims = load_providers_from_dir(tmp_path, only_paths={"b"})

        assert [s.name for s in shims] == ["b--chat_completions"]

    def test_only_paths_selects_a_subtree(self, tmp_path: Path):
        _leaf(tmp_path / "vendor" / "llm", "x")
        _leaf(tmp_path, "a")

        shims = load_providers_from_dir(tmp_path, only_paths={"vendor"})

        assert [s.name for s in shims] == ["x--chat_completions"]

    def test_an_empty_selection_loads_nothing(self, tmp_path: Path):
        _three_shims(tmp_path)

        assert load_providers_from_dir(tmp_path, only_paths=set()) == []

    def test_a_declined_leaf_is_not_imported(self, tmp_path: Path):
        """Selection happens before the module import, so a broken one is moot."""
        _three_shims(tmp_path)
        (tmp_path / "b" / "transforms.py").write_text(
            "this is not python(\n", encoding="utf-8"
        )

        shims = load_providers_from_dir(
            tmp_path, on_error="raise", keep_names=lambda name: "b--" not in name
        )

        assert [s.name for s in shims] == [
            "a--chat_completions",
            "c--chat_completions",
        ]


class TestConfigSelection:
    """``shims.include`` / ``shims.exclude`` reaching the remote spec."""

    def test_the_selection_is_folded_into_the_preset_remote(self):
        sources = parse_shims_section(
            {
                "presets": ["free"],
                "include": {"free": ["llm7--chat_completions"]},
                "exclude": {"free": ["kilo_free--chat_completions"]},
            }
        )

        remote = sources.remotes[0]
        assert remote["include"] == ["llm7--chat_completions"]
        assert remote["exclude"] == ["kilo_free--chat_completions"]

    def test_an_explicit_remote_may_carry_its_own_selection(self):
        sources = parse_shims_section(
            {
                "remotes": [
                    {
                        "url": "https://github.com/acme/pack",
                        "include": ["x--chat_completions"],
                    }
                ]
            }
        )

        assert sources.remotes[0]["include"] == ["x--chat_completions"]

    def test_an_explicit_remote_overrides_the_preset_selection(self):
        sources = parse_shims_section(
            {
                "presets": ["free"],
                "include": {"free": ["llm7--chat_completions"]},
                "remotes": [
                    {
                        "url": "https://github.com/Oaklight/llm-rosetta-shim-free",
                        "include": ["ovhcloud--chat_completions"],
                    }
                ],
            }
        )

        assert sources.remotes[0]["include"] == ["ovhcloud--chat_completions"]

    def test_an_unknown_preset_key_is_skipped_unless_strict(self):
        section = {"presets": ["free"], "include": {"ghost": ["x"]}}

        assert parse_shims_section(section).remotes
        with pytest.raises(ValueError, match="ghost"):
            parse_shims_section({**section, "strict": True})

    @pytest.mark.parametrize(
        "section",
        [
            {"presets": ["free"], "include": ["x"]},
            {"presets": ["free"], "include": {"free": "x"}},
            {"presets": ["free"], "include": {"free": [1]}},
            {"presets": ["free"], "include": {"free": [""]}},
            {"remotes": [{"url": "https://github.com/o/r", "include": "x"}]},
            {"remotes": [{"url": "https://github.com/o/r", "exclude": [""]}]},
        ],
    )
    def test_bad_shapes_raise(self, section):
        with pytest.raises(ValueError):
            parse_shims_section(section)

    def test_a_selection_for_an_undeclared_preset_warns(self, captured):
        sources = parse_shims_section({"presets": ["argo"], "include": {"free": ["x"]}})

        assert [remote["url"] for remote in sources.remotes] == [
            "https://github.com/Oaklight/llm-rosetta-shim-argo"
        ]
        assert any("does not declare" in message for message in captured)


class TestStrictUnmatched:
    """An include that matched nothing stops a strict load, and only that."""

    @staticmethod
    def _patch(monkeypatch):
        monkeypatch.setattr(
            shim_sources,
            "_load_local_dir",
            lambda directory: ShimSourceStatus(
                kind="path", location=str(directory), unmatched=("ghost--*",)
            ),
        )

    def test_strict_raises(self, monkeypatch):
        self._patch(monkeypatch)
        with pytest.raises(ShimSourceError, match="matched nothing"):
            rebuild_shims(["x"], strict=True)

    def test_non_strict_returns_the_report(self, monkeypatch):
        self._patch(monkeypatch)

        report = rebuild_shims(["x"])

        assert report.failures() == []
        assert report.sources[0].unmatched == ("ghost--*",)


class TestCoreFloor:
    """A package that tracks its branch states the oldest core it needs."""

    def test_a_floor_the_running_core_meets_loads(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=0.1'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert status.shim_names == ("a--chat_completions",)

    def test_a_floor_above_the_running_core_is_refused(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=99.0.0'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert not status.ok
        error = status.error or ""
        assert "needs llm-rosetta >=" in error
        assert "upgrade the gateway" in error

    @pytest.mark.parametrize("body", ["core: '>='\n", "core: '^0.15'\n", "core: 15\n"])
    def test_an_unusable_floor_is_refused(self, tmp_path: Path, body: str):
        _leaf(tmp_path, "a")
        _manifest(tmp_path, f"version: 1\n{body}shims: []\n")

        with pytest.raises(ShimManifestError):
            read_shim_manifest(tmp_path)

    def test_the_floor_is_recorded_on_the_manifest(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=0.15.0'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.core == ">=0.15.0"

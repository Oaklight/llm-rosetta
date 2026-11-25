"""The ``llm-rosetta`` dispatcher, its command contract and ``ext`` (#1703 E7)."""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rosetta import cli
from llm_rosetta.contracts import SURFACE_VERSION, command
from llm_rosetta.contracts.command import (
    COMMAND_ENTRY_POINT_GROUP,
    CORE_COMMAND_NAMES,
    CommandConflict,
    CommandLoadError,
    CommandNameError,
    CommandSpec,
    UnknownCommand,
    command_catalog,
    register_command,
    registered_commands,
    resolve_command,
)
from llm_rosetta.extensions import _cli as ext_cli
from llm_rosetta.extensions import (
    ExtensionInstallError,
    _discovery,
    configure_extensions_dir,
)
from llm_rosetta.shims import list_shims
from llm_rosetta.shims._loader import _PLUGIN_SOURCES
from llm_rosetta.shims.provider_shim import unregister_shim


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Own the command registry, the entry points and the extension state."""
    saved = dict(command._REGISTRY)
    command._REGISTRY.clear()
    monkeypatch.setattr(command, "_entry_points", lambda: [])
    monkeypatch.setattr(_discovery, "_entry_points", lambda group: [])
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
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


def _entry(name: str, value: str, dist: str = "dist-a") -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value, dist=SimpleNamespace(name=dist))


def _module(tmp_path: Path, monkeypatch, name: str, body: str) -> str:
    """Write an importable module and return its name."""
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    return name


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #


class TestSpec:
    @pytest.mark.parametrize("name", ["", "Serve", "1x", "a_b", "a b", "-x"])
    def test_a_bad_name_is_refused(self, name):
        with pytest.raises(CommandNameError):
            CommandSpec(name, "s", "pkg.mod:fn")

    @pytest.mark.parametrize("target", ["", "pkg.mod", "pkg.mod:", ":fn", "a b:c"])
    def test_a_bad_target_is_refused(self, target):
        with pytest.raises(CommandNameError):
            CommandSpec("x", "s", target)

    def test_a_good_spec(self):
        spec = CommandSpec("my-cmd2", "s", "pkg.mod:fn")
        assert (spec.name, spec.target) == ("my-cmd2", "pkg.mod:fn")

    def test_the_surface_version_did_not_move(self):
        assert SURFACE_VERSION == 1

    def test_the_group_name(self):
        assert COMMAND_ENTRY_POINT_GROUP == "llm_rosetta.commands"


class TestRegistration:
    def test_a_core_name_cannot_be_taken(self):
        for name in CORE_COMMAND_NAMES:
            with pytest.raises(CommandConflict, match="llm-rosetta"):
                register_command(CommandSpec(name, "s", "pkg.mod:fn"), owner="evil")
        assert registered_commands() == ()

    def test_a_core_name_declared_as_an_entry_point_is_ignored_and_reported(
        self, monkeypatch
    ):
        monkeypatch.setattr(command, "_entry_points", lambda: [_entry("ext", "p.m:f")])

        catalog = command_catalog()

        assert "ext" not in catalog.commands
        assert any("'ext'" in problem for problem in catalog.problems)

    def test_two_owners_for_one_name_fail_naming_both(self):
        register_command(CommandSpec("x", "s", "a.m:f"), owner="owner-a")

        with pytest.raises(CommandConflict) as info:
            register_command(CommandSpec("x", "s", "b.m:f"), owner="owner-b")

        assert "owner-a" in str(info.value) and "owner-b" in str(info.value)
        assert registered_commands()[0].owner == "owner-a"

    def test_the_same_owner_and_target_is_a_no_op(self):
        spec = CommandSpec("x", "s", "a.m:f")
        register_command(spec, owner="owner-a")
        register_command(spec, owner="owner-a")
        assert len(registered_commands()) == 1

    def test_the_same_owner_with_another_target_is_a_conflict(self):
        register_command(CommandSpec("x", "s", "a.m:f"), owner="owner-a")
        with pytest.raises(CommandConflict):
            register_command(CommandSpec("x", "s", "a.m:g"), owner="owner-a")

    def test_the_owner_defaults_to_the_top_level_package(self):
        register_command(CommandSpec("x", "s", "pkg.sub.mod:fn"))
        assert registered_commands()[0].owner == "pkg"


class TestCatalog:
    def test_an_entry_point_and_the_same_owner_registering_it_agree(self, monkeypatch):
        monkeypatch.setattr(
            command, "_entry_points", lambda: [_entry("x", "a.m:f", "owner-a")]
        )
        register_command(CommandSpec("x", "s", "a.m:f"), owner="owner-a")

        catalog = command_catalog()

        assert catalog.conflicts == {}
        assert catalog.commands["x"].source == "registered"

    def test_an_entry_point_against_another_owner_is_a_conflict(self, monkeypatch):
        monkeypatch.setattr(
            command, "_entry_points", lambda: [_entry("x", "b.m:f", "owner-b")]
        )
        register_command(CommandSpec("x", "s", "a.m:f"), owner="owner-a")

        catalog = command_catalog()

        assert "x" not in catalog.commands
        assert (
            "owner-a" in catalog.conflicts["x"] and "owner-b" in catalog.conflicts["x"]
        )
        with pytest.raises(CommandConflict):
            resolve_command("x")

    def test_two_entry_points_with_one_name_conflict(self, monkeypatch):
        monkeypatch.setattr(
            command,
            "_entry_points",
            lambda: [_entry("x", "a.m:f", "dist-a"), _entry("x", "b.m:f", "dist-b")],
        )

        catalog = command_catalog()

        assert "dist-a" in catalog.conflicts["x"] and "dist-b" in catalog.conflicts["x"]

    def test_a_bad_entry_point_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(
            command,
            "_entry_points",
            lambda: [_entry("Bad_Name", "a.m:f"), _entry("good", "a.m:f")],
        )

        catalog = command_catalog()

        assert list(catalog.commands) == ["good"]
        assert len(catalog.problems) == 1


class TestResolve:
    def test_the_target_is_imported_only_when_resolved(self, tmp_path, monkeypatch):
        name = _module(
            tmp_path,
            monkeypatch,
            "lazy_cmd_mod",
            """
            def main(argv=None):
                return 7
            """,
        )
        register_command(CommandSpec("lazy", "s", f"{name}:main"), owner="t")

        assert name not in sys.modules
        command_catalog()
        assert name not in sys.modules
        assert cli.main(["help"]) == 0
        assert name not in sys.modules

        func = resolve_command("lazy")

        assert name in sys.modules
        assert func([]) == 7

    def test_dispatch_imports_only_the_named_command(self, tmp_path, monkeypatch):
        a = _module(
            tmp_path, monkeypatch, "cmd_mod_a", "def main(argv=None):\n return 0\n"
        )
        b = _module(
            tmp_path, monkeypatch, "cmd_mod_b", "def main(argv=None):\n return 5\n"
        )
        register_command(CommandSpec("aa", "s", f"{a}:main"), owner="t")
        register_command(CommandSpec("bb", "s", f"{b}:main"), owner="t")

        assert cli.main(["bb"]) == 5
        assert b in sys.modules and a not in sys.modules

    def test_the_arguments_after_the_name_reach_the_command(
        self, tmp_path, monkeypatch
    ):
        name = _module(
            tmp_path,
            monkeypatch,
            "cmd_args_mod",
            """
            SEEN = []
            def main(argv=None):
                SEEN.append(list(argv))
                return 0
            """,
        )
        register_command(CommandSpec("argcmd", "s", f"{name}:main"), owner="t")

        assert cli.main(["argcmd", "--flag", "v", "pos"]) == 0
        assert sys.modules[name].SEEN == [["--flag", "v", "pos"]]

    def test_an_unknown_name_raises_unknown_command(self):
        with pytest.raises(UnknownCommand):
            resolve_command("nope")

    def test_a_missing_module_raises_load_error_and_chains_the_cause(self):
        register_command(CommandSpec("x", "s", "no_such_module_xyz:main"), owner="t")
        with pytest.raises(CommandLoadError) as info:
            resolve_command("x")
        assert isinstance(info.value.__cause__, ImportError)

    def test_a_missing_attribute_raises_load_error(self, tmp_path, monkeypatch):
        name = _module(tmp_path, monkeypatch, "cmd_noattr", "X = 1\n")
        register_command(CommandSpec("x", "s", f"{name}:main"), owner="t")
        with pytest.raises(CommandLoadError, match="no 'main'"):
            resolve_command("x")

    def test_a_non_callable_target_raises_load_error(self, tmp_path, monkeypatch):
        name = _module(tmp_path, monkeypatch, "cmd_notcall", "main = 3\n")
        register_command(CommandSpec("x", "s", f"{name}:main"), owner="t")
        with pytest.raises(CommandLoadError, match="not callable"):
            resolve_command("x")


# --------------------------------------------------------------------------- #
# the dispatcher
# --------------------------------------------------------------------------- #


class TestHelp:
    def test_it_lists_core_and_registered_commands_with_owners(self, capsys):
        register_command(
            CommandSpec("serve", "Start it", "llm_rosetta_gateway.cli:serve_main"),
            owner="llm-rosetta-gateway",
        )

        assert cli.main([]) == 0

        out = capsys.readouterr().out
        for needle in ("ext", "help", "serve", "Start it", "llm-rosetta-gateway"):
            assert needle in out

    def test_no_arguments_and_help_print_the_same_list(self, capsys):
        cli.main([])
        first = capsys.readouterr().out
        cli.main(["help"])
        second = capsys.readouterr().out
        assert first == second

    def test_it_imports_no_command_module(self, monkeypatch, capsys):
        """A registered command's summary is in its spec; nothing is imported."""
        register_command(
            CommandSpec("serve", "Start it", "llm_rosetta_gateway.cli:serve_main"),
            owner="llm-rosetta-gateway",
        )
        seen: list[str] = []
        real = importlib.import_module
        monkeypatch.setattr(
            command.importlib,
            "import_module",
            lambda name, *a, **k: seen.append(name) or real(name, *a, **k),
        )
        before = {n for n in sys.modules if n.startswith("llm_rosetta_gateway")}

        assert cli.main(["help"]) == 0

        assert seen == []
        assert {n for n in sys.modules if n.startswith("llm_rosetta_gateway")} == before

    def test_an_entry_point_summary_is_the_docstring_first_line(
        self, tmp_path, monkeypatch, capsys
    ):
        name = _module(
            tmp_path,
            monkeypatch,
            "cmd_doc_mod",
            '''
            def main(argv=None):
                """Do the thing.

                More detail.
                """
                return 0
            ''',
        )
        monkeypatch.setattr(
            command, "_entry_points", lambda: [_entry("thing", f"{name}:main")]
        )

        cli.main(["help"])

        assert "Do the thing." in capsys.readouterr().out

    def test_a_conflicting_name_is_shown_as_unavailable(self, monkeypatch, capsys):
        monkeypatch.setattr(
            command,
            "_entry_points",
            lambda: [_entry("x", "a.m:f", "dist-a"), _entry("x", "b.m:f", "dist-b")],
        )

        cli.main(["help"])

        out = capsys.readouterr().out
        assert "unavailable" in out and "dist-b" in out

    def test_help_for_a_command_runs_it_with_help(self, tmp_path, monkeypatch):
        name = _module(
            tmp_path,
            monkeypatch,
            "cmd_help_mod",
            """
            SEEN = []
            def main(argv=None):
                SEEN.append(list(argv))
                return 0
            """,
        )
        register_command(CommandSpec("hh", "s", f"{name}:main"), owner="t")

        assert cli.main(["help", "hh"]) == 0
        assert sys.modules[name].SEEN == [["--help"]]

    def test_the_command_may_exit_with_system_exit(self, tmp_path, monkeypatch):
        name = _module(
            tmp_path,
            monkeypatch,
            "cmd_exit_mod",
            """
            def main(argv=None):
                raise SystemExit(4)
            """,
        )
        register_command(CommandSpec("ex", "s", f"{name}:main"), owner="t")

        assert cli.main(["ex"]) == 4


class TestUnknown:
    def test_exit_2_and_the_available_commands(self, capsys):
        register_command(CommandSpec("serve", "s", "a.m:f"), owner="t")

        assert cli.main(["frobnicate"]) == 2

        err = capsys.readouterr().err
        assert "frobnicate" in err
        assert "ext, help, serve" in err

    def test_without_the_gateway_it_says_where_the_commands_come_from(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(cli.importlib_util, "find_spec", lambda name: None)

        assert cli.main(["serve"]) == 2

        assert "llm-rosetta-gateway" in capsys.readouterr().err

    def test_with_the_gateway_installed_it_does_not_say_so(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.importlib_util, "find_spec", lambda name: object())

        assert cli.main(["serve"]) == 2

        assert "pip install" not in capsys.readouterr().err

    def test_a_conflicting_name_exits_2_naming_both_owners(self, monkeypatch, capsys):
        monkeypatch.setattr(
            command,
            "_entry_points",
            lambda: [_entry("x", "a.m:f", "dist-a"), _entry("x", "b.m:f", "dist-b")],
        )

        assert cli.main(["x"]) == 2

        err = capsys.readouterr().err
        assert "dist-a" in err and "dist-b" in err

    def test_a_broken_target_exits_1(self, capsys):
        register_command(CommandSpec("x", "s", "no_such_module_xyz:main"), owner="t")

        assert cli.main(["x"]) == 1
        assert "cannot import" in capsys.readouterr().err

    def test_the_version(self, capsys):
        assert cli.main(["--version"]) == 0
        assert capsys.readouterr().out.startswith("llm-rosetta ")


def test_python_dash_m_prints_the_list_and_starts_nothing():
    result = subprocess.run(
        [sys.executable, "-m", "llm_rosetta"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "commands:" in result.stdout and "ext" in result.stdout


# --------------------------------------------------------------------------- #
# ext
# --------------------------------------------------------------------------- #


def _manifest(ext_id="acme", version="0.1.0", surface=None, requires="", external=""):
    surface = SURFACE_VERSION if surface is None else surface
    return (
        f"id: {ext_id}\nname: {ext_id} test\nversion: {version}\n"
        f"maturity: experimental\nrequires:\n  surface: {surface}\n  ir: 1\n"
        f"{requires}provides:\n  shims: [{ext_id}--chat_completions]\n{external}"
    )


def _wheel(tmp_path: Path, ext_id="acme", version="0.1.0", manifest=None) -> Path:
    path = tmp_path / f"{ext_id}-{version}-py3-none-any.whl"
    base = f"llm_rosetta_ext/{ext_id}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}/plugin.yaml", manifest or _manifest(ext_id, version))
        zf.writestr(
            f"{base}/provider.yaml", "connection:\n  base_url: https://x.invalid\n"
        )
        zf.writestr(
            f"{base}/llm/chat_completions/provider.yaml",
            f"name: {ext_id}--chat_completions\nmodel_type: llm\n"
            "format: chat_completions\n",
        )
        zf.writestr(f"{ext_id}-{version}.dist-info/METADATA", "Name: x\n")
        zf.writestr(f"{ext_id}-{version}.dist-info/RECORD", f"{base}/plugin.yaml,,\n")
    return path


def _place(directory: Path, ext_id: str, manifest: str) -> None:
    """Put an extension folder straight into a directory (no installer checks)."""
    folder = directory / "llm_rosetta_ext" / ext_id
    folder.mkdir(parents=True)
    (folder / "plugin.yaml").write_text(manifest, encoding="utf-8")
    (folder / "__init__.py").write_text("", encoding="utf-8")
    (folder / "provider.yaml").write_text(
        "connection:\n  base_url: https://x.invalid\n", encoding="utf-8"
    )
    lane = folder / "llm" / "chat_completions"
    lane.mkdir(parents=True)
    (lane / "provider.yaml").write_text(
        f"name: {ext_id}--chat_completions\nmodel_type: llm\n"
        "format: chat_completions\n",
        encoding="utf-8",
    )


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "exts"
    path.mkdir()
    return path


def _ext(*argv: str) -> int:
    return cli.main(["ext", *argv])


class TestExtInstallListRemove:
    def test_install_a_wheel_then_list_then_remove(self, tmp_path, home, capsys):
        wheel = _wheel(tmp_path)

        assert _ext("install", str(wheel), "--dir", str(home)) == 0
        assert "installed acme 0.1.0" in capsys.readouterr().out

        assert _ext("list", "--dir", str(home)) == 0
        row = next(
            line for line in capsys.readouterr().out.splitlines() if "acme" in line
        )
        assert "loaded" in row and "0.1.0" in row and "namespace" in row

        assert _ext("remove", "acme", "--dir", str(home)) == 0
        assert not (home / "llm_rosetta_ext" / "acme").exists()
        capsys.readouterr()

        assert _ext("list", "--dir", str(home)) == 0
        assert "acme" not in capsys.readouterr().out

    def test_install_checks_a_given_sha256(self, tmp_path, home, capsys):
        wheel = _wheel(tmp_path)

        assert (
            _ext("install", str(wheel), "--sha256", "0" * 64, "--dir", str(home)) == 1
        )
        assert "sha256 mismatch" in capsys.readouterr().err
        assert not (home / "llm_rosetta_ext").exists()

    def test_install_accepts_the_right_sha256(self, tmp_path, home):
        import hashlib

        wheel = _wheel(tmp_path)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()

        assert _ext("install", str(wheel), "--sha256", digest, "--dir", str(home)) == 0

    def test_a_url_needs_a_sha256(self, home, capsys):
        with pytest.raises(SystemExit) as info:
            ext_cli.ext_main(
                ["install", "https://example.invalid/a.whl", "--dir", str(home)]
            )
        assert info.value.code == 2
        assert _ext("install", "https://example.invalid/a.whl", "--dir", str(home)) == 2
        assert "--sha256" in capsys.readouterr().err

    def test_a_url_with_a_sha256_goes_to_the_url_installer(
        self, home, monkeypatch, capsys
    ):
        seen = {}

        def fake(url, *, sha256, dest=None, timeout=30.0):
            seen.update(url=url, sha256=sha256)
            return SimpleNamespace(id="acme", version="1", path=home)

        monkeypatch.setattr(ext_cli, "install_wheel_url", fake)

        assert (
            _ext(
                "install",
                "https://h.invalid/a.whl",
                "--sha256",
                "ab",
                "--dir",
                str(home),
            )
            == 0
        )
        assert seen == {"url": "https://h.invalid/a.whl", "sha256": "ab"}

    def test_a_missing_file_exits_1(self, tmp_path, home, capsys):
        assert _ext("install", str(tmp_path / "none.whl"), "--dir", str(home)) == 1
        assert "cannot read" in capsys.readouterr().err

    def test_a_directory_instead_of_a_wheel_exits_1(self, tmp_path, home, capsys):
        assert _ext("install", str(tmp_path), "--dir", str(home)) == 1
        assert "cannot read" in capsys.readouterr().err

    def test_a_bad_wheel_exits_1(self, tmp_path, home, capsys):
        bad = tmp_path / "bad.whl"
        bad.write_bytes(b"not a zip")

        assert _ext("install", str(bad), "--dir", str(home)) == 1
        assert "not a valid wheel" in capsys.readouterr().err

    def test_pip_without_pip_exits_2_with_the_message(self, home, monkeypatch, capsys):
        monkeypatch.setattr(ext_cli, "pip_available", lambda: False)

        assert _ext("install", "--pip", "acme", "--dir", str(home)) == 2
        assert "pip is not available" in capsys.readouterr().err

    def test_pip_with_pip_installs_what_it_built(self, home, monkeypatch, capsys):
        seen = {}

        class Installed:
            id = "acme"
            version = "1"
            path = home / "llm_rosetta_ext" / "acme"

        monkeypatch.setattr(ext_cli, "pip_available", lambda: True)
        monkeypatch.setattr(
            ext_cli,
            "install_with_pip",
            lambda spec: seen.update(spec=spec) or Installed(),
        )

        assert _ext("install", "--pip", "acme==1", "--dir", str(home)) == 0
        assert seen == {"spec": "acme==1"}
        assert "installed acme 1" in capsys.readouterr().out

    def test_a_pip_failure_exits_1_with_its_message(self, home, monkeypatch, capsys):
        def boom(spec):
            raise ExtensionInstallError("pip could not build a wheel for 'x'")

        monkeypatch.setattr(ext_cli, "pip_available", lambda: True)
        monkeypatch.setattr(ext_cli, "install_with_pip", boom)

        assert _ext("install", "--pip", "x", "--dir", str(home)) == 1
        assert "could not build a wheel" in capsys.readouterr().err

    def test_install_needs_a_source(self, home):
        assert _ext("install", "--dir", str(home)) == 2

    def test_a_wheel_and_pip_together_are_refused(self, home, tmp_path):
        assert (
            _ext("install", str(_wheel(tmp_path)), "--pip", "x", "--dir", str(home))
            == 2
        )

    def test_remove_of_an_absent_extension_exits_1(self, home, capsys):
        assert _ext("remove", "ghost", "--dir", str(home)) == 1
        assert "not installed" in capsys.readouterr().err

    def test_remove_refuses_a_path_as_an_id(self, home, capsys):
        assert _ext("remove", "../x", "--dir", str(home)) == 1

    def test_list_shows_an_installed_one_that_cannot_load(self, home, capsys):
        _place(home, "broken", "id: [unterminated\n")

        assert _ext("list", "--dir", str(home)) == 0

        row = next(
            line for line in capsys.readouterr().out.splitlines() if "broken" in line
        )
        assert "failed" in row or "not loadable" in row

    def test_list_of_an_empty_directory(self, home, capsys):
        assert _ext("list", "--dir", str(home)) == 0
        assert "no extensions found" in capsys.readouterr().out

    def test_the_dir_flag_does_not_stick(self, home):
        from llm_rosetta.extensions import extensions_dir

        _ext("list", "--dir", str(home))
        assert extensions_dir() != home


class TestExtDoctor:
    def test_a_sound_extension_exits_0(self, home, capsys):
        _place(home, "acme", _manifest("acme"))

        assert _ext("doctor", "--dir", str(home)) == 0
        assert "ok" in capsys.readouterr().out

    def test_nothing_installed_exits_0(self, home):
        assert _ext("doctor", "--dir", str(home)) == 0

    def test_an_incompatible_manifest_exits_1(self, home, capsys):
        _place(home, "acme", _manifest("acme", surface=SURFACE_VERSION + 5))

        assert _ext("doctor", "--dir", str(home)) == 1
        out = capsys.readouterr().out
        assert "incompatible" in out and "surface" in out

    def test_a_missing_dependency_exits_1(self, home, capsys):
        _place(
            home,
            "acme",
            _manifest("acme", requires="  extensions: [base_ext]\n"),
        )

        assert _ext("doctor", "--dir", str(home)) == 1
        assert "'base_ext'" in capsys.readouterr().out

    def test_a_present_dependency_is_fine(self, home):
        _place(home, "base_ext", _manifest("base_ext"))
        _place(
            home,
            "acme",
            _manifest("acme", requires="  extensions: [base_ext]\n"),
        )

        assert _ext("doctor", "--dir", str(home)) == 0

    def _external(
        self, check: str, minimum: str | None = None, hint: str | None = None
    ):
        lines = f"external:\n  - name: tool\n    check: {check}\n"
        if minimum:
            lines += f'    minimum: "{minimum}"\n'
        if hint:
            lines += f'    install_hint: "{hint}"\n'
        return lines

    def test_an_external_program_that_is_ok(self, home, capsys):
        ext = self._external(
            f"[{sys.executable!r}, '-c', \"print('tool 3.1.4')\"]", "3.0"
        )
        _place(home, "acme", _manifest("acme", external=ext))

        assert _ext("doctor", "--dir", str(home)) == 0
        assert "external tool: ok 3.1.4" in capsys.readouterr().out

    def test_an_external_program_that_is_too_old_shows_the_hint(self, home, capsys):
        ext = self._external(
            f"[{sys.executable!r}, '-c', \"print('tool 1.0.0')\"]",
            "2.0",
            "pip install tool",
        )
        _place(home, "acme", _manifest("acme", external=ext))

        assert _ext("doctor", "--dir", str(home)) == 1
        out = capsys.readouterr().out
        assert "too_old" in out and "pip install tool" in out

    def test_an_external_check_that_cannot_run_is_a_problem(
        self, home, monkeypatch, capsys
    ):
        ext = self._external("[tool]", "2.0")
        _place(home, "acme", _manifest("acme", external=ext))

        def refuse(program):
            raise ValueError("minimum 'x' is not a dotted version")

        monkeypatch.setattr(ext_cli, "check_external", refuse)

        assert _ext("doctor", "--dir", str(home)) == 1
        assert "dotted version" in capsys.readouterr().out

    def test_an_external_program_that_is_missing(self, home, capsys):
        ext = self._external(
            "[no-such-program-xyz-123]", None, "get it from the vendor"
        )
        _place(home, "acme", _manifest("acme", external=ext))

        assert _ext("doctor", "--dir", str(home)) == 1
        out = capsys.readouterr().out
        assert "missing" in out and "get it from the vendor" in out

    def test_an_external_program_with_no_version_is_unknown(self, home, capsys):
        ext = self._external(f"[{sys.executable!r}, '-c', \"print('hello')\"]", "2.0")
        _place(home, "acme", _manifest("acme", external=ext))

        assert _ext("doctor", "--dir", str(home)) == 1
        assert "unknown" in capsys.readouterr().out

    def test_a_not_loaded_extension_gives_its_reason(self, home, capsys):
        _place(home, "acme", _manifest("other_id"))  # id does not match the folder

        assert _ext("doctor", "--dir", str(home)) == 1
        assert "not loaded" in capsys.readouterr().out

    def test_an_unreadable_manifest_exits_1(self, home, capsys):
        _place(home, "acme", "id: [unterminated\n")

        assert _ext("doctor", "--dir", str(home)) == 1

    def test_one_id_checks_only_that_one(self, home, capsys):
        _place(home, "good", _manifest("good"))
        _place(home, "bad", _manifest("bad", surface=SURFACE_VERSION + 5))

        assert _ext("doctor", "good", "--dir", str(home)) == 0
        out = capsys.readouterr().out
        assert "good" in out and "bad" not in out
        assert _ext("doctor", "bad", "--dir", str(home)) == 1

    def test_an_unknown_id_exits_1(self, home, capsys):
        assert _ext("doctor", "ghost", "--dir", str(home)) == 1
        assert "no extension 'ghost'" in capsys.readouterr().err


class TestExtScaffold:
    def test_new_then_check(self, tmp_path, capsys):
        assert _ext("new", "acme_free", "--dest", str(tmp_path / "packages")) == 0
        folder = tmp_path / "packages" / "acme_free"
        assert (folder / "plugin.yaml").is_file()

        assert _ext("check", "--no-build", str(folder)) == 0

    def test_new_refuses_a_bad_id(self, tmp_path, capsys):
        assert _ext("new", "Not-An-Id", "--dest", str(tmp_path)) != 0

    def test_check_reports_a_broken_folder(self, tmp_path):
        _ext("new", "acme_free", "--dest", str(tmp_path))
        (tmp_path / "acme_free" / "plugin.yaml").write_text(
            "id: [x\n", encoding="utf-8"
        )

        assert _ext("check", "--no-build", str(tmp_path / "acme_free")) != 0

    def test_ext_without_an_action_prints_help_and_exits_2(self, capsys):
        assert _ext() == 2
        assert "usage" in capsys.readouterr().out

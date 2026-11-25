"""An extension adds commands to ``llm-rosetta <area> <verb>`` (#1704)."""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from llm_rosetta import cli
from llm_rosetta.contracts import SURFACE_VERSION, command
from llm_rosetta.contracts.extension import (
    command_area,
    parse_extension_manifest,
    validate_extension_manifest,
)
from llm_rosetta.devtools import ext_scaffold
from llm_rosetta.extensions import (
    _commands,
    _discovery,
    configure_extensions_dir,
    extension_areas,
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
    monkeypatch.delenv(cli.DEBUG_ENV, raising=False)
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


SUMMARIES = {
    "run": "Print the arguments.",
    "boom": "Fail with an unhandled exception.",
    "leave": "Exit with a message.",
    "pending": "Return a code of its own.",
}


def _manifest(ext_id: str, verbs=("run",), surface=None, shims=False) -> str:
    surface = SURFACE_VERSION if surface is None else surface
    provides = "  commands:\n" + "".join(
        f'    {verb}: {{ en: "{SUMMARIES.get(verb, verb)}" }}\n' for verb in verbs
    )
    if shims:
        provides = f"  shims: [{ext_id}--chat_completions]\n" + provides
    return (
        f"id: {ext_id}\nname: {ext_id} test\nversion: 0.1.0\n"
        f"maturity: experimental\nrequires:\n  surface: {surface}\n  ir: 1\n"
        f"provides:\n{provides}"
        'summary:\n  en: "A test extension."\n'
    )


COMMANDS_PY = '''
"""Test commands."""
import sys


def run(argv):
    """Print the arguments."""
    print("ran", *argv)
    return 0


def boom(argv):
    """Fail with an unhandled exception."""
    raise RuntimeError("it broke")


def leave(argv):
    """Exit with a message."""
    raise SystemExit("leaving")


def pending(argv):
    """Return a code of its own."""
    return 3


COMMANDS = {"run": run, "boom": boom, "leave": leave, "pending": pending}
'''


def _place(
    directory: Path, ext_id: str, manifest: str, commands: str | None = COMMANDS_PY
) -> Path:
    folder = directory / "llm_rosetta_ext" / ext_id
    folder.mkdir(parents=True)
    (folder / "plugin.yaml").write_text(manifest, encoding="utf-8")
    (folder / "__init__.py").write_text("", encoding="utf-8")
    if commands is not None:
        (folder / "commands.py").write_text(textwrap.dedent(commands), encoding="utf-8")
    return folder


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "exts"
    path.mkdir()
    configure_extensions_dir(path)
    return path


ALL_VERBS = ("run", "boom", "leave", "pending")


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def _document(**provides):
    return {
        "id": "acme_tool",
        "name": "Acme",
        "version": "0.1.0",
        "maturity": "experimental",
        "requires": {"surface": SURFACE_VERSION, "ir": 1},
        "provides": provides,
    }


class TestManifest:
    def test_an_extension_may_provide_only_commands(self):
        manifest = parse_extension_manifest(
            _document(commands={"run-it": {"en": "Run it.", "zh": "运行。"}})
        )
        assert manifest.commands == {"run-it": {"en": "Run it.", "zh": "运行。"}}
        assert manifest.shims == () and manifest.transports == ()

    def test_the_area_spells_underscores_as_hyphens(self):
        assert command_area("claude_cli") == "claude-cli"
        assert command_area("argo") == "argo"

    @pytest.mark.parametrize("verb", ["Run", "run_it", "1run", "", "run it"])
    def test_a_bad_verb_is_refused(self, verb):
        assert validate_extension_manifest(_document(commands={verb: {"en": "x"}}))

    def test_a_verb_needs_an_english_summary(self):
        assert validate_extension_manifest(_document(commands={"run": {"zh": "x"}}))
        assert validate_extension_manifest(_document(commands={"run": "x"}))

    def test_a_list_of_verbs_is_refused(self):
        assert validate_extension_manifest(_document(commands=["run"]))

    def test_an_extension_cannot_take_a_core_command_name(self):
        document = _document(commands={"run": {"en": "x"}})
        document["id"] = "help"
        problems = validate_extension_manifest(document)
        assert any("core command name" in p for p in problems), problems

    def test_nothing_provided_is_still_refused(self):
        assert validate_extension_manifest(_document())


# --------------------------------------------------------------------------- #
# areas, read from manifests only
# --------------------------------------------------------------------------- #


class TestAreas:
    def test_an_area_is_read_without_importing_the_extension(self, home):
        _place(home, "acme_tool", _manifest("acme_tool", ALL_VERBS))
        areas, problems = _areas_after_activation(home)
        assert problems == []
        area = areas["acme-tool"]
        assert area.ext_id == "acme_tool"
        assert area.verbs == ALL_VERBS
        assert area.summary == "A test extension."
        assert area.summaries["run"] == "Print the arguments."
        assert "llm_rosetta_ext.acme_tool.commands" not in sys.modules

    def test_an_extension_without_commands_owns_no_area(self, home):
        folder = _place(home, "plain", _manifest("plain", ("x",), shims=True))
        text = (folder / "plugin.yaml").read_text()
        text = text[: text.index("  commands:")]
        (folder / "plugin.yaml").write_text(text)
        areas, _ = _areas_after_activation(home)
        assert areas == {}

    def test_a_bad_manifest_is_reported_and_the_rest_still_work(self, home):
        _place(home, "good", _manifest("good"))
        _place(home, "broken", "id: broken\n")
        areas, problems = _areas_after_activation(home)
        assert "good" in areas
        assert any("broken" in p for p in problems), problems

    def test_an_incompatible_extension_has_an_area_that_will_not_run(self, home):
        _place(home, "old", _manifest("old", surface=SURFACE_VERSION + 1))
        areas, _ = _areas_after_activation(home)
        assert areas["old"].problem


def _areas_after_activation(home: Path):
    from llm_rosetta.extensions import activate_extensions_dir

    activate_extensions_dir(home)
    return extension_areas()


# --------------------------------------------------------------------------- #
# running a verb
# --------------------------------------------------------------------------- #


class TestRun:
    @pytest.fixture(autouse=True)
    def _ext(self, home):
        _place(home, "acme_tool", _manifest("acme_tool", ALL_VERBS))

    def test_the_arguments_after_the_verb_reach_the_command(self, capsys):
        assert cli.main(["acme-tool", "run", "a", "--b"]) == 0
        assert capsys.readouterr().out.strip() == "ran a --b"

    def test_no_verb_prints_the_verbs_and_exits_2(self, capsys):
        assert cli.main(["acme-tool"]) == 2
        out = capsys.readouterr().out
        assert "run" in out and "Print the arguments." in out

    @pytest.mark.parametrize("flag", ["--help", "-h", "help"])
    def test_help_for_the_area_exits_0(self, flag, capsys):
        assert cli.main(["acme-tool", flag]) == 0
        assert "Fail with an unhandled exception." in capsys.readouterr().out

    def test_help_command_for_the_area(self, capsys):
        assert cli.main(["help", "acme-tool"]) == 0
        assert "Print the arguments." in capsys.readouterr().out

    def test_an_unknown_verb_exits_2_and_lists_the_verbs(self, capsys):
        assert cli.main(["acme-tool", "nope"]) == 2
        err = capsys.readouterr().err
        assert "unknown command 'nope'" in err and "run, boom" in err

    def test_an_unhandled_exception_is_one_line_and_exit_1(self, capsys):
        assert cli.main(["acme-tool", "boom"]) == 1
        err = capsys.readouterr().err.strip()
        assert err == "llm-rosetta acme-tool boom: it broke"

    def test_the_debug_variable_lets_it_raise(self, monkeypatch):
        monkeypatch.setenv(cli.DEBUG_ENV, "1")
        with pytest.raises(RuntimeError, match="it broke"):
            cli.main(["acme-tool", "boom"])

    def test_a_system_exit_with_a_message_is_exit_1(self, capsys):
        assert cli.main(["acme-tool", "leave"]) == 1
        assert "leaving" in capsys.readouterr().err

    def test_a_code_of_its_own_is_kept(self):
        assert cli.main(["acme-tool", "pending"]) == 3

    def test_the_area_is_listed_by_help_with_its_owner(self, capsys):
        assert cli.main(["help"]) == 0
        out = capsys.readouterr().out
        assert "acme-tool" in out and "A test extension." in out
        assert "[extension acme_tool]" in out

    def test_an_unknown_command_lists_the_area(self, capsys):
        assert cli.main(["zzz"]) == 2
        assert "acme-tool" in capsys.readouterr().err


class TestFailures:
    def test_a_missing_commands_module_exits_1_with_the_reason(self, home, capsys):
        _place(home, "bare", _manifest("bare"), commands=None)
        assert cli.main(["bare", "run"]) == 1
        assert "commands module cannot be imported" in capsys.readouterr().err

    def test_a_module_without_the_mapping_exits_1(self, home, capsys):
        _place(home, "nomap", _manifest("nomap"), commands="x = 1\n")
        assert cli.main(["nomap", "run"]) == 1
        assert "COMMANDS" in capsys.readouterr().err

    def test_a_listed_verb_with_no_callable_exits_1(self, home, capsys):
        _place(home, "hole", _manifest("hole"), commands="COMMANDS = {}\n")
        assert cli.main(["hole", "run"]) == 1
        assert "no callable" in capsys.readouterr().err

    def test_an_incompatible_extension_exits_1(self, home, capsys):
        _place(home, "old", _manifest("old", surface=SURFACE_VERSION + 1))
        assert cli.main(["old", "run"]) == 1
        assert "surface" in capsys.readouterr().err

    def test_help_lists_the_verbs_without_importing_the_module(self, home, capsys):
        _place(home, "bare", _manifest("bare"), commands=None)
        assert cli.main(["help", "bare"]) == 0
        assert "run  Print the arguments." in capsys.readouterr().out
        assert "llm_rosetta_ext.bare.commands" not in sys.modules

    def test_the_summary_does_not_come_from_a_docstring(self, home, capsys):
        _place(
            home,
            "nodoc",
            _manifest("nodoc"),
            commands="def run(argv):\n    return 0\nCOMMANDS = {'run': run}\n",
        )
        assert cli.main(["help", "nodoc"]) == 0
        assert "run  Print the arguments." in capsys.readouterr().out


class TestOrder:
    def test_a_registered_command_wins_over_an_area_of_the_same_name(
        self, home, capsys
    ):
        _place(home, "serve", _manifest("serve"))
        command.register_command(
            command.CommandSpec("serve", "Start it", f"{__name__}:_serve")
        )
        assert cli.main(["serve", "go"]) == 0
        assert "served go" in capsys.readouterr().out
        cli.main(["help"])
        assert "[extension serve]" not in capsys.readouterr().out

    def test_dispatching_a_registered_command_scans_no_extension(self, monkeypatch):
        command.register_command(
            command.CommandSpec("serve", "Start it", f"{__name__}:_serve")
        )

        def boom():
            raise AssertionError("scanned the extensions")

        monkeypatch.setattr(cli, "_areas", boom)
        assert cli.main(["serve"]) == 0

    def test_help_survives_a_failing_scan(self, monkeypatch, capsys):
        monkeypatch.setattr(
            _commands, "discover", lambda: (_ for _ in ()).throw(OSError("denied"))
        )
        assert cli.main(["help"]) == 0
        assert "ext" in capsys.readouterr().out


def _serve(argv=None):
    """Start it."""
    print("served", *(argv or ()))
    return 0


# --------------------------------------------------------------------------- #
# the scaffold and the check
# --------------------------------------------------------------------------- #


class TestScaffold:
    def test_new_with_a_command_makes_a_folder_that_checks_and_runs(
        self, tmp_path, home, capsys
    ):
        assert (
            cli.main(
                [
                    "ext",
                    "new",
                    "acme_tool",
                    "--command",
                    "run-it",
                    "--dest",
                    str(home / "llm_rosetta_ext"),
                ]
            )
            == 0
        )
        capsys.readouterr()
        folder = home / "llm_rosetta_ext" / "acme_tool"
        assert cli.main(["ext", "check", str(folder), "--no-build"]) == 0
        assert cli.main(["acme-tool", "run-it", "x"]) == 0
        assert "acme_tool run-it x" in capsys.readouterr().out

    def test_a_bad_verb_is_refused(self, tmp_path, capsys):
        assert (
            cli.main(
                [
                    "ext",
                    "new",
                    "acme_tool",
                    "--command",
                    "Run_It",
                    "--dest",
                    str(tmp_path),
                ]
            )
            == 2
        )
        assert "verb" in capsys.readouterr().err


class TestCheck:
    def _folder(self, tmp_path, commands: str | None) -> Path:
        files = ext_scaffold.render_extension(
            "acme_tool", format_="chat_completions", command="run-it"
        )
        folder = tmp_path / "acme_tool"
        for relative, text in files.items():
            if relative == "commands.py":
                if commands is None:
                    continue
                text = commands
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return folder

    def test_a_missing_commands_module_is_a_problem(self, tmp_path):
        result = ext_scaffold.check_extension(self._folder(tmp_path, None), build=False)
        assert any("no commands module" in p for p in result.problems), result.problems

    def test_a_module_without_the_mapping_is_a_problem(self, tmp_path):
        result = ext_scaffold.check_extension(
            self._folder(tmp_path, "x = 1\n"), build=False
        )
        assert any("no COMMANDS mapping" in p for p in result.problems)

    def test_a_mapping_that_disagrees_with_the_manifest_is_a_problem(self, tmp_path):
        result = ext_scaffold.check_extension(
            self._folder(tmp_path, "COMMANDS = {'other': print}\n"), build=False
        )
        assert any("provides.commands lists" in p for p in result.problems)

"""The gateway's commands for the ``llm-rosetta`` dispatcher (#1703 E7).

``llm-rosetta-gateway [args]`` is ``llm-rosetta serve [args]`` when the
arguments are empty or start with an option, and the named command otherwise.
Every invocation the 0.13.0 CLI accepted keeps working.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta import cli as core_cli
from llm_rosetta.api import CommandSpec, register_command, resolve_command
from llm_rosetta.contracts import command as command_contract
from llm_rosetta.contracts.command import (
    CommandConflict,
    command_catalog,
    registered_commands,
)
from llm_rosetta_gateway import cli
from llm_rosetta_gateway.maintenance import migrate_0140
from llm_rosetta_gateway.observability.usage import cli as usage_cli

PYPROJECT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "llm-rosetta-gateway"
    / "pyproject.toml"
)
EXPECTED = ("serve", "init", "add", "credential", "password", "db", "usage", "migrate")


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """Start from no registrations and no installed package metadata."""
    saved = dict(command_contract._REGISTRY)
    command_contract._REGISTRY.clear()
    monkeypatch.setattr(command_contract, "_entry_points", lambda: [])
    yield
    command_contract._REGISTRY.clear()
    command_contract._REGISTRY.update(saved)


def _declared_entry_points() -> list[SimpleNamespace]:
    """The ``llm_rosetta.commands`` entries of the gateway's pyproject."""
    text = PYPROJECT.read_text(encoding="utf-8")
    section = text.split('[project.entry-points."llm_rosetta.commands"]', 1)[1]
    section = re.split(r"^\[", section, maxsplit=1, flags=re.M)[0]
    found = re.findall(r'^([a-z][a-z0-9-]*)\s*=\s*"([^"]+)"', section, flags=re.M)
    dist = SimpleNamespace(name="llm-rosetta-gateway")
    return [SimpleNamespace(name=n, value=v, dist=dist) for n, v in found]


@pytest.fixture
def served(monkeypatch, tmp_path):
    """A config file, with the server's three collaborators replaced.

    Yields a dict that records what the server start received.
    """
    config_file = tmp_path / "config.jsonc"
    config_file.write_text("{}\n", encoding="utf-8")
    seen: dict[str, Any] = {"config_file": str(config_file)}
    fake = SimpleNamespace(
        verbose=False,
        log_bodies=False,
        log_format="text",
        host="cfg-host",
        port=1234,
        socket=None,
        tls_cert=None,
        tls_key=None,
        providers={},
        models={},
    )

    def load(path, args):
        seen["load"] = (path, args.config, args.proxy, args.data_dir)
        return fake, None

    async def create_app(config, *, config_path, shim_source_report):
        seen["app"] = (config, config_path)
        return "app"

    async def run_gateway(app, host, port, *, socket, ssl_context):
        seen["run"] = (app, host, port, socket, ssl_context)

    import llm_rosetta_gateway.app as app_module

    monkeypatch.setattr(cli, "_load_startup_config", load)
    monkeypatch.setattr(cli, "_report_shim_sources", lambda report: None)
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(app_module, "create_app", create_app)
    monkeypatch.setattr(app_module, "run_gateway", run_gateway)
    return seen


# --------------------------------------------------------------------------- #
# the server start
# --------------------------------------------------------------------------- #


class TestTheServerStart:
    def test_the_documented_invocation_reaches_the_server(self, served):
        code = cli.main(
            ["--config", served["config_file"], "--host", "Y", "--no-banner"]
        )

        assert code == 0
        assert served["load"][:2] == (served["config_file"], served["config_file"])
        assert served["app"] == (served["app"][0], served["config_file"])
        assert served["run"] == ("app", "Y", 1234, None, None)

    def test_the_port_and_the_socket_pass_through(self, served):
        cli.main(
            [
                "-c",
                served["config_file"],
                "--port",
                "99",
                "-S",
                "/x.sock",
                "--no-banner",
            ]
        )

        assert served["run"] == ("app", "cfg-host", 99, "/x.sock", None)

    def test_the_overrides_reach_the_config_loader(self, served):
        cli.main(
            [
                "--config",
                served["config_file"],
                "--proxy",
                "socks5://p",
                "--data-dir",
                "/d",
                "--no-banner",
            ]
        )

        assert served["load"] == (
            served["config_file"],
            served["config_file"],
            "socks5://p",
            "/d",
        )

    def test_the_explicit_serve_command_is_the_same_start(self, served):
        cli.main(
            ["serve", "--config", served["config_file"], "--host", "Y", "--no-banner"]
        )
        explicit = dict(served)
        served.pop("run")
        cli.main(["--config", served["config_file"], "--host", "Y", "--no-banner"])

        assert explicit["run"] == served["run"]

    def test_options_before_the_serve_word_are_kept(self, served):
        cli.main(
            ["--no-banner", "--host", "Y", "serve", "--config", served["config_file"]]
        )

        assert served["run"][1] == "Y"

    def test_no_config_is_exit_1(self, monkeypatch, served, tmp_path):
        monkeypatch.setattr(cli, "discover_config", lambda path: None)

        with pytest.raises(SystemExit) as info:
            cli.main(["--no-banner"])

        assert info.value.code == 1
        assert "run" not in served

    def test_edit_opens_the_editor_and_does_not_serve(self, monkeypatch, served):
        opened = []
        monkeypatch.setattr(cli, "_open_in_editor", lambda path: opened.append(path))

        assert cli.main(["--edit", "--config", "c.jsonc"]) == 0
        assert opened == ["c.jsonc"] and "run" not in served

    def test_the_version_flag(self, capsys):
        with pytest.raises(SystemExit) as info:
            cli.main(["--version"])

        assert info.value.code == 0
        assert capsys.readouterr().out.startswith("llm-rosetta-gateway ")

    def test_an_unknown_command_is_exit_2_with_the_choices(self, capsys):
        assert cli.main(["frobnicate"]) == 2
        err = capsys.readouterr().err
        assert "frobnicate" in err and "credential" in err

    def test_options_before_an_unknown_command_are_refused(self, capsys):
        with pytest.raises(SystemExit) as info:
            cli.main(["--config", "x.jsonc", "frobnicate"])

        assert info.value.code == 2
        assert "options before 'frobnicate'" in capsys.readouterr().err

    def test_an_installed_extension_command_runs_from_the_gateway_binary(
        self, tmp_path, monkeypatch, capsys
    ):
        """The Nuitka binary's only entry is this ``main``: ``<bin> acme run``."""
        import importlib
        import sys

        from llm_rosetta.api import configure_extensions_dir

        folder = tmp_path / "llm_rosetta_ext" / "acme"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("", encoding="utf-8")
        (folder / "plugin.yaml").write_text(
            "id: acme\nname: acme\nversion: 0.1.0\nmaturity: experimental\n"
            "requires:\n  surface: 1\n  ir: 1\nprovides:\n  commands:\n    run: { en: x }\n",
            encoding="utf-8",
        )
        (folder / "commands.py").write_text(
            "def run(argv):\n    print('ran', *argv)\n    return 0\n"
            "COMMANDS = {'run': run}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "path", list(sys.path))
        configure_extensions_dir(tmp_path)
        try:
            assert cli.main(["acme", "run", "x"]) == 0
            assert capsys.readouterr().out.strip() == "ran x"
        finally:
            configure_extensions_dir(None)
            for name in [
                n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"
            ]:
                del sys.modules[name]
            importlib.invalidate_caches()

    def test_a_word_after_double_dash_is_not_a_command(self, tmp_path):
        target = tmp_path / "config.jsonc"

        with pytest.raises(SystemExit) as info:
            cli.main(["--config", str(target), "--", "init"])

        assert info.value.code == 2
        assert not target.exists()

    def test_the_shims_scaffold_commands_are_gone(self, capsys):
        for word in ("shims",):
            assert cli.main([word, "new", "acme"]) == 2
            assert word in capsys.readouterr().err


class TestCoreCommandsFromTheGatewayEntry:
    """The Nuitka binary has only this entry point, and no ``llm-rosetta`` script.

    An operator without Python or pip installs an extension through it, so the
    core-owned commands must be reachable here.
    """

    def test_ext_install_and_list_work(self, tmp_path, capsys):
        from _ext_wheels import build_wheel

        wheel = tmp_path / "acme-0.1.0-py3-none-any.whl"
        wheel.write_bytes(build_wheel("acme", "0.1.0", ["acme--chat_completions"]))
        target = tmp_path / "extensions"

        assert cli.main(["ext", "install", str(wheel), "--dir", str(target)]) == 0
        assert (target / "llm_rosetta_ext" / "acme" / "plugin.yaml").is_file()
        capsys.readouterr()
        assert cli.main(["ext", "list", "--dir", str(target)]) == 0
        assert "acme" in capsys.readouterr().out

    def test_help_lists_the_gateways_commands_and_the_core_ones(self, capsys):
        assert cli.main(["help"]) == 0
        out = capsys.readouterr().out
        for word in ("serve", "credential", "ext", "help"):
            assert word in out

    def test_options_before_a_core_command_are_refused(self, capsys):
        with pytest.raises(SystemExit) as info:
            cli.main(["--config", "x.jsonc", "ext", "list"])

        assert info.value.code == 2
        assert "options before 'ext'" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the other commands keep working
# --------------------------------------------------------------------------- #


class TestInitAddDb:
    def test_init(self, tmp_path, capsys):
        target = tmp_path / "gw" / "config.jsonc"

        assert cli.main(["init", "--config", str(target)]) == 0
        assert target.is_file()
        assert "Created config" in capsys.readouterr().out

    def test_init_refuses_an_existing_config(self, tmp_path):
        target = tmp_path / "config.jsonc"
        cli.main(["init", "--config", str(target)])

        with pytest.raises(SystemExit) as info:
            cli.main(["init", "--config", str(target)])

        assert info.value.code == 1

    def test_the_config_flag_before_the_command(self, tmp_path):
        target = tmp_path / "config.jsonc"

        assert cli.main(["--config", str(target), "init"]) == 0
        assert target.is_file()

    def test_add_provider_and_model(self, tmp_path):
        target = tmp_path / "config.jsonc"
        cli.main(["init", "--config", str(target)])

        cli.main(
            [
                "add",
                "--config",
                str(target),
                "provider",
                "acme",
                "--api-key",
                "${ACME_KEY}",
                "--base-url",
                "https://acme.invalid/v1",
            ]
        )
        cli.main(
            ["--config", str(target), "add", "model", "acme-1", "--provider", "acme"]
        )

        from llm_rosetta_gateway.config import load_config

        data = load_config(str(target))
        assert data["providers"]["acme"] == {
            "api_key": "${ACME_KEY}",
            "base_url": "https://acme.invalid/v1",
        }
        assert "acme-1" in json.dumps(data["models"])

    def test_add_without_a_type_prints_help(self, capsys):
        assert cli.main(["add"]) == 0
        assert "provider" in capsys.readouterr().out

    def test_db_arguments_reach_the_handler(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "_cmd_db_cleanup_logs", lambda a: seen.update(vars(a)))

        cli.main(["db", "--config", "c.jsonc", "cleanup-logs", "--max-age-days", "7"])

        assert seen["max_age_days"] == 7 and seen["config"] == "c.jsonc"
        assert seen["data_dir"] is None

    def test_db_data_dir_before_the_command_is_carried(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "_cmd_db_cleanup", lambda a: seen.update(vars(a)))

        cli.main(["--config", "c.jsonc", "--data-dir", "/d", "db", "cleanup"])

        assert seen["data_dir"] == "/d" and seen["config"] == "c.jsonc"
        assert seen["max_age_days"] == 90

    def test_db_data_dir_after_the_command(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli, "_cmd_db_cleanup_errors", lambda a: seen.update(vars(a))
        )

        cli.main(["db", "--data-dir", "/d2", "cleanup-errors"])

        assert seen["data_dir"] == "/d2"

    def test_db_export_errors_options(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            cli, "_cmd_db_export_errors", lambda a: seen.update(vars(a))
        )

        cli.main(["db", "export-errors", "--start", "2026-01-01", "-o", "x.tgz"])

        assert (seen["start"], seen["output"], seen["end"]) == (
            "2026-01-01",
            "x.tgz",
            None,
        )

    def test_db_without_an_action_prints_help(self, capsys):
        assert cli.main(["db"]) == 0
        assert "cleanup-logs" in capsys.readouterr().out

    def test_the_password_command_and_its_old_name(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cli, "_cmd_set_password", lambda a: seen.append(a.config))

        cli.main(["password", "a.jsonc"])
        cli.main(["set-password", "b.jsonc"])
        cli.main(["password", "--config", "c.jsonc"])
        cli.main(["--config", "d.jsonc", "password"])
        cli.main(["password"])

        assert seen == ["a.jsonc", "b.jsonc", "c.jsonc", "d.jsonc", None]

    def test_credential_gets_the_overrides_given_before_it(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "_cmd_credential", lambda a: seen.update(vars(a)))

        cli.main(["--proxy", "socks5://p", "--config", "c", "credential", "chatgpt"])

        assert seen["proxy"] == "socks5://p" and seen["config"] == "c"
        assert seen["provider"] == "chatgpt" and seen["tls_cert"] is None


class TestUsageAndMigrate:
    def test_usage_gets_its_arguments(self, monkeypatch):
        seen = []
        monkeypatch.setattr(usage_cli, "main", lambda argv=None: seen.append(argv) or 3)

        assert cli.main(["usage", "--db", "x", "--json"]) == 3
        assert seen == [["--db", "x", "--json"]]

    def test_migrate_gets_its_arguments(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            migrate_0140, "main", lambda argv=None: seen.append(argv) or 2
        )

        assert cli.main(["migrate", "--config", "c.jsonc", "--write"]) == 2
        assert seen == [["--config", "c.jsonc", "--write"]]

    def test_the_command_exit_code_is_the_return_value(self, monkeypatch):
        monkeypatch.setattr(usage_cli, "main", lambda argv=None: 0)
        assert cli.main(["usage"]) == 0


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_main_registers_every_command_without_package_metadata(self, served):
        assert registered_commands() == ()

        cli.main(["--config", served["config_file"], "--no-banner"])

        names = [item.spec.name for item in registered_commands()]
        assert sorted(names) == sorted(EXPECTED)
        assert {item.owner for item in registered_commands()} == {"llm-rosetta-gateway"}
        assert command_catalog().conflicts == {}

    def test_serve_resolves_to_the_function_main_runs(self):
        cli.register_commands()

        assert resolve_command("serve") is cli.serve_main
        assert resolve_command("usage") is usage_cli.main
        assert resolve_command("migrate") is migrate_0140.main

    def test_registering_twice_is_a_no_op(self):
        cli.register_commands()
        cli.register_commands()

        assert len(registered_commands()) == len(EXPECTED)

    def test_the_dispatcher_runs_serve_after_explicit_registration(self, served):
        cli.register_commands()

        code = core_cli.main(
            ["serve", "--config", served["config_file"], "--host", "Y", "--no-banner"]
        )

        assert code == 0
        assert served["run"][1] == "Y"

    def test_another_owner_cannot_take_a_gateway_name(self):
        cli.register_commands()

        with pytest.raises(CommandConflict) as info:
            register_command(
                CommandSpec("serve", "s", "x.y:z"),
                owner="intruder",
            )

        assert "llm-rosetta-gateway" in str(info.value) and "intruder" in str(
            info.value
        )


class TestConsoleScripts:
    """One released console script stays; the unreleased ones are commands."""

    CORE = PYPROJECT.parents[1] / "llm-rosetta" / "pyproject.toml"

    @staticmethod
    def _scripts(path: Path) -> dict[str, str]:
        text = path.read_text(encoding="utf-8")
        section = text.split("[project.scripts]", 1)[1]
        section = re.split(r"^\[", section, maxsplit=1, flags=re.M)[0]
        return dict(re.findall(r'^([a-z][a-z-]*)\s*=\s*"([^"]+)"', section, flags=re.M))

    def test_the_gateway_keeps_only_its_released_script(self):
        assert self._scripts(PYPROJECT) == {
            "llm-rosetta-gateway": "llm_rosetta_gateway:main"
        }

    def test_core_ships_the_dispatcher(self):
        assert self._scripts(self.CORE) == {"llm-rosetta": "llm_rosetta.cli:main"}

    def test_the_entry_script_target_is_the_function_main_runs(self):
        import llm_rosetta_gateway

        assert llm_rosetta_gateway.main is cli.main


class TestEntryPoints:
    def test_pyproject_declares_exactly_the_registered_commands(self):
        declared = {e.name: e.value for e in _declared_entry_points()}

        assert declared == {spec.name: spec.target for spec in cli._COMMANDS}
        assert tuple(declared) == EXPECTED

    def test_serve_through_entry_point_discovery_reaches_the_same_function(
        self, monkeypatch, served
    ):
        monkeypatch.setattr(command_contract, "_entry_points", _declared_entry_points)
        assert registered_commands() == ()

        catalog = command_catalog()
        assert catalog.conflicts == {} and catalog.problems == ()
        assert resolve_command("serve") is cli.serve_main

        code = core_cli.main(
            ["serve", "--config", served["config_file"], "--host", "Y", "--no-banner"]
        )

        assert code == 0
        assert served["run"] == ("app", "Y", 1234, None, None)

    def test_declaring_and_registering_agree(self, monkeypatch):
        monkeypatch.setattr(command_contract, "_entry_points", _declared_entry_points)
        cli.register_commands()

        catalog = command_catalog()

        assert catalog.conflicts == {}
        assert sorted(catalog.commands) == sorted(EXPECTED)

    def test_every_entry_point_has_a_docstring_summary(self):
        for entry in _declared_entry_points():
            command = command_contract.RegisteredCommand(
                command_contract.CommandSpec(entry.name, "", entry.value),
                "llm-rosetta-gateway",
                "entry point",
            )
            assert command_contract.command_summary(command), entry.name

    def test_help_lists_the_gateway_commands_from_the_entry_points(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(command_contract, "_entry_points", _declared_entry_points)

        assert core_cli.main(["help"]) == 0

        out = capsys.readouterr().out
        for name in EXPECTED:
            assert re.search(rf"^\s+{name}\s", out, flags=re.M), name
        assert "llm-rosetta-gateway" in out

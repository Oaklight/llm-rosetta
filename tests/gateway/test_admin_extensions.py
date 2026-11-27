"""The admin panel's read-only view of the installed extensions (epic #1703 E6)."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from _ext_wheels import build_wheel, plugin_yaml
from llm_rosetta.api import (
    configure_extensions_dir,
    install_wheel,
    load_providers,
)
from llm_rosetta_gateway.admin.routes.config import extensions as extensions_route
from llm_rosetta_gateway.app.admin_context import GatewayAdminContext


@pytest.fixture
def ext_dir(tmp_path: Path) -> Path:
    path = tmp_path / "extensions"
    path.mkdir()
    configure_extensions_dir(path)
    return path


def _install(ext_dir: Path, ext_id: str, manifest: str | None = None, **kw) -> None:
    wheel = build_wheel(
        ext_id, "0.1.0", [f"{ext_id}--chat_completions"], manifest=manifest, **kw
    )
    install_wheel(wheel, dest=ext_dir)


def _by_id() -> dict[str, dict]:
    return {row["id"]: row for row in GatewayAdminContext().extension_listing()}


def _manifest_with_external(ext_id: str) -> str:
    base = plugin_yaml(
        ext_id,
        "0.1.0",
        [f"{ext_id}--chat_completions"],
        transports=("pigeon",),
        requires_extensions=("lanes",),
    )
    return base + textwrap.dedent(
        f"""\
        external:
          - name: python
            check: ["{sys.executable}", "--version"]
          - name: nowhere
            check: ["no-such-program-for-llm-rosetta", "--version"]
            install_hint: install nowhere
        """
    )


class TestTheListing:
    def test_a_loaded_extension_is_described_in_full(self, ext_dir):
        _install(ext_dir, "acme", _manifest_with_external("acme"))
        load_providers()

        row = _by_id()["acme"]

        assert (row["version"], row["state"], row["reason"]) == (
            "0.1.0",
            "loaded",
            None,
        )
        assert row["source"] == "namespace"
        assert row["shims"] == ["acme--chat_completions"]
        assert row["transports"] == ["pigeon"]
        assert row["requires"] == ["lanes"]
        assert row["missing_dependencies"] == ["lanes"]
        assert row["maturity"] == "beta"
        programs = {item["name"]: item for item in row["external"]}
        assert programs["python"]["state"] == "ok"
        assert programs["nowhere"]["state"] == "missing"
        assert programs["nowhere"]["install_hint"] == "install nowhere"
        json.dumps(row)

    def test_an_installed_dependency_is_not_missing(self, ext_dir):
        _install(ext_dir, "lanes")
        _install(ext_dir, "acme", _manifest_with_external("acme"))
        load_providers()

        rows = _by_id()

        assert rows["acme"]["missing_dependencies"] == []
        assert rows["lanes"]["state"] == "loaded"

    def test_an_extension_that_failed_to_load_says_why(self, ext_dir):
        _install(
            ext_dir,
            "acme",
            members={
                "llm/chat_completions/transforms.py": "raise RuntimeError('boom')\n"
            },
        )
        load_providers()

        row = _by_id()["acme"]

        assert row["state"] == "failed"
        assert "boom" in row["reason"]
        assert row["shims"] == []

    def test_an_extension_on_disk_that_is_not_loaded_yet_is_listed(self, ext_dir):
        _install(ext_dir, "acme")

        row = _by_id()["acme"]

        assert row["state"] == "installed"
        assert row["source"] == "extensions directory"
        assert row["version"] == "0.1.0"
        assert "reload" in row["reason"]
        assert row["path"].endswith("llm_rosetta_ext/acme")

    def test_no_extension_gives_an_empty_list(self, ext_dir):
        assert GatewayAdminContext().extension_listing() == []


def _manifest_with_cli(
    ext_id: str, *, minimum: str | None = None, install: bool = True
) -> str:
    base = plugin_yaml(ext_id, "0.1.0", [f"{ext_id}--chat_completions"])
    lines = [
        "external:",
        "  - name: acmecli",
        '    check: ["acmecli", "--version"]',
    ]
    if minimum:
        lines.append(f'    minimum: "{minimum}"')
    if install:
        lines += [
            "    install:",
            '      npm: "@acme/cli"',
            '      version: "2.1.299"',
        ]
    return base + "\n".join(lines) + "\n"


def _put_cli(tools: Path, ext_id: str, version: str, needs: str | None = None) -> None:
    """What ``ext install-cli`` leaves: an entry link that reports *version*.

    *needs* names a command the program runs first, as an npm script runs node.
    """
    link = tools / ext_id / "bin" / "acmecli"
    link.parent.mkdir(parents=True)
    guard = f"{needs} --version >/dev/null || exit $?\n" if needs else ""
    link.write_text(f"#!/bin/sh\n{guard}echo acmecli {version}\n")
    link.chmod(0o755)


@pytest.fixture
def tools(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "tools"
    monkeypatch.setenv("LLM_ROSETTA_TOOLS_DIR", str(path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    return path


class TestAnExternalProgramHasThreeStates:
    """The extension card says: missing, older than the minimum, or ready (#1820)."""

    @staticmethod
    def _program(ext_id: str = "acme") -> dict:
        return _by_id()[ext_id]["external"][0]

    def test_a_program_that_is_not_installed_is_missing_and_names_the_command(
        self, ext_dir, tools
    ):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        load_providers()

        program = self._program()

        assert program["status"] == "missing"
        assert program["install_command"] == "llm-rosetta ext install-cli acme"
        assert program["install"] == {"version": "2.1.299", "source": "npm"}

    def test_a_program_older_than_the_minimum_is_too_old(self, ext_dir, tools):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        _put_cli(tools, "acme", "2.0.0")
        load_providers()

        program = self._program()

        assert (program["status"], program["found"]) == ("too_old", "2.0.0")
        assert program["install_command"] == "llm-rosetta ext install-cli acme"

    def test_a_program_whose_node_is_gone_is_broken_and_names_the_command(
        self, ext_dir, tools
    ):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        _put_cli(tools, "acme", "2.1.299", needs="node")
        load_providers()

        program = self._program()

        assert (program["state"], program["status"]) == ("broken", "broken")
        assert program["install_command"] == "llm-rosetta ext install-cli acme"

    def test_the_shared_node_link_lets_the_same_program_run(self, ext_dir, tools):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        _put_cli(tools, "acme", "2.1.299", needs="node")
        node = tools / "bin" / "node"
        node.parent.mkdir(parents=True)
        node.write_text("#!/bin/sh\necho v24.0.0\n")
        node.chmod(0o755)
        load_providers()

        assert self._program()["status"] == "ready"

    def test_a_program_installed_by_install_cli_is_ready_outside_path(
        self, ext_dir, tools
    ):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        _put_cli(tools, "acme", "2.1.299")
        load_providers()

        program = self._program()

        assert (program["status"], program["found"]) == ("ready", "2.1.299")
        assert program["install_command"] is None

    def test_a_program_with_no_install_spec_has_no_command(self, ext_dir, tools):
        _install(ext_dir, "acme", _manifest_with_cli("acme", install=False))
        load_providers()

        program = self._program()

        assert program["status"] == "missing"
        assert program["install"] is None
        assert program["install_command"] is None

    def test_a_second_program_names_itself_in_the_command(self, ext_dir, tools):
        manifest = _manifest_with_cli("acme") + (
            "  - name: other\n"
            '    check: ["other", "--version"]\n'
            "    install:\n"
            '      npm: "other-cli"\n'
            '      version: "1.0.0"\n'
        )
        _install(ext_dir, "acme", manifest)
        load_providers()

        commands = {
            p["name"]: p["install_command"] for p in _by_id()["acme"]["external"]
        }

        assert commands["other"] == "llm-rosetta ext install-cli acme --program other"

    def test_the_listing_stays_json_safe(self, ext_dir, tools):
        _install(ext_dir, "acme", _manifest_with_cli("acme", minimum="2.1.299"))
        load_providers()
        json.dumps(GatewayAdminContext().extension_listing())


class TestTheEndpoint:
    @staticmethod
    def _call(app: object):
        response = asyncio.run(
            extensions_route.get_extensions(SimpleNamespace(app=app))
        )
        return response.status_code, json.loads(response.body)

    def test_it_returns_the_listing(self, ext_dir):
        _install(ext_dir, "acme")
        load_providers()

        status, body = self._call(SimpleNamespace(admin=GatewayAdminContext()))

        assert status == 200
        assert [row["id"] for row in body["extensions"]] == ["acme"]

    def test_a_failure_is_an_error_body_not_an_exception(self):
        class Broken:
            def extension_listing(self):
                raise RuntimeError("no listing today")

        status, body = self._call(SimpleNamespace(admin=Broken()))

        assert status == 500
        assert "no listing today" in body["error"]

    def test_the_route_is_registered(self):
        import inspect

        from llm_rosetta_gateway.admin.routes import config as config_routes

        text = inspect.getsource(config_routes.register_config_routes)
        assert '"/admin/api/extensions", methods=["GET"]' in text


def _result(name: str, version: str, found: str | None = "2.1.299") -> SimpleNamespace:
    """What ``install_cli`` returns, as far as the panel reads it."""
    return SimpleNamespace(
        program=name, state="installed", version=version, link="/t/bin/x", found=found
    )


class TestInstallingFromThePanel:
    """``POST /admin/api/extensions/<id>/install-cli`` installs what the manifest pins (#1820)."""

    @pytest.fixture
    def calls(self, monkeypatch) -> list:
        from llm_rosetta_gateway.ops import extensions as ops_extensions

        recorded: list = []

        def fake(ext_id, program, **kwargs):
            recorded.append((ext_id, program.name, program.install.version, kwargs))
            return _result(program.name, program.install.version)

        monkeypatch.setattr(ops_extensions, "install_cli", fake)
        return recorded

    @staticmethod
    def _post(app, ext_id, body):
        class _Request:
            def __init__(self):
                self.app = app

            def json(self):
                if isinstance(body, Exception):
                    raise body
                return body

        response = asyncio.run(
            extensions_route.post_extension_install_cli(_Request(), ext_id)
        )
        return response.status_code, json.loads(response.body)

    @staticmethod
    def _app(ops_log=None):
        from llm_rosetta_gateway.ops import OpsContext

        app = SimpleNamespace(ops_ctx=OpsContext(ops_log=ops_log))
        app.admin = GatewayAdminContext(app)
        return app

    def test_it_installs_the_program_at_the_version_the_manifest_pins(
        self, ext_dir, calls
    ):
        _install(ext_dir, "acme", _manifest_with_cli("acme"))

        status, body = self._post(self._app(), "acme", {"program": "acmecli"})

        assert status == 200
        assert body == {
            "extension": "acme",
            "program": "acmecli",
            "version": "2.1.299",
            "state": "installed",
            "found": "2.1.299",
        }
        assert calls == [("acme", "acmecli", "2.1.299", {})]

    def test_the_program_may_be_left_out_when_there_is_only_one(self, ext_dir, calls):
        _install(ext_dir, "acme", _manifest_with_cli("acme"))

        status, _ = self._post(self._app(), "acme", {})

        assert status == 200 and calls[0][1] == "acmecli"

    @pytest.mark.parametrize(
        "body",
        [
            {"version": "9.9.9"},
            {"program": "acmecli", "url": "https://evil.test/x"},
            {"program": "acmecli", "command": "rm -rf /"},
            {"program": ["acmecli"]},
            ["acmecli"],
        ],
    )
    def test_nothing_but_the_program_name_is_accepted(self, ext_dir, calls, body):
        _install(ext_dir, "acme", _manifest_with_cli("acme"))

        status, payload = self._post(self._app(), "acme", body)

        assert (status, payload["reason"]) == (400, "bad_body")
        assert calls == []

    def test_invalid_json_is_refused(self, ext_dir, calls):
        _install(ext_dir, "acme", _manifest_with_cli("acme"))

        status, _ = self._post(self._app(), "acme", ValueError("not json"))

        assert status == 400 and calls == []

    @pytest.mark.parametrize(
        ("ext_id", "body", "status", "reason"),
        [
            ("nope", {}, 404, "unknown_extension"),
            ("acme", {"program": "other"}, 404, "unknown_program"),
        ],
    )
    def test_an_unknown_extension_or_program_is_a_404(
        self, ext_dir, calls, ext_id, body, status, reason
    ):
        _install(ext_dir, "acme", _manifest_with_cli("acme"))

        got, payload = self._post(self._app(), ext_id, body)

        assert (got, payload["reason"]) == (status, reason)
        assert calls == []

    def test_a_program_with_no_install_spec_is_a_409(self, ext_dir, calls):
        _install(ext_dir, "acme", _manifest_with_cli("acme", install=False))

        status, payload = self._post(self._app(), "acme", {"program": "acmecli"})

        assert (status, payload["reason"]) == (409, "no_install_spec")
        assert calls == []

    def test_two_programs_need_a_name(self, ext_dir, calls):
        manifest = _manifest_with_cli("acme") + (
            "  - name: second\n"
            '    check: ["second", "--version"]\n'
            "    install:\n"
            '      npm: "@acme/second"\n'
            '      version: "1.0.0"\n'
        )
        _install(ext_dir, "acme", manifest)

        status, payload = self._post(self._app(), "acme", {})

        assert (status, payload["reason"]) == (400, "program_required")

    def test_a_second_install_of_the_same_extension_is_a_409_while_one_runs(
        self, ext_dir, monkeypatch
    ):
        import threading

        from llm_rosetta_gateway.ops import extensions as ops_extensions

        _install(ext_dir, "acme", _manifest_with_cli("acme"))
        started, release = threading.Event(), threading.Event()

        def slow(ext_id, program, **kwargs):
            started.set()
            release.wait(5)
            return _result(program.name, "2.1.299", found=None)

        monkeypatch.setattr(ops_extensions, "install_cli", slow)
        app = self._app()

        async def scenario():
            first = asyncio.create_task(
                app.admin.install_extension_cli("acme", "acmecli")
            )
            await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
            from llm_rosetta_gateway.admin_api import ExtensionInstallRefused

            with pytest.raises(ExtensionInstallRefused) as seen:
                await app.admin.install_extension_cli("acme", "acmecli")
            release.set()
            await first
            again = await app.admin.install_extension_cli("acme", "acmecli")
            return seen.value, again

        refused, again = asyncio.run(scenario())

        assert (refused.status, refused.reason) == (409, "busy")
        assert again["state"] == "installed"

    def test_a_failed_install_is_a_502_with_the_reason_and_is_recorded(
        self, ext_dir, monkeypatch
    ):
        from unittest.mock import AsyncMock, MagicMock

        from llm_rosetta.api import CliInstallError
        from llm_rosetta_gateway.ops import extensions as ops_extensions

        def failing(ext_id, program, **kwargs):
            raise CliInstallError("npm exited with 1: ERR 404")

        monkeypatch.setattr(ops_extensions, "install_cli", failing)
        _install(ext_dir, "acme", _manifest_with_cli("acme"))
        ops_log = MagicMock()
        ops_log.add = AsyncMock()

        status, payload = self._post(self._app(ops_log), "acme", {"program": "acmecli"})

        assert (status, payload["reason"]) == (502, "failed")
        assert "ERR 404" in payload["error"]
        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == "extension_cli_install"
        assert entry.details["extension"] == "acme"

    def test_a_success_is_recorded_without_the_environment(self, ext_dir, calls):
        from unittest.mock import AsyncMock, MagicMock

        _install(ext_dir, "acme", _manifest_with_cli("acme"))
        ops_log = MagicMock()
        ops_log.add = AsyncMock()

        self._post(self._app(ops_log), "acme", {"program": "acmecli"})

        entry = ops_log.add.call_args[0][0]
        assert set(entry.details) == {
            "extension",
            "program",
            "version",
            "state",
            "found",
        }

    def test_the_route_is_registered_behind_the_providers_tab(self):
        import inspect

        from llm_rosetta_gateway.admin.routes import config as config_routes

        text = inspect.getsource(config_routes.register_config_routes)
        assert '"/admin/api/extensions/<ext_id>/install-cli", methods=["POST"]' in text
        assert '_guard("providers", post_extension_install_cli)' in text

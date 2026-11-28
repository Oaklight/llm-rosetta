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


# -- managing the extensions themselves (#1829) -------------------------------


def _ops_app():
    from unittest.mock import AsyncMock, MagicMock

    from llm_rosetta_gateway.ops import OpsContext

    log = MagicMock()
    log.add = AsyncMock()
    app = SimpleNamespace(ops_ctx=OpsContext(ops_log=log))
    app.admin = GatewayAdminContext(app)
    return app, log


def _request(app, body=None):
    class _Request:
        def __init__(self):
            self.app = app

        def json(self):
            if isinstance(body, Exception):
                raise body
            return body

    return _Request()


def _call(handler, app, *args, body=None):
    response = asyncio.run(handler(_request(app, body), *args))
    return response.status_code, json.loads(response.body)


def _tools(ext_dir: Path, ext_id: str, *, npm: bool = False) -> Path:
    root = ext_dir.parent / "tools" / ext_id
    (root / ("npm-1.0.0" if npm else "release-1.0.0")).mkdir(parents=True)
    (root / "installed.json").write_text("{}")
    return root


@pytest.fixture
def reloads(monkeypatch) -> list:
    seen: list = []

    async def fake(request):
        seen.append(1)
        return None

    monkeypatch.setattr(extensions_route, "_reload_shims", fake)
    return seen


class TestRemovingAnExtension:
    def test_the_plan_lists_what_would_go_and_changes_nothing(self, ext_dir):
        _install(ext_dir, "acme")
        root = _tools(ext_dir, "acme")
        app, _ = _ops_app()

        status, plan = _call(extensions_route.get_extension_removal_plan, app, "acme")

        assert status == 200 and plan["origin"] == "directory" and plan["removable"]
        paths = [row["path"] for row in plan["delete"]]
        assert str(ext_dir / "llm_rosetta_ext" / "acme") in paths
        assert str(root) in paths
        assert (ext_dir / "llm_rosetta_ext" / "acme").is_dir() and root.is_dir()

    def test_an_unknown_extension_has_no_plan(self, ext_dir):
        app, _ = _ops_app()
        status, body = _call(extensions_route.get_extension_removal_plan, app, "ghost")
        assert (status, body["reason"]) == (404, "unknown_extension")

    def test_it_removes_the_extension_and_its_programs_and_reloads(
        self, ext_dir, reloads
    ):
        _install(ext_dir, "acme")
        root = _tools(ext_dir, "acme")
        app, log = _ops_app()

        status, body = _call(
            extensions_route.post_extension_remove, app, "acme", body={}
        )

        assert status == 200 and body["extension"] == "acme"
        assert body["failed"] == [] and body["reload_error"] is None
        assert not (ext_dir / "llm_rosetta_ext" / "acme").exists()
        assert not root.exists()
        assert reloads == [1]
        entry = log.add.call_args[0][0]
        assert entry.event_type == "extension_remove"
        assert (
            entry.details["extension"] == "acme" and entry.details["replace"] is False
        )

    def test_a_replacement_is_marked_in_the_record(self, ext_dir, reloads):
        _install(ext_dir, "acme")
        app, log = _ops_app()
        _call(
            extensions_route.post_extension_remove,
            app,
            "acme",
            body={"replace": True},
        )
        assert log.add.call_args[0][0].details["replace"] is True

    def test_an_image_extension_is_refused_and_nothing_is_deleted(
        self, ext_dir, tmp_path, monkeypatch, reloads
    ):
        from llm_rosetta_gateway.shims import extension_view

        site = tmp_path / "site-packages" / "acme"
        site.mkdir(parents=True)
        root = _tools(ext_dir, "acme")
        monkeypatch.setattr(
            extension_view, "locate_extension", lambda ext_id: (site, "0.1.0")
        )
        app, log = _ops_app()

        status, body = _call(
            extensions_route.post_extension_remove, app, "acme", body={}
        )

        assert (status, body["reason"]) == (409, "not_removable")
        assert "rebuild the image" in body["error"]
        assert site.is_dir() and root.is_dir() and reloads == []
        log.add.assert_not_called()

    def test_an_unknown_extension_is_a_404(self, ext_dir, reloads):
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_remove, app, "ghost", body={}
        )
        assert (status, body["reason"]) == (404, "unknown_extension")
        assert reloads == []

    def test_a_second_operation_on_the_same_extension_is_refused(
        self, ext_dir, reloads
    ):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        app.admin._installing.add("acme")
        status, body = _call(
            extensions_route.post_extension_remove, app, "acme", body={}
        )
        assert (status, body["reason"]) == (409, "busy")
        assert (ext_dir / "llm_rosetta_ext" / "acme").is_dir()

    @pytest.mark.parametrize(
        "body",
        [{"path": "/etc"}, {"replace": "yes"}, {"replace": True, "tools": "x"}, []],
    )
    def test_nothing_but_replace_is_accepted(self, ext_dir, reloads, body):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        status, answer = _call(
            extensions_route.post_extension_remove, app, "acme", body=body
        )
        assert (status, answer["reason"]) == (400, "bad_body")
        assert (ext_dir / "llm_rosetta_ext" / "acme").is_dir() and reloads == []

    def test_a_failed_reload_is_reported_not_swallowed(self, ext_dir, monkeypatch):
        _install(ext_dir, "acme")

        async def boom(request, path, **kw):
            raise RuntimeError("config broke")

        monkeypatch.setattr(extensions_route, "_reload_gateway_config", boom)
        monkeypatch.setattr(extensions_route, "_get_config_path", lambda r: "x.jsonc")
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_remove, app, "acme", body={}
        )
        assert status == 200
        assert "shim reload failed" in body["reload_error"]
        assert "config broke" in body["reload_error"]


CATALOGUE = {
    "sources": [
        {
            "id": "contrib",
            "url": "https://example.test/manifest.json",
            "ok": True,
            "vendors": [
                {
                    "vendor": "acme",
                    "installable": "0.2.0",
                    "newest": "0.2.0",
                    "versions": ["0.2.0", "0.1.0"],
                },
                {
                    "vendor": "same",
                    "installable": "0.1.0",
                    "newest": "0.1.0",
                    "versions": ["0.1.0"],
                },
                {
                    "vendor": "future",
                    "installable": None,
                    "newest": "9.0.0",
                    "newest_problem": "requires surface 9",
                    "versions": ["9.0.0"],
                },
            ],
        }
    ]
}


class TestCheckingForUpdates:
    VERSIONS = {"acme": "0.1.0", "same": "0.1.0", "future": "0.1.0", "alone": "0.1.0"}

    def test_each_extension_is_judged_the_same_way(self):
        from llm_rosetta_gateway.shims.extension_view import update_report

        rows = {r["id"]: r for r in update_report(self.VERSIONS, CATALOGUE)}

        assert rows["acme"]["status"] == "update"
        assert (rows["acme"]["current"], rows["acme"]["available"]) == (
            "0.1.0",
            "0.2.0",
        )
        assert rows["acme"]["source"] == "contrib"
        assert rows["same"]["status"] == "current"
        assert rows["future"]["status"] == "no_compatible"
        assert rows["future"]["error"] == "requires surface 9"
        assert rows["alone"]["status"] == "not_published"

    def test_an_unreadable_source_is_never_reported_as_up_to_date(self):
        from llm_rosetta_gateway.shims.extension_view import update_report

        catalogue = {
            "sources": [
                {
                    "id": "contrib",
                    "ok": False,
                    "error": "timed out after 10 s",
                    "vendors": [],
                }
            ]
        }
        row = update_report({"acme": "0.1.0"}, catalogue)[0]
        assert row["status"] == "failed"
        assert "contrib: timed out after 10 s" in row["error"]

    def test_a_cached_copy_does_not_confirm_that_it_is_up_to_date(self):
        from llm_rosetta_gateway.shims.extension_view import update_report

        catalogue = {
            "sources": [
                {
                    "id": "contrib",
                    "ok": True,
                    "stale": True,
                    "error": "timed out after 10 s",
                    "url": "u",
                    "vendors": [
                        {
                            "vendor": "same",
                            "installable": "0.1.0",
                            "versions": ["0.1.0"],
                        },
                        {
                            "vendor": "acme",
                            "installable": "0.2.0",
                            "versions": ["0.2.0"],
                        },
                    ],
                }
            ]
        }
        rows = {
            r["id"]: r
            for r in update_report(
                {"same": "0.1.0", "acme": "0.1.0", "alone": "0.1.0"}, catalogue
            )
        }
        assert rows["same"]["status"] == "failed"
        assert "cached copy only" in rows["same"]["error"]
        assert rows["acme"]["status"] == "update"
        assert rows["alone"]["status"] == "failed"
        assert "timed out" in rows["alone"]["error"]

    def test_the_ids_limit_the_report(self):
        from llm_rosetta_gateway.shims.extension_view import update_report

        rows = update_report(self.VERSIONS, CATALOGUE, ["same"])
        assert [r["id"] for r in rows] == ["same"]

    def test_the_route_refetches_the_sources_and_answers(self, ext_dir, monkeypatch):
        _install(ext_dir, "acme")
        seen: list = []

        def catalogue(self, config, report, *, refresh=False):
            seen.append(refresh)
            return CATALOGUE

        monkeypatch.setattr(GatewayAdminContext, "shim_source_catalogue", catalogue)
        app, _ = _ops_app()

        status, body = _call(
            extensions_route.post_extension_check_updates, app, body={"ids": ["acme"]}
        )

        assert status == 200 and seen == [True]
        assert [(u["id"], u["status"]) for u in body["updates"]] == [("acme", "update")]

    def test_a_failure_to_read_says_why(self, ext_dir, monkeypatch):
        def catalogue(self, config, report, *, refresh=False):
            raise RuntimeError("no network")

        monkeypatch.setattr(GatewayAdminContext, "shim_source_catalogue", catalogue)
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_check_updates, app, body={}
        )
        assert status == 502 and "no network" in body["error"]

    @pytest.mark.parametrize("body", [{"ids": "acme"}, {"ids": [1]}, {"x": 1}, []])
    def test_a_malformed_body_is_refused(self, body):
        app, _ = _ops_app()
        status, answer = _call(
            extensions_route.post_extension_check_updates, app, body=body
        )
        assert (status, answer["reason"]) == (400, "bad_body")


class TestUpdatingAnExtension:
    @pytest.fixture
    def installs(self, monkeypatch) -> list:
        import llm_rosetta.api as api

        seen: list = []

        def fake_load(url):
            seen.append(("load", url))
            return {"index": url}

        def fake_install(ext_id, index, *, version=None):
            seen.append(("install", ext_id, version))
            return [SimpleNamespace(id=ext_id, version=version)]

        monkeypatch.setattr(api, "load_index", fake_load)
        monkeypatch.setattr(api, "install_from_index", fake_install)
        monkeypatch.setattr(
            GatewayAdminContext,
            "shim_source_catalogue",
            lambda self, config, report, *, refresh=False: CATALOGUE,
        )
        return seen

    def test_it_installs_the_newest_release_and_reloads(
        self, ext_dir, installs, reloads
    ):
        _install(ext_dir, "acme")
        app, log = _ops_app()

        status, body = _call(
            extensions_route.post_extension_update, app, "acme", body={}
        )

        assert status == 200
        assert (body["before"], body["after"]) == ("0.1.0", "0.2.0")
        assert installs == [
            ("load", "https://example.test/manifest.json"),
            ("install", "acme", "0.2.0"),
        ]
        assert reloads == [1]
        entry = log.add.call_args[0][0]
        assert entry.event_type == "extension_update"
        assert (entry.details["before"], entry.details["after"]) == ("0.1.0", "0.2.0")

    def test_a_published_version_can_be_chosen(self, ext_dir, installs, reloads):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_update,
            app,
            "acme",
            body={"version": "0.1.0"},
        )
        assert status == 200 and installs[-1] == ("install", "acme", "0.1.0")

    def test_a_version_the_source_does_not_publish_is_refused(
        self, ext_dir, installs, reloads
    ):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_update,
            app,
            "acme",
            body={"version": "7.7.7"},
        )
        assert (status, body["reason"]) == (400, "no_release")
        assert installs == [] and reloads == []

    def test_an_extension_no_source_publishes_is_a_404(
        self, ext_dir, installs, reloads
    ):
        _install(ext_dir, "alone")
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_update, app, "alone", body={}
        )
        assert (status, body["reason"]) == (404, "no_release")

    def test_an_image_extension_is_refused(
        self, ext_dir, tmp_path, monkeypatch, installs, reloads
    ):
        from llm_rosetta_gateway.shims import extension_view

        monkeypatch.setattr(
            extension_view, "locate_extension", lambda ext_id: (tmp_path, "0.1.0")
        )
        app, _ = _ops_app()
        status, body = _call(
            extensions_route.post_extension_update, app, "acme", body={}
        )
        assert (status, body["reason"]) == (409, "not_removable")
        assert installs == []

    def test_a_failing_install_is_a_502_and_is_recorded(
        self, ext_dir, installs, reloads, monkeypatch
    ):
        import llm_rosetta.api as api

        def refuse(ext_id, index, *, version=None):
            raise api.ExtensionInstallError("sha256 mismatch")

        monkeypatch.setattr(api, "install_from_index", refuse)
        _install(ext_dir, "acme")
        app, log = _ops_app()
        status, body = _call(
            extensions_route.post_extension_update, app, "acme", body={}
        )
        assert (status, body["reason"]) == (502, "failed")
        assert "sha256 mismatch" in body["error"]
        assert log.add.call_args[0][0].details["error"] == "sha256 mismatch"
        assert reloads == []

    def test_a_second_operation_is_refused(self, ext_dir, installs, reloads):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        app.admin._installing.add("acme")
        status, body = _call(
            extensions_route.post_extension_update, app, "acme", body={}
        )
        assert (status, body["reason"]) == (409, "busy")
        assert installs == []

    @pytest.mark.parametrize("body", [{"url": "https://evil.test"}, {"version": 2}, []])
    def test_only_a_version_string_is_accepted(self, ext_dir, installs, reloads, body):
        _install(ext_dir, "acme")
        app, _ = _ops_app()
        status, answer = _call(
            extensions_route.post_extension_update, app, "acme", body=body
        )
        assert (status, answer["reason"]) == (400, "bad_body")
        assert installs == []


class TestWhichShimsNeedWhich:
    def test_a_program_names_the_shims_that_need_it(self, ext_dir, monkeypatch):
        from llm_rosetta_gateway.shims import extension_view

        _install(ext_dir, "acme", _manifest_with_external("acme"))
        load_providers()
        needs = {"acme--chat_completions": ["python"]}
        monkeypatch.setattr(
            extension_view,
            "get_shim",
            lambda name: SimpleNamespace(needs=needs.get(name)),
        )

        programs = {p["name"]: p for p in _by_id()["acme"]["external"]}

        assert programs["python"]["used_by"] == ["acme--chat_completions"]
        assert programs["nowhere"]["used_by"] == []

    def test_a_shim_without_needs_uses_every_program(self, ext_dir, monkeypatch):
        from llm_rosetta_gateway.shims import extension_view

        _install(ext_dir, "acme", _manifest_with_external("acme"))
        load_providers()
        monkeypatch.setattr(
            extension_view, "get_shim", lambda name: SimpleNamespace(needs=None)
        )
        programs = {p["name"]: p for p in _by_id()["acme"]["external"]}
        assert programs["python"]["used_by"] == ["acme--chat_completions"]
        assert programs["nowhere"]["used_by"] == ["acme--chat_completions"]

    def test_naming_node_adds_the_managed_node_row(self, ext_dir, monkeypatch):
        from llm_rosetta_gateway.shims import extension_view

        _install(ext_dir, "acme", _manifest_with_external("acme"))
        load_providers()
        monkeypatch.setattr(
            extension_view, "get_shim", lambda name: SimpleNamespace(needs=["node"])
        )
        programs = {p["name"]: p for p in _by_id()["acme"]["external"]}
        assert programs["node"]["used_by"] == ["acme--chat_completions"]
        assert programs["python"]["used_by"] == []

    def test_a_directory_install_reports_its_origin(self, ext_dir):
        _install(ext_dir, "acme")
        row = _by_id()["acme"]
        assert row["origin"] == "directory" and row["remove_hint"] is None

    def test_an_image_extension_reports_its_origin_and_how_to_remove_it(
        self, ext_dir, tmp_path, monkeypatch
    ):
        from llm_rosetta_gateway.shims import extension_view

        site = tmp_path / "site-packages" / "llm_rosetta_ext" / "argo"
        site.mkdir(parents=True)
        item = SimpleNamespace(
            id="argo",
            manifest=None,
            source="entry point",
            state="loaded",
            reason=None,
            path=site,
            shims=(),
        )
        monkeypatch.setattr(extension_view, "list_extensions", lambda: [item])

        row = _by_id()["argo"]

        assert row["origin"] == "image"
        assert "rebuild the image" in row["remove_hint"]

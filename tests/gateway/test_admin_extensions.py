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

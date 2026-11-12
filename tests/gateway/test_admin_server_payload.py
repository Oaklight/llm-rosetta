"""The ``server`` section of the admin config payload.

One value there is not a copy of the file: ``open_on_no_keys`` is resolved
from the live config, because the API Keys page has to state which of two
opposite outcomes is in force — and the same value drives the startup warning
(#1502 §2.2).  This pins that single source, and keeps the masking the
sanitiser has always done.
"""

from __future__ import annotations

from llm_rosetta_gateway.admin.routes.config.payload import _effective_server_section
from llm_rosetta_gateway.config import GatewayConfig


def _config(server: dict | None = None) -> GatewayConfig:
    return GatewayConfig({"providers": {}, "models": {}, "server": server or {}})


def test_absent_switch_reads_as_closed():
    section = _effective_server_section({}, _config())

    assert section["open_on_no_keys"] is False


def test_the_resolved_value_wins_over_the_file():
    # A hand-written string is truthy in the file; the parsed config is not.
    section = _effective_server_section({"open_on_no_keys": "false"}, _config())

    assert section["open_on_no_keys"] is False


def test_an_open_gateway_says_so():
    section = _effective_server_section({}, _config({"open_on_no_keys": True}))

    assert section["open_on_no_keys"] is True


def test_the_value_is_always_a_bool():
    assert isinstance(_effective_server_section({}, _config())["open_on_no_keys"], bool)
    assert isinstance(
        _effective_server_section(
            {"open_on_no_keys": 1}, _config({"open_on_no_keys": True})
        )["open_on_no_keys"],
        bool,
    )


def test_the_password_stays_off_the_wire():
    section = _effective_server_section(
        {"admin_password": "hunter2", "port": 8765},
        _config({"admin_password": "hunter2"}),
    )

    assert "admin_password" not in section
    assert section["port"] == 8765


class TestTheEndpoint:
    """The route, not just the helper: a refactor must not stop calling it."""

    def _payload(self, monkeypatch, server: dict) -> dict:
        import asyncio
        import json
        from types import SimpleNamespace

        from llm_rosetta_gateway.admin.routes.config import payload

        raw = {"providers": {}, "server": server}
        monkeypatch.setattr(payload, "_get_config_path", lambda request: "/tmp/c.jsonc")
        monkeypatch.setattr(
            payload,
            "_get_config_io",
            lambda request: SimpleNamespace(load_raw=lambda path: raw),
        )
        monkeypatch.setattr(payload, "_shim_status_payload", lambda request: {})
        monkeypatch.setattr(payload, "_shim_removal_advisories", lambda r, doc: [])
        monkeypatch.setattr(payload, "_shim_source_presets_payload", lambda: [])
        cfg = _config({**server, "admin_password": "hunter2"})
        cfg.credential_visible = False
        cfg.api_keys_db = None
        cfg.admin_session_idle_minutes = 30
        cfg.admin_session_remember_days = 30
        request = SimpleNamespace(
            app=SimpleNamespace(
                gateway_config=cfg,
                auth_state=SimpleNamespace(admin_password=None),
            )
        )
        return json.loads(asyncio.run(payload.get_config(request)).body)

    def test_the_section_carries_the_resolved_switch(self, monkeypatch) -> None:
        body = self._payload(monkeypatch, {})

        assert body["server"]["open_on_no_keys"] is False
        assert "admin_password" not in body["server"]

    def test_an_open_gateway_reaches_the_panel(self, monkeypatch) -> None:
        body = self._payload(monkeypatch, {"open_on_no_keys": True})

        assert body["server"]["open_on_no_keys"] is True

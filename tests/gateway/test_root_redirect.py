"""Tests for configurable root redirect (server.root_redirect)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from llm_rosetta.gateway.config import GatewayConfig


def _minimal_raw(**server_overrides) -> dict:
    raw = {
        "providers": {
            "test": {
                "api_key": "sk-test",
                "base_url": "https://api.example.com",
                "type": "openai",
            }
        },
        "models": {"gpt-test": "test"},
        "server": {},
    }
    raw["server"].update(server_overrides)
    return raw


def _find_root_get_route(app):
    for route in app._routes:
        if route.pattern.fullmatch("/") and "GET" in route.methods:
            return route
    return None


class TestRootRedirectConfig:
    def test_default_none(self):
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.root_redirect is None

    def test_set_to_admin(self):
        cfg = GatewayConfig(_minimal_raw(root_redirect="/admin"))
        assert cfg.root_redirect == "/admin"

    def test_set_to_custom_path(self):
        cfg = GatewayConfig(_minimal_raw(root_redirect="/dashboard"))
        assert cfg.root_redirect == "/dashboard"


class TestRootRedirectRoute:
    def test_root_route_registered_when_configured(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw(root_redirect="/admin"))
        app = create_app(cfg)
        assert _find_root_get_route(app) is not None

    def test_root_route_not_registered_when_disabled(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw())
        app = create_app(cfg)
        assert _find_root_get_route(app) is None

    def test_redirect_returns_307_to_admin(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw(root_redirect="/admin"))
        app = create_app(cfg)
        route = _find_root_get_route(app)
        assert route is not None

        request = MagicMock()
        resp = asyncio.run(route.handler(request))
        assert resp.status_code == 307
        assert resp.headers["Location"] == "/admin"

    def test_redirect_to_custom_path(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw(root_redirect="/dashboard"))
        app = create_app(cfg)
        route = _find_root_get_route(app)
        assert route is not None

        request = MagicMock()
        resp = asyncio.run(route.handler(request))
        assert resp.status_code == 307
        assert resp.headers["Location"] == "/dashboard"

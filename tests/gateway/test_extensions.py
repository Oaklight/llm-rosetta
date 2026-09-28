"""Tests for GatewayExtensions — composable create_app."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.app import GatewayExtensions, create_app
from llm_rosetta.gateway.config import GatewayConfig


def _minimal_cfg(**overrides) -> GatewayConfig:
    raw: dict = {
        "providers": {},
        "server": {"open_on_no_keys": True},
    }
    raw.update(overrides)
    return GatewayConfig(raw)


def _route_patterns(app) -> set[str]:
    """Collect all registered route regex patterns from the app."""
    patterns = set()
    for r in getattr(app, "_routes", []):
        patterns.add(r.pattern.pattern)
    for r in getattr(app, "_static_routes", []):
        patterns.add(r.pattern.pattern)
    return patterns


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_no_extensions(self):
        app = await create_app(_minimal_cfg())
        assert hasattr(app, "auth_state")
        assert hasattr(app, "transport")

    @pytest.mark.asyncio
    async def test_none_extensions(self):
        app = await create_app(_minimal_cfg(), extensions=None)
        assert hasattr(app, "auth_state")


class TestSkipDefaultRoutes:
    @pytest.mark.asyncio
    async def test_skip_default_routes(self):
        ext = GatewayExtensions(skip_default_routes=True)
        app = await create_app(_minimal_cfg(), extensions=ext)
        patterns = _route_patterns(app)
        assert patterns, "route introspection returned nothing"
        assert "^/v1/chat/completions$" not in patterns

    @pytest.mark.asyncio
    async def test_default_routes_present(self):
        app = await create_app(_minimal_cfg())
        patterns = _route_patterns(app)
        assert patterns, "route introspection returned nothing"
        assert "^/v1/chat/completions$" in patterns


class TestCustomTransport:
    @pytest.mark.asyncio
    async def test_custom_transport(self):
        class MockTransport:
            pass

        ext = GatewayExtensions(transport=MockTransport())
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert isinstance(app.transport, MockTransport)  # ty: ignore[unresolved-attribute]

    @pytest.mark.asyncio
    async def test_default_transport(self):
        from llm_rosetta.gateway.transport import HttpTransport

        app = await create_app(_minimal_cfg())
        assert isinstance(app.transport, HttpTransport)  # ty: ignore[unresolved-attribute]


class TestSkipBuiltinAuth:
    @pytest.mark.asyncio
    async def test_skip_builtin_auth(self):
        ext = GatewayExtensions(skip_builtin_auth=True)
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert hasattr(app, "auth_state")
        assert hasattr(app, "internal_token")

    @pytest.mark.asyncio
    async def test_builtin_auth_present_by_default(self):
        app = await create_app(_minimal_cfg())
        assert hasattr(app, "auth_state")


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self):
        ext = GatewayExtensions(enable_rate_limiting=False)
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert not hasattr(app, "rate_limit_state")

    @pytest.mark.asyncio
    async def test_rate_limiting_enabled_by_default(self):
        app = await create_app(_minimal_cfg())
        assert hasattr(app, "rate_limit_state")


class TestMaxBodySize:
    @pytest.mark.asyncio
    async def test_custom_body_size(self):
        ext = GatewayExtensions(max_body_size=100 * 1024 * 1024)
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert app.max_body_size == 100 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_default_body_size(self):
        app = await create_app(_minimal_cfg())
        assert app.max_body_size == 50_000_000


class TestExtraRoutes:
    @pytest.mark.asyncio
    async def test_extra_routes_registered(self):
        async def my_handler(request):
            return {"ok": True}

        ext = GatewayExtensions(
            extra_routes=[("/custom/endpoint", ["GET"], my_handler)],
        )
        app = await create_app(_minimal_cfg(), extensions=ext)
        patterns = _route_patterns(app)
        assert patterns, "route introspection returned nothing"
        assert "^/custom/endpoint$" in patterns


class TestExtraHooks:
    @pytest.mark.asyncio
    async def test_before_request_handlers_wired(self):
        async def my_hook(request):
            return None

        ext = GatewayExtensions(before_hooks=[my_hook])
        app = await create_app(_minimal_cfg(), extensions=ext)
        hooks = getattr(app, "_before_request_handlers", [])
        assert my_hook in hooks, "before_hook not found in app hook list"

    @pytest.mark.asyncio
    async def test_after_request_handlers_wired(self):
        async def my_hook(request, response):
            return response

        ext = GatewayExtensions(after_hooks=[my_hook])
        app = await create_app(_minimal_cfg(), extensions=ext)
        hooks = getattr(app, "_after_request_handlers", [])
        assert my_hook in hooks, "after_hook not found in app hook list"


class TestAdminSetup:
    @pytest.mark.asyncio
    async def test_skip_admin_setup(self):
        ext = GatewayExtensions(skip_admin_setup=True)
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert not hasattr(app, "admin_custom_head")

    @pytest.mark.asyncio
    async def test_admin_branding_forwarded(self):
        ext = GatewayExtensions(
            branding={"title": "My Gateway", "subtitle": "admin"},
            disabled_tabs=["keys"],
        )
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert hasattr(app, "admin_custom_head")
        assert "keys" in app.disabled_tabs  # ty: ignore[unresolved-attribute]


class TestCombinedExtensions:
    @pytest.mark.asyncio
    async def test_full_downstream_config(self):
        class CustomTransport:
            pass

        async def auth_hook(request):
            return None

        async def security_hook(request):
            return None

        async def custom_handler(request):
            return {"ok": True}

        ext = GatewayExtensions(
            transport=CustomTransport(),
            max_body_size=100 * 1024 * 1024,
            skip_default_routes=True,
            skip_builtin_auth=True,
            enable_rate_limiting=False,
            skip_admin_setup=True,
            before_hooks=[auth_hook, security_hook],
            extra_routes=[
                ("/admin/api/custom/env", ["GET"], custom_handler),
                ("/admin/api/custom/env", ["PUT"], custom_handler),
            ],
        )
        app = await create_app(_minimal_cfg(), extensions=ext)
        assert isinstance(app.transport, CustomTransport)  # ty: ignore[unresolved-attribute]
        assert app.max_body_size == 100 * 1024 * 1024
        assert not hasattr(app, "rate_limit_state")
        patterns = _route_patterns(app)
        assert patterns, "route introspection returned nothing"
        assert "^/v1/chat/completions$" not in patterns
        assert "^/admin/api/custom/env$" in patterns

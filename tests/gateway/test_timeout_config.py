"""Tests for configurable gateway timeouts (server.upstream_timeout, server.read_timeout)."""

from __future__ import annotations

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


class TestTimeoutConfig:
    def test_defaults(self):
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.upstream_timeout == 300.0
        assert cfg.read_timeout == 300.0

    def test_custom_upstream_timeout(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout=600))
        assert cfg.upstream_timeout == 600.0

    def test_custom_read_timeout(self):
        cfg = GatewayConfig(_minimal_raw(read_timeout=120))
        assert cfg.read_timeout == 120.0

    def test_both_custom(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout=900, read_timeout=60))
        assert cfg.upstream_timeout == 900.0
        assert cfg.read_timeout == 60.0

    def test_float_coercion_from_int(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout=10, read_timeout=20))
        assert isinstance(cfg.upstream_timeout, float)
        assert isinstance(cfg.read_timeout, float)

    def test_float_coercion_from_string(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout="450", read_timeout="90"))
        assert cfg.upstream_timeout == 450.0
        assert cfg.read_timeout == 90.0

    def test_clamp_zero(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout=0, read_timeout=0))
        assert cfg.upstream_timeout == 1.0
        assert cfg.read_timeout == 1.0

    def test_clamp_negative(self):
        cfg = GatewayConfig(_minimal_raw(upstream_timeout=-10, read_timeout=-5))
        assert cfg.upstream_timeout == 1.0
        assert cfg.read_timeout == 1.0


class TestTimeoutWiring:
    def test_transport_receives_upstream_timeout(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw(upstream_timeout=600))
        app = create_app(cfg)
        assert app.transport._pool._timeout == 600.0  # ty: ignore[unresolved-attribute]

    def test_transport_receives_default_timeout(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw())
        app = create_app(cfg)
        assert app.transport._pool._timeout == 300.0  # ty: ignore[unresolved-attribute]

    def test_app_receives_read_timeout(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw(read_timeout=120))
        app = create_app(cfg)
        assert app.read_timeout == 120.0

    def test_app_receives_default_read_timeout(self):
        from llm_rosetta.gateway.app import create_app

        cfg = GatewayConfig(_minimal_raw())
        app = create_app(cfg)
        assert app.read_timeout == 300.0

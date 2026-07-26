"""Tests for gateway configuration parsing and validation."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.config import GatewayConfig


def _minimal_raw(**server_overrides) -> dict:
    """Return a minimal valid config dict with optional server overrides."""
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


class TestAdminPasswordUnresolvedEnvVar:
    """admin_password must not contain unresolved ${...} placeholders."""

    def test_reject_unresolved_placeholder(self):
        raw = _minimal_raw(admin_password="${ADMIN_PASSWORD}")
        with pytest.raises(ValueError, match="unresolved"):
            GatewayConfig(raw)

    def test_reject_partial_placeholder(self):
        raw = _minimal_raw(admin_password="prefix-${SOME_VAR}-suffix")
        with pytest.raises(ValueError, match="unresolved"):
            GatewayConfig(raw)

    def test_accept_literal_password(self):
        raw = _minimal_raw(admin_password="my-secret-password")
        cfg = GatewayConfig(raw)
        assert cfg.admin_password == "my-secret-password"

    def test_accept_none(self):
        raw = _minimal_raw()
        cfg = GatewayConfig(raw)
        assert cfg.admin_password is None


class TestOpenOnNoKeys:
    """server.open_on_no_keys controls anonymous access when no keys exist."""

    def test_defaults_to_false(self):
        # Secure by default: absent flag → closed.
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.open_on_no_keys is False

    def test_explicit_true(self):
        cfg = GatewayConfig(_minimal_raw(open_on_no_keys=True))
        assert cfg.open_on_no_keys is True

    def test_explicit_false(self):
        cfg = GatewayConfig(_minimal_raw(open_on_no_keys=False))
        assert cfg.open_on_no_keys is False

    def test_coerced_to_bool(self):
        # Truthy/falsy JSON values are normalised to real bools.
        assert GatewayConfig(_minimal_raw(open_on_no_keys=1)).open_on_no_keys is True
        assert GatewayConfig(_minimal_raw(open_on_no_keys=0)).open_on_no_keys is False

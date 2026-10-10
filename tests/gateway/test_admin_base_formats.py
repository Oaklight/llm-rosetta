"""Tests for the ``base_formats`` payload of the admin config route."""

from __future__ import annotations

from llm_rosetta.gateway.admin.routes import config as admin_config
from llm_rosetta.gateway.providers import BASE_FORMATS
from llm_rosetta.shims.providers import load_providers

_KEYS = {
    "name",
    "default_base_url",
    "default_api_key_env",
    "supports_custom_tools",
    "recommended_provider",
}


def test_base_formats_shape():
    load_providers()
    payload = admin_config._base_formats_payload()

    assert [b["name"] for b in payload] == list(BASE_FORMATS)
    for entry in payload:
        assert set(entry) == _KEYS
        assert "base" not in entry
        assert entry["default_base_url"]


def test_recommended_provider_null_when_shim_absent(monkeypatch):
    """An unregistered recommendation degrades to None, not a dangling name."""
    monkeypatch.setattr(admin_config, "get_shim", lambda name: None)

    payload = admin_config._base_formats_payload()

    assert all(b["recommended_provider"] is None for b in payload)


def test_recommended_provider_points_at_shim():
    load_providers()
    by_name = {b["name"]: b for b in admin_config._base_formats_payload()}

    assert by_name["openai_chat"]["recommended_provider"] == "openai"
    assert by_name["google_generate"]["recommended_provider"] == "google"
    # Self-recommendation and no-shim cases degrade to None, so the payload is
    # self-describing and the UI shows no hint.
    assert by_name["anthropic"]["recommended_provider"] is None
    assert by_name["open_responses"]["recommended_provider"] is None


def test_supports_custom_tools_defaults():
    load_providers()
    by_name = {b["name"]: b for b in admin_config._base_formats_payload()}

    assert by_name["openai_chat"]["supports_custom_tools"] is True
    assert by_name["anthropic"]["supports_custom_tools"] is False

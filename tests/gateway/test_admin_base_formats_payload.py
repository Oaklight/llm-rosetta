"""Tests for the ``base_formats`` payload helper of the admin config route.

These pin ``_base_formats_payload()`` directly. The route that serves it
(``get_config`` at ``/admin/api/config``) is not exercised here — only the
helper's shape, the custom-tools defaults, and the recommendation null-guard.
"""

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
        # A default host is expected for the vendor standards, but is not
        # universal — open_responses has none, so only its *type* is pinned.
        assert isinstance(entry["default_base_url"], str)
    by_name = {b["name"]: b for b in payload}
    assert by_name["chat_completions"]["default_base_url"]


def test_recommended_provider_null_when_shim_absent(monkeypatch):
    """With no registered provider names, no recommendation is emitted."""
    monkeypatch.setattr(admin_config, "list_shims", lambda: [])

    payload = admin_config._base_formats_payload()

    assert all(b["recommended_provider"] is None for b in payload)


def test_recommended_provider_points_at_shim():
    load_providers()
    by_name = {b["name"]: b for b in admin_config._base_formats_payload()}

    assert by_name["chat_completions"]["recommended_provider"] == "openai"
    # `google_generate`'s vendor shim is named `google_generate`, so the
    # recommendation is a self-recommendation and degrades to None (no dangling
    # name that isn't in the picker).
    assert by_name["google_generate"]["recommended_provider"] is None
    # Self-recommendation and no-shim cases degrade to None, so the payload is
    # self-describing and the UI shows no hint.
    assert by_name["anthropic"]["recommended_provider"] is None
    assert by_name["open_responses"]["recommended_provider"] is None


def test_supports_custom_tools_defaults():
    load_providers()
    by_name = {b["name"]: b for b in admin_config._base_formats_payload()}

    assert by_name["chat_completions"]["supports_custom_tools"] is True
    assert by_name["anthropic"]["supports_custom_tools"] is False

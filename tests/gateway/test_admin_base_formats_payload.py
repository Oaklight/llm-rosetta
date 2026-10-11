"""Tests for the ``base_formats`` payload helper of the admin config route.

These pin ``_base_formats_payload()`` directly. The route that serves it
(``get_config`` at ``/admin/api/config``) is not exercised here — only the
helper's shape and the custom-tools defaults.
"""

from __future__ import annotations

from llm_rosetta.gateway.admin.routes import config as admin_config
from llm_rosetta.gateway.providers import BASE_FORMATS
from llm_rosetta.shims.providers import load_providers

_KEYS = {
    "name",
    "placeholder_base_url",
    "placeholder_api_key_env",
    "supports_custom_tools",
}


def test_base_formats_shape():
    load_providers()
    payload = admin_config._base_formats_payload()

    assert [b["name"] for b in payload] == list(BASE_FORMATS)
    for entry in payload:
        assert set(entry) == _KEYS
        assert "base" not in entry
        # Vendor URL / key env are *hint* strings; the JS renders them as input
        # placeholders, never as pre-filled values.
        assert isinstance(entry["placeholder_base_url"], str)
        assert isinstance(entry["placeholder_api_key_env"], str)
        assert "default_base_url" not in entry
        assert "default_api_key_env" not in entry
    by_name = {b["name"]: b for b in payload}
    assert by_name["chat_completions"]["placeholder_base_url"]
    assert by_name["chat_completions"]["placeholder_api_key_env"]


def test_supports_custom_tools_defaults():
    load_providers()
    by_name = {b["name"]: b for b in admin_config._base_formats_payload()}

    assert by_name["chat_completions"]["supports_custom_tools"] is True
    assert by_name["messages"]["supports_custom_tools"] is False

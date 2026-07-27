"""Tests for GatewayConfig fine-grained parsing (v6.2.0 refactor)."""

from __future__ import annotations

from llm_rosetta.gateway.config import GatewayConfig


def _cfg(**overrides):
    base = {
        "providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-x",
                "type": "openai_chat",
            }
        },
        "models": {},
    }
    base.update(overrides)
    return GatewayConfig(base)


class TestReasoningOverrides:
    def test_override_extracted(self):
        c = _cfg(
            models={
                "m1": {
                    "provider": "openai",
                    "reasoning_override": {"disabled": "omit"},
                },
                "m2": {"provider": "openai"},
            }
        )
        assert c.model_reasoning_overrides == {"m1": {"disabled": "omit"}}

    def test_empty_override_ignored(self):
        c = _cfg(models={"m1": {"provider": "openai", "reasoning_override": {}}})
        assert c.model_reasoning_overrides == {}


class TestUrlTemplates:
    def test_url_and_stream_templates(self):
        c = _cfg(
            models={
                "m1": {
                    "provider": "openai",
                    "url_template": "{base_url}/chat",
                    "stream_url_template": "{base_url}/stream",
                },
                "m2": {"provider": "openai", "url_template": "{base_url}/only"},
            }
        )
        assert c.model_url_templates == {
            "m1": "{base_url}/chat",
            "m2": "{base_url}/only",
        }
        assert c.model_stream_url_templates == {"m1": "{base_url}/stream"}

    def test_string_model_entry_ignored(self):
        c = _cfg(models={"m1": "openai"})
        assert c.model_url_templates == {}
        assert c.model_stream_url_templates == {}


class TestFlattenSystem:
    def test_gemini_auto_detected(self):
        c = _cfg(models={"gemini-2.0-pro": "openai"})
        assert c.model_flatten_system == {"gemini-2.0-pro": True}

    def test_explicit_true(self):
        c = _cfg(models={"m1": {"provider": "openai", "flatten_system": True}})
        assert c.model_flatten_system == {"m1": True}

    def test_explicit_false_overrides_gemini_default(self):
        c = _cfg(models={"gemini-pro": {"provider": "openai", "flatten_system": False}})
        assert c.model_flatten_system == {"gemini-pro": False}

    def test_no_pattern_match(self):
        c = _cfg(models={"gpt-4": "openai"})
        assert c.model_flatten_system == {}


class TestLegacyApiKey:
    def test_legacy_api_key_wrapped(self):
        c = _cfg(server={"api_key": "sk-legacy"})
        assert c.api_keys == [
            {
                "id": "default",
                "key": "sk-legacy",
                "label": "default",
                "created": "",
            }
        ]
        assert c.api_key == "sk-legacy"
        assert "sk-legacy" in c.api_key_set
        assert c.api_key_labels["sk-legacy"] == "default"

    def test_api_keys_preferred(self):
        c = _cfg(
            server={
                "api_key": "sk-legacy",
                "api_keys": [
                    {"id": "a", "key": "sk-new", "label": "new", "created": ""}
                ],
            }
        )
        assert c.api_keys[0]["key"] == "sk-new"
        assert c.api_key == "sk-new"

    def test_empty_api_keys(self):
        c = _cfg()
        assert c.api_keys == []
        assert c.api_key is None
        assert c.api_key_set == frozenset()

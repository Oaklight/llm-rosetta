"""Tests for custom upstream URL template override (#374).

Covers:
- Per-provider url_template / stream_url_template override
- Per-model url_template / stream_url_template override
- ProviderInfo.with_url_templates() shallow-copy helper
"""

from __future__ import annotations


from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.providers import build_provider_info
from llm_rosetta.gateway.transport.provider_info import ProviderInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_info(
    *,
    name: str = "test",
    api_key: str = "sk-test",
    base_url: str = "https://api.example.com",
    url_template: str = "{base_url}/chat/completions",
    stream_url_template: str | None = None,
) -> ProviderInfo:
    return ProviderInfo(
        name,
        api_key=api_key,
        base_url=base_url,
        auth_header_fn=lambda k: {"Authorization": f"Bearer {k}"},
        url_template=url_template,
        stream_url_template=stream_url_template,
    )


def _make_config(
    *,
    providers: dict | None = None,
    models: dict | None = None,
) -> GatewayConfig:
    """Return a GatewayConfig with sensible defaults."""
    raw = {
        "providers": providers
        or {
            "default": {
                "api_key": "sk-test",
                "base_url": "https://api.example.com",
                "type": "openai_chat",
            },
        },
        "models": models or {"gpt-test": "default"},
        "server": {},
    }
    return GatewayConfig(raw)


# ---------------------------------------------------------------------------
# ProviderInfo.with_url_templates
# ---------------------------------------------------------------------------


class TestProviderInfoWithUrlTemplates:
    def test_override_url_template(self):
        pi = _make_provider_info()
        clone = pi.with_url_templates(url_template="{base_url}/custom/endpoint")
        assert clone.upstream_url("m") == "https://api.example.com/custom/endpoint"
        # Original unchanged
        assert pi.upstream_url("m") == "https://api.example.com/chat/completions"

    def test_override_stream_url_template(self):
        pi = _make_provider_info()
        clone = pi.with_url_templates(stream_url_template="{base_url}/stream/{model}")
        assert (
            clone.upstream_url("gpt-5", stream=True)
            == "https://api.example.com/stream/gpt-5"
        )
        # Non-stream still uses original
        assert (
            clone.upstream_url("gpt-5", stream=False)
            == "https://api.example.com/chat/completions"
        )

    def test_override_both(self):
        pi = _make_provider_info()
        clone = pi.with_url_templates(
            url_template="{base_url}/v2/generate",
            stream_url_template="{base_url}/v2/stream",
        )
        assert clone.upstream_url("m") == "https://api.example.com/v2/generate"
        assert (
            clone.upstream_url("m", stream=True) == "https://api.example.com/v2/stream"
        )

    def test_override_none_is_noop(self):
        pi = _make_provider_info()
        clone = pi.with_url_templates()
        assert clone.upstream_url("m") == pi.upstream_url("m")

    def test_shares_key_ring(self):
        pi = _make_provider_info()
        clone = pi.with_url_templates(url_template="{base_url}/other")
        assert clone.key_ring is pi.key_ring


# ---------------------------------------------------------------------------
# Per-provider url_template override via build_provider_info
# ---------------------------------------------------------------------------


class TestPerProviderUrlTemplate:
    def test_custom_url_template(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://host.example.com",
            "url_template": "{base_url}/v1/endpoints/chat/completions",
        }
        pi = build_provider_info("openai_chat", cfg)
        assert (
            pi.upstream_url("gpt-5")
            == "https://host.example.com/v1/endpoints/chat/completions"
        )

    def test_custom_stream_url_template(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://host.example.com/v1",
            "stream_url_template": "{base_url}/endpoints/stream/{model}",
        }
        pi = build_provider_info("openai_chat", cfg)
        # Non-stream uses registry default
        assert (
            pi.upstream_url("gpt-5", stream=False)
            == "https://host.example.com/v1/chat/completions"
        )
        # Stream uses override
        assert (
            pi.upstream_url("gpt-5", stream=True)
            == "https://host.example.com/v1/endpoints/stream/gpt-5"
        )

    def test_both_templates(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://host.example.com",
            "url_template": "{base_url}/custom/generate",
            "stream_url_template": "{base_url}/custom/stream",
        }
        pi = build_provider_info("openai_chat", cfg)
        assert pi.upstream_url("m") == "https://host.example.com/custom/generate"
        assert (
            pi.upstream_url("m", stream=True)
            == "https://host.example.com/custom/stream"
        )

    def test_no_override_uses_registry_default(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://host.example.com/v1",
        }
        pi = build_provider_info("openai_chat", cfg)
        assert (
            pi.upstream_url("gpt-5") == "https://host.example.com/v1/chat/completions"
        )

    def test_model_placeholder_in_template(self):
        cfg = {
            "api_key": "sk-test",
            "base_url": "https://host.example.com",
            "url_template": "{base_url}/models/{model}/generate",
        }
        pi = build_provider_info("openai_chat", cfg)
        assert (
            pi.upstream_url("my-model")
            == "https://host.example.com/models/my-model/generate"
        )


# ---------------------------------------------------------------------------
# Per-model url_template override via GatewayConfig
# ---------------------------------------------------------------------------


class TestPerModelUrlTemplate:
    def test_model_url_template_override(self):
        cfg = _make_config(
            models={
                "custom-model": {
                    "provider": "default",
                    "url_template": "{base_url}/v2/custom/completions",
                },
            },
        )
        route, pinfo = cfg.resolve("openai_chat", "custom-model")
        assert (
            pinfo.upstream_url("custom-model")
            == "https://api.example.com/v2/custom/completions"
        )

    def test_model_stream_url_template_override(self):
        cfg = _make_config(
            models={
                "custom-model": {
                    "provider": "default",
                    "stream_url_template": "{base_url}/v2/stream/{model}",
                },
            },
        )
        route, pinfo = cfg.resolve("openai_chat", "custom-model")
        # Non-stream: uses provider default
        assert (
            pinfo.upstream_url("custom-model", stream=False)
            == "https://api.example.com/chat/completions"
        )
        # Stream: uses model override
        assert (
            pinfo.upstream_url("custom-model", stream=True)
            == "https://api.example.com/v2/stream/custom-model"
        )

    def test_model_without_override_uses_provider_default(self):
        cfg = _make_config(
            models={
                "normal-model": "default",
                "custom-model": {
                    "provider": "default",
                    "url_template": "{base_url}/v2/custom",
                },
            },
        )
        _, pinfo_normal = cfg.resolve("openai_chat", "normal-model")
        _, pinfo_custom = cfg.resolve("openai_chat", "custom-model")

        assert (
            pinfo_normal.upstream_url("normal-model")
            == "https://api.example.com/chat/completions"
        )
        assert (
            pinfo_custom.upstream_url("custom-model")
            == "https://api.example.com/v2/custom"
        )

    def test_model_override_does_not_mutate_provider(self):
        """Per-model override should not affect other models on the same provider."""
        cfg = _make_config(
            models={
                "model-a": "default",
                "model-b": {
                    "provider": "default",
                    "url_template": "{base_url}/custom",
                },
            },
        )
        # Resolve model-b first (with override)
        _, pinfo_b = cfg.resolve("openai_chat", "model-b")
        # Then model-a (no override) — should use provider default
        _, pinfo_a = cfg.resolve("openai_chat", "model-a")

        assert pinfo_b.upstream_url("model-b") == "https://api.example.com/custom"
        assert (
            pinfo_a.upstream_url("model-a")
            == "https://api.example.com/chat/completions"
        )

    def test_per_model_overrides_per_provider(self):
        """Per-model template takes precedence over per-provider template."""
        cfg = _make_config(
            providers={
                "custom-provider": {
                    "api_key": "sk-test",
                    "base_url": "https://api.example.com",
                    "type": "openai_chat",
                    "url_template": "{base_url}/provider-level",
                },
            },
            models={
                "model-default": "custom-provider",
                "model-override": {
                    "provider": "custom-provider",
                    "url_template": "{base_url}/model-level",
                },
            },
        )
        _, pinfo_default = cfg.resolve("openai_chat", "model-default")
        _, pinfo_override = cfg.resolve("openai_chat", "model-override")

        assert (
            pinfo_default.upstream_url("model-default")
            == "https://api.example.com/provider-level"
        )
        assert (
            pinfo_override.upstream_url("model-override")
            == "https://api.example.com/model-level"
        )

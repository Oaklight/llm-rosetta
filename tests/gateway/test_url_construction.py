"""Tests for URL construction across all endpoint types.

Validates that base_url + path produces correct upstream URLs for
chat, embedding, and rerank endpoints.
"""

from __future__ import annotations

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.embeddings import _resolve_embedding_provider


def _config_with_provider(
    base_url: str,
    *,
    embedding_format: str | None = None,
    embedding_path: str | None = None,
    rerank_format: str | None = None,
    rerank_path: str | None = None,
    extra_models: dict | None = None,
) -> GatewayConfig:
    provider: dict = {
        "api_key": "sk-test",
        "base_url": base_url,
        "type": "openai_chat",
    }
    if embedding_format:
        provider["embedding_format"] = embedding_format
    if embedding_path:
        provider["embedding_path"] = embedding_path
    if rerank_format:
        provider["rerank_format"] = rerank_format
    if rerank_path:
        provider["rerank_path"] = rerank_path

    models: dict = {
        "gpt-4": "test-provider",
        "embed-model": {
            "provider": "test-provider",
            "type": "embedding" if embedding_format else "llm",
        },
    }
    if extra_models:
        models.update(extra_models)

    raw: dict = {
        "providers": {"test-provider": provider},
        "models": models,
        "server": {"open_on_no_keys": True},
    }
    return GatewayConfig(raw)


# ── Chat URL construction ────────────────────────────────────────────


class TestChatUrlConstruction:
    def test_base_url_with_v1(self):
        cfg = _config_with_provider("https://api.openai.com/v1")
        _, pinfo = cfg.resolve("openai_chat", "gpt-4")
        url = pinfo.upstream_url("gpt-4")
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_base_url_without_v1(self):
        cfg = _config_with_provider("https://api.deepseek.com")
        _, pinfo = cfg.resolve("openai_chat", "gpt-4")
        url = pinfo.upstream_url("gpt-4")
        assert url == "https://api.deepseek.com/chat/completions"

    def test_base_url_trailing_slash_stripped(self):
        cfg = _config_with_provider("https://api.openai.com/v1/")
        _, pinfo = cfg.resolve("openai_chat", "gpt-4")
        url = pinfo.upstream_url("gpt-4")
        assert url == "https://api.openai.com/v1/chat/completions"


# ── Embedding URL construction ───────────────────────────────────────


class TestEmbeddingUrlConstruction:
    def test_base_url_no_v1_default_path(self):
        cfg = _config_with_provider(
            "https://api.example.com",
            embedding_format="openai",
        )
        body: dict = {"model": "embed-model", "input": "test"}
        resolved = _resolve_embedding_provider(cfg, "embed-model", body)
        assert resolved is not None
        assert resolved.upstream_url == "https://api.example.com/v1/embeddings"

    def test_base_url_with_v1_default_path(self):
        """base_url with /v1 + default /v1/embeddings — normalized, no double /v1."""
        cfg = _config_with_provider(
            "https://api.openai.com/v1",
            embedding_format="openai",
        )
        body: dict = {"model": "embed-model", "input": "test"}
        resolved = _resolve_embedding_provider(cfg, "embed-model", body)
        assert resolved is not None
        assert resolved.upstream_url == "https://api.openai.com/v1/embeddings"

    def test_base_url_with_v1_custom_path_no_v1(self):
        cfg = _config_with_provider(
            "https://api.openai.com/v1",
            embedding_format="openai",
            embedding_path="/embeddings",
        )
        body: dict = {"model": "embed-model", "input": "test"}
        resolved = _resolve_embedding_provider(cfg, "embed-model", body)
        assert resolved is not None
        assert resolved.upstream_url == "https://api.openai.com/v1/embeddings"

    def test_custom_path_v4_embed(self):
        cfg = _config_with_provider(
            "https://api.custom.com",
            embedding_format="openai",
            embedding_path="/v4/embed",
        )
        body: dict = {"model": "embed-model", "input": "test"}
        resolved = _resolve_embedding_provider(cfg, "embed-model", body)
        assert resolved is not None
        assert resolved.upstream_url == "https://api.custom.com/v4/embed"

    def test_chat_fallback_no_embedding_format(self):
        cfg = _config_with_provider("https://api.openai.com/v1")
        body: dict = {"model": "gpt-4", "input": "test"}
        resolved = _resolve_embedding_provider(cfg, "gpt-4", body)
        assert resolved is not None
        assert resolved.upstream_url == "https://api.openai.com/v1/embeddings"


# ── Rerank URL construction ──────────────────────────────────────────


class TestRerankUrlConstruction:
    def _rerank_upstream_url(self, cfg: GatewayConfig, model: str) -> str:
        route = cfg.resolve_rerank(model)
        return route.provider_info.upstream_url("")

    def test_base_url_no_v1_default_path(self):
        cfg = _config_with_provider(
            "https://api.jina.ai",
            rerank_format="jina",
            extra_models={
                "rerank-model": {"provider": "test-provider", "type": "rerank"},
            },
        )
        url = self._rerank_upstream_url(cfg, "rerank-model")
        assert url == "https://api.jina.ai/v1/rerank"

    def test_base_url_with_v1_default_path(self):
        """base_url with /v1 + default /v1/rerank — normalized, no double /v1."""
        cfg = _config_with_provider(
            "https://api.example.com/v1",
            rerank_format="jina",
            extra_models={
                "rerank-model": {"provider": "test-provider", "type": "rerank"},
            },
        )
        url = self._rerank_upstream_url(cfg, "rerank-model")
        assert url == "https://api.example.com/v1/rerank"

    def test_custom_rerank_path(self):
        cfg = _config_with_provider(
            "https://api.cohere.com",
            rerank_format="cohere",
            rerank_path="/v2/rerank",
            extra_models={
                "rerank-model": {"provider": "test-provider", "type": "rerank"},
            },
        )
        url = self._rerank_upstream_url(cfg, "rerank-model")
        assert url == "https://api.cohere.com/v2/rerank"

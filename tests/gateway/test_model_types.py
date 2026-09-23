"""Tests for the model type registry (gateway/model_types.py)."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.model_types import (
    MODEL_TYPE_REGISTRY,
    ModelTypeDescriptor,
    RouteSpec,
    all_model_types,
    get_model_type,
    register_model_type,
    registered_type_names,
)


# ---------------------------------------------------------------------------
# Built-in registration tests
# ---------------------------------------------------------------------------


class TestBuiltinRegistration:
    """Verify that built-in types are registered at import time."""

    def test_llm_registered(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.name == "llm"

    def test_embedding_registered(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.name == "embedding"

    def test_rerank_registered(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.name == "rerank"

    def test_builtin_count(self) -> None:
        # At minimum, three built-in types must exist
        assert len(all_model_types()) >= 3

    def test_registered_type_names(self) -> None:
        names = registered_type_names()
        assert "llm" in names
        assert "embedding" in names
        assert "rerank" in names


# ---------------------------------------------------------------------------
# LLM descriptor details
# ---------------------------------------------------------------------------


class TestLLMDescriptor:
    """Verify the LLM type descriptor properties."""

    def test_llm_routes(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        paths = [r.path for r in desc.routes]
        assert "/v1/chat/completions" in paths
        assert "/v1/messages" in paths
        assert "/v1/responses" in paths

    def test_llm_formats(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert "openai_chat" in desc.formats
        assert "anthropic" in desc.formats

    def test_llm_pipeline_is_none(self) -> None:
        """LLM uses the full converter pipeline; pipeline should be None."""
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.pipeline is None

    def test_llm_supports_streaming(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.supports_streaming is True


# ---------------------------------------------------------------------------
# Embedding descriptor details
# ---------------------------------------------------------------------------


class TestEmbeddingDescriptor:
    """Verify the embedding type descriptor properties."""

    def test_embedding_routes(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        paths = [r.path for r in desc.routes]
        assert "/v1/embeddings" in paths

    def test_embedding_formats(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert "openai" in desc.formats
        assert "jina" in desc.formats
        assert "cohere" in desc.formats
        assert "voyage" in desc.formats

    def test_embedding_no_streaming(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.supports_streaming is False

    def test_embedding_config_keys(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.config_format_key == "embedding_format"
        assert desc.config_path_key == "embedding_path"
        assert desc.default_path == "/v1/embeddings"


# ---------------------------------------------------------------------------
# Rerank descriptor details
# ---------------------------------------------------------------------------


class TestRerankDescriptor:
    """Verify the rerank type descriptor properties."""

    def test_rerank_routes(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        paths = [r.path for r in desc.routes]
        assert "/v1/rerank" in paths
        assert "/v2/rerank" in paths

    def test_rerank_formats(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert "jina" in desc.formats
        assert "cohere" in desc.formats
        assert "voyage" in desc.formats

    def test_rerank_no_streaming(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.supports_streaming is False

    def test_rerank_config_keys(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.config_format_key == "rerank_format"
        assert desc.config_path_key == "rerank_path"
        assert desc.default_path == "/v1/rerank"


# ---------------------------------------------------------------------------
# Registry CRUD tests
# ---------------------------------------------------------------------------


class TestRegistryCRUD:
    """Test register / get / all / names operations."""

    def test_get_nonexistent_returns_none(self) -> None:
        assert get_model_type("nonexistent_type_xyz") is None

    def test_register_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_model_type(ModelTypeDescriptor(name="llm"))

    def test_register_custom_type(self) -> None:
        """Register a custom type, verify it appears, then clean up."""
        custom_name = "_test_custom_type_42"
        try:
            desc = ModelTypeDescriptor(
                name=custom_name,
                routes=[RouteSpec("/v1/custom", methods=["POST"])],
                formats=["custom_fmt"],
                supports_streaming=False,
                badge_class="cap-badge-custom",
            )
            register_model_type(desc)

            assert get_model_type(custom_name) is not None
            assert custom_name in registered_type_names()
            assert desc in all_model_types()
        finally:
            # Clean up to avoid polluting the global registry
            MODEL_TYPE_REGISTRY.pop(custom_name, None)

    def test_all_model_types_returns_list(self) -> None:
        result = all_model_types()
        assert isinstance(result, list)
        assert all(isinstance(d, ModelTypeDescriptor) for d in result)


# ---------------------------------------------------------------------------
# RouteSpec tests
# ---------------------------------------------------------------------------


class TestRouteSpec:
    """Test RouteSpec dataclass defaults and values."""

    def test_default_methods(self) -> None:
        spec = RouteSpec("/test")
        assert spec.methods == ["POST"]
        assert spec.source_format is None

    def test_custom_methods(self) -> None:
        spec = RouteSpec("/test", methods=["GET", "POST"], source_format="openai_chat")
        assert spec.methods == ["GET", "POST"]
        assert spec.source_format == "openai_chat"


# ---------------------------------------------------------------------------
# Config integration: format lists from registry
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Verify that GatewayConfig format lists align with the registry."""

    def test_embedding_formats_match_registry(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        desc = get_model_type("embedding")
        assert desc is not None
        # The class-level list and registry formats should have the same
        # elements (order may differ).
        assert set(GatewayConfig.EMBEDDING_FORMATS) == set(desc.formats)

    def test_rerank_formats_match_registry(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        desc = get_model_type("rerank")
        assert desc is not None
        assert set(GatewayConfig.RERANK_FORMATS) == set(desc.formats)

    def test_formats_for_type_helper(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        desc = get_model_type("embedding")
        assert desc is not None
        assert GatewayConfig._formats_for_type("embedding") == list(desc.formats)
        assert GatewayConfig._formats_for_type("nonexistent") == []


# ---------------------------------------------------------------------------
# Config resolution: _distribute_typed_models uses registry
# ---------------------------------------------------------------------------


class TestDistributeTypedModels:
    """Verify that _distribute_typed_models works via registry lookup."""

    def test_embedding_model_distributed(self) -> None:
        """An embedding model should be moved from the LLM pool."""
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "jina": {
                    "api_key": "test",
                    "base_url": "https://api.jina.ai",
                    "type": "openai_chat",
                    "embedding_format": "jina",
                },
            },
            "models": {
                "jina-embeddings-v3": {
                    "provider": "jina",
                    "type": "embedding",
                },
                "gpt-4": {
                    "provider": "jina",
                },
            },
        }
        config = GatewayConfig(raw)
        # Embedding model should be in embedding_models, not in LLM models
        assert "jina-embeddings-v3" in config.embedding_models
        assert "jina-embeddings-v3" not in config.models
        # LLM model should remain in the LLM pool
        assert "gpt-4" in config.models

    def test_rerank_model_distributed(self) -> None:
        """A rerank model should be moved from the LLM pool."""
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "jina": {
                    "api_key": "test",
                    "base_url": "https://api.jina.ai",
                    "type": "openai_chat",
                    "rerank_format": "jina",
                },
            },
            "models": {
                "jina-reranker-v2": {
                    "provider": "jina",
                    "type": "rerank",
                },
            },
        }
        config = GatewayConfig(raw)
        assert "jina-reranker-v2" in config.rerank_models
        assert "jina-reranker-v2" not in config.models


# ---------------------------------------------------------------------------
# Admin route integration: _build_model_entry validates type
# ---------------------------------------------------------------------------


class TestAdminModelEntry:
    """Verify that admin _build_model_entry validates against registry."""

    def test_valid_type_accepted(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _build_model_entry

        entry = _build_model_entry({"type": "embedding"}, "provider-a")
        assert entry["type"] == "embedding"

    def test_llm_type_omitted_from_entry(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _build_model_entry

        entry = _build_model_entry({"type": "llm"}, "provider-a")
        assert "type" not in entry

    def test_unknown_type_falls_back_to_llm(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _build_model_entry

        entry = _build_model_entry({"type": "nonexistent_xyz"}, "provider-a")
        # Unknown type should fall back to LLM (no "type" key in entry)
        assert "type" not in entry

    def test_no_type_defaults_to_llm(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _build_model_entry

        entry = _build_model_entry({}, "provider-a")
        assert "type" not in entry


# ---------------------------------------------------------------------------
# Admin config endpoint: model_types metadata
# ---------------------------------------------------------------------------


class TestAdminModelTypeMetadata:
    """Verify the _get_model_type_metadata helper."""

    def test_metadata_contains_builtins(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _get_model_type_metadata

        metadata = _get_model_type_metadata()
        names = [m["name"] for m in metadata]
        assert "llm" in names
        assert "embedding" in names
        assert "rerank" in names

    def test_metadata_has_expected_fields(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _get_model_type_metadata

        metadata = _get_model_type_metadata()
        for entry in metadata:
            assert "name" in entry
            assert "formats" in entry
            assert "supports_streaming" in entry
            assert "badge_class" in entry
            assert "routes" in entry

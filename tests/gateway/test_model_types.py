"""Tests for the model type registry (gateway/model_types.py)."""

from __future__ import annotations


import pytest

from llm_rosetta.gateway.model_types import (
    ModelTypeDescriptor,
    RouteSpec,
    _reset_registry,
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
        # At minimum, four built-in types must exist
        assert len(all_model_types()) >= 4

    def test_registered_type_names(self) -> None:
        names = registered_type_names()
        assert "llm" in names
        assert "embedding" in names
        assert "rerank" in names
        assert "decision" in names


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

    def test_llm_is_llm(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.is_llm is True

    def test_llm_icon_svg(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.icon_svg.startswith("<svg")
        assert "</svg>" in desc.icon_svg

    def test_llm_color(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.color == "var(--green)"


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

    def test_embedding_is_not_llm(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.is_llm is False

    def test_embedding_icon_svg(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.icon_svg.startswith("<svg")
        assert "</svg>" in desc.icon_svg

    def test_embedding_color(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.color == "var(--blue)"


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

    def test_rerank_is_not_llm(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.is_llm is False

    def test_rerank_icon_svg(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.icon_svg.startswith("<svg")
        assert "</svg>" in desc.icon_svg

    def test_rerank_color(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.color == "var(--orange)"


class TestDecisionDescriptor:
    """Verify the decision type descriptor properties."""

    def test_decision_registered(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.name == "decision"

    def test_decision_routes(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        paths = [r.path for r in desc.routes]
        assert "/v1/decision" in paths
        assert "/v1/systemone" in paths

    def test_decision_formats(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert "typesafe" in desc.formats

    def test_decision_no_streaming(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.supports_streaming is False

    def test_decision_config_keys(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.config_format_key == "decision_format"
        assert desc.config_path_key == "decision_path"
        assert desc.default_path == "/v1/systemone"

    def test_decision_badge_class(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.badge_class == "cap-badge-decision"

    def test_decision_is_not_llm(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.is_llm is False

    def test_decision_icon_svg(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.icon_svg.startswith("<svg")
        assert "</svg>" in desc.icon_svg

    def test_decision_color(self) -> None:
        desc = get_model_type("decision")
        assert desc is not None
        assert desc.color == "#8b5cf6"


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
            _reset_registry()

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
        # The static method delegates to the registry — results should
        # match the descriptor's formats.
        assert set(GatewayConfig.embedding_formats()) == set(desc.formats)

    def test_rerank_formats_match_registry(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        desc = get_model_type("rerank")
        assert desc is not None
        assert set(GatewayConfig.rerank_formats()) == set(desc.formats)

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
            assert "config_format_key" in entry
            assert "config_path_key" in entry
            assert "default_path" in entry
            assert "icon_svg" in entry
            assert "color" in entry
            assert "is_llm" in entry

    def test_metadata_llm_is_llm_true(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _get_model_type_metadata

        metadata = _get_model_type_metadata()
        llm_entry = next(m for m in metadata if m["name"] == "llm")
        assert llm_entry["is_llm"] is True
        assert llm_entry["icon_svg"].startswith("<svg")
        assert llm_entry["color"] == "var(--green)"

    def test_metadata_non_llm_is_llm_false(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _get_model_type_metadata

        metadata = _get_model_type_metadata()
        for entry in metadata:
            if entry["name"] != "llm":
                assert entry["is_llm"] is False

    def test_metadata_embedding_config_keys(self) -> None:
        from llm_rosetta.gateway.admin.routes.config import _get_model_type_metadata

        metadata = _get_model_type_metadata()
        emb = next(m for m in metadata if m["name"] == "embedding")
        assert emb["config_format_key"] == "embedding_format"
        assert emb["config_path_key"] == "embedding_path"
        assert emb["default_path"] == "/v1/embeddings"


# ---------------------------------------------------------------------------
# Pipeline field tests
# ---------------------------------------------------------------------------


class TestPipelineField:
    """Verify that non-LLM descriptors have a callable pipeline."""

    def test_embedding_pipeline_returns_handler(self) -> None:
        desc = get_model_type("embedding")
        assert desc is not None
        assert desc.pipeline is not None
        handler = desc.pipeline()
        assert callable(handler)

    def test_rerank_pipeline_returns_handler(self) -> None:
        desc = get_model_type("rerank")
        assert desc is not None
        assert desc.pipeline is not None
        handler = desc.pipeline()
        assert callable(handler)

    def test_llm_pipeline_is_none(self) -> None:
        desc = get_model_type("llm")
        assert desc is not None
        assert desc.pipeline is None


# ---------------------------------------------------------------------------
# _reset_registry helper tests
# ---------------------------------------------------------------------------


class TestResetRegistry:
    """Verify the _reset_registry test helper."""

    def test_reset_clears_custom_types(self) -> None:
        """Custom types should be gone after reset."""
        custom_name = "_test_reset_custom"
        register_model_type(ModelTypeDescriptor(name=custom_name))
        assert get_model_type(custom_name) is not None

        _reset_registry()
        assert get_model_type(custom_name) is None

    def test_reset_preserves_builtins(self) -> None:
        """Built-in types should be re-registered after reset."""
        _reset_registry()
        assert get_model_type("llm") is not None
        assert get_model_type("embedding") is not None
        assert get_model_type("rerank") is not None


# ---------------------------------------------------------------------------
# No-pipeline warning path test
# ---------------------------------------------------------------------------


class TestNoPipelineWarning:
    """Verify that _register_non_llm_routes warns on pipeline=None."""

    def test_no_pipeline_logs_warning(self) -> None:
        """A non-LLM type with pipeline=None should produce a warning."""
        from unittest.mock import MagicMock, patch

        from llm_rosetta.gateway.app import _register_non_llm_routes

        custom_name = "_test_no_pipeline_type"
        try:
            register_model_type(
                ModelTypeDescriptor(
                    name=custom_name,
                    routes=[RouteSpec("/v1/custom_no_pipeline")],
                    formats=["custom"],
                    pipeline=None,  # deliberately no handler
                )
            )

            mock_app = MagicMock()
            mock_config = MagicMock()

            # Patch the gateway logger (propagate=False prevents caplog
            # from seeing it) and capture warning calls directly.
            with patch("llm_rosetta.gateway.app.logger") as mock_logger:
                _register_non_llm_routes(mock_app, mock_config)

            # Verify that a warning was logged mentioning the custom type
            warning_calls = [call.args for call in mock_logger.warning.call_args_list]
            assert any(
                custom_name in str(args) and "no pipeline" in str(args)
                for args in warning_calls
            ), f"Expected warning about {custom_name!r} having no pipeline"

            # Verify that no routes were registered for the pipeline-less type
            route_calls = [call.args[0] for call in mock_app.route.call_args_list]
            assert "/v1/custom_no_pipeline" not in route_calls
        finally:
            _reset_registry()

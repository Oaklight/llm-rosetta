"""Model type registry for streamlined new-type onboarding.

Provides a declarative ``ModelTypeDescriptor`` registry where adding a new
model type (e.g. "decision") requires defining one descriptor plus a
pipeline function — no changes needed in ``app.py``, ``config.py``, or
admin routes.

Built-in types registered at import time: ``llm``, ``embedding``, ``rerank``.

LLM is special: it uses the full converter pipeline via ``_proxy_handler``
in ``app.py``, so its ``pipeline`` is ``None``.  Non-LLM types (embedding,
rerank) point to their dedicated pipeline-handler callables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteSpec:
    """Describes a single HTTP route for a model type.

    Attributes:
        path: URL path (e.g. ``"/v1/embeddings"``).
        methods: HTTP methods (e.g. ``["POST"]``).
        source_format: Optional source format identifier for LLM routes
            (e.g. ``"openai_chat"``).  ``None`` for non-LLM types whose
            handler does its own format detection.
    """

    path: str
    methods: list[str] = field(default_factory=lambda: ["POST"])
    source_format: str | None = None


@dataclass(frozen=True)
class ModelTypeDescriptor:
    """Declarative description of a model type.

    Attributes:
        name: Canonical type name (``"llm"``, ``"embedding"``, ``"rerank"``).
        routes: HTTP routes this type responds on.
        formats: Provider API format strings this type supports.
        pipeline: A zero-argument callable that returns the async handler
            ``(request, config) -> Response`` for non-LLM types.  The
            callable is invoked lazily at route registration time to
            avoid circular imports.  ``None`` for LLM, which uses the
            full converter pipeline via ``_proxy_handler``.
        supports_streaming: Whether streaming is relevant for this type.
        badge_class: CSS class for the admin UI badge.
        config_format_key: Provider config key for the type's format
            (e.g. ``"embedding_format"``).  Used by config resolution.
        config_path_key: Provider config key for the type's endpoint path
            (e.g. ``"embedding_path"``).  Used by config resolution.
        default_path: Default upstream path for this type's endpoint
            (e.g. ``"/v1/embeddings"``).
        icon_svg: Inline SVG string for the capability icon.  Consumed
            by the admin UI for badges and filters.
        color: CSS color string (e.g. ``"var(--green)"``) for badges
            and status dots in the admin UI.
        is_llm: ``True`` only for the ``"llm"`` type, indicating that
            LLM-specific UI elements (shim type, URL templates, tool
            toggles) should be shown.
    """

    name: str
    routes: list[RouteSpec] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    pipeline: Callable[[], Callable[..., Any]] | None = None
    supports_streaming: bool = False
    badge_class: str = ""
    config_format_key: str | None = None
    config_path_key: str | None = None
    default_path: str | None = None
    icon_svg: str = ""
    color: str = ""
    is_llm: bool = False


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

MODEL_TYPE_REGISTRY: dict[str, ModelTypeDescriptor] = {}


def register_model_type(descriptor: ModelTypeDescriptor) -> None:
    """Register a model type descriptor.

    Args:
        descriptor: The type descriptor to register.

    Raises:
        ValueError: If a type with the same name is already registered.
    """
    if descriptor.name in MODEL_TYPE_REGISTRY:
        raise ValueError(f"Model type '{descriptor.name}' is already registered")
    MODEL_TYPE_REGISTRY[descriptor.name] = descriptor


def get_model_type(name: str) -> ModelTypeDescriptor | None:
    """Return the descriptor for *name*, or ``None`` if not registered."""
    return MODEL_TYPE_REGISTRY.get(name)


def all_model_types() -> list[ModelTypeDescriptor]:
    """Return all registered type descriptors in registration order."""
    return list(MODEL_TYPE_REGISTRY.values())


def registered_type_names() -> list[str]:
    """Return the names of all registered model types."""
    return list(MODEL_TYPE_REGISTRY.keys())


def _reset_registry() -> None:
    """Reset the registry to built-in types only.

    Intended for tests that register custom types and need a clean slate
    without leaking state into other tests.
    """
    MODEL_TYPE_REGISTRY.clear()
    _register_builtins()


# ---------------------------------------------------------------------------
# Built-in type registrations
# ---------------------------------------------------------------------------


def _embedding_pipeline() -> Callable[..., Any]:
    """Lazy import wrapper for the embedding handler.

    Returned by the embedding descriptor's ``pipeline`` field.  Defers
    the import of ``gateway.embeddings`` to route-registration time so
    that ``model_types`` can be imported without pulling in the full
    handler dependency graph (which would cause circular imports).
    """
    from llm_rosetta.gateway.pipelines.embeddings import handle_embeddings

    return handle_embeddings


def _rerank_pipeline() -> Callable[..., Any]:
    """Lazy import wrapper for the rerank handler.

    Same rationale as :func:`_embedding_pipeline`.
    """
    from llm_rosetta.gateway.pipelines.rerank import handle_rerank

    return handle_rerank


def _register_builtins() -> None:
    """Register the built-in model types: llm, embedding, rerank."""

    # --- LLM ---
    # LLM uses the full converter pipeline via _proxy_handler in app.py,
    # so pipeline=None.  Its routes are registered explicitly in app.py
    # because each format has its own handler with complex logic.
    register_model_type(
        ModelTypeDescriptor(
            name="llm",
            routes=[
                RouteSpec("/v1/chat/completions", source_format="openai_chat"),
                RouteSpec("/v1/messages", source_format="anthropic"),
                RouteSpec("/v1/responses", source_format="openai_responses"),
                RouteSpec(
                    "/v1beta/models/<path:model_path>",
                    source_format="google_generate",
                ),
                RouteSpec("/v1beta/interactions", source_format="google_interactions"),
            ],
            formats=[
                "openai_chat",
                "anthropic",
                "openai_responses",
                "google_generate",
                "google_interactions",
            ],
            pipeline=None,
            supports_streaming=True,
            badge_class="cap-badge-llm",
            icon_svg=(
                '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
                ' stroke-width="1.5" stroke-linecap="round"'
                ' stroke-linejoin="round"><path d="M2 4c0-1.1.9-2 2-2h8a2'
                ' 2 0 012 2v5a2 2 0 01-2 2H5l-3 3V4z"/></svg>'
            ),
            color="var(--green)",
            is_llm=True,
        )
    )

    # --- Embedding ---
    register_model_type(
        ModelTypeDescriptor(
            name="embedding",
            routes=[
                RouteSpec("/v1/embeddings"),
            ],
            formats=["openai", "cohere", "jina", "voyage"],
            pipeline=_embedding_pipeline,
            supports_streaming=False,
            badge_class="cap-badge-embedding",
            config_format_key="embedding_format",
            config_path_key="embedding_path",
            default_path="/v1/embeddings",
            icon_svg=(
                '<svg viewBox="0 0 16 16" fill="currentColor">'
                '<circle cx="4" cy="5" r="1.3"/>'
                '<circle cx="10" cy="3" r="1.3"/>'
                '<circle cx="12" cy="9" r="1.3"/>'
                '<circle cx="6" cy="11" r="1.3"/>'
                '<circle cx="3" cy="9" r="1"/>'
                '<circle cx="9" cy="7" r="1"/>'
                '<circle cx="13" cy="13" r="1"/></svg>'
            ),
            color="var(--blue)",
        )
    )

    # --- Rerank ---
    register_model_type(
        ModelTypeDescriptor(
            name="rerank",
            routes=[
                RouteSpec("/v1/rerank"),
                RouteSpec("/v2/rerank"),
            ],
            formats=["jina", "cohere", "voyage"],
            pipeline=_rerank_pipeline,
            supports_streaming=False,
            badge_class="cap-badge-rerank",
            config_format_key="rerank_format",
            config_path_key="rerank_path",
            default_path="/v1/rerank",
            icon_svg=(
                '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
                ' stroke-width="1.5" stroke-linecap="round">'
                '<path d="M4 3h8M4 7h6M4 11h10"/>'
                '<path d="M13 1.5l1.5 1.5-1.5 1.5" stroke-linejoin="round"/>'
                '<path d="M3 9.5L1.5 11 3 12.5"'
                ' stroke-linejoin="round"/></svg>'
            ),
            color="var(--orange)",
        )
    )


def _decision_pipeline() -> Callable[..., Any]:
    """Lazy import wrapper for the decision handler."""
    from llm_rosetta.gateway.pipelines.decision import handle_decision

    return handle_decision


def _register_decision() -> None:
    """Register the decision model type.

    Currently TypeSafe System One (Jev) is the only native decision API,
    so ``"typesafe"`` is the sole format.  LLM-backed decision (via
    structured output) uses the chat pipeline, not this route.
    """
    register_model_type(
        ModelTypeDescriptor(
            name="decision",
            routes=[
                RouteSpec("/v1/decision"),
                RouteSpec("/v1/systemone"),
            ],
            formats=["typesafe"],
            pipeline=_decision_pipeline,
            supports_streaming=False,
            badge_class="cap-badge-decision",
            config_format_key="decision_format",
            config_path_key="decision_path",
            default_path="/v1/systemone",
            icon_svg=(
                '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"'
                ' stroke-width="1.5" stroke-linecap="round"'
                ' stroke-linejoin="round">'
                '<path d="M8 2v4M8 6l-4 4M8 6l4 4"/>'
                '<circle cx="8" cy="2" r="1"/>'
                '<circle cx="4" cy="11" r="1"/>'
                '<circle cx="12" cy="11" r="1"/>'
                '<path d="M4 12v2M12 12v2"/></svg>'
            ),
            color="var(--purple)",
        )
    )


_register_builtins()
_register_decision()

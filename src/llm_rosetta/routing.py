"""Routing layer — model/provider/shim resolution contract.

Defines the :class:`ResolvedRoute` data contract and :class:`Router`
protocol for resolving a model name into a target provider, shim config,
capabilities, and reasoning overrides.

This module lives in the core library (no network or gateway deps) so
any consumer (gateway, argo-proxy, CLI tools) can depend on it.

Transport config (e.g. :class:`~gateway.transport.ProviderInfo`) is
**not** part of the route — it's transport-specific and returned
separately by each :class:`Router` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_rosetta.auto_detect import ProviderType


@dataclass(slots=True, frozen=True)
class ResolvedRoute:
    """Result of resolving a model name to a target route.

    Contains all the information the pipeline and transport layers need
    to process a request, except transport-specific config (e.g.
    ``ProviderInfo``), which is returned alongside by the router.

    Attributes:
        source_provider: API standard of the incoming request
            (e.g. ``"openai_chat"``).
        target_provider: API standard of the upstream provider
            (e.g. ``"anthropic"``).
        provider_name: User-configured provider name
            (e.g. ``"deepseek"``, ``"my-openai"``).
        shim_name: Shim/type identifier for transform lookup
            (e.g. ``"volcengine--openai_chat"``), or ``None``.
        upstream_model: Actual model ID to send upstream, or ``None``
            when the gateway model name is used as-is.
        model_capabilities: Declared capabilities of the model
            (e.g. ``["text", "vision"]``).
        reasoning_override: Per-model reasoning config override from
            the admin UI / config, or ``None``.
    """

    source_provider: ProviderType
    target_provider: ProviderType
    provider_name: str
    shim_name: str | None = None
    upstream_model: str | None = None
    model_capabilities: list[str] = field(default_factory=lambda: ["text"])
    reasoning_override: dict[str, Any] | None = None


class Router(Protocol):
    """Interface for resolving a model name to a target route.

    Each deployment (gateway, argo-proxy, etc.) provides its own
    implementation backed by its config system.  The pipeline and
    transport layers consume the resulting :class:`ResolvedRoute`
    without knowing how it was produced.
    """

    def resolve(self, source_provider: ProviderType, model: str) -> ResolvedRoute:
        """Resolve *model* to a :class:`ResolvedRoute`.

        Implementations may return additional transport-specific config
        (e.g. ``tuple[ResolvedRoute, ProviderInfo]``), but the
        protocol contract only guarantees ``ResolvedRoute``.  Callers
        that need transport config should use the concrete type.

        Args:
            source_provider: API standard of the incoming request.
            model: Model name as specified by the client.

        Returns:
            A :class:`ResolvedRoute` with target provider, shim,
            capabilities, and reasoning config.

        Raises:
            KeyError: If the model is not in the routing table.
        """
        ...

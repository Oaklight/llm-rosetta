"""Unified shim pipeline — single entry points for shim-driven transforms.

Provides two public functions that encapsulate all shim-driven behavior
so consumers (gateway, argo-proxy, etc.) don't need to manually wire
each shim field.  This is the "Phase 2a" of the conversion pipeline:

- :func:`setup_shim_context` — inject reasoning config into the
  :class:`~llm_rosetta.converters.base.context.ConversionContext`
  before conversion begins.
- :func:`apply_shim_to_ir` — apply all IR-level shim transforms
  (image stripping, image limit, tool call unwind) after
  source → IR conversion.

Both functions accept a ``ProviderShim`` instance, a registered shim
name (``str``), or ``None`` (no-op).
"""

from __future__ import annotations

import re
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims.provider_shim import ProviderShim, ReasoningCapability, get_shim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_shim(shim: ProviderShim | str | None) -> ProviderShim | None:
    """Resolve a shim argument to a ProviderShim instance.

    Args:
        shim: A ProviderShim instance, a registered name, or None.

    Returns:
        The resolved ProviderShim, or None.
    """
    if shim is None:
        return None
    if isinstance(shim, ProviderShim):
        return shim
    return get_shim(shim)


def _apply_config_reasoning_override(
    base: ReasoningCapability,
    override: dict[str, Any],
) -> ReasoningCapability:
    """Merge config-level reasoning overrides onto a base capability.

    Only fields present in *override* are replaced; the rest inherit
    from *base*.

    Args:
        base: The base reasoning capability from the shim.
        override: A dict of field overrides (e.g. from admin UI).

    Returns:
        A new ReasoningCapability with merged values.
    """
    return ReasoningCapability(
        disabled=override.get("disabled", base.disabled),
        effort_field=override.get("effort_field", base.effort_field),
        max_effort=override.get("max_effort", base.max_effort),
        thinking_type=override.get("thinking_type", base.thinking_type),
        unsigned_reasoning_blocks=override.get(
            "unsigned_reasoning_blocks", base.unsigned_reasoning_blocks
        ),
        effort_map=override.get("effort_map", base.effort_map),
        budget_tokens_default_ratio=override.get(
            "budget_tokens_default_ratio", base.budget_tokens_default_ratio
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_shim_context(
    ctx: ConversionContext,
    shim: ProviderShim | str | None,
    *,
    model: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Inject shim-driven options into a ConversionContext.

    Currently injects ``reasoning_cap`` so converters produce the correct
    thinking/reasoning output for the target provider.

    Resolution priority (highest first):

    1. *config_override* — per-model override from external config
       (e.g. gateway admin UI).
    2. ``shim.model_reasoning[model]`` — per-model override from the
       provider YAML.
    3. ``shim.reasoning`` — provider-level default.

    Args:
        ctx: Conversion context to mutate.
        shim: ProviderShim instance, registered name, or None (no-op).
        model: Upstream model ID (for per-model reasoning overrides).
        config_override: External reasoning override (highest priority).
    """
    resolved = _resolve_shim(shim)
    if resolved is None:
        return

    cap = resolved.reasoning
    # Model-level override (keyed by upstream model ID)
    if model and resolved.model_reasoning and model in resolved.model_reasoning:
        cap = resolved.model_reasoning[model]
    # Config-level override (from admin UI, keyed by gateway model name)
    if cap is not None and config_override:
        cap = _apply_config_reasoning_override(cap, config_override)
    if cap is not None:
        ctx.options["reasoning_cap"] = cap


def apply_shim_to_ir(
    ir_request: dict[str, Any],
    shim: ProviderShim | str | None,
    *,
    upstream_model: str | None = None,
    model_capabilities: list[str] | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Apply all shim-driven IR-level transforms in canonical order.

    Operations applied (in order):

    1. **Strip non-vision images** — if *model_capabilities* is provided
       and does not include ``"vision"``, replace all images with text
       placeholders.  Driven by the caller, not by the shim.
    2. **Enforce image count limit** — if the shim declares
       ``max_images`` (and the upstream model matches
       ``max_images_pattern`` when set), truncate excess images.
    3. **Unwind parallel tool calls** — if the shim declares
       ``unwind_parallel_tool_calls`` (and the upstream model matches
       ``unwind_parallel_tool_calls_pattern`` when set), split parallel
       tool calls into sequential call-result pairs.

    All operations are no-ops when the corresponding shim field is unset
    or when the pattern guard doesn't match.

    Args:
        ir_request: The IR request dict (modified in-place where possible).
        shim: ProviderShim instance, registered name, or None (no-op).
        upstream_model: The upstream model ID (for pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
            When ``None``, the non-vision image stripping step is skipped.
        request_id: Request identifier for logging.

    Returns:
        The (possibly modified) IR request dict.
    """
    # 1. Strip images for non-vision models (caller-driven, not shim-driven)
    if model_capabilities is not None and "vision" not in model_capabilities:
        from llm_rosetta.converters.base.helpers.image_limit import (
            strip_images_for_non_vision,
        )

        ir_request = strip_images_for_non_vision(
            ir_request, model=upstream_model or "", request_id=request_id
        )

    resolved = _resolve_shim(shim)
    if resolved is None:
        return ir_request

    # 2. Enforce per-shim image count limit
    if resolved.max_images is not None:
        apply_limit = True
        if resolved.max_images_pattern is not None:
            apply_limit = bool(
                upstream_model
                and re.search(resolved.max_images_pattern, upstream_model)
            )
        if apply_limit:
            from llm_rosetta.converters.base.helpers.image_limit import truncate_images

            ir_request = truncate_images(
                ir_request, resolved.max_images, request_id=request_id
            )

    # 3. Unwind parallel tool calls
    if resolved.unwind_parallel_tool_calls:
        apply_unwind = True
        if resolved.unwind_parallel_tool_calls_pattern is not None:
            apply_unwind = bool(
                upstream_model
                and re.search(
                    resolved.unwind_parallel_tool_calls_pattern, upstream_model
                )
            )
        if apply_unwind:
            from llm_rosetta.converters.base.helpers.tool_call_unwind import (
                unwind_parallel_tool_calls_ir,
            )

            ir_request = unwind_parallel_tool_calls_ir(ir_request)

    return ir_request

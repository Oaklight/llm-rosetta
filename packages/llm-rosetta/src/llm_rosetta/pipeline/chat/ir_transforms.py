"""Chat-pipeline IR transform entry points and their deprecated aliases."""

import logging
import warnings
from typing import Any, Protocol

from llm_rosetta.ir.passes import (
    enforce_reasoning,
)
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.conversion_warnings import WarningSink
from llm_rosetta.contracts.shim import ProviderShim
from llm_rosetta.contracts.resolver import resolve_shim
from llm_rosetta.ir.primitives.ir import (
    TransformContext,
    apply_ir_transforms as _apply_ir_transforms_exec,
)

logger = logging.getLogger("llm_rosetta.pipeline.chat")


class _HasIntrinsicTransforms(Protocol):
    _INTRINSIC_IR_TRANSFORMS: tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_context(
    ctx: ConversionContext,
    shim: ProviderShim | str | None,
    *,
    model: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Deprecated: use :func:`llm_rosetta.ir.passes.enforce_reasoning`."""

    warnings.warn(
        "configure_context is deprecated; use capabilities.enforce_reasoning()",
        DeprecationWarning,
        stacklevel=2,
    )
    enforce_reasoning(ctx, shim, model=model, config_override=config_override)


def apply_ir_transforms(
    ir_request: dict[str, Any],
    shim: ProviderShim | str | None,
    *,
    upstream_model: str | None = None,
    model_capabilities: list[str] | None = None,
    request_id: str = "-",
    hoist_system_messages: bool = True,
    converter: _HasIntrinsicTransforms | None = None,
    warnings: WarningSink | None = None,
) -> dict[str, Any]:
    """Apply converter-intrinsic and shim-driven IR-level transforms.

    Intrinsic transforms (from ``converter._INTRINSIC_IR_TRANSFORMS``) run
    first — they are format-level prerequisites (e.g. hoisting system
    messages for APIs that don't support them inline).  Shim transforms
    follow, with duplicates removed so an idempotent transform is not
    executed twice.

    Args:
        ir_request: The IR request dict.  Some operations mutate in-place,
            others return a new dict — **always use the return value**.
        shim: ProviderShim instance, registered name, or None.
        upstream_model: The upstream model ID (for pattern matching).
        model_capabilities: Model capability list (e.g. ``["text", "vision"]``).
            When ``None``, transforms that check capabilities treat the
            model as unknown and skip capability-dependent operations.
        request_id: Request identifier for logging.
        hoist_system_messages: Whether hoisting transforms should fire.
        converter: Target converter instance.  When provided, its
            ``_INTRINSIC_IR_TRANSFORMS`` are prepended to the transform
            chain.
        warnings: The conversion's sink.  A transform that rewrites the
            request reports the change through it; ``None`` leaves the
            reports to the conversion in progress, then to the log.

    Returns:
        The IR request dict after all applicable transforms.  Always
        assign the return value: ``ir = apply_ir_transforms(ir, shim, ...)``.
    """
    intrinsic: tuple = getattr(converter, "_INTRINSIC_IR_TRANSFORMS", ())
    resolved = resolve_shim(shim)
    shim_transforms: tuple = resolved.ir_transforms if resolved else ()

    # Merge: intrinsic first, then shim (dedup by repr to avoid re-running
    # the same idempotent transform).
    if intrinsic and shim_transforms:
        seen = {repr(t) for t in intrinsic}
        combined = intrinsic + tuple(t for t in shim_transforms if repr(t) not in seen)
    else:
        combined = intrinsic or shim_transforms

    if not combined:
        return ir_request

    ctx = TransformContext(
        model=upstream_model or "",
        model_capabilities=model_capabilities,
        request_id=request_id,
        hoist_system_messages=hoist_system_messages,
        warnings=warnings,
    )
    return _apply_ir_transforms_exec(combined, ir_request, ctx)


# ---------------------------------------------------------------------------
# Deprecated aliases (backward compatibility with v0.6.x)
# ---------------------------------------------------------------------------


def setup_shim_context(*args: Any, **kwargs: Any) -> None:
    """Deprecated: use :func:`configure_context`."""

    warnings.warn(
        "setup_shim_context is deprecated; use configure_context()",
        DeprecationWarning,
        stacklevel=2,
    )
    return configure_context(*args, **kwargs)


def apply_shim_to_ir(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated: use :func:`apply_ir_transforms`."""

    warnings.warn(
        "apply_shim_to_ir is deprecated; use apply_ir_transforms()",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_ir_transforms(*args, **kwargs)

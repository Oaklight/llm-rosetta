"""Capability enforcement — adapt IR input to model capabilities.

This module handles **platform-level** capability constraints that apply
regardless of provider dialect.  Every model has capabilities (vision,
audio, tools, reasoning, etc.) and the pipeline must adapt the IR
request to match what the model can actually process.

This is distinct from **shim transforms** (provider-specific dialect
adaptation) and from **converter logic** (API-standard translation).

Functions follow the ``enforce_*`` naming convention:

- :func:`enforce_reasoning` — configure reasoning output mode (pre-IR)
- :func:`strip_reasoning_for_non_reasoning` — strip reasoning for non-reasoning models (post-IR)
- :func:`enforce_vision` — strip images for non-vision models (post-IR)
- :func:`enforce_custom_tools` — downgrade custom tools for non-supporting providers (post-IR)

Called by :class:`~llm_rosetta.pipeline.ConversionPipeline` at the
appropriate pipeline stages.
"""

from __future__ import annotations

import copy

import logging
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    resolve_shim,
)

logger = logging.getLogger(__name__)


def _apply_config_reasoning_override(
    base: ReasoningCapability,
    override: dict[str, Any],
) -> ReasoningCapability:
    """Merge config-level reasoning overrides onto a base capability.

    Only fields present in *override* are replaced; the rest inherit
    from *base*.

    Legacy compat: accepts old field names (``thinking_type``,
    ``disabled``, ``budget_tokens_default_ratio``) from existing
    admin UI configs and maps them to the new schema.
    """
    raw_range = override.get("effort_range", base.effort_range)
    effort_range = tuple(raw_range) if isinstance(raw_range, list) else raw_range

    # Legacy compat: old admin UI sends thinking_type/disabled as scalars.
    thinking_modes = override.get("thinking_modes", base.thinking_modes)
    if "thinking_type" in override and "thinking_modes" not in override:
        tt = override["thinking_type"]
        if base.thinking_modes:
            thinking_modes = {**base.thinking_modes}
            for ir_mode, prov_val in list(thinking_modes.items()):
                if ir_mode in ("auto", "enabled"):
                    thinking_modes[ir_mode] = tt
        else:
            thinking_modes = {"auto": tt, "enabled": tt, "disabled": "disabled"}

    budget = override.get(
        "budget_ratio",
        override.get("budget_tokens_default_ratio", base.budget_ratio),
    )

    return ReasoningCapability(
        thinking_modes=thinking_modes,
        thinking_default=override.get("thinking_default", base.thinking_default),
        effort_field=override.get("effort_field", base.effort_field),
        effort_range=effort_range,
        budget_ratio=budget,
        visibility_modes=override.get("visibility_modes", base.visibility_modes),
        unsigned_blocks=override.get("unsigned_blocks", base.unsigned_blocks),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enforce_reasoning(
    ctx: ConversionContext,
    shim: ProviderShim | str | None,
    *,
    model: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Configure reasoning capability in the conversion context.

    Injects ``reasoning_cap`` into *ctx* so converters produce the
    correct thinking/reasoning output for the target provider.

    Must be called **before** source → IR conversion (converters read
    ``ctx.options["reasoning_cap"]`` during parsing).

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
    resolved = resolve_shim(shim)
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


def strip_reasoning_for_non_reasoning(
    ir_request: dict[str, Any],
    *,
    model_capabilities: list[str] | None = None,
    model: str = "",
    request_id: str = "-",
) -> dict[str, Any]:
    """Strip reasoning config from the IR request if the model lacks reasoning capability.

    Must be called **after** source → IR conversion (operates on the IR
    dict, not the raw provider body).

    No-op when *model_capabilities* is ``None`` (unknown) or includes
    ``"reasoning"``.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        model_capabilities: Declared capabilities of the model.
        model: Upstream model identifier (for logging).
        request_id: Request identifier (for logging).

    Returns:
        The IR request with reasoning config removed, or the original
        request if the model has reasoning capability.
    """
    if model_capabilities is None or "reasoning" in model_capabilities:
        return ir_request

    reasoning = ir_request.pop("reasoning", None)
    if reasoning:
        logger.info(
            "[%s] model=%s: stripped reasoning config (model lacks 'reasoning' capability)",
            request_id,
            model,
        )

    return ir_request


def enforce_vision(
    ir_request: dict[str, Any],
    *,
    model_capabilities: list[str] | None = None,
    model: str = "",
    request_id: str = "-",
) -> dict[str, Any]:
    """Strip images from the IR request if the model lacks vision capability.

    Must be called **after** source → IR conversion (operates on the IR
    dict, not the raw provider body).

    No-op when *model_capabilities* is ``None`` (unknown) or includes
    ``"vision"``.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        model_capabilities: Declared capabilities of the model.
        model: Upstream model identifier (for logging).
        request_id: Request identifier (for logging).

    Returns:
        The IR request with images replaced by text placeholders, or
        the original request if the model has vision capability.
    """
    if model_capabilities is None or "vision" in model_capabilities:
        return ir_request

    from llm_rosetta.converters.base.helpers.image_limit import (
        strip_images_for_non_vision,
    )

    return strip_images_for_non_vision(ir_request, model=model, request_id=request_id)


# ---------------------------------------------------------------------------
# Custom tool enforcement
# ---------------------------------------------------------------------------

_CUSTOM_TOOL_SYNTH_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "Free-form text input for the tool.",
        },
    },
    "required": ["input"],
}


def enforce_custom_tools(
    ir_request: dict[str, Any],
    *,
    shim: ProviderShim | str | None = None,
    config_override: bool | None = None,
) -> dict[str, Any]:
    """Downgrade custom tools to functions for providers that lack support.

    Must be called **after** source → IR conversion.  When the effective
    supports value is False, each IR tool with ``type == "custom"`` is
    rewritten to ``type = "function"`` with a synthesised JSON schema
    wrapping the input as ``{"input": string}``.  The original type is
    preserved in ``metadata["provider_type"]`` so the response path can
    restore it.

    Resolution: ``config_override`` carries the pre-resolved value from
    ``config.resolve()`` (config override > shim default > False) and is
    authoritative when not ``None``.  Direct callers may omit it and pass
    ``shim`` instead, in which case the shim's default is used.

    ``None`` and ``False`` are distinct: ``None`` means "unset, fall back to
    the shim", while ``False`` means "this provider does not support custom
    tools" and must not be overridden by a shim that claims otherwise.

    No-op when the effective value is True.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        shim: Provider shim (name or object).  Used as fallback only when
            ``config_override`` is ``None``.
        config_override: Pre-resolved supports_custom_tools value, or
            ``None`` to defer to ``shim``.

    Returns:
        The IR request with custom tools downgraded, or the original
        request unchanged.
    """
    supports = config_override
    if supports is None and shim is not None:
        resolved = resolve_shim(shim) if isinstance(shim, str) else shim
        supports = resolved.supports_custom_tools if resolved is not None else False
    if supports:
        return ir_request

    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    # Check if any custom tools exist before copying
    if not any(isinstance(t, dict) and t.get("type") == "custom" for t in tools):
        return ir_request

    # Deep-copy tools to avoid mutating cached entries
    import copy

    tools = copy.deepcopy(tools)
    ir_request["tools"] = tools

    changed = False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "custom":
            continue
        changed = True
        tool["type"] = "function"

        meta = tool.get("metadata") or {}
        meta["_downgraded_from"] = "custom"
        fmt = meta.get("format")
        tool["metadata"] = meta

        if not tool.get("parameters"):
            tool["parameters"] = copy.deepcopy(_CUSTOM_TOOL_SYNTH_PARAMS)

        if fmt:
            fmt_type = fmt.get("type", "unknown")
            fmt_syntax = fmt.get("syntax") or fmt.get("grammar", {}).get("syntax", "")
            hint = f"[Output format: {fmt_type}"
            if fmt_syntax:
                hint += f", syntax: {fmt_syntax}"
            hint += "]"
            desc = tool.get("description", "")
            tool["description"] = f"{desc}\n\n{hint}" if desc else hint

    if changed:
        tc = ir_request.get("tool_choice")
        if isinstance(tc, dict) and tc.get("tool_type") == "custom":
            del tc["tool_type"]

    return ir_request


def get_custom_tool_names(ir_request: dict[str, Any]) -> frozenset[str]:
    """Return names of tools that were downgraded from custom to function.

    Looks for ``metadata._downgraded_from == "custom"`` on each tool
    definition in the IR request — the marker set by
    :func:`enforce_custom_tools`.
    """
    names: set[str] = set()
    for tool in ir_request.get("tools") or []:
        if (
            isinstance(tool, dict)
            and (tool.get("metadata") or {}).get("_downgraded_from") == "custom"
        ):
            name = tool.get("name")
            if name:
                names.add(name)
    return frozenset(names)


def restore_custom_tool_calls(
    ir_response: dict[str, Any],
    *,
    custom_tool_names: frozenset[str],
) -> None:
    """Re-tag downgraded tool calls as custom in the IR response.

    Mutates *ir_response* in place.  For each tool call whose
    ``tool_name`` is in *custom_tool_names*, sets
    ``tool_type = "custom"``.  Called on the non-streaming response
    path after Target → IR conversion.

    Input unwrapping (extracting raw text from the ``{"input": ...}``
    JSON wrapper) is handled downstream by the source converter's
    IR → provider serialisation for custom tool calls.
    """
    if not custom_tool_names:
        return

    for choice in ir_response.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and part.get("tool_name") in custom_tool_names
            ):
                part["tool_type"] = "custom"

    for msg in ir_response.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and part.get("tool_name") in custom_tool_names
            ):
                part["tool_type"] = "custom"


def unwrap_custom_tool_input(raw: str) -> str:
    """Recover a custom tool's raw text from the downgraded JSON wrapper.

    The synthesised schema is ``{"input": string}``, so a well-formed
    call arrives as ``{"input": "..."}``.  Anything else is returned
    untouched.
    """
    import json

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, dict) and list(parsed.keys()) == ["input"]:
        value = parsed["input"]
        return value if isinstance(value, str) else json.dumps(value)
    return raw


# ---------------------------------------------------------------------------
# Oversized tool description relocation
# ---------------------------------------------------------------------------

_POINTER_TEMPLATE = "[Full documentation for '{}' provided separately.]"


def relocate_oversized_tool_descriptions(
    ir_request: dict[str, Any],
    *,
    max_description_length: int | None = None,
    request_id: str = "-",
) -> dict[str, Any]:
    """Move oversized tool descriptions into a late system message.

    When a tool's description exceeds *max_description_length*, the full
    text is moved into an appended system message and the tool's
    description is replaced with a short pointer.  The downstream
    ``hoist_late_system_messages`` transform adapts the system message
    for each provider automatically.

    No-op when *max_description_length* is ``None`` (provider has no
    limit) or no tools exceed the threshold.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        max_description_length: Maximum allowed description length, or
            ``None`` to skip relocation entirely.
        request_id: For logging.

    Returns:
        The IR request with oversized descriptions relocated, or the
        original request unchanged.
    """
    if max_description_length is None:
        return ir_request

    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    oversized: list[tuple[int, str, str]] = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        desc = tool.get("description", "")
        if len(desc) > max_description_length:
            oversized.append((i, tool.get("name", f"tool_{i}"), desc))

    if not oversized:
        return ir_request

    logger = logging.getLogger(__name__)

    tools = copy.deepcopy(tools)
    ir_request = {**ir_request, "tools": tools}

    sections: list[str] = []
    for idx, name, full_desc in oversized:
        tools[idx]["description"] = _POINTER_TEMPLATE.format(name)
        meta = tools[idx].get("metadata") or {}
        meta["_description_relocated"] = True
        tools[idx]["metadata"] = meta
        sections.append(f"## Tool: {name}\n\n{full_desc}")

    relocated_text = "\n\n---\n\n".join(sections)
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": [{"type": "text", "text": relocated_text}],
    }

    messages = list(ir_request.get("messages", []))
    messages.append(system_msg)
    ir_request["messages"] = messages

    logger.debug(
        "[%s] relocated %d oversized tool description(s) to late system message",
        request_id,
        len(oversized),
    )
    return ir_request

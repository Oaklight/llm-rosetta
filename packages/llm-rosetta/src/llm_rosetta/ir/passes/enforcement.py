"""Capability enforcement — adapt an IR request to what the target can process.

Platform-level constraints that apply regardless of provider dialect: reasoning,
vision, intrinsic (provider-hosted) tools, and custom tools.  Driven by the model
capabilities and the target shim, never by a provider name.
"""

from __future__ import annotations

import copy

import logging
from typing import Any

from ..tools.intrinsic import (
    intrinsic_kind_of as _intrinsic_kind_of,
    make_intrinsic_tool_definition as _make_intrinsic_tool_definition,
)

from llm_rosetta.converters.base.context import ConversionContext, StreamContext
from ..primitives.image_limit import strip_images_for_non_vision
from ..primitives.parts import iter_tool_call_parts
from llm_rosetta.conversion_warnings import WarningSink, report_drop
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    match_model_reasoning,
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
        replay_missing=override.get("replay_missing", base.replay_missing),
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
    2. ``shim.model_reasoning`` — per-model override from the provider YAML.
       An exact upstream-ID key wins; otherwise the keys are tried in
       declaration order as shell-style patterns, so a family such as
       ``deepseek-*`` covers every model that lane serves.
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

    cap = match_model_reasoning(resolved, model) or resolved.reasoning
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


def backfill_reasoning_replay(
    ir_request: dict[str, Any],
    *,
    reasoning_cap: ReasoningCapability | None = None,
    model: str = "",
    request_id: str = "-",
) -> dict[str, Any]:
    """Give every assistant turn a reasoning part the target must replay.

    A thinking-mode upstream may require the model's reasoning to be passed
    back: for a request carrying ``tools``, the history must carry the
    reasoning field on every assistant message — tool call or not — or the
    upstream answers ``400 The `reasoning_content` in the thinking mode must
    be passed back to the API``.  A history ending on a ``user`` turn needs
    nothing; one ending on a ``tool`` result trips the check even without the
    ``tools`` parameter, and one ending on an assistant turn only does when
    tools are sent.

    The requirement is declared by the target model
    (``ReasoningCapability.replay_missing == "fill_empty"``), not by a
    vendor transform, so every endpoint that fronts the same model inherits
    it.  The pass is format-neutral: it appends an empty ``reasoning`` part to
    each assistant message that has none, and the target converter renders it
    in its own dialect (Chat Completions' ``reasoning_content: ""``).  An
    empty value satisfies the upstream — verified live on ``deepseek-flash``
    — which keeps a client that dropped the reasoning when replaying history
    across a format boundary from failing.

    Must be called **after** source → IR conversion: an assistant message
    that already carries a reasoning part is left untouched, so the pass is
    idempotent.  Like the other enforcement passes it mutates *ir_request* in
    place and returns it; callers must use the return value, and must not
    keep a reference to the pre-pass IR for a later comparison.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        reasoning_cap: The target's resolved reasoning capability.  ``None``
            or ``replay_missing == "as_is"`` is a no-op.
        model: Upstream model identifier (for logging).
        request_id: Request identifier (for logging).

    Returns:
        The IR request with an empty reasoning part on each assistant message
        that lacked one, or the original request when the target does not
        require a replay.
    """
    if reasoning_cap is None or reasoning_cap.replay_missing != "fill_empty":
        return ir_request

    messages = ir_request.get("messages")
    if not messages or not isinstance(messages, list):
        return ir_request
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") == "user":
        return ir_request
    # Ending on a tool result trips the check even without the tools
    # parameter; a trailing assistant turn only does when tools are sent.
    if last.get("role") != "tool" and not ir_request.get("tools"):
        return ir_request

    filled = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "reasoning"
            for part in content
        ):
            continue
        content.insert(0, {"type": "reasoning", "reasoning": ""})
        filled += 1

    if filled:
        logger.debug(
            "[%s] model=%s: backfilled an empty reasoning part on %d assistant "
            "message(s); the client replayed history without the model's reasoning",
            request_id,
            model,
            filled,
        )
    return ir_request


def enforce_vision(
    ir_request: dict[str, Any],
    *,
    model_capabilities: list[str] | None = None,
    model: str = "",
    request_id: str = "-",
    warnings: WarningSink | None = None,
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
        warnings: Sink that receives the report, so a stripped image is a
            recorded loss rather than only a log line.

    Returns:
        The IR request with images replaced by text placeholders, or
        the original request if the model has vision capability.
    """
    if model_capabilities is None or "vision" in model_capabilities:
        return ir_request

    return strip_images_for_non_vision(
        ir_request, model=model, request_id=request_id, warnings=warnings
    )


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


def resolve_intrinsic_tools(
    ir_request: dict[str, Any],
    *,
    shim: ProviderShim | str | None = None,
    same_format: bool = False,
    allow_name_promotion: bool = False,
    request_id: str = "-",
    warnings: Any = None,
) -> dict[str, Any]:
    """Reconcile intrinsic tool definitions with the target provider's shim.

    A provider's shim declares which intrinsic kinds it supports
    (``ToolsConfig.intrinsic_tools``).  This function:

    - **promotes** a function tool whose name matches a supported intrinsic
      kind into an IR ``type="intrinsic"`` definition (for client formats,
      like OpenAI Chat, that cannot declare server tools natively);
    - **drops** an intrinsic definition whose kind the target does not
      support.

    A provider with no shim, or an empty ``intrinsic_tools`` list, supports
    no intrinsic tools — everything intrinsic is dropped.  Same-format
    conversions are left untouched (a provider always understands its own
    native declarations).  ``shim`` accepts a name as well as an object, like
    the sibling gates; a name that does not resolve is treated as no shim.

    ``warnings`` is the conversion's sink, when the caller has one: a dropped
    definition is reported there rather than only counted in the log line.
    """
    if same_format:
        return ir_request

    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    resolved = resolve_shim(shim) if isinstance(shim, str) else shim
    supported = set(
        getattr(getattr(resolved, "tools", None), "intrinsic_tools", ()) or ()
    )

    new_tools: list[Any] = []
    dropped = 0
    promoted = 0
    for t in tools:
        if not isinstance(t, dict):
            new_tools.append(t)
            continue
        ttype = t.get("type")
        if ttype == "intrinsic":
            kind = _intrinsic_kind_of(t)
            if kind in supported:
                new_tools.append(t)
            else:
                # ``request_id`` defaults to ``-``: only a real one is worth
                # spending the message's prefix on.
                where = f"[{request_id}] " if request_id not in ("", "-") else ""
                report_drop(
                    f"{where}target does not support intrinsic kind "
                    f"{kind!r}; dropping the definition",
                    subject=kind,
                    warnings=warnings,
                )
                dropped += 1
            continue
        if allow_name_promotion and ttype == "function" and t.get("name") in supported:
            new_tools.append(
                _make_intrinsic_tool_definition(
                    t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters", {}),
                )
            )
            promoted += 1
            continue
        new_tools.append(t)

    if not dropped and not promoted:
        return ir_request

    ir_request = dict(ir_request)
    ir_request["tools"] = new_tools
    logger.info(
        "[%s] intrinsic tool definitions: promoted %d, dropped %d (target supports %s)",
        request_id,
        promoted,
        dropped,
        sorted(supported) or "none",
    )
    return ir_request


def drop_nameless_tools(
    ir_request: dict[str, Any],
    *,
    request_id: str = "-",
    warnings: Any = None,
) -> dict[str, Any]:
    """Drop tool definitions that carry no name.

    Every dialect identifies a tool by its name, so a nameless definition can
    only be sent as an empty string — and both measured upstreams reject that:
    Chat Completions answers ``400 Invalid 'tools[0].function.name': empty
    string`` and Messages ``400 tools.0.custom.name: String should have at
    least 1 character`` (measured 2026-10-23).  A nameless definition is what
    an IR request carries when a client body holds a tool *type* the source
    reader does not know and no name to fall back on; dropping it here and
    saying so is a better answer than an upstream 400 that names a position in
    a body the client did not write.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        request_id: Request identifier for the report's prefix.
        warnings: The conversion's sink, when the caller has one.

    Returns:
        The IR request without its nameless tool definitions, or the original
        request when every tool has a name.
    """
    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    kept: list[Any] = []
    dropped: list[str] = []
    for index, tool in enumerate(tools):
        if isinstance(tool, dict):
            name = tool.get("name")
            if not (isinstance(name, str) and name.strip()):
                # The definition has no name to identify it by, so the report
                # names its position and the type it declared.
                dropped.append(f"tools[{index}] (type {tool.get('type')!r})")
                continue
        kept.append(tool)

    if not dropped:
        return ir_request

    where = f"[{request_id}] " if request_id not in ("", "-") else ""
    trimmed: dict[str, Any] = {**ir_request, "tools": kept}

    # A private choice that forces the tool by name has nothing left to name
    # once the only tool it could mean was the nameless one.  Only the empty
    # name is handled here: a choice naming a tool that is absent for another
    # reason (an ambiguous namespace, say) is deliberately left as sent and
    # reported elsewhere, so stripping it here would contradict that.
    choice = trimmed.get("tool_choice")
    orphaned_choice = False
    if isinstance(choice, dict) and choice.get("mode") == "tool":
        forced = choice.get("tool_name")
        if not (isinstance(forced, str) and forced.strip()):
            trimmed.pop("tool_choice", None)
            orphaned_choice = True

    report_drop(
        f"{where}dropped {len(dropped)} tool definition(s) with no name: "
        f"{', '.join(dropped)}; every wire format identifies a tool by its "
        "name, and an empty one is rejected"
        + (
            ", so the 'tool_choice' forcing it was removed as well"
            if orphaned_choice
            else ""
        ),
        subject="tools",
        warnings=warnings,
    )
    return trimmed


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
    wrapping the input as ``{"input": string}``.  The downgrade is recorded
    as ``metadata["_downgraded_from"] = "custom"`` so the response path can
    restore the original type.

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


class CustomToolRestorer:
    """Put back the custom type on calls the request leg downgraded.

    :func:`enforce_custom_tools` gives a custom tool a function shape — a
    synthetic ``{"input": string}`` schema — so the target answers with an
    ordinary function call carrying JSON-wrapped text.  The response leg has
    to re-tag that call and recover the raw text.

    The two legs are shaped differently: a body carries each call whole, a
    stream carries it as JSON fragments that must not reach the client
    half-parsed.  One object owns both so the per-call mark is written once;
    the stream leg is why it holds state rather than being a function.
    """

    def __init__(
        self,
        custom_tool_names: frozenset[str],
        *,
        context: StreamContext | None = None,
    ) -> None:
        """Build a restorer for the names the request leg downgraded.

        Args:
            custom_tool_names: The names :func:`enforce_custom_tools` turned
                into functions.  Empty means nothing was downgraded.
            context: The stream context whose tool-call registration follows
                the type; ``None`` on the body leg, which has no context.
        """
        self._names = custom_tool_names
        self._context = context
        self._buffers: dict[str, str] = {}
        #: The choice each buffered call opened on.  A finish belongs to one
        #: choice, and flushing another choice's call there would hand the
        #: client a half-written wrapper.
        self._choices: dict[str, Any] = {}

    def _marks(self, name: Any) -> bool:
        """Whether *name* is one the request leg downgraded."""
        return isinstance(name, str) and name in self._names

    def restore_calls(self, ir_response: dict[str, Any]) -> None:
        """Re-tag every downgraded call in a response body.

        Mutates *ir_response* in place.  Extracting the raw text from the
        ``{"input": ...}`` wrapper is the source converter's job when it
        serialises a custom call, so the mark is the whole of the body leg.

        Args:
            ir_response: An IR response body.
        """
        if not self._names:
            return
        for part in iter_tool_call_parts(ir_response):
            if self._marks(part.get("tool_name")):
                part["tool_type"] = "custom"

    def restore_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-tag downgraded calls in one chunk of stream events.

        A start event is re-tagged and registered with the context, so the
        source converter reads one type from both.  A delta of a marked call
        is buffered instead of emitted — it is a fragment of the JSON wrapper
        — and the finish flushes the buffers it owns as a single unwrapped
        delta before the finish event itself.

        "The buffers it owns" matters on a multi-choice stream: a finish
        carries the choice it terminates, so only that choice's calls are
        flushed.  A call or a finish with no attribution is ambiguous, and
        then everything is flushed, which is what a single-choice stream
        wants anyway.

        A choice whose finish never arrives — a truncated upstream — keeps
        its buffers to the end of the stream and they are never written.
        Dropping wrapper fragments is the intended outcome, not an oversight:
        nothing here knows that call finished.  There is deliberately no
        end-of-stream flush; one would belong where the stream knows it ended.

        Args:
            events: The IR events one upstream chunk produced.

        Returns:
            The events to emit, in their original order, with each buffered
            argument delta replaced by its unwrapped form at the finish.
        """
        if not self._names:
            return events

        staged: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                staged.append(event)
                continue
            etype = event.get("type")

            if etype == "tool_call_start" and self._marks(event.get("tool_name")):
                event["tool_type"] = "custom"
                call_id = event.get("tool_call_id", "")
                if call_id:
                    self._buffers[call_id] = ""
                    self._choices[call_id] = event.get("choice_index")
                    if self._context is not None:
                        self._context.set_tool_call_type(call_id, "custom")
                staged.append(event)
                continue

            if (
                etype == "tool_call_delta"
                and event.get("tool_call_id") in self._buffers
            ):
                self._buffers[event["tool_call_id"]] += (
                    event.get("arguments_delta") or ""
                )
                continue

            if etype == "finish" and self._buffers:
                choice = event.get("choice_index")
                for call_id, raw in list(self._buffers.items()):
                    if choice is not None and self._choices.get(call_id) not in (
                        None,
                        choice,
                    ):
                        continue
                    staged.append(
                        {
                            "type": "tool_call_delta",
                            "tool_call_id": call_id,
                            "arguments_delta": unwrap_custom_tool_input(raw),
                        }
                    )
                    del self._buffers[call_id]
                    self._choices.pop(call_id, None)

            staged.append(event)
        return staged


def restore_custom_tool_calls(
    ir_response: dict[str, Any],
    *,
    custom_tool_names: frozenset[str],
) -> None:
    """Re-tag downgraded tool calls as custom in a response body.

    The one-shot body spelling of :class:`CustomToolRestorer`, for callers
    that never see a stream.  Mutates *ir_response* in place.
    """
    CustomToolRestorer(custom_tool_names).restore_calls(ir_response)


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

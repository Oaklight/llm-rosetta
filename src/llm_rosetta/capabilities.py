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
- :func:`dedupe_tool_definitions` — collapse duplicate tool definitions (post-IR)

Called by :class:`~llm_rosetta.pipeline.ConversionPipeline` at the
appropriate pipeline stages.
"""

from __future__ import annotations

import copy

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta.converters.base.tools.intrinsic import (
    get_definition_kind as _get_definition_kind,
    is_intrinsic_part as _is_intrinsic_part,
    make_intrinsic_tool_definition as _make_intrinsic_tool_definition,
)

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.helpers.tool_group import (
    GENERATED_GROUP_MARKER,
)
from llm_rosetta.converters.base.helpers.truncate import truncate_with_digest
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


def _has_intrinsic_parts(messages: list[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content") or []:
            if _is_intrinsic_part(part):
                return True
    return False


def _strip_intrinsic_from_messages(messages: list[Any]) -> tuple[list[Any], int]:
    new_messages: list[Any] = []
    stripped = 0
    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        filtered = [p for p in content if not _is_intrinsic_part(p)]
        stripped += len(content) - len(filtered)
        if filtered:
            new_messages.append({**msg, "content": filtered})
        elif msg.get("role") != "tool":
            new_messages.append({**msg, "content": filtered})
    return new_messages, stripped


def strip_intrinsic_tools(
    ir_request: dict[str, Any],
    *,
    same_format: bool = False,
    request_id: str = "-",
) -> dict[str, Any]:
    """Strip intrinsic (provider-hosted) tools from the IR request for cross-format conversion.

    Intrinsic tools (web search, code execution, etc.) are provider-specific
    server-side capabilities. They pass through in same-format conversions but
    must be stripped in cross-format conversions where the target provider
    cannot interpret them.

    Removes intrinsic tool definitions, tool_call parts, and tool_result parts.
    Tool-role messages that become empty after stripping are removed entirely.

    No-op when ``same_format`` is True.

    .. deprecated:: 0.15.0
       No longer used by the pipeline, which keeps intrinsic history for the
       target converter to map (:func:`resolve_intrinsic_tools` gates
       definitions only).  Kept for backward compatibility (public since before this
       change); slated for removal in a future release.  Do not add new
       callers — the two paths would drift.
    """
    if same_format:
        return ir_request

    tools = ir_request.get("tools")
    has_intrinsic_defs = tools and any(
        isinstance(t, dict) and t.get("type") == "intrinsic" for t in tools
    )

    messages = ir_request.get("messages")
    has_parts = messages and _has_intrinsic_parts(messages)

    if not has_intrinsic_defs and not has_parts:
        return ir_request

    ir_request = dict(ir_request)
    stripped_count = 0

    if has_intrinsic_defs:
        ir_request["tools"] = [
            t
            for t in tools
            if not (isinstance(t, dict) and t.get("type") == "intrinsic")
        ]
        stripped_count += len(tools) - len(ir_request["tools"])

    if has_parts:
        ir_request["messages"], parts_stripped = _strip_intrinsic_from_messages(
            messages
        )
        stripped_count += parts_stripped

    if stripped_count:
        logger.info(
            "[%s] stripped %d intrinsic tool element(s) for cross-format conversion",
            request_id,
            stripped_count,
        )

    return ir_request


def resolve_intrinsic_tools(
    ir_request: dict[str, Any],
    *,
    shim: Any = None,
    same_format: bool = False,
    allow_name_promotion: bool = False,
    request_id: str = "-",
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
    native declarations).
    """
    if same_format:
        return ir_request

    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    supported = set(getattr(getattr(shim, "tools", None), "intrinsic_tools", ()) or ())

    new_tools: list[Any] = []
    dropped = 0
    promoted = 0
    for t in tools:
        if not isinstance(t, dict):
            new_tools.append(t)
            continue
        ttype = t.get("type")
        if ttype == "intrinsic":
            kind = _get_definition_kind(t)
            if kind in supported:
                new_tools.append(t)
            else:
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


@dataclass(frozen=True)
class ToolNameMap:
    """Bidirectional map between client tool identities and upstream names.

    To the client a tool is identified by ``(name, namespace)``; upstream it
    is a single flat name, because no target format carries a namespace.
    Anything that renames a tool on the request leg registers both spellings
    here; the request leg translates history calls with :meth:`to_upstream`
    and the response leg translates them back with :meth:`to_client`.

    Both directions fall back to the input name, so call sites can translate
    unconditionally. A name the map cannot attribute to exactly one tool
    falls back too — guessing would route the call to the wrong handler.
    """

    _upstream: dict[tuple[str, str | None], str] = field(default_factory=dict)
    _client: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    # Bare client name -> its single upstream spelling, or None when it
    # resolves to several. Names needing no translation are left out.
    _bare: dict[str, str | None] = field(default_factory=dict)
    _contested: frozenset[str] = frozenset()
    # Names declared outside any namespace. Recorded even for tools the rest
    # of the map leaves out: a top-level tool is never renamed, so it earns
    # no entry elsewhere, yet it is exactly what gets asked about.
    _top_level: frozenset[str] = frozenset()
    # Client names where every tool answering to one sits in a *generated*
    # group — one we named because the container had none. A client has no
    # namespace to echo back for those, so a warning that advises exactly that
    # would be describing an action it cannot take.
    _no_client_group: frozenset[str] = frozenset()

    def has_top_level(self, name: str) -> bool:
        """Whether some tool declares this name outside any namespace.

        Not "is this a name we send upstream": a tool from a nameless
        container also keeps its bare spelling, but a bare call still cannot
        be meant for it — the client has no name to point at it with.
        """
        return name in self._top_level

    def is_contested(self, upstream_name: str) -> bool:
        """Whether this upstream name was declared by more than one tool.

        :meth:`to_client` cannot attribute such a name, so it falls back and
        drops the namespace — indistinguishable from a tool that was never
        renamed, which is why these are recorded rather than inferred.

        Only a nameless container reaches this: a named one always gets its
        tools a spelling nobody else has.
        """
        return upstream_name in self._contested

    def has_client_group(self, name: str) -> bool:
        """Whether some tool answering to this client name sits in a group the
        client named — and so carries a namespace a call could echo back.

        False when the only groups involved are ones we generated, which is
        what makes the ambiguity advice actionable or not.
        """
        return name not in self._no_client_group

    def is_ambiguous(self, name: str) -> bool:
        """Whether this bare client name leads to several upstream spellings.

        Not the same question as :meth:`is_contested`, and neither implies
        the other. Tools that collide but were qualified reach the wire under
        different names, so the bare name is ambiguous; tools that could not
        be qualified share one upstream name, so the bare name resolves and
        it is the way *back* that is lost.
        """
        return name in self._bare and self._bare[name] is None

    def to_upstream(self, name: str, namespace: str | None = None) -> str:
        """Map a client ``(name, namespace)`` to the name the provider knows.

        Without a namespace — ``tool_choice`` has nowhere to put one — the
        name resolves only while a single tool answers to it.
        """
        direct = self._upstream.get((name, namespace))
        if direct is not None:
            return direct
        if namespace is None:
            # Missing and ambiguous both fall back: nothing to translate to,
            # or too many.
            return self._bare.get(name) or name
        return name

    def to_client(self, upstream_name: str) -> tuple[str, str | None]:
        """Map a provider name back to the client's ``(name, namespace)``."""
        return self._client.get(upstream_name, (upstream_name, None))

    def __bool__(self) -> bool:
        return bool(self._client or self._upstream)


#: Longest tool name the strictest targets accept.  OpenAI and Anthropic both
#: publish ``^[a-zA-Z0-9_-]+$`` with a cap of this length; every other target is
#: more permissive, so one budget covers them all.
MAX_TOOL_NAME_LEN = 64


def _is_generated_group(tool: Mapping[str, Any]) -> bool:
    """Whether the tool's group name was chosen by us rather than by the client."""
    metadata = tool.get("metadata")
    return isinstance(metadata, dict) and bool(metadata.get(GENERATED_GROUP_MARKER))


def _qualify_tool_name(
    tool_group: str, name: str, used_names: set[str], warnings: list[str]
) -> str:
    """Build a name for a grouped tool that no other tool upstream has.

    Always succeeds.  The upstream spelling is ours to choose — the client
    never sees it, addressing its tools by ``(name, tool_group)`` and having
    them restored through :class:`ToolNameMap` — so rather than give up and let
    two tools share one name, which silently misroutes every call naming it, we
    keep going until the name is unique.

    Preferences, best first:

    1. ``{group}_{name}`` — readable, and what nearly every caller gets.
    2. The group truncated to fit.  Still readable.
    3. A digest suffix, which is not.  The model reads a tool's name as a hint
       about what it does, so this one costs prompting quality and earns a
       warning, but a degraded hint beats a misrouted call.

    Steps 1 and 2 can also be taken by an unrelated tool, so each candidate is
    checked against *used_names* before it is accepted.
    """
    qualified = f"{tool_group}_{name}"
    if len(qualified) > MAX_TOOL_NAME_LEN:
        max_group = MAX_TOOL_NAME_LEN - len(name) - 1
        if max_group > 0:
            qualified = f"{tool_group[:max_group]}_{name}"
    if len(qualified) <= MAX_TOOL_NAME_LEN and qualified not in used_names:
        return qualified

    # Seed from the pair that must stay distinct, never from list position:
    # identical requests have to produce identical names or every one is a
    # prompt-cache miss. The counter only widens the search when a digest is
    # itself taken, and terminates because the tool list is finite.
    max_attempts = 1000
    attempt = 0
    while attempt < max_attempts:
        seed = (
            f"{tool_group}\0{name}\0{attempt}" if attempt else f"{tool_group}\0{name}"
        )
        generated = truncate_with_digest(
            qualified, MAX_TOOL_NAME_LEN, seed=seed, force=True
        )
        if generated not in used_names:
            break
        attempt += 1
    else:
        raise RuntimeError(
            f"Tool name qualification exhausted {max_attempts} attempts "
            f"for {name!r} in namespace {tool_group!r}"
        )

    warnings.append(
        f"Tool {name!r} in namespace {tool_group!r} cannot be qualified within "
        f"{MAX_TOOL_NAME_LEN} characters without colliding with another tool, "
        f"so it goes upstream under the generated name {generated!r}; calls "
        "still route correctly, but the model sees a less meaningful name"
    )
    return generated


def _fold_generated_duplicates(
    tools: list[Any], dropped: list[int], dropped_names: Counter[str]
) -> None:
    """Record, in *dropped*, the byte-identical children of one generated group.

    A generated group name is ours — it exists only because that container had
    no name for the client to address its tools by — so nothing client-side
    keeps an identical pair apart, and a strict upstream answers 400 to the
    second of them (#913).  Run before any renaming, which is what would
    otherwise make the copies differ.
    """
    identical: dict[str, list[Any]] = defaultdict(list)
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not _is_generated_group(tool):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        if tool in identical[name]:
            dropped.append(index)
            dropped_names[name] += 1
            continue
        identical[name].append(tool)


def _qualify_collisions(
    tools: list[Any],
    gone: set[int],
    result: list[Any],
    used_names: set[str],
    warnings: list[str],
) -> bool:
    """Qualify every grouped tool whose name still collides.

    Top-level tools keep their name — nothing to prefix them with — and the
    grouped ones move aside.  A group whose duplicates were all folded away
    has nothing left to tell apart, so its survivor keeps the bare name too,
    which is why the collisions are counted from what survived rather than
    from what came in.

    Returns whether anything was renamed.
    """
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, tool in enumerate(tools):
        if index in gone or not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            by_name[name].append(index)

    renamed_any = False
    for name, indices in by_name.items():
        if len(indices) <= 1:
            continue
        for index in indices:
            tool = tools[index]
            group = tool.get("tool_group")
            if not group:
                continue
            qualified = _qualify_tool_name(group, name, used_names, warnings)
            renamed = dict(tool)
            renamed["name"] = qualified
            # The response leg undoes the qualification with this.
            metadata = renamed.get("metadata")
            renamed["metadata"] = {
                **(metadata if isinstance(metadata, dict) else {}),
                "_original_name": name,
            }
            result[index] = renamed
            used_names.add(qualified)
            renamed_any = True
    return renamed_any


def qualify_tool_names(
    ir_request: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Give every grouped tool an upstream name no other tool has.

    A group exists only inside the IR.  Of the wire formats this library
    speaks, only the Responses family's container carries one; every other
    target has a flat list where a name is the only thing telling two tools
    apart.  So a target that cannot carry a container gets names that carry the
    group instead — ``{group}_{name}`` — and the client's ``(name, group)``
    identity is restored on the way back by :class:`ToolNameMap`.

    This runs against the *target*, which is why it is here and not in the
    source converter that flattened the container: whether the prefix is needed
    at all depends on what the target can express, and the source side is not
    what knows that.  Every target is flat today, so every collision is
    qualified.

    Tools sharing a name are treated by where they came from:

    - **Top-level tools keep the name.**  They came from no group, so there is
      nothing to prefix, and the grouped tool moves aside instead.
    - **Grouped tools are qualified.**  A group name the client chose always
      gets its children a spelling nobody else has, however long that name or
      however crowded the request.
    - **A generated group's byte-identical children collapse instead.**  That
      group is ours — it exists only because the container had no name for the
      client to address it by — so there is no client-side identity to keep
      apart, and the pair is one duplicate that a strict upstream would answer
      400 to, twice over (#913).

    Renamed tools record the pre-rename name in ``metadata._original_name``,
    which is what the response leg undoes the qualification with.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        warnings: Optional list to append qualification reports to.

    Returns:
        The IR request with colliding grouped tools qualified and generated
        duplicates dropped, or the original request unchanged when nothing
        collided.
    """
    tools = ir_request.get("tools")
    if not tools:
        return ir_request

    dropped: list[int] = []
    dropped_names: Counter[str] = Counter()
    _fold_generated_duplicates(tools, dropped, dropped_names)
    gone = set(dropped)

    result: list[Any] = list(tools)
    used_names: set[str] = {
        tool["name"]
        for index, tool in enumerate(tools)
        if index not in gone
        and isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
    }
    renamed_any = _qualify_collisions(
        tools, gone, result, used_names, warnings if warnings is not None else []
    )

    if dropped and warnings is not None:
        for name in sorted(dropped_names):
            warnings.append(
                f"Tool {name!r} was declared {dropped_names[name] + 1} times with "
                "an identical definition inside the same tool group; "
                f"{dropped_names[name]} duplicate(s) were dropped"
            )

    if not renamed_any and not dropped:
        return ir_request
    return {
        **ir_request,
        "tools": [t for i, t in enumerate(result) if i not in gone],
    }


def build_tool_name_map(ir_request: dict[str, Any]) -> ToolNameMap:
    """Derive a :class:`ToolNameMap` from the IR request's tool definitions.

    Tools harvested from a container carry ``tool_group`` (set during
    flattening).  Colliding names are additionally rewritten to
    ``{group}_{name}`` by ``qualify_tool_names``, which records the
    pre-rename name in ``metadata._original_name``.

    Tools whose name survived unchanged and that carry no group are
    skipped — there is nothing to translate, and leaving them out keeps the
    map falsy for the common case.
    """
    tools = [
        t
        for t in (ir_request.get("tools") or [])
        if isinstance(t, dict) and t.get("name")
    ]
    # Two tools still share one upstream name when a nameless container left
    # nothing to qualify them with; a call naming it belongs to neither.
    claimants = Counter(t["name"] for t in tools)

    upstream: dict[tuple[str, str | None], str] = {}
    client: dict[str, tuple[str, str | None]] = {}
    sole: dict[str, str | None] = {}
    # Only names a group was meant to distinguish: two top-level tools
    # sharing a name lose nothing on the way back, having none to drop.
    #
    # `is None`, not truthiness: a nameless container gives "", which is the
    # worst case rather than an absent one. Only the harvest writes
    # ``tool_group``, so its presence marks a tool as having come from a
    # container.
    contested = {
        t["name"]
        for t in tools
        if claimants[t["name"]] > 1 and t.get("tool_group") is not None
    }
    # `is None` again, and with more at stake: reading "" as top-level would
    # claim a bare call meant that tool, when the client has no name to point
    # at it with and the call is in fact ambiguous.
    top_level = {t["name"] for t in tools if t.get("tool_group") is None}

    named_claim: set[str] = set()
    generated_claim: set[str] = set()

    for tool in tools:
        upstream_name = tool["name"]
        meta = tool.get("metadata") or {}
        tool_group = tool.get("tool_group")
        client_name = meta.get("_original_name") or upstream_name

        # Tracked for every tool, including untranslated ones: an untouched
        # top-level `exec` is what makes a namespaced `exec` unresolvable.
        if sole.setdefault(client_name, upstream_name) != upstream_name:
            sole[client_name] = None

        if tool_group is not None and not _is_generated_group(tool):
            named_claim.add(client_name)
        else:
            generated_claim.add(client_name)

        if tool_group is None and client_name == upstream_name:
            continue
        # Forward, the key keeps the real group — including a generated one.
        # A client call cannot name a generated group, so it never matches:
        # the lookup falls through to the bare name, which is ambiguous and
        # reported, rather than silently resolving to a tool the client had
        # no way to ask for.
        upstream[(client_name, tool_group)] = upstream_name
        if claimants[upstream_name] == 1:
            # Backward, a generated group is dropped.  It is ours, not the
            # client's — it exists only because that container had no name —
            # so handing it back would return a name the client never wrote.
            client[upstream_name] = (
                client_name,
                None if _is_generated_group(tool) else tool_group,
            )

    return ToolNameMap(
        upstream,
        client,
        # Ambiguous entries are kept; resolvable ones only when they say
        # something ``to_upstream``'s own fallback would not.
        {k: v for k, v in sole.items() if v is None or v != k},
        frozenset(contested),
        frozenset(top_level),
        frozenset(generated_claim - named_claim),
    )


def _to_client_identity(target: dict[str, Any], name_map: ToolNameMap) -> None:
    """Rewrite one IR tool call from its upstream name to the client's.

    Works on any dict carrying ``tool_name`` — both a ToolCallPart and a
    ``tool_call_start`` stream event qualify.  No-op when the name was never
    rewritten.
    """
    upstream_name = target.get("tool_name", "")
    client_name, namespace = name_map.to_client(upstream_name)
    if client_name == upstream_name and namespace is None:
        return
    target["tool_name"] = client_name
    if namespace is None:
        return
    pm = target.get("provider_metadata")
    if not isinstance(pm, dict):
        pm = {}
        target["provider_metadata"] = pm
    # The call side still carries the group under the wire name, unlike the
    # definition side where it is the first-class ``tool_group`` field.
    # Both say the same thing; unifying them means giving a tool call a
    # ``tool_group`` of its own, which is the next step on #953.
    pm["namespace"] = namespace


def _select_upstream_name(
    name: str,
    *,
    selector: str,
    name_map: ToolNameMap,
    declared: set[Any],
    warnings: list[str],
) -> str:
    """Re-spell one name used by a tool selector, warning if it resolves to nothing.

    Selectors carry no namespace, so the map can only answer while a single
    tool claims the bare name. Otherwise the name goes upstream unchanged and
    matches nothing, silently as far as the provider is concerned — hence
    *selector* in the warning. The two causes are reported apart because a
    shared name is ours to explain and an undeclared one is the client's typo.
    """
    chosen = name_map.to_upstream(name)
    if chosen in declared:
        return chosen
    if name_map.is_ambiguous(name):
        warnings.append(
            f"{selector} names {chosen!r}, which is not among the tools sent "
            "upstream — the name is shared by tools in more than one namespace "
            f"and {selector} has no namespace to disambiguate it"
        )
    else:
        warnings.append(
            f"{selector} names {chosen!r}, which is not among the tools sent "
            "upstream — no tool of that name was declared"
        )
    return chosen


def _apply_upstream_allowed_tools(
    allowed: Any,
    *,
    name_map: ToolNameMap,
    declared: set[Any],
    warnings: list[str],
) -> None:
    """Re-spell the tool names inside an ``allowed_tools`` extension.

    Passed through from the provider request verbatim, so it arrives either
    as a bare list of entries or wrapped in ``{"mode", "tools"}``, and an
    entry is either a name or a ``{"type", "name"}`` dict.  Mutates in place;
    anything of another shape is left alone.
    """
    unwrapped: Any = allowed.get("tools") if isinstance(allowed, dict) else allowed
    if not isinstance(unwrapped, list):
        return
    entries: list[Any] = unwrapped
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            entries[i] = _select_upstream_name(
                entry,
                selector="allowed_tools",
                name_map=name_map,
                declared=declared,
                warnings=warnings,
            )
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            entry["name"] = _select_upstream_name(
                entry["name"],
                selector="allowed_tools",
                name_map=name_map,
                declared=declared,
                warnings=warnings,
            )


def _respell_history_tool_calls(
    ir_request: dict[str, Any],
    *,
    name_map: ToolNameMap,
) -> tuple[set[str], set[str]]:
    """Re-spell history tool calls in place, and sort the broken ones.

    Mutates the ``tool_call`` parts of *ir_request* to their upstream
    spelling, and returns ``(unresolved, unattributable)`` — the two ways a
    namespace-less history call goes wrong, kept apart because the client can
    only act on one. ``unresolved``: the name matches no declared tool, and
    echoing the namespace back would have fixed it. ``unattributable``: it
    matches several, and nothing the client sends can separate them.
    """
    unresolved: set[str] = set()
    unattributable: set[str] = set()
    for msg in ir_request.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "tool_call":
                continue
            pm = part.get("provider_metadata")
            namespace = pm.get("namespace") if isinstance(pm, dict) else None
            client_name = part.get("tool_name", "")
            upstream_name = name_map.to_upstream(client_name, namespace)
            part["tool_name"] = upstream_name
            if namespace is not None:
                continue
            if name_map.is_contested(upstream_name):
                # Tested against the *upstream* name, and first: a third tool
                # sharing the name makes both branches true, and this is the
                # more specific answer. The other would then advise echoing
                # the namespace back, which cannot help when the namespaces
                # are precisely what could not be folded in.
                # Do not reorder these.
                unattributable.add(client_name)
            elif name_map.is_ambiguous(client_name) and not name_map.has_top_level(
                client_name
            ):
                # `is_ambiguous` alone over-reports: when a top-level tool and
                # a namespaced one share a bare name, omitting the namespace
                # is how the client says it meant the top-level one, and the
                # fallback lands on exactly that tool.
                if name_map.has_client_group(client_name):
                    unresolved.add(client_name)
                else:
                    # Every group involved is one we generated, so telling the
                    # client to echo a namespace back describes something it
                    # cannot do — it never wrote one. Nothing it sends can
                    # separate these, which is the other arm's answer.
                    unattributable.add(client_name)
    return unresolved, unattributable


def apply_upstream_tool_names(
    ir_request: dict[str, Any],
    *,
    name_map: ToolNameMap,
    warnings: list[str],
) -> None:
    """Re-spell history tool calls to match the request's tool definitions.

    Mutates *ir_request* in place.  A client that received a namespaced call
    echoes it back as ``(bare name, namespace)``; the tool definitions in the
    same request may have been flattened to a qualified upstream name.  Left
    alone, the assistant message would name a function the request does not
    declare.  The inverse of :func:`restore_client_tool_names`.

    ``tool_choice`` and the ``allowed_tools`` extension are re-spelled here
    too. Neither carries a namespace, so both resolve by name alone or not at
    all, and either one left naming no declared tool is reported in
    *warnings* — even when nothing was renamed, since such a selector is
    broken on its own terms.

    A history call is reported only when it omits a namespace several tools
    need, the one failure this flattening causes. Naming a tool the request
    no longer declares is the client's own history to reconcile.
    """
    declared = {
        t.get("name") for t in ir_request.get("tools") or [] if isinstance(t, dict)
    }

    choice = ir_request.get("tool_choice")
    if isinstance(choice, dict) and choice.get("tool_name"):
        choice["tool_name"] = _select_upstream_name(
            choice["tool_name"],
            selector="tool_choice",
            name_map=name_map,
            declared=declared,
            warnings=warnings,
        )

    extensions = ir_request.get("provider_extensions")
    if isinstance(extensions, dict):
        for key in ("allowed_tools", "_open_responses_allowed_tools"):
            _apply_upstream_allowed_tools(
                extensions.get(key),
                name_map=name_map,
                declared=declared,
                warnings=warnings,
            )

    # Only the history rewrite below depends on a rename having happened:
    # with an empty map every name resolves to itself and nothing is
    # ambiguous, so the loop would write each name back unchanged.
    if not name_map:
        return

    unresolved, unattributable = _respell_history_tool_calls(
        ir_request, name_map=name_map
    )

    for client_name in sorted(unresolved):
        warnings.append(
            f"A history tool call names {client_name!r} with no namespace, but "
            "tools in more than one namespace declare that name, so it goes "
            "upstream matching none of them — echo back the 'namespace' field "
            "returned alongside the call to identify which tool it was"
        )

    for client_name in sorted(unattributable):
        warnings.append(
            f"A history tool call names {client_name!r} with no namespace, and "
            "more than one of the tools declaring that name went upstream "
            "under it, so the provider cannot tell which one the call was for "
            "— see the warnings about that name for why their namespaces could "
            "not be folded in"
        )


def _warn_contested(
    seen: set[str],
    name_map: ToolNameMap,
    warnings: list[str] | None,
    already_warned: set[str] | None = None,
) -> None:
    """Report provider calls the map had to hand back without a namespace.

    This is the leg where the failure becomes visible — the client receives
    a call it cannot route — and the one leg that cannot repair it.

    *already_warned*, when given, spans more than one call and suppresses
    names reported before; *seen* covers only the current chunk.
    """
    if warnings is None:
        return
    for name in sorted(n for n in seen if name_map.is_contested(n)):
        if already_warned is not None:
            if name in already_warned:
                continue
            already_warned.add(name)
        warnings.append(
            f"The provider called {name!r}, but more than one declared tool "
            "went upstream under that name, so the namespace it belongs to "
            "cannot be determined and is omitted from the returned call"
        )


def restore_client_tool_names(
    ir_response: dict[str, Any],
    *,
    name_map: ToolNameMap,
    warnings: list[str] | None = None,
) -> None:
    """Restore client-facing tool names on an IR response.

    Mutates *ir_response* in place.  For each tool call whose ``tool_name``
    was rewritten on the request leg, restores the client's spelling and
    records any namespace under ``provider_metadata.namespace`` so the source
    converter can emit it.

    Chat Completions cannot represent a namespace, so the provider echoes
    back only the flat name and the client could not otherwise route the call.
    Called on the non-streaming path after Target → IR conversion.
    """
    if not name_map:
        return

    seen: set[str] = set()

    def _restore_parts(msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_call":
                seen.add(part.get("tool_name", ""))
                _to_client_identity(part, name_map)

    for choice in ir_response.get("choices") or []:
        if isinstance(choice, dict):
            _restore_parts(choice.get("message"))

    for msg in ir_response.get("messages") or []:
        _restore_parts(msg)

    _warn_contested(seen, name_map, warnings)


def restore_client_tool_name_events(
    ir_events: list[dict[str, Any]],
    *,
    name_map: ToolNameMap,
    warnings: list[str] | None = None,
    already_warned: set[str] | None = None,
) -> None:
    """Restore client-facing tool names on streamed tool calls.

    Mutates the ``tool_call_start`` events in *ir_events* in place.  The
    source converter reads the namespace back off ``provider_metadata`` and
    carries it onto the later done/completed items.  The streaming
    counterpart of :func:`restore_client_tool_names`.

    *already_warned* is the caller's whole-stream set of reported names.
    Pass one: this runs per chunk, and a provider that puts each tool call in
    its own chunk — the anthropic wire format's normal shape — would otherwise
    warn once per chunk where the non-streaming leg warns once in total.
    """
    if not name_map:
        return

    seen: set[str] = set()
    for event in ir_events:
        if isinstance(event, dict) and event.get("type") == "tool_call_start":
            seen.add(event.get("tool_name", ""))
            _to_client_identity(event, name_map)

    _warn_contested(seen, name_map, warnings, already_warned)


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


def dedupe_tool_definitions(
    ir_request: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Collapse duplicate tool definitions and report name collisions.

    A request may declare the same tool name more than once.  Upstreams
    disagree on whether that is legal — Anthropic, DeepSeek, xAI and Google
    all answer 400, while OpenAI and several others accept it — so the same
    body succeeds or fails depending on the target provider.

    Two cases are treated differently:

    - **Exact duplicates** — same name *and* identical definition — collapse
      to the first occurrence.  Lossless, and the request stops being rejected
      by the strict upstreams above.
    - **Conflicting duplicates** — same name, differing definition — are left
      untouched and reported in *warnings*.  The definitions describe
      different tools, so silently keeping one would drop a legitimate tool
      and hide a client-side mistake no converter can repair.

    Tools harvested from a *named* container are exempt: the group is part
    of their identity, so two byte-identical definitions there are still two
    distinct tools, left to ``qualify_tool_names`` / ``ToolNameMap``.  A
    container with no name of its own is not exempt, and its children
    collapse like any other duplicate — see the comment at the exemption
    below.

    Equality is on the whole definition dict, so this is not semantic dedup:
    any incidental per-copy difference (``metadata``, an optional field set on
    one copy only) downgrades a pair from "exact duplicate" to "conflicting".
    Erring that way is deliberate — it only ever keeps a tool, never drops one.

    Must be called after source → IR conversion and after the intrinsic /
    custom tool passes, and **before**
    :func:`relocate_oversized_tool_descriptions`: collapsing first keeps a
    dropped tool's description out of the relocated system message.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        warnings: Optional list to append collision reports to.

    Returns:
        The IR request with exact duplicates collapsed, or the original
        request unchanged when nothing was duplicated.
    """
    tools = ir_request.get("tools")
    if not tools or len(tools) < 2:
        return ir_request

    kept: list[Any] = []
    # name -> distinct definitions seen so far, in first-seen order
    definitions: dict[str, list[Any]] = {}
    dropped: Counter[str] = Counter()

    for tool in tools:
        if not isinstance(tool, dict):
            kept.append(tool)
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            kept.append(tool)
            continue
        # A tool harvested from a container carries its identity in
        # ``tool_group``, so two byte-identical definitions there are still
        # two tools the client addresses separately.  Left to the group
        # machinery (``qualify_tool_names`` / ``ToolNameMap``).
        #
        # Only a *named* container qualifies — the check is truthiness, not
        # ``is not None``.  A container with no name of its own yields "", and
        # "" qualifies nothing: ``qualify_tool_names`` has nothing to build a
        # unique name from, so its children keep one bare upstream name and
        # cannot be told apart, by the upstream or by the client that would
        # have to address them.  No provider implementing ``namespace``
        # accepts an empty name either — OpenAI, DeepSeek and Volcengine all
        # reject the container outright — so there is no identity to preserve
        # here and the byte-identical ones collapse like any other duplicate
        # (#913).  Conflicting definitions still survive, as everywhere else.
        #
        # Deliberately the opposite reading to `build_tool_name_map` above,
        # which tests the same ``""`` with ``is None`` and calls it *not*
        # top-level.  The questions differ — could a bare call be ambiguous /
        # is there an identity here worth preserving — so both answers are
        # correct and neither is a stale copy of the other.
        if tool.get("tool_group"):
            kept.append(tool)
            continue
        seen = definitions.setdefault(name, [])
        if any(tool == existing for existing in seen):
            dropped[name] += 1
            continue
        seen.append(tool)
        kept.append(tool)

    conflicted = {
        name: len(defs) for name, defs in definitions.items() if len(defs) > 1
    }
    if not dropped and not conflicted:
        return ir_request

    if warnings is not None:
        for name in sorted(dropped):
            warnings.append(
                f"Tool {name!r} was declared {dropped[name] + 1} times with an "
                f"identical definition; {dropped[name]} duplicate(s) were dropped"
            )
        for name in sorted(conflicted):
            warnings.append(
                f"Tool name {name!r} is declared with {conflicted[name]} "
                "distinct definitions and is forwarded as-is; upstreams that "
                "require unique tool names will reject it"
            )

    if not dropped:
        return ir_request

    return {**ir_request, "tools": kept}

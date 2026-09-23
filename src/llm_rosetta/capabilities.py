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
from collections import Counter
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ToolNameMap:
    """Bidirectional map between client tool identities and upstream names.

    To the client a tool is identified by ``(name, namespace)``; upstream it
    is a single flat name, because no target format carries a namespace.
    Anything that rewrites a tool name on the request leg registers both
    spellings here, and every leg that names a tool translates through it:

    - request: history tool calls sent by the client, via :meth:`to_upstream`
    - response: tool calls echoed by the provider, via :meth:`to_client`

    Both directions fall back to the input name, so call sites can translate
    unconditionally without knowing whether a rewrite happened.  A name the
    map cannot attribute to exactly one tool falls back too: guessing would
    route the call to the wrong handler, which is worse than not translating.
    """

    _upstream: dict[tuple[str, str | None], str] = field(default_factory=dict)
    _client: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    # Bare client name -> the single upstream spelling it resolves to, or
    # None when it resolves to several.  One field rather than a mapping plus
    # a set of its own None-valued keys: they could only ever disagree.
    # Names that need no translation are left out, as in ``_upstream``.
    _bare: dict[str, str | None] = field(default_factory=dict)
    _contested: frozenset[str] = frozenset()

    def is_contested(self, upstream_name: str) -> bool:
        """Whether this upstream name was declared by more than one tool.

        :meth:`to_client` cannot attribute such a name, so it falls back and
        the namespace is dropped.  That is indistinguishable from the far
        more common case of a tool that was never renamed, which is why the
        contested names are recorded rather than inferred from the fallback.
        """
        return upstream_name in self._contested

    def is_ambiguous(self, name: str) -> bool:
        """Whether this bare client name leads to several upstream spellings.

        Distinguishes the two ways :meth:`to_upstream` can decline to
        translate a name given without a namespace: several tools claim it
        under different upstream names, or no tool does.  Only a caller
        reporting the failure needs to care.

        Not the same question as :meth:`is_contested`, and neither implies
        the other.  Tools that collide *and* were successfully qualified
        reach the wire under different names, so the bare name is ambiguous;
        tools whose qualification was refused share one upstream name, so the
        bare name resolves cleanly and it is the way *back* that is lost.
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


def build_tool_name_map(ir_request: dict[str, Any]) -> ToolNameMap:
    """Derive a :class:`ToolNameMap` from the IR request's tool definitions.

    Tools harvested from a ``namespace`` container carry
    ``metadata.namespace`` (set during flattening).  Colliding names are
    additionally rewritten to ``{namespace}_{name}`` by
    ``_dedup_ir_tool_names``, which records the pre-rename name in
    ``metadata._original_name``.

    Tools whose name survived unchanged and that carry no namespace are
    skipped — there is nothing to translate, and leaving them out keeps the
    map falsy for the common case.
    """
    tools = [
        t
        for t in (ir_request.get("tools") or [])
        if isinstance(t, dict) and t.get("name")
    ]
    # Qualification can fail — the bare name already fills the 64-char budget,
    # or the qualified spelling is taken — and then two tools go upstream
    # under one name.  A call naming it belongs to neither in particular.
    claimants = Counter(t["name"] for t in tools)

    upstream: dict[tuple[str, str | None], str] = {}
    client: dict[str, tuple[str, str | None]] = {}
    sole: dict[str, str | None] = {}
    # Only names a namespace was meant to distinguish: two plain top-level
    # tools sharing a name lose nothing on the way back, since neither had a
    # namespace to drop.
    contested = {
        t["name"]
        for t in tools
        if claimants[t["name"]] > 1 and (t.get("metadata") or {}).get("namespace")
    }

    for tool in tools:
        upstream_name = tool["name"]
        meta = tool.get("metadata") or {}
        namespace = meta.get("namespace")
        client_name = meta.get("_original_name") or upstream_name

        # Tracked for every tool, including untranslated ones: an untouched
        # top-level `exec` is what makes a namespaced `exec` unresolvable.
        if sole.setdefault(client_name, upstream_name) != upstream_name:
            sole[client_name] = None

        if namespace is None and client_name == upstream_name:
            continue
        upstream[(client_name, namespace)] = upstream_name
        if claimants[upstream_name] == 1:
            client[upstream_name] = (client_name, namespace)

    return ToolNameMap(
        upstream,
        client,
        # Ambiguous entries are kept; resolvable ones only when they say
        # something ``to_upstream``'s own fallback would not.
        {k: v for k, v in sole.items() if v is None or v != k},
        frozenset(contested),
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
        _apply_upstream_allowed_tools(
            extensions.get("allowed_tools"),
            name_map=name_map,
            declared=declared,
            warnings=warnings,
        )

    # Only the history rewrite below depends on a rename having happened:
    # with an empty map every name resolves to itself and nothing is
    # ambiguous, so the loop would write each name back unchanged.
    if not name_map:
        return

    # Two different ways a namespace-less history call goes wrong, told apart
    # because the client can only act on one of them.  ``unresolved``: the
    # name reaches the provider matching no declared tool, and echoing the
    # namespace back would have fixed it.  ``unattributable``: it matches
    # several, and nothing the client sends can separate them — the tools
    # needed different upstream names and could not be given any.
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
                # Checked first, and against the *upstream* name: these tools
                # kept their bare spelling, so the call resolves to a name we
                # really are sending and the ``declared`` test below would pass
                # it.  Being sent is not the same as being attributable.
                unattributable.add(client_name)
            elif name_map.is_ambiguous(client_name) and upstream_name not in declared:
                # ``is_ambiguous`` alone over-reports.  A top-level tool and a
                # namespaced one can share a bare name, and then the name is
                # ambiguous in the map yet the call is not: omitting the
                # namespace is how a client says it meant the top-level tool,
                # and the fallback lands on exactly that tool's upstream name.
                # Asking whether the result is a name we are actually sending
                # tells the two apart, and the tool list can answer it where
                # the map cannot — an unrenamed top-level tool is left out of
                # the map entirely.
                unresolved.add(client_name)

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

"""Tool names and tool groups — the identity a client gave a tool.

A client may address a tool by ``(name, group)``.  Not every target can carry that
structure on the wire, so this module holds the format-agnostic half: the
bidirectional :class:`ToolNameMap`, the qualification that makes colliding names
unique for a flat target, and the duplicate-definition pass beside it.
"""

from __future__ import annotations


import logging
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


from llm_rosetta.converters.base.helpers.tool_group import (
    GENERATED_GROUP_MARKER,
    tool_call_group,
)
from llm_rosetta.converters.base.tools.intrinsic import is_intrinsic_part
from llm_rosetta.converters.base.helpers.truncate import truncate_with_digest

logger = logging.getLogger(__name__)


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
    # Names declared outside any namespace. Recorded even for tools the rest
    # of the map leaves out: a top-level tool is never renamed, so it earns
    # no entry elsewhere, yet it is exactly what gets asked about.
    _top_level: frozenset[str] = frozenset()
    # Client names where every tool answering to one sits in a *generated*
    # group — one we named because the container had none. A client has no
    # namespace to echo back for those, so a warning that advises exactly that
    # would be describing an action it cannot take.
    _no_client_group: frozenset[str] = frozenset()
    # Reverse index keyed by ``(upstream name, group)``, filled for the targets
    # that carry a container.  There the upstream name is the client's own
    # child name, so a name alone stops identifying a tool — a top-level
    # ``wait`` and one inside ``tasks`` share it — and only the group the wire
    # echoes beside the name tells them apart.
    _client_grouped: dict[tuple[str, str | None], tuple[str, str | None]] = field(
        default_factory=dict
    )

    def has_top_level(self, name: str) -> bool:
        """Whether some tool declares this name outside any namespace.

        Not "is this a name we send upstream": a tool from a nameless
        container also keeps its bare spelling, but a bare call still cannot
        be meant for it — the client has no name to point at it with.
        """
        return name in self._top_level

    def has_client_group(self, name: str) -> bool:
        """Whether some tool answering to this client name sits in a group the
        client named — and so carries a namespace a call could echo back.

        False when the only groups involved are ones we generated, which is
        what makes the ambiguity advice actionable or not.
        """
        return name not in self._no_client_group

    def is_ambiguous(self, name: str) -> bool:
        """Whether this bare client name leads to several upstream spellings.

        Tools that collide are qualified apart, so they reach the wire under
        different names and the bare name stops resolving to any one of them.
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

    def to_client(
        self, upstream_name: str, group: str | None = None
    ) -> tuple[str, str | None]:
        """Map a provider name back to the client's ``(name, namespace)``.

        *group* is the namespace the upstream echoed beside the name.  A
        target that carries a container produces one, and there the upstream
        name is the client's own child name — so a top-level ``wait`` and one
        inside ``tasks`` reach the wire under the same name, and this is what
        separates them.  A flat target echoes none, and the name is the whole
        identity.
        """
        if group is not None:
            grouped = self._client_grouped.get((upstream_name, group))
            if grouped is not None:
                return grouped
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
    *,
    same_group_only: bool = False,
) -> bool:
    """Qualify every grouped tool whose name still collides.

    Top-level tools keep their name — nothing to prefix them with — and the
    grouped ones move aside.  A group whose duplicates were all folded away
    has nothing left to tell apart, so its survivor keeps the bare name too,
    which is why the collisions are counted from what survived rather than
    from what came in.

    *same_group_only* is for a target that carries a container.  There the
    group travels on the wire, so only the members of one container have to be
    told apart — measured, a container child may share its name with a
    top-level tool and with a child of another container.  Collisions are then
    counted per ``(name, group)``, and only a group's own members move aside.

    Returns whether anything was renamed.
    """
    by_key: dict[Any, list[int]] = defaultdict(list)
    for index, tool in enumerate(tools):
        if index in gone or not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            key = (name, tool.get("tool_group")) if same_group_only else name
            by_key[key].append(index)

    renamed_any = False
    for indices in by_key.values():
        if len(indices) <= 1:
            continue
        for index in indices:
            tool = tools[index]
            group = tool.get("tool_group")
            if not group:
                continue
            name = tool["name"]
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
    carries_groups: bool = False,
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
    what knows that.

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

    *carries_groups* narrows the second rule to what the target needs.  A
    target that puts the group back on the wire carries it there, so a child
    only has to be unique among its own container's members: measured, every
    container-capable upstream accepts a child named like a top-level tool and
    like a child of another container.  Qualifying against the whole list would
    rename for a collision that cannot happen and hand the model a longer name
    than the client wrote.  The third rule is unaffected — a strict upstream
    rejects a repeated name *inside* one container either way.

    Renamed tools record the pre-rename name in ``metadata._original_name``,
    which is what the response leg undoes the qualification with.

    Args:
        ir_request: The IR request dict — **always use the return value**.
        warnings: Optional list to append qualification reports to.
        carries_groups: Whether the target puts each group back on the wire as
            a container, so a child's name only has to be unique within its
            own group.

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
        tools,
        gone,
        result,
        used_names,
        warnings if warnings is not None else [],
        same_group_only=carries_groups,
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

    Every grouped tool is also entered under ``(upstream name, group)``.
    Against a target that carries a container a child goes out under the
    client's own name, so the name alone stops identifying it: a top-level
    ``wait`` and one inside ``tasks`` are both ``wait`` on that wire, and the
    group the upstream echoes beside the name is what separates them.
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
    client_grouped: dict[tuple[str, str | None], tuple[str, str | None]] = {}
    sole: dict[str, str | None] = {}
    # `is None`, not truthiness: only the harvest writes ``tool_group``, so its
    # presence is what marks a tool as having come from a container — and a
    # container the client left nameless gives "", which is the worst case
    # rather than an absent one.
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
        # Keyed on both halves of what a container-carrying target hands back.
        # There the upstream name is the client's own child name, so this is
        # the only entry that tells two tools sharing that name apart.  A
        # generated group is dropped, as in the name-keyed map beside it: it is
        # ours, not the client's.
        client_grouped[(upstream_name, tool_group)] = (
            client_name,
            None if _is_generated_group(tool) else tool_group,
        )
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
        frozenset(top_level),
        frozenset(generated_claim - named_claim),
        client_grouped,
    )


def _drop_legacy_namespace(target: dict[str, Any]) -> None:
    """Remove the pre-``tool_group`` metadata spelling, if present.

    The call side writes and clears its group through the first-class
    ``tool_group`` field; a legacy ``provider_metadata["namespace"]`` left
    beside it would be a second copy able to disagree.
    """
    pm = target.get("provider_metadata")
    if isinstance(pm, dict):
        pm.pop("namespace", None)


def _to_client_identity(target: dict[str, Any], name_map: ToolNameMap) -> None:
    """Rewrite one IR tool call from its upstream name to the client's.

    Works on any dict carrying ``tool_name`` — both a ToolCallPart and a
    ``tool_call_start`` stream event qualify.  Idempotent: a name that was
    never rewritten is written back unchanged, so the effect is nothing.

    The group travels in the first-class ``tool_group`` field, matching the
    definition side.  The legacy ``provider_metadata["namespace"]`` spelling
    is read (through :func:`tool_call_group`) and cleared, so the two can never
    disagree.
    """
    upstream_name = target.get("tool_name", "")
    # The group the upstream echoed beside the name.  Only a target that
    # carries a container produces one; on a flat target the name is the whole
    # identity, and this is absent.
    wire_group = tool_call_group(target)
    client_name, namespace = name_map.to_client(upstream_name, group=wire_group)
    target["tool_name"] = client_name
    if namespace is not None:
        target["tool_group"] = namespace
    elif wire_group is not None:
        # A group the client never wrote — one we generated for a nameless
        # container — is not the client's to see, so the field goes rather
        # than being handed back.
        target.pop("tool_group", None)
    _drop_legacy_namespace(target)


def _select_upstream_name(
    name: str,
    *,
    selector: str,
    name_map: ToolNameMap,
    declared: set[Any],
    warnings: list[str],
    group: str | None = None,
) -> str:
    """Re-spell one name used by a tool selector, warning if it resolves to nothing.

    *group* is the namespace the selector carried, when its shape has room for
    one — an IR ``allowed_tools`` identity does.  Without it the map can only
    answer while a single tool claims the bare name; with it the exact tool is
    named and the ambiguity cannot arise.

    Otherwise the name goes upstream unchanged and matches nothing, silently
    as far as the provider is concerned — hence *selector* in the warning.
    The two causes are reported apart because a shared name is ours to explain
    and an undeclared one is the client's typo.
    """
    chosen = name_map.to_upstream(name, group)
    if chosen in declared:
        return chosen
    if group is None and name_map.is_ambiguous(name):
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
            group = entry.get("namespace")
            entry["name"] = _select_upstream_name(
                entry["name"],
                selector="allowed_tools",
                name_map=name_map,
                declared=declared,
                warnings=warnings,
                group=group if isinstance(group, str) else None,
            )


def _respell_allowed_tools_ir(
    choice: Any,
    *,
    name_map: ToolNameMap,
    declared: set[Any],
    warnings: list[str],
) -> None:
    """Re-spell the identities in an IR tool choice's allow-list.

    The IR keeps the group the source stated, so a restriction naming a tool
    whose bare name is shared still resolves; the group is dropped from the
    ref once the name carries the qualification, so the two cannot disagree.
    Mutates in place.
    """
    if not isinstance(choice, dict):
        return
    refs = choice.get("allowed_tools")
    if not isinstance(refs, list):
        return
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name")
        if not isinstance(name, str) or not name:
            continue
        group = ref.get("tool_group")
        ref["name"] = _select_upstream_name(
            name,
            selector="allowed_tools",
            name_map=name_map,
            declared=declared,
            warnings=warnings,
            group=group if isinstance(group, str) and group else None,
        )
        ref.pop("tool_group", None)


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
            if is_intrinsic_part(part):
                # A server call is named by its kind, not by a tool
                # definition: re-spelling it against the client's tool list
                # would ask the target to replay a server tool under a name
                # the client made up.  History intrinsics the target cannot
                # render have already been degraded to ordinary function
                # calls (``tool_type == "function"``) by
                # :func:`degrade_intrinsic_history`, so they are re-spelled
                # here like any other client call.
                continue
            namespace = tool_call_group(part)
            client_name = part.get("tool_name", "")
            upstream_name = name_map.to_upstream(client_name, namespace)
            part["tool_name"] = upstream_name
            if namespace is not None:
                continue
            if name_map.is_ambiguous(client_name) and not name_map.has_top_level(
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

    ``tool_choice`` and its ``allowed_tools`` restriction are re-spelled here
    too, along with the legacy ``allowed_tools`` extension. A bare selector
    carries no namespace, so it resolves by name alone or not at all; an IR
    ``allowed_tools`` identity does carry the group the source stated, so it
    resolves through it and the group is spent — no wire spelling in this
    position has a slot for one. Either way, a selector left naming no
    declared tool is reported in *warnings* — even when nothing was renamed,
    since such a selector is broken on its own terms.

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
    _respell_allowed_tools_ir(
        choice, name_map=name_map, declared=declared, warnings=warnings
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


def restore_client_tool_names(
    ir_response: dict[str, Any],
    *,
    name_map: ToolNameMap,
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

    def _restore_parts(msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "tool_call":
                _to_client_identity(part, name_map)

    for choice in ir_response.get("choices") or []:
        if isinstance(choice, dict):
            _restore_parts(choice.get("message"))

    for msg in ir_response.get("messages") or []:
        _restore_parts(msg)


def restore_client_tool_name_events(
    ir_events: list[dict[str, Any]],
    *,
    name_map: ToolNameMap,
) -> None:
    """Restore client-facing tool names on streamed tool calls.

    Mutates the ``tool_call_start`` events in *ir_events* in place.  The
    source converter reads the namespace back off ``provider_metadata`` and
    carries it onto the later done/completed items.  The streaming
    counterpart of :func:`restore_client_tool_names`.

    The streaming counterpart of :func:`restore_client_tool_names`, and unlike
    it this runs per chunk.
    """
    if not name_map:
        return

    for event in ir_events:
        if isinstance(event, dict) and event.get("type") == "tool_call_start":
            _to_client_identity(event, name_map)


def dedupe_tool_definitions(
    ir_request: dict[str, Any],
    *,
    warnings: list[str] | None = None,
    collisions: dict[str, int] | None = None,
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
        collisions: Optional mapping to fill with the conflicting names and
            their distinct-definition counts (``{name: count}``).  Cleared
            first, so a reused dict never keeps stale entries.  Empty when
            nothing conflicts — the exact duplicates that were collapsed do
            not appear here, only the ones left in place.

    Returns:
        The IR request with exact duplicates collapsed, or the original
        request unchanged when nothing was duplicated.
    """
    if collisions is not None:
        collisions.clear()

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
    if collisions is not None:
        collisions.update(conflicted)
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

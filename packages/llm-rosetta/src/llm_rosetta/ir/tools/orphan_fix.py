"""Tool call/result pairing, repaired once for every target dialect.

A request history can reach this library in a state no provider accepts: a
tool call nothing answered, a result whose declaration is missing, two
declarations sharing one id.  The repairs are the same for every target — a
repair in one writer is not a repair — so they happen here, once, on the IR,
before any dialect writes it.

The pairing is **positional and occurrence-aware**:

* a result answers the earliest *preceding* declaration it shares an id
  with that no result has answered yet — both lists keep the client's order;
* a result with no preceding declaration is dropped (an id nothing declares
  and an id declared later are different mistakes, and the warning says
  which);
* duplicate ids are renamed deterministically, first occurrence keeping the
  id the client used;
* every declaration still unanswered gets one synthetic result whose content
  names this library as its source.

A set of ids cannot express any of this: two declarations of ``q1`` are
"answered" by one ``q1`` result under set semantics, which is why the Chat
writer used to carry a second, occurrence-aware copy of the repair.  That
copy is gone; :meth:`BaseConverter.request_to_provider` runs this pass.

Intrinsic (provider-hosted) calls are answered by themselves: the provider ran
them, no client owes a result, and a synthetic one would fabricate a turn.

Also provides ``strip_orphaned_tool_config`` which removes ``tool_choice``
/ ``tool_config`` when no tool definitions are present (Codex CLI context
compaction workaround).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from llm_rosetta.conversion_warnings import report_drop
from llm_rosetta.types.ir import (
    IRInputItem,
    is_message,
    is_tool_call_part,
    is_tool_result_part,
)
from llm_rosetta.types.ir.request import IRRequest
from .call_id import sanitize_tool_call_id
from .intrinsic import is_intrinsic_part

logger = logging.getLogger(__name__)


# Content of the synthetic ``tool`` result the fixers emit for a call nothing
# answered. It is content this library invents, so it names its own provenance
# instead of reading as something the tool returned.
UNANSWERED_TOOL_CALL = "[llm-rosetta: no tool output was recorded for this call]"


class UnrepairableToolHistory(ValueError):
    """A tool history that cannot be repaired without inventing a turn.

    Raised when dropping the orphans leaves no message at all: forwarding an
    empty ``messages`` array tells the client nothing, while naming the
    orphaned id says what is wrong with its history.  The converter boundary
    turns this into the conversion error a gateway answers with a 400.

    Attributes:
        orphaned_ids: The ids whose results were dropped.
    """

    def __init__(self, message: str, *, orphaned_ids: Sequence[str] = ()) -> None:
        self.orphaned_ids = tuple(orphaned_ids)
        super().__init__(message)


# ==================== Shared Helpers ====================


def extract_part_ids(parts: Sequence[Any], part_type: str, id_key: str) -> set[str]:
    """Extract IDs from content parts/items matching a given type.

    Generic helper reused across IR, Anthropic, and OpenAI Responses
    orphaned-tool-call fixers.

    Args:
        parts: List of dicts (content parts, blocks, or flat items).
        part_type: Value of the ``type`` key to match.
        id_key: Key whose value is collected as an ID.

    Returns:
        Set of extracted ID strings.
    """
    return {
        part[id_key]
        for part in parts
        if isinstance(part, dict) and part.get("type") == part_type and part.get(id_key)
    }


def log_orphan_warnings(
    log: logging.Logger,
    call_ids: Sequence[str],
    result_ids: Sequence[str],
    call_label: str = "tool_call",
    result_label: str = "tool_result",
) -> None:
    """Log warnings for orphaned tool calls and results.

    Shared by the legacy per-format ``fix_orphaned_tool_calls`` functions.

    Args:
        log: Logger instance to emit warnings on.
        call_ids: IDs of orphaned calls that were patched.
        result_ids: IDs of orphaned results that were removed.
        call_label: Human-readable label for call items (e.g. "tool_use").
        result_label: Human-readable label for result items.
    """
    if call_ids:
        log.warning(
            "Fixed %d orphaned %s(s) by injecting synthetic results: %s",
            len(call_ids),
            call_label,
            ", ".join(call_ids),
        )
    if result_ids:
        log.warning(
            "Removed %d orphaned %s(s) with no matching %s: %s",
            len(result_ids),
            result_label,
            call_label,
            ", ".join(result_ids),
        )


# ==================== IR-Level Pairing Normalisation ====================


@dataclass
class _Declaration:
    """One tool-call part, as an occurrence in the request.

    Occurrence identity is the whole point: two parts sharing an id are two
    declarations, and a result answers exactly one of them.  The message and
    the part are held by reference so neither pruning nor renaming has to
    track an index that a removal elsewhere would invalidate.
    """

    message: Any
    part: Any
    original_id: str
    wire_id: str
    intrinsic: bool
    answered: bool = False
    result_part: Any = None

    @property
    def label(self) -> str:
        """The id to name in a warning."""
        return self.original_id or self.wire_id


@dataclass
class _Pairing:
    """What one walk over the history found."""

    declarations: list[_Declaration] = field(default_factory=list)
    #: ``(id, declared_later)`` per dropped client result.
    dropped: list[tuple[str, bool]] = field(default_factory=list)
    #: Messages that held nothing but the dropped results.
    emptied: list[Any] = field(default_factory=list)


def _content_list(message: Any) -> list[Any] | None:
    """Return the message's content list, or ``None`` when there is none."""
    if not is_message(message):
        return None
    content = message.get("content")
    return content if isinstance(content, list) else None


def _is_client_result(part: Any) -> bool:
    """Whether *part* is a tool result a client is expected to have sent."""
    return is_tool_result_part(part) and not is_intrinsic_part(part)


def _copy_messages(messages: Sequence[IRInputItem]) -> list[IRInputItem]:
    """Shallow-copy every message that carries a content list.

    The pass promises not to modify the caller's IR.  Parts are shared until
    one of them has to change; a changed part is replaced by a copy inside
    its message.
    """
    copied: list[IRInputItem] = []
    for message in messages:
        content = _content_list(message)
        if content is None:
            copied.append(message)
        else:
            copied.append(cast(IRInputItem, {**message, "content": list(content)}))
    return copied


def _replace_part_in(message: Any, old: Any, new: Any) -> None:
    """Swap *old* for *new* inside *message*'s content, by identity."""
    content = _content_list(message)
    if content is None:
        return
    for index, part in enumerate(content):
        if part is old:
            content[index] = new
            return


def _reserve_ids(messages: Sequence[IRInputItem]) -> set[str]:
    """Collect every tool-call id the request already uses.

    A generated id must collide with none of them — declarations *and*
    results — including ids whose declaration this pass has not reached yet.
    """
    reserved: set[str] = set()
    for message in messages:
        for part in _content_list(message) or ():
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("tool_call", "tool_result"):
                continue
            call_id = part.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                reserved.add(call_id)
    return reserved


def _collect_declared_ids(messages: Sequence[IRInputItem]) -> set[str]:
    """Every tool-call id the request declares anywhere."""
    declared: set[str] = set()
    for message in messages:
        if not is_message(message) or message.get("role") != "assistant":
            continue
        for part in _content_list(message) or ():
            if is_tool_call_part(part):
                call_id = part.get("tool_call_id")
                if isinstance(call_id, str) and call_id:
                    declared.add(call_id)
    return declared


def _answer_intrinsic(part: Any, pending: dict[str, list[_Declaration]]) -> None:
    """Mark the intrinsic declaration a server result belongs to as answered."""
    call_id = part.get("tool_call_id")
    stack = pending.get(call_id) if isinstance(call_id, str) and call_id else None
    if not stack:
        return
    for declaration in reversed(stack):
        if declaration.intrinsic and not declaration.answered:
            declaration.answered = True
            return


def _bind_declaration(
    message: Any,
    part: Any,
    pending: dict[str, list[_Declaration]],
    pairing: _Pairing,
) -> None:
    """Record one declaration occurrence, in message order."""
    call_id = part.get("tool_call_id")
    intrinsic = is_intrinsic_part(part)
    declaration = _Declaration(
        message=message,
        part=part,
        original_id=call_id if isinstance(call_id, str) else "",
        wire_id=call_id if isinstance(call_id, str) else "",
        intrinsic=intrinsic,
        # A provider-hosted call is answered by the provider.
        answered=intrinsic,
    )
    pairing.declarations.append(declaration)
    if declaration.original_id:
        pending.setdefault(declaration.original_id, []).append(declaration)


def _bind_result(
    part: Any,
    pending: dict[str, list[_Declaration]],
    pairing: _Pairing,
    declared_later: set[str],
) -> tuple[bool, bool]:
    """Pair or drop one result part.

    Returns:
        ``(keep, client)`` — whether the part stays in its message, and
        whether it was a client result rather than a server one.
    """
    if not _is_client_result(part):
        # A server result answers its own call and is never an orphan:
        # nothing on the client side would declare it.
        _answer_intrinsic(part, pending)
        return True, False

    call_id = part.get("tool_call_id")
    stack = pending.get(call_id) if isinstance(call_id, str) and call_id else None
    if stack:
        # First in, first answered: when two declarations share an id, the
        # order of the results is the only signal left, and it is the order
        # the client sent them in.
        declaration = stack.pop(0)
        declaration.answered = True
        declaration.result_part = part
        return True, True

    pairing.dropped.append(
        (
            call_id if isinstance(call_id, str) else "",
            isinstance(call_id, str) and call_id in declared_later,
        )
    )
    return False, True


def _pair_message(
    message: Any,
    pending: dict[str, list[_Declaration]],
    pairing: _Pairing,
    declared_later: set[str],
) -> None:
    """Pair one message's parts, pruning the results that pair with nothing."""
    content = _content_list(message)
    if content is None:
        return
    kept: list[Any] = []
    had_client_result = False
    for part in content:
        if is_tool_call_part(part):
            _bind_declaration(message, part, pending, pairing)
            kept.append(part)
            continue
        if is_tool_result_part(part):
            keep, client = _bind_result(part, pending, pairing, declared_later)
            had_client_result = had_client_result or client
            if keep:
                kept.append(part)
            continue
        kept.append(part)
    if had_client_result and not kept:
        pairing.emptied.append(message)
    if len(kept) != len(content):
        content[:] = kept


def _pair_results(messages: list[IRInputItem]) -> _Pairing:
    """Pair every client result with the nearest preceding declaration."""
    pairing = _Pairing()
    pending: dict[str, list[_Declaration]] = {}
    declared_later = _collect_declared_ids(messages)
    for message in messages:
        _pair_message(message, pending, pairing, declared_later)
    return pairing


def _rename_duplicates(
    messages: Sequence[IRInputItem],
    declarations: Sequence[_Declaration],
    warnings: Any,
) -> None:
    """Give every repeated declaration after the first its own wire id.

    Deterministic: the second occurrence of ``q1`` is ``q1_duplicate_2``
    unless that id is already in the request.  The result paired with an
    occurrence is renamed with it, so a client replay keeps working.
    """
    reserved = _reserve_ids(messages)
    seen: dict[str, int] = {}
    for declaration in declarations:
        original = declaration.original_id
        if not original:
            continue
        occurrence = seen.get(original, 0) + 1
        seen[original] = occurrence
        if occurrence == 1:
            continue

        suffix = occurrence
        while True:
            candidate = sanitize_tool_call_id(f"{original}_duplicate_{suffix}")
            if candidate not in reserved:
                break
            suffix += 1
        reserved.add(candidate)
        declaration.wire_id = candidate

        renamed = cast(dict[str, Any], {**declaration.part, "tool_call_id": candidate})
        _replace_part_in(declaration.message, declaration.part, renamed)
        declaration.part = renamed
        if declaration.result_part is not None:
            renamed_result = cast(
                dict[str, Any],
                {**declaration.result_part, "tool_call_id": candidate},
            )
            for message in messages:
                content = _content_list(message)
                if content is not None and any(
                    part is declaration.result_part for part in content
                ):
                    _replace_part_in(message, declaration.result_part, renamed_result)
                    break
            declaration.result_part = renamed_result

        report_drop(
            f"Renamed duplicate tool_call_id='{original}' to '{candidate}'",
            warnings=warnings,
            subject=original,
        )


def normalize_tool_pairing(
    messages: Sequence[IRInputItem],
    *,
    warnings: Any = None,
    placeholder: str = UNANSWERED_TOOL_CALL,
) -> list[IRInputItem]:
    """Repair tool call/result pairing once, for every target dialect.

    The repairs, in the order they have to happen:

    1. **Pair.**  A client result answers the nearest *preceding* declaration
       with the same id, in the order both appear.  A result with no such
       declaration is dropped — an
       id nothing declares and one declared later are both unrepresentable,
       and the warning names which mistake it was.
    2. **Rename.**  Repeated declarations keep the client's id on the first
       occurrence; later ones get a deterministic, collision-free id, and the
       result paired with them is renamed along.
    3. **Fill.**  Every remaining unanswered client declaration gets one
       synthetic ``tool`` result carrying :data:`UNANSWERED_TOOL_CALL`.
       Intrinsic calls are skipped: the provider ran them, so a synthetic
       result would fabricate a turn the client never owed.

    The pass is idempotent — a second run finds every declaration answered
    and every id unique — and never modifies the caller's IR.

    Args:
        messages: IR messages to repair.
        warnings: Where repairs are recorded; a
            :class:`~llm_rosetta.conversion_warnings.WarningSink` (the
            conversion's own list) when a caller has one.
        placeholder: Content for the synthetic result.

    Returns:
        A new messages list with the pairing repaired.

    Raises:
        UnrepairableToolHistory: When dropping orphans leaves no message at
            all.  An empty ``messages`` array is not a repair.
    """
    work = _copy_messages(messages)
    if not work:
        return work

    pairing = _pair_results(work)

    for call_id, declared_later in pairing.dropped:
        reason = (
            "it arrives before its declaration"
            if declared_later
            else "no tool call declares it"
        )
        report_drop(
            f"Dropped tool result id={call_id!r}: {reason}",
            warnings=warnings,
            subject=call_id,
        )

    emptied = set(map(id, pairing.emptied))
    kept_messages = [message for message in work if id(message) not in emptied]
    if pairing.dropped and not kept_messages:
        orphaned = ", ".join(sorted({call_id for call_id, _ in pairing.dropped}))
        raise UnrepairableToolHistory(
            "Tool history has no message left after dropping orphaned tool "
            f"result(s): {orphaned}. Each tool_result needs a tool_call in a "
            "preceding message.",
            orphaned_ids=[call_id for call_id, _ in pairing.dropped],
        )

    _rename_duplicates(kept_messages, pairing.declarations, warnings)

    by_message: dict[int, list[_Declaration]] = {}
    for declaration in pairing.declarations:
        if declaration.answered or declaration.intrinsic or not declaration.wire_id:
            continue
        by_message.setdefault(id(declaration.message), []).append(declaration)

    filled: list[IRInputItem] = []
    for message in kept_messages:
        filled.append(message)
        content = _content_list(message)
        if content is None:
            continue
        for declaration in by_message.get(id(message), ()):
            if not any(part is declaration.part for part in content):
                # Its message survived without the declaration; nothing to
                # answer on this turn.
                continue
            filled.append(
                cast(
                    IRInputItem,
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": declaration.wire_id,
                                "result": placeholder,
                            }
                        ],
                    },
                )
            )
            report_drop(
                "Inserted placeholder for unanswered tool_call_id="
                f"'{declaration.label}'",
                warnings=warnings,
                subject=declaration.label,
            )

    return filled


def fix_orphaned_tool_calls_ir(
    messages: Sequence[IRInputItem],
    *,
    placeholder: str = UNANSWERED_TOOL_CALL,
) -> list[IRInputItem]:
    """Deprecated alias of :func:`normalize_tool_pairing`.

    Kept for callers that name the old repair; it now performs the whole
    occurrence-aware normalisation, not only the set-difference one.

    Args:
        messages: IR messages (any iterable of Message dicts).
        placeholder: Content string for injected synthetic tool results.

    Returns:
        A new messages list with the pairing repaired.
    """
    return normalize_tool_pairing(messages, placeholder=placeholder)


# ==================== Orphaned Tool Config Fix (IR level) ====================


def strip_orphaned_tool_config(ir_request: IRRequest) -> list[str]:
    """Strip ``tool_choice`` and ``tool_config`` when no tools are defined.

    Codex CLI context compaction can remove all tool definitions from a
    request while keeping ``tool_choice`` (e.g. ``"auto"``).  This produces
    an invalid request that upstream APIs reject with *"tool_choice is set
    but no tools are provided"*.

    This is part of the same problem family as
    :func:`normalize_tool_pairing` (orphaned tool_call/result pairing) and
    ``_reorder_tool_messages`` (tool message ordering) — all stem from
    Codex context compaction breaking request structural integrity.

    The request dict is modified **in-place**.

    Args:
        ir_request: IR request dict (mutated in-place).

    Returns:
        One warning per stripped field.  The caller records them — the
        converters append them to their context's sink, which is what logs
        them — so this function is deliberately silent.
    """
    tools = ir_request.get("tools")
    has_tools = bool(tools)

    if has_tools:
        return []

    # Cast to plain dict for mutation — IRRequest is a TypedDict at
    # type-check time but a regular dict at runtime.
    request_dict = cast(dict[str, Any], ir_request)

    warnings: list[str] = []
    for name in ("tool_choice", "tool_config"):
        if name in request_dict:
            value = request_dict.pop(name)
            warnings.append(
                f"Stripped orphaned '{name}' (value: {value!r}) — "
                "no tool definitions present in request (Codex context "
                "compaction workaround)"
            )

    return warnings

"""The ledger's columns, and the canonical total that fills one of them.

Every usage figure the project stores or shows is one of six columns.  This
module is the only place that says what they are, which IR field each one comes
from, and which single value counts as "the total".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["IR_FIELDS", "LEDGER_COLUMNS", "token_total", "usage_columns"]

#: The token columns of a ledger row, in display order.  ``total_tokens`` is the
#: canonical total; the rest are a breakdown of it, not parts that add up to it
#: (reasoning rides inside the output, and cache is additive for some standards
#: and a subset of the input for others — see
#: :mod:`llm_rosetta.observability.usage.semantics`).
LEDGER_COLUMNS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)

#: Ledger column -> the IR usage field behind it.
IR_FIELDS: Mapping[str, str] = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "total_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_creation_tokens": "cache_creation_tokens",
    "reasoning_tokens": "reasoning_tokens",
}


def token_total(usage: Mapping[str, Any] | None) -> int | None:
    """The canonical token total of an IR usage dict.

    The provider-reported ``total_tokens`` wins whenever the provider sends
    one; a total the provider stands behind is not ours to recompute, and for
    the OpenAI-shaped standards it sits below the additive sum on purpose
    (their cached counters are part of the prompt count).  Only when no total
    is sent — the Messages API never sends one — is it computed additively:
    input + output + cache read + cache write.

    :meth:`~llm_rosetta.observability.persistence.PersistenceManager.backfill_total_tokens`
    applies the same rule to rows an older writer left blank or derived too low,
    which is why a stored total survives a restart unchanged.

    ``reasoning_tokens`` is deliberately never added: it is a subset of the
    output tokens for every provider that reports it.

    Returns:
        The total, or ``None`` when *usage* names no token count at all.
    """
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is None:
        parts = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("cache_read_tokens"),
            usage.get("cache_creation_tokens"),
        )
        if all(part is None for part in parts):
            return None
        total = sum(part or 0 for part in parts)
    return int(total)


def usage_columns(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """The ledger columns of an IR usage dict, the canonical total included.

    A column the provider did not report is left out rather than zeroed, so a
    caller can tell "no cache in play" from "cache reported as zero".

    Returns:
        A subset of :data:`LEDGER_COLUMNS`, suitable for splatting into
        ``RequestLogEntry.create`` or ``MetricsCollector.record_request``.
    """
    if not usage:
        return {}
    columns: dict[str, int] = {}
    for column, field in IR_FIELDS.items():
        value = usage.get(field)
        if isinstance(value, int):
            columns[column] = value
    total = token_total(usage)
    if total is not None:
        columns["total_tokens"] = total
    return columns

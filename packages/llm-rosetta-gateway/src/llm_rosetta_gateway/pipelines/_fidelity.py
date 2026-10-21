"""Same-format fidelity shadow check for the non-LLM handlers.

The chat proxy compares every same-format request and response with its IR
round-trip and logs the differences under ``[FIDELITY]``.  The embedding,
rerank and decision handlers now convert same-format traffic through the IR
as well, so they run the same check.

This is a warning channel, not a gate: a conversion that raises already fails
the request with a ``400``, and a critical diff only decides the severity of
the warning.  Unlike the chat path there is no error dump.

Every same-format request and response runs the check — there is no sampling
and no debug gate — and a diff is logged at ``WARNING`` under ``[FIDELITY]``,
once per direction.  The comparison itself is the fast ``critical`` mode
(~0.01 ms); an identical body logs nothing, so the log volume tracks actual
round-trip loss rather than request volume.
"""

from __future__ import annotations

from typing import Any, Literal

from ..fidelity_check import compare_same_format
from ..logging import get_logger, log_fidelity_warning

logger = get_logger()


def check_same_format_fidelity(
    model_type: str,
    source_format: str,
    target_format: str,
    original: dict[str, Any] | None,
    converted: dict[str, Any],
    direction: Literal["request", "response"],
    *,
    model: str | None = None,
) -> None:
    """Warn when a same-format IR round-trip changed a critical field.

    *original* is ``None`` when the handler did not capture a passthrough
    baseline (it only does so for a same-format request), and the check is
    skipped.  Cross-format conversions are skipped too: the source and target
    shapes are expected to differ.
    """
    summary = compare_same_format(
        source_format,
        target_format,
        original,
        converted,
        direction,
        model_type=model_type,
    )
    if summary is None:
        return
    log_fidelity_warning(
        summary.diff_count,
        summary.max_severity,
        direction,
        model=model,
        source_provider=source_format,
    )

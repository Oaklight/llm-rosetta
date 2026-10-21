"""One same-format fidelity check, two reporting modes.

The chat proxy and the typed (embedding/rerank/decision) handlers both compare
a same-format IR round-trip with its original body and report the differences
under ``[FIDELITY]``.  They differ only in *reporting*: the chat path records
the summary in its profile and dumps an error on a critical diff, while the
typed path logs and moves on.  The comparison itself lives here once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class FidelitySummary:
    """What one comparison found.

    Attributes:
        diff_count: Number of differing paths.
        max_severity: The most severe effective severity seen.
        critical: Whether a critical-severity difference was found.
        diffs: The first 20 differences, as ``{path, kind, severity}``.
    """

    diff_count: int
    max_severity: str
    critical: bool
    diffs: list[dict[str, Any]] = field(default_factory=list)


def compare_same_format(
    source_format: str,
    target_format: str,
    original: dict[str, Any] | None,
    converted: dict[str, Any],
    direction: Literal["request", "response"],
    *,
    model_type: str = "llm",
) -> FidelitySummary | None:
    """Compare a same-format round-trip, or return ``None`` when skipped.

    The check is skipped when *original* is ``None`` (the caller did not keep
    a passthrough baseline) or when the formats differ — a cross-format
    conversion is expected to change the shape.  A failure inside the checker
    is swallowed: fidelity is a warning channel, never a gate.
    """
    if original is None or source_format != target_format:
        return None
    try:
        from llm_rosetta.fidelity import FidelityChecker

        checker = FidelityChecker(
            mode="critical", format_name=source_format, model_type=model_type
        )
        diffs = checker.compare(original, converted, direction=direction)
    except Exception:
        return None
    if not diffs:
        return None
    max_severity = max(diff.effective_severity for diff in diffs)
    return FidelitySummary(
        diff_count=len(diffs),
        max_severity=max_severity,
        critical=max_severity == "critical",
        diffs=[
            {
                "path": diff.path,
                "kind": diff.kind,
                "severity": diff.effective_severity,
            }
            for diff in diffs[:20]
        ],
    )

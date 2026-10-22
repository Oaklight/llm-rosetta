"""The same-format fidelity shadow check and its error dump."""

from __future__ import annotations

from typing import Any, Literal


from llm_rosetta.routing import ResolvedRoute

from llm_rosetta_gateway.observability.error_dump import dump_error

from ..fidelity_check import compare_same_format
from ..logging import get_logger, log_fidelity_warning

logger = get_logger()


def _check_fidelity(
    source_provider: str,
    target_provider: str,
    original: dict[str, Any],
    converted: dict[str, Any],
    direction: Literal["request", "response"],
    profile: dict[str, Any],
) -> bool:
    """Run the fidelity diff for same-format routes.

    Returns ``True`` if any critical-severity diffs were found.  Results are
    injected into *profile* under ``"fidelity"``.
    """
    summary = compare_same_format(
        source_provider, target_provider, original, converted, direction
    )
    if summary is None:
        return False
    profile.setdefault("fidelity", {})[direction] = {
        "diff_count": summary.diff_count,
        "max_severity": summary.max_severity,
        "diffs": summary.diffs,
    }
    log_fidelity_warning(
        summary.diff_count,
        summary.max_severity,
        direction,
        source_provider=source_provider,
    )
    return summary.critical


async def _fidelity_check_and_dump(
    route: ResolvedRoute,
    original: dict[str, Any],
    converted: dict[str, Any],
    direction: Literal["request", "response"],
    profile: dict[str, Any],
    model: str,
    persistence: Any | None,
    entry_id: str | None,
) -> None:
    """Run fidelity diff and dump on critical severity."""
    if _check_fidelity(
        route.source_provider,
        route.target_provider,
        original,
        converted,
        direction,
        profile,
    ):
        await dump_error(
            persistence,
            request_body=original,
            converted_body=converted,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            error_phase="fidelity",
            request_log_id=entry_id,
        )

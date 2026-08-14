"""Round-trip fidelity checker for same-format conversion.

Compares an original request/response body against its IR round-tripped
version to detect information loss.  Two modes:

- **critical**: check only fields known to cause breakage (fast, ~0.01ms)
- **full**: canonical JSON comparison of entire body (~1ms for 50KB)

Usage::

    checker = FidelityChecker(mode="critical")
    diffs = checker.compare_request(original, roundtripped)
    if diffs:
        logger.warning("Round-trip fidelity loss: %s", diffs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FidelityMode = Literal["critical", "full"]


# ============================================================================
# Critical field paths — known fragile spots in round-trips
# ============================================================================

# Request fields that converters may alter
_COMMON_REQUEST_PATHS: list[str] = [
    "model",
    "stream",
]

_COMMON_RESPONSE_PATHS: list[str] = [
    "id",
    "model",
    "usage",
]

_CRITICAL_REQUEST_BY_FORMAT: dict[str, list[str]] = {
    "openai_chat": [
        "max_tokens",
        "max_completion_tokens",
        "tools.*.type",
        "tools.*.function.name",
        "tool_choice",
        "messages.*.role",
        "messages.*.content.*.type",
        "messages.*.content.*.tool_call_id",
        "messages.*.tool_calls.*.type",
        "messages.*.tool_calls.*.function.name",
    ],
    "anthropic": [
        "max_tokens",
        "system",
        "messages.*.role",
        "messages.*.content.*.type",
        "messages.*.content.*.tool_use_id",
        "tools.*.name",
        "tool_choice.type",
    ],
    "openai_responses": [
        "input.*.type",
        "input.*.role",
        "input.*.content.*.type",
        "input.*.call_id",
        "input.*.name",
        "input.*.output",
        "tools.*.type",
        "tools.*.name",
    ],
    "google": [
        "contents.*.role",
        "contents.*.parts.*.text",
        "contents.*.parts.*.functionCall.name",
        "contents.*.parts.*.functionResponse.name",
        "tools.*.functionDeclarations.*.name",
        "systemInstruction.parts.*.text",
        "generationConfig.maxOutputTokens",
        "generationConfig.temperature",
    ],
}

_CRITICAL_RESPONSE_BY_FORMAT: dict[str, list[str]] = {
    "openai_chat": [
        "object",
        "choices.*.finish_reason",
        "choices.*.message.role",
        "choices.*.message.content.*.type",
        "choices.*.message.tool_calls.*.type",
        "choices.*.message.tool_calls.*.id",
        "choices.*.message.tool_calls.*.function.name",
    ],
    "anthropic": [
        "type",
        "stop_reason",
        "content.*.type",
        "content.*.id",
        "content.*.name",
        "content.*.tool_use_id",
    ],
    "openai_responses": [
        "object",
        "status",
        "output.*.type",
        "output.*.id",
        "output.*.content.*.type",
        "output.*.call_id",
        "output.*.name",
        "output.*.status",
    ],
    "google": [
        "candidates.*.content.role",
        "candidates.*.content.parts.*.text",
        "candidates.*.content.parts.*.functionCall.name",
        "candidates.*.finishReason",
        "usageMetadata.promptTokenCount",
        "usageMetadata.candidatesTokenCount",
        "usageMetadata.totalTokenCount",
    ],
}


def _get_critical_paths(
    format_name: str | None,
    common: list[str],
    by_format: dict[str, list[str]],
) -> list[str]:
    """Return critical paths for a specific format, or all if format is None."""
    if format_name is not None and format_name in by_format:
        return common + by_format[format_name]
    # Unknown format or None: check all paths (union)
    all_paths = list(common)
    for paths in by_format.values():
        all_paths.extend(paths)
    return all_paths


# ============================================================================
# Diff result
# ============================================================================


@dataclass
class FidelityDiff:
    """A single field-level difference found during fidelity comparison."""

    path: str
    original: Any = field(repr=False, default=None)
    roundtripped: Any = field(repr=False, default=None)
    kind: str = ""  # "missing", "added", "changed", "type_changed"

    def __str__(self) -> str:
        if self.kind == "missing":
            return f"{self.path}: missing after round-trip"
        if self.kind == "added":
            return f"{self.path}: added by round-trip"
        if self.kind == "type_changed":
            return (
                f"{self.path}: type changed "
                f"{type(self.original).__name__} → {type(self.roundtripped).__name__}"
            )
        return f"{self.path}: changed"


# ============================================================================
# Path extraction
# ============================================================================


def _get_at_path(obj: Any, segments: list[str]) -> list[tuple[str, Any]]:
    """Extract values at a dotted path with ``*`` wildcard for arrays.

    Returns list of (resolved_path, value) pairs.
    """
    if not segments:
        return [("", obj)]

    head, *rest = segments

    if head == "*":
        if not isinstance(obj, list):
            return []
        results = []
        for i, item in enumerate(obj):
            for sub_path, val in _get_at_path(item, rest):
                full = f"[{i}]{('.' + sub_path) if sub_path else ''}"
                results.append((full, val))
        return results

    if isinstance(obj, dict) and head in obj:
        results = []
        for sub_path, val in _get_at_path(obj[head], rest):
            full = f"{head}{('.' + sub_path) if sub_path else ''}"
            results.append((full, val))
        return results

    return []


def _extract_field(obj: Any, path: str) -> list[tuple[str, Any]]:
    """Extract all values matching a dotted path pattern."""
    segments = path.split(".")
    return _get_at_path(obj, segments)


# ============================================================================
# Comparison logic
# ============================================================================


def _compare_critical(
    original: dict[str, Any],
    roundtripped: dict[str, Any],
    paths: list[str],
) -> list[FidelityDiff]:
    """Compare only critical field paths between two dicts."""
    diffs: list[FidelityDiff] = []

    for path_pattern in paths:
        orig_values = dict(_extract_field(original, path_pattern))
        rt_values = dict(_extract_field(roundtripped, path_pattern))

        all_paths = set(orig_values.keys()) | set(rt_values.keys())
        for resolved in sorted(all_paths):
            full_path = f"{path_pattern}→{resolved}" if resolved else path_pattern
            if resolved not in orig_values:
                diffs.append(
                    FidelityDiff(
                        path=full_path,
                        roundtripped=rt_values[resolved],
                        kind="added",
                    )
                )
            elif resolved not in rt_values:
                diffs.append(
                    FidelityDiff(
                        path=full_path,
                        original=orig_values[resolved],
                        kind="missing",
                    )
                )
            else:
                ov, rv = orig_values[resolved], rt_values[resolved]
                if type(ov) is not type(rv):
                    diffs.append(
                        FidelityDiff(
                            path=full_path,
                            original=ov,
                            roundtripped=rv,
                            kind="type_changed",
                        )
                    )
                elif ov != rv:
                    diffs.append(
                        FidelityDiff(
                            path=full_path,
                            original=ov,
                            roundtripped=rv,
                            kind="changed",
                        )
                    )

    return diffs


def _diff_recursive(
    original: Any,
    roundtripped: Any,
    path: str,
    diffs: list[FidelityDiff],
) -> None:
    """Recursively compare two values, reporting leaf-level differences."""
    if type(original) is not type(roundtripped):
        diffs.append(
            FidelityDiff(
                path=path,
                original=original,
                roundtripped=roundtripped,
                kind="type_changed",
            )
        )
        return

    if isinstance(original, dict):
        all_keys = set(original.keys()) | set(roundtripped.keys())
        for key in sorted(all_keys):
            child_path = f"{path}.{key}" if path else key
            if key not in original:
                diffs.append(
                    FidelityDiff(
                        path=child_path,
                        roundtripped=roundtripped[key],
                        kind="added",
                    )
                )
            elif key not in roundtripped:
                diffs.append(
                    FidelityDiff(
                        path=child_path,
                        original=original[key],
                        kind="missing",
                    )
                )
            else:
                _diff_recursive(original[key], roundtripped[key], child_path, diffs)
        return

    if isinstance(original, list):
        for i in range(max(len(original), len(roundtripped))):
            child_path = f"{path}[{i}]"
            if i >= len(original):
                diffs.append(
                    FidelityDiff(
                        path=child_path,
                        roundtripped=roundtripped[i],
                        kind="added",
                    )
                )
            elif i >= len(roundtripped):
                diffs.append(
                    FidelityDiff(
                        path=child_path,
                        original=original[i],
                        kind="missing",
                    )
                )
            else:
                _diff_recursive(original[i], roundtripped[i], child_path, diffs)
        return

    # Leaf comparison
    if original != roundtripped:
        diffs.append(
            FidelityDiff(
                path=path,
                original=original,
                roundtripped=roundtripped,
                kind="changed",
            )
        )


def _compare_full(
    original: dict[str, Any],
    roundtripped: dict[str, Any],
) -> list[FidelityDiff]:
    """Full recursive leaf-level diff."""
    diffs: list[FidelityDiff] = []
    _diff_recursive(original, roundtripped, "", diffs)
    return diffs


# ============================================================================
# FidelityChecker
# ============================================================================


class FidelityChecker:
    """Compare original and round-tripped bodies for fidelity loss.

    Args:
        mode: ``"critical"`` checks only format-specific fragile fields
            (~0.01ms). ``"full"`` does recursive leaf-level diff (~1ms
            for 50KB).
        format_name: API format to check against (``"openai_chat"``,
            ``"anthropic"``, ``"openai_responses"``).  When ``None``,
            checks all formats' paths (slower but safe when format is
            unknown).
    """

    def __init__(
        self,
        mode: FidelityMode = "critical",
        format_name: str | None = None,
    ) -> None:
        self.mode = mode
        self.format_name = format_name

    def compare_request(
        self,
        original: dict[str, Any],
        roundtripped: dict[str, Any],
    ) -> list[FidelityDiff]:
        """Compare an original request body against its round-tripped version."""
        if self.mode == "critical":
            paths = _get_critical_paths(
                self.format_name, _COMMON_REQUEST_PATHS, _CRITICAL_REQUEST_BY_FORMAT
            )
            return _compare_critical(original, roundtripped, paths)
        return _compare_full(original, roundtripped)

    def compare_response(
        self,
        original: dict[str, Any],
        roundtripped: dict[str, Any],
    ) -> list[FidelityDiff]:
        """Compare an original response body against its round-tripped version."""
        if self.mode == "critical":
            paths = _get_critical_paths(
                self.format_name, _COMMON_RESPONSE_PATHS, _CRITICAL_RESPONSE_BY_FORMAT
            )
            return _compare_critical(original, roundtripped, paths)
        return _compare_full(original, roundtripped)

    def compare(
        self,
        original: dict[str, Any],
        roundtripped: dict[str, Any],
        *,
        direction: Literal["request", "response"] = "request",
    ) -> list[FidelityDiff]:
        """Compare bodies in the specified direction."""
        if direction == "request":
            return self.compare_request(original, roundtripped)
        return self.compare_response(original, roundtripped)

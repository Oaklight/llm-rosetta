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

from llm_rosetta.provider_names import canonical_provider_name

FidelityMode = Literal["critical", "full"]

#: Severity levels for fidelity differences, most severe first.
FidelitySeverity = Literal["critical", "warning", "info"]

#: Default severity derived from a diff's :attr:`FidelityDiff.kind`.
#:
#: - ``missing`` — a field was dropped by the round-trip (information loss).
#: - ``type_changed`` — incompatible types, likely to break consumers.
#: - ``changed`` — value altered, impact depends on the field.
#: - ``added`` — extra field introduced, usually harmless.
DEFAULT_SEVERITY_BY_KIND: dict[str, FidelitySeverity] = {
    "missing": "critical",
    "type_changed": "critical",
    "changed": "warning",
    "added": "info",
}


def severity_for_kind(kind: str) -> FidelitySeverity:
    """Return the default severity for a diff ``kind``.

    Unknown kinds default to ``"warning"`` so a new diff category is never
    silently classified as harmless.
    """
    return DEFAULT_SEVERITY_BY_KIND.get(kind, "warning")


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
    "chat_completions": [
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
    "messages": [
        "max_tokens",
        "system",
        "messages.*.role",
        "messages.*.content.*.type",
        "messages.*.content.*.tool_use_id",
        "tools.*.name",
        "tool_choice.type",
    ],
    "open_responses": [
        "input.*.type",
        "input.*.role",
        "input.*.content.*.type",
        "input.*.call_id",
        "input.*.name",
        "input.*.output",
        "tools.*.type",
        "tools.*.name",
    ],
    "google_generate": [
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
    "chat_completions": [
        "object",
        "choices.*.finish_reason",
        "choices.*.message.role",
        "choices.*.message.content.*.type",
        "choices.*.message.tool_calls.*.type",
        "choices.*.message.tool_calls.*.id",
        "choices.*.message.tool_calls.*.function.name",
    ],
    "messages": [
        "type",
        "stop_reason",
        "content.*.type",
        "content.*.id",
        "content.*.name",
        "content.*.tool_use_id",
    ],
    "open_responses": [
        "object",
        "status",
        "output.*.type",
        "output.*.id",
        "output.*.content.*.type",
        "output.*.call_id",
        "output.*.name",
        "output.*.status",
    ],
    "google_generate": [
        "candidates.*.content.role",
        "candidates.*.content.parts.*.text",
        "candidates.*.content.parts.*.functionCall.name",
        "candidates.*.finishReason",
        "usageMetadata.promptTokenCount",
        "usageMetadata.candidatesTokenCount",
        "usageMetadata.totalTokenCount",
    ],
}


# Non-LLM model types serve one route for several dialects, so the critical
# paths are the union across their formats: a path that does not exist in the
# source dialect is absent from both sides of the comparison and never diffs.
_CRITICAL_REQUEST_BY_TYPE: dict[str, list[str]] = {
    "embedding": [
        "input",
        "texts",
        "encoding_format",
        "dimensions",
        "output_dimension",
        "taskType",
        "task",
        "input_type",
        "embedding_types",
        "embedding_type",
        "output_dtype",
        "truncate",
        "truncation",
        "autoTruncate",
        "user",
        "title",
        "normalized",
        "late_chunking",
        "return_multivector",
        "content.parts.*.text",
        "requests.*.model",
        "requests.*.content.parts.*.text",
    ],
    "rerank": [
        "query",
        "documents",
        "top_n",
        "top_k",
        "return_documents",
        "max_chunks_per_doc",
    ],
    "decision": [
        "state",
        "images",
        "questions.*.type",
        "questions.*.instructions",
        "questions.*.criteria",
        "questions.*.criteria.*",
    ],
}

_CRITICAL_RESPONSE_BY_TYPE: dict[str, list[str]] = {
    "embedding": [
        "object",
        "id",
        "texts",
        "data.*.index",
        "data.*.embedding",
        "embeddings.float",
        "embeddings.int8",
        "embeddings.uint8",
        "embeddings.binary",
        "embeddings.ubinary",
        "embedding.values",
        "embeddings.*.values",
        "usage.prompt_tokens",
        "usage.total_tokens",
        "usageMetadata.promptTokenCount",
        "usageMetadata.totalTokenCount",
        "meta.billed_units.input_tokens",
        "meta.tokens.input_tokens",
    ],
    "rerank": [
        "object",
        "id",
        "results.*.index",
        "results.*.relevance_score",
        "results.*.document",
        "usage.total_tokens",
        "usage.prompt_tokens",
        "meta.billed_units",
        "meta.tokens.input_tokens",
    ],
    "decision": [
        "object",
        "answers.*.type",
        "answers.*.probability",
        "answers.*.choice",
        "answers.*.score",
        "answers.*.probabilities",
        "answers.*.confidence",
        "usage.input_tokens",
        "usage.output_tokens",
    ],
}


def _get_critical_paths(
    format_name: str | None,
    common: list[str],
    by_format: dict[str, list[str]],
) -> list[str]:
    """Return critical paths for a specific format, or all if format is None."""
    if format_name == "google":
        format_name = "google_generate"  # legacy alias
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
    #: Explicit severity override; ``None`` defers to :func:`severity_for_kind`.
    severity: str | None = None

    @property
    def effective_severity(self) -> str:
        """Severity used for baselines: the override or the kind-derived default."""
        if self.severity is not None:
            return self.severity
        return severity_for_kind(self.kind)

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
    """Extract values at a dotted path with a ``*`` wildcard.

    ``*`` iterates a list's items or a dict's values, so a path can name a
    per-item field without knowing the container's shape (``answers.*.probability``
    for a dict keyed by question id, ``data.*.embedding`` for a list).
    Returns list of (resolved_path, value) pairs.
    """
    if not segments:
        return [("", obj)]

    head, *rest = segments

    if head == "*":
        if isinstance(obj, list):
            results = []
            for i, item in enumerate(obj):
                for sub_path, val in _get_at_path(item, rest):
                    full = f"[{i}]{('.' + sub_path) if sub_path else ''}"
                    results.append((full, val))
            return results
        if isinstance(obj, dict):
            results = []
            for key in sorted(obj):
                for sub_path, val in _get_at_path(obj[key], rest):
                    full = f"{key}{('.' + sub_path) if sub_path else ''}"
                    results.append((full, val))
            return results
        return []

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
        format_name: API format to check against (``"chat_completions"``,
            ``"messages"``, ``"open_responses"``).  When ``None``,
            checks all formats' paths (slower but safe when format is
            unknown).
        model_type: Capability category.  For a non-LLM type
            (``embedding``, ``rerank``, ``decision``) the critical paths
            are the union over that family's dialects, because one route
            serves all of them; *format_name* only labels the warning.
    """

    def __init__(
        self,
        mode: FidelityMode = "critical",
        format_name: str | None = None,
        model_type: str = "llm",
    ) -> None:
        self.mode = mode
        self.model_type = model_type
        # Accept legacy base/format spellings (e.g. ``openai_chat``) in the
        # internal format name without emitting a duplicate warning.
        self.format_name = (
            canonical_provider_name(format_name) if format_name is not None else None
        )

    def _request_paths(self) -> list[str]:
        by_type = _CRITICAL_REQUEST_BY_TYPE.get(self.model_type)
        if by_type is not None:
            return _COMMON_REQUEST_PATHS + by_type
        return _get_critical_paths(
            self.format_name, _COMMON_REQUEST_PATHS, _CRITICAL_REQUEST_BY_FORMAT
        )

    def _response_paths(self) -> list[str]:
        by_type = _CRITICAL_RESPONSE_BY_TYPE.get(self.model_type)
        if by_type is not None:
            return _COMMON_RESPONSE_PATHS + by_type
        return _get_critical_paths(
            self.format_name, _COMMON_RESPONSE_PATHS, _CRITICAL_RESPONSE_BY_FORMAT
        )

    def compare_request(
        self,
        original: dict[str, Any],
        roundtripped: dict[str, Any],
    ) -> list[FidelityDiff]:
        """Compare an original request body against its round-tripped version."""
        if self.mode == "critical":
            return _compare_critical(original, roundtripped, self._request_paths())
        return _compare_full(original, roundtripped)

    def compare_response(
        self,
        original: dict[str, Any],
        roundtripped: dict[str, Any],
    ) -> list[FidelityDiff]:
        """Compare an original response body against its round-tripped version."""
        if self.mode == "critical":
            return _compare_critical(original, roundtripped, self._response_paths())
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

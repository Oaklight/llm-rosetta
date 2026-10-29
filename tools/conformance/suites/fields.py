"""Response field coverage — the response surface of every chat format.

The leg-agreement harness (#1239) compares what the two writers *emit*.  This
suite covers what the schema *declares*: every field of a format's response
type must be recorded in the ledger, either as part of the turn's comparable
content or with a reason it is not compared.

A field the SDK declares and the ledger does not have fails the case, so a new
upstream field is a reviewed change rather than an invisible one.  A field the
ledger has and the SDK no longer declares is reported, not failed: an upstream
deprecation does not mean our behaviour changed.

Response slice of #1223's field-coverage property; tracked as P0b of #1238.
Run it with ``python -m tools.conformance run --suite fields``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import Probe, ProbeFailure, ProbeSkip

_BASELINE = Path(__file__).resolve().parents[1] / "baselines/response-fields.json"

#: Classifications a ledger entry can carry.  ``content`` is the comparable
#: turn content; everything else needs a reason unless it is envelope/terminal/
#: usage.
_CLASSIFICATIONS = frozenset(
    {
        "content",
        "envelope",
        "terminal",
        "usage",
        "provider_only",
        "echoed_request",
        "not_on_wire",
        "not_compared",
    }
)
_REASON_REQUIRED = frozenset(
    {"provider_only", "echoed_request", "not_on_wire", "not_compared"}
)

#: Leaf names that carry a provider's own metadata, not turn content.
_PROVIDER_ONLY = frozenset(
    {
        "annotations",
        "audio",
        "citation_metadata",
        "citations",
        "container",
        "content_logprobs",
        "grounding_metadata",
        "logprobs",
        "logprobs_result",
        "media_resolution",
        "metadata",
        "moderation",
        "part_metadata",
        "safety_ratings",
        "service_tier",
        "signature",
        "system_fingerprint",
        "thought_signature",
        "url_context_metadata",
        "video_metadata",
    }
)
_ENVELOPE = frozenset(
    {
        "id",
        "object",
        "type",
        "model",
        "created",
        "created_at",
        "completed_at",
        "updated",
        "responseId",
        "modelVersion",
        "index",
        "sequence_number",
        "output_index",
        "content_index",
        "block_index",
        "item_id",
        "event_type",
        "role",
        "call_id",
    }
)
_TERMINAL = frozenset(
    {
        "finish_reason",
        "stop_reason",
        "stop_sequence",
        "stop_details",
        "incomplete_details",
        "error",
        "status",
        "finishReason",
        "block_reason",
    }
)

#: Fields that are not part of the translated turn and are not walked.
_RESPONSES_ECHOED = frozenset(
    {
        "instructions",
        "tools",
        "tool_choice",
        "text",
        "reasoning",
        "temperature",
        "top_p",
        "max_output_tokens",
        "parallel_tool_calls",
        "background",
        "conversation",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "truncation",
        "user",
        "safety_identifier",
        "max_tool_calls",
        "top_logprobs",
    }
)
_GOOGLE_SDK_ONLY = frozenset(
    {"sdk_http_response", "parsed", "automatic_function_calling_history"}
)


@dataclass(frozen=True)
class FormatSchema:
    """One format's response root and the union variants worth walking."""

    name: str
    root: Any
    expand: dict[str, frozenset[str]]
    skip: dict[str, str]
    source: str = "sdk"


def _hints(tp: Any) -> dict[str, Any]:
    """Return a model's field annotations, for pydantic and TypedDict types."""
    try:
        pydantic = importlib.import_module("pydantic")
    except ModuleNotFoundError:  # pragma: no cover - pydantic is a test extra
        pydantic = None
    if (
        pydantic is not None
        and isinstance(tp, type)
        and issubclass(tp, pydantic.BaseModel)
    ):
        return {name: field.annotation for name, field in tp.model_fields.items()}
    return typing.get_type_hints(tp)


def _members(ann: Any) -> list[type]:
    """Return the SDK classes reachable through Union/List/Annotated."""
    found: list[type] = []
    seen: set[type] = set()

    def walk(node: Any) -> None:
        if typing.get_origin(node) is typing.Annotated:
            walk(typing.get_args(node)[0])
            return
        if typing.get_origin(node) is not None:
            for arg in typing.get_args(node):
                walk(arg)
            return
        if isinstance(node, type) and node not in seen:
            seen.add(node)
            found.append(node)

    walk(ann)
    return [
        cls
        for cls in found
        if cls.__module__.startswith(
            ("openai.types", "anthropic.types", "google.genai")
        )
    ]


def _is_list(ann: Any) -> bool:
    """Whether a field annotation wraps its members in a list/iterable."""
    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        return _is_list(typing.get_args(ann)[0])
    if origin is typing.Union:
        return any(_is_list(arg) for arg in typing.get_args(ann))
    if origin is None:
        return False
    return origin in (list, typing.Iterable, typing.Sequence)


def _walk(
    tp: Any,
    *,
    expand: dict[str, frozenset[str]],
    skip: dict[str, str],
    prefix: str = "",
    depth: int = 0,
    seen: set[type] | None = None,
    fields: dict[str, str] | None = None,
    variants: dict[str, list[str]] | None = None,
    max_depth: int = 4,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Walk a response type into ``{path: leaf_name}`` and ``{path: variants}``."""
    if fields is None:
        fields = {}
    if variants is None:
        variants = {}
    if seen is None:
        seen = set()
    if depth > max_depth:
        return fields, variants

    for name, ann in _hints(tp).items():
        path = f"{prefix}.{name}" if prefix else name
        fields[path] = name
        if path in skip:
            continue
        classes = _members(ann)
        child_prefix = path + "[]" if _is_list(ann) else path
        if len(classes) > 1:
            variants[child_prefix] = sorted(cls.__name__ for cls in classes)
            for cls in classes:
                if (
                    cls.__name__ in expand.get(child_prefix, frozenset())
                    and cls not in seen
                ):
                    seen.add(cls)
                    _walk(
                        cls,
                        expand=expand,
                        skip=skip,
                        prefix=child_prefix,
                        depth=depth + 1,
                        seen=seen,
                        fields=fields,
                        variants=variants,
                        max_depth=max_depth,
                    )
            continue
        for cls in classes:
            if cls in seen:
                continue
            seen.add(cls)
            _walk(
                cls,
                expand=expand,
                skip=skip,
                prefix=child_prefix,
                depth=depth + 1,
                seen=seen,
                fields=fields,
                variants=variants,
                max_depth=max_depth,
            )
    return fields, variants


def _classify(path: str, leaf: str, skip: dict[str, str]) -> tuple[str, str]:
    """Return ``(classification, reason)`` for one field path."""
    if path in skip:
        return skip[path], _skip_reason(leaf, skip[path])
    if leaf in _ENVELOPE:
        return "envelope", ""
    if leaf in _TERMINAL:
        return "terminal", ""
    if any(part in ("logprobs", "citations") for part in path.split(".")):
        return "provider_only", _provider_only_reason(leaf)
    if "usage" in path.lower() or leaf.lower().endswith(("token", "tokens")):
        return "usage", ""
    if leaf in _PROVIDER_ONLY:
        return "provider_only", _provider_only_reason(leaf)
    return "content", ""


def _skip_reason(leaf: str, classification: str) -> str:
    if classification == "echoed_request":
        return (
            "the Responses API echoes the request configuration; it is not a "
            "translated output field"
        )
    if classification == "not_on_wire":
        return "SDK-only helper; never appears on the wire"
    return f"{leaf} is not part of the translated turn"


def _provider_only_reason(leaf: str) -> str:
    reasons = {
        "logprobs": "per-token logprobs have no streamed counterpart",
        "annotations": "whole-block annotations are body-only",
        "citations": "whole-block citations are body-only; the stream carries deltas",
        "service_tier": "response-level service tier has no streamed counterpart",
        "system_fingerprint": "response-level system fingerprint has no streamed counterpart",
        "container": "Anthropic container metadata is provider-specific",
        "signature": "opaque provider signature; not comparable across legs",
        "thought_signature": "opaque Google signature; not comparable across legs",
        "safety_ratings": "provider safety metadata; not part of the turn content",
        "grounding_metadata": "provider grounding metadata; not part of the turn content",
    }
    return reasons.get(leaf, "provider-specific metadata; not part of the turn content")


def _interactions_inventory() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Hand-authored response surface for the Interactions API (no SDK type)."""
    fields = {
        "id": "id",
        "object": "object",
        "model": "model",
        "status": "status",
        "created": "created",
        "updated": "updated",
        "steps[]": "steps",
        "steps[].type": "type",
        "steps[].id": "id",
        "steps[].name": "name",
        "steps[].arguments": "arguments",
        "steps[].call_id": "call_id",
        "steps[].signature": "signature",
        "steps[].result": "result",
        "steps[].is_error": "is_error",
        "steps[].content[]": "content",
        "steps[].content[].type": "type",
        "steps[].content[].text": "text",
        "steps[].summary[]": "summary",
        "steps[].summary[].type": "type",
        "steps[].summary[].text": "text",
        "usage": "usage",
        "usage.total_input_tokens": "total_input_tokens",
        "usage.total_output_tokens": "total_output_tokens",
        "usage.total_tokens": "total_tokens",
    }
    return fields, {}


def _schema_definitions() -> dict[str, FormatSchema]:
    """Load the SDK response roots and the union variants to walk."""
    from anthropic.types import Message
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion
    from openai.types.responses import Response

    return {
        "chat_completions": FormatSchema(
            name="chat_completions",
            root=ChatCompletion,
            expand={
                "choices[].message.tool_calls[]": frozenset(
                    {
                        "ChatCompletionMessageFunctionToolCall",
                        "ChatCompletionMessageCustomToolCall",
                    }
                )
            },
            skip={
                "moderation": "provider_only",
                "choices[].message.annotations": "provider_only",
            },
        ),
        "open_responses": FormatSchema(
            name="open_responses",
            root=Response,
            expand={
                "output[]": frozenset(
                    {
                        "ResponseOutputMessage",
                        "ResponseFunctionToolCall",
                        "ResponseCustomToolCall",
                        "ResponseReasoningItem",
                        "ResponseFunctionWebSearch",
                    }
                ),
                "output[].content[]": frozenset(
                    {"ResponseOutputText", "ResponseOutputRefusal", "Content"}
                ),
                "output[].summary[]": frozenset({"Summary"}),
                "output[].action": frozenset(
                    {"ActionSearch", "ActionOpenPage", "ActionFind"}
                ),
            },
            skip={
                **{name: "echoed_request" for name in _RESPONSES_ECHOED},
                "moderation": "provider_only",
                "service_tier": "provider_only",
                "metadata": "provider_only",
                "output[].content[].annotations": "provider_only",
                "output[].caller": "envelope",
            },
        ),
        "messages": FormatSchema(
            name="messages",
            root=Message,
            expand={
                "content[]": frozenset(
                    {
                        "TextBlock",
                        "ThinkingBlock",
                        "RedactedThinkingBlock",
                        "ToolUseBlock",
                        "ServerToolUseBlock",
                    }
                )
            },
            skip={
                "container": "provider_only",
                "content[].citations": "provider_only",
                "content[].caller": "envelope",
            },
        ),
        "google_generate": FormatSchema(
            name="google_generate",
            root=GenerateContentResponse,
            expand={},
            skip={
                **{name: "not_on_wire" for name in _GOOGLE_SDK_ONLY},
                "candidates[].citation_metadata": "provider_only",
                "candidates[].grounding_metadata": "provider_only",
                "candidates[].safety_ratings": "provider_only",
                "candidates[].logprobs_result": "provider_only",
                "candidates[].url_context_metadata": "provider_only",
                "candidates[].content.parts[].media_resolution": "provider_only",
                "candidates[].content.parts[].video_metadata": "provider_only",
                "candidates[].content.parts[].part_metadata": "provider_only",
                "prompt_feedback.safety_ratings": "provider_only",
                "usage_metadata.cache_tokens_details": "usage",
                "usage_metadata.candidates_tokens_details": "usage",
                "usage_metadata.prompt_tokens_details": "usage",
                "usage_metadata.tool_use_prompt_tokens_details": "usage",
            },
        ),
    }


def inventory() -> dict[str, dict[str, Any]]:
    """Build the current response inventory for every format."""
    formats: dict[str, dict[str, Any]] = {}
    for name, schema in _schema_definitions().items():
        fields, variants = _walk(schema.root, expand=schema.expand, skip=schema.skip)
        formats[name] = {
            "source": schema.source,
            "fields": fields,
            "variants": variants,
            "skip": schema.skip,
        }
    fields, variants = _interactions_inventory()
    formats["google_interactions"] = {
        "source": "no_sdk",
        "fields": fields,
        "variants": variants,
        "skip": {},
    }
    return formats


def _expanded_variants(format_name: str, path: str) -> frozenset[str]:
    """Return the variant names the walker expands for a format/path."""
    schema = _schema_definitions().get(format_name)
    if schema is None:
        return frozenset()
    return schema.expand.get(path, frozenset())


def _sdk_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("openai", "anthropic", "google-genai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def read_baseline() -> dict[str, Any]:
    """Read the committed ledger, or an empty one when it is absent."""
    if not _BASELINE.exists():
        return {}
    import json

    return json.loads(_BASELINE.read_text())


class _FieldsProbe:
    """One format's schema fields against the ledger."""

    def __init__(
        self,
        name: str,
        current: dict[str, Any],
        baseline: dict[str, Any],
        sdk_versions: dict[str, str],
    ) -> None:
        self.name = name
        self._current = current
        self._baseline = baseline
        self._sdk_versions = sdk_versions

    def run(self) -> dict[str, Any]:
        if not self._baseline:
            raise ProbeSkip("no ledger committed yet")
        recorded = self._baseline.get("formats", {}).get(self.name)
        if recorded is None:
            raise ProbeSkip("format not in the ledger; regenerate it")
        if (
            self._current["source"] == "sdk"
            and self._baseline.get("sdk_versions") != self._sdk_versions
        ):
            raise ProbeSkip(
                "SDK versions changed since the ledger; refresh it with "
                "`python -c 'from tools.conformance.suites.fields import "
                "write_baseline; write_baseline(seed=True)'`"
            )

        current_fields = set(self._current["fields"])
        recorded_fields = set(recorded.get("fields", {}))
        added = sorted(current_fields - recorded_fields)
        removed = sorted(recorded_fields - current_fields)
        if added:
            raise ProbeFailure(f"new fields not in the ledger: {added}")

        for path, entry in recorded.get("fields", {}).items():
            classification = entry.get("class")
            if classification not in _CLASSIFICATIONS:
                raise ProbeFailure(f"{path}: unclassified or unknown class")
            if classification in _REASON_REQUIRED and not entry.get("reason"):
                raise ProbeFailure(f"{path}: {classification} needs a reason")

        current_variants = self._current["variants"]
        recorded_variants = recorded.get("variants", {})
        added_variants = sorted(set(current_variants) - set(recorded_variants))
        if added_variants:
            raise ProbeFailure(
                f"new union variants not in the ledger: {added_variants}"
            )
        for path, names in current_variants.items():
            known = recorded_variants.get(path, {})
            new_names = sorted(set(names) - set(known))
            if new_names:
                raise ProbeFailure(f"{path}: new variants {new_names}")
            for name in names:
                entry = known[name]
                classification = entry.get("class")
                if classification not in _CLASSIFICATIONS:
                    raise ProbeFailure(f"{path} {name}: unclassified variant")
                if classification in _REASON_REQUIRED and not entry.get("reason"):
                    raise ProbeFailure(
                        f"{path} {name}: {classification} needs a reason"
                    )

        return {
            "fields": len(current_fields),
            "variants": len(current_variants),
            "added": added,
            "removed": removed,
        }


class _BrokenProbe:
    name = "inventory"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def run(self) -> None:
        raise self._error


def build() -> tuple[list[Probe], dict[str, Any]]:
    try:
        current = inventory()
    except Exception as exc:  # noqa: BLE001 - reported as an error case
        return [_BrokenProbe(exc)], {"sdk_versions": {}}
    baseline = read_baseline()
    versions = _sdk_versions()
    probes: list[Probe] = [
        _FieldsProbe(name, data, baseline, versions) for name, data in current.items()
    ]
    return probes, {"sdk_versions": versions}


def write_baseline(*, seed: bool = False) -> Path:
    """Regenerate the ledger.

    Existing classifications are preserved.  New fields are added as
    ``unclassified`` so the suite fails until a human classifies them; pass
    ``seed=True`` to fill them with the heuristic classification instead (the
    first generation only).
    """
    import json

    current = inventory()
    old = read_baseline()
    old_formats = old.get("formats", {})
    formats: dict[str, Any] = {}
    for name, data in current.items():
        old_fields = old_formats.get(name, {}).get("fields", {})
        fields: dict[str, Any] = {}
        for path, leaf in data["fields"].items():
            if path in old_fields:
                fields[path] = old_fields[path]
                continue
            classification, reason = _classify(path, leaf, data["skip"])
            fields[path] = {
                "class": classification if seed else "unclassified",
                "reason": reason if seed else "",
            }
        old_variants = old_formats.get(name, {}).get("variants", {})
        variants: dict[str, dict[str, dict[str, str]]] = {}
        for path, names in data["variants"].items():
            known = old_variants.get(path, {})
            entries: dict[str, dict[str, str]] = {}
            for variant in names:
                if variant in known:
                    entries[variant] = known[variant]
                elif variant in _expanded_variants(name, path):
                    entries[variant] = {"class": "content", "reason": ""}
                else:
                    entries[variant] = {
                        "class": "not_compared",
                        "reason": (
                            "variant not exercised by the leg-agreement harness; "
                            "its IR mapping is not asserted"
                        ),
                    }
            variants[path] = entries
        formats[name] = {"fields": fields, "variants": variants}

    payload = {
        "schema_version": 1,
        "suite": "fields",
        "sdk_versions": _sdk_versions(),
        "formats": formats,
    }
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _BASELINE

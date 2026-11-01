"""Request field coverage — what the read leg does with every request field.

The response counterpart (:mod:`tools.conformance.suites.fields`) records what
each format's response type declares.  This suite does the same for requests:
every field a format declares is recorded with a class saying what the
conversion does with it.

The classes, each of which describes the *read* leg (``request_from_provider``
— source format to IR):

- ``mapped`` — the reader stores the field in the IR, or in
  ``provider_extensions`` when the IR has no home for it but the field must
  survive a same-format round trip.
- ``dropped`` — the reader ignores the field.  The reason names the missing
  concept; several are open gaps the ledger exists to expose.
- ``not_on_wire`` — an SDK-only helper (``http_options``, ``timeout``) that
  never reaches the wire and so is not a request field at all.

A field the format declares and the ledger does not record fails the case, so
a new upstream request field is a reviewed change rather than an invisible
one.  A ledger entry the format no longer declares is reported, not failed: an
upstream deprecation does not mean our behaviour changed.

The ledger is the reviewed statement.  This module can walk a format's request
type, but whether a field is mapped or dropped is not derivable from the type
alone — it depends on what each reader reads, directly or through a list of
names.  So new paths are written ``unclassified`` and the suite fails until a
human reads the corresponding ``_do_request_from_provider`` and records the
answer.  ``tools/conformance`` is developer-only, so this costs a maintainer a
reading, once per upstream field.

Request slice of #1223's field-coverage property.  Run it with
``python -m tools.conformance run --suite request-fields``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from ..model import Probe, ProbeFailure, ProbeSkip
from .fields import FormatSchema, _walk

_BASELINE = Path(__file__).resolve().parents[1] / "baselines/request-fields.json"

#: Classes a request field can carry.
_CLASSIFICATIONS = frozenset({"mapped", "dropped", "not_on_wire"})
#: Classes that must say why.
_REASON_REQUIRED = frozenset({"dropped"})

#: SDK helpers that never reach the wire.
_NOT_ON_WIRE = frozenset(
    {
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
        "http_options",
        "automatic_function_calling",
        "should_return_http_response",
    }
)


def _classify(path: str, leaf: str) -> tuple[str, str]:
    """Return the class the walker can decide on its own, else ``unclassified``.

    Only one class is structural: a field in :data:`_NOT_ON_WIRE` cannot be a
    wire field.  Everything else needs a reader inspection, which the ledger
    records.

    Args:
        path: The dotted field path.
        leaf: The field's own name.

    Returns:
        ``(classification, reason)``.
    """
    if leaf in _NOT_ON_WIRE:
        return "not_on_wire", "SDK-only helper; never sent on the wire"
    return "unclassified", ""


#: Subtrees recorded but not walked, with the class of the recorded path.
#:
#: A message list is one field whose *contents* are the part surface the
#: cross-leg harness and the per-part tests own, so the ledger records the list
#: and stops.  A provider-defined config object (``moderation``,
#: ``image_config``) is the same shape: one request field whose value is a
#: provider's own vocabulary.  The class says what the reader does with the
#: container, not with its keys.
_SKIP: dict[str, dict[str, str]] = {
    "chat_completions": {
        "messages": "mapped",
        "functions": "dropped",
        "audio": "dropped",
        "moderation": "dropped",
        "prediction": "dropped",
        "prompt_cache_options": "dropped",
        "web_search_options": "dropped",
    },
    "open_responses": {
        "input": "mapped",
        # Stored whole in ``provider_extensions``; the keys are the vendor's.
        "conversation": "mapped",
        "prompt": "mapped",
        "context_management": "mapped",
        "moderation": "mapped",
        "prompt_cache_options": "mapped",
        "access_programs": "mapped",
    },
    "messages": {
        "messages": "mapped",
        # Stored whole as ``cache_hint`` on a part; the keys are Anthropic's.
        "system.cache_control": "mapped",
        "output_config.format": "dropped",
        "metadata": "dropped",
        "cache_control": "dropped",
        "container": "dropped",
        "diagnostics": "dropped",
    },
    "google_generate": {
        "contents[].parts": "mapped",
        # The schema's keys are JSON Schema keywords, not request fields; the
        # converter hands the schema through whole.
        "response_schema": "mapped",
        "http_options": "not_on_wire",
        "automatic_function_calling": "not_on_wire",
        "audio_transcription_config": "dropped",
        "generation_output_config": "dropped",
        "image_config": "dropped",
        "speech_config": "dropped",
        "routing_config": "dropped",
        "model_armor_config": "dropped",
        "model_selection_config": "dropped",
        "safety_settings": "dropped",
        "media_resolution": "dropped",
        "tool_config.retrieval_config": "dropped",
    },
}


def _chat_schema() -> FormatSchema:
    """The Chat Completions request root (the non-streaming variant)."""
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
    )

    return FormatSchema(
        name="chat_completions",
        root=CompletionCreateParamsNonStreaming,
        expand={},
        skip=_SKIP["chat_completions"],
    )


def _responses_schema() -> FormatSchema:
    """The Responses request root (the non-streaming variant)."""
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )

    return FormatSchema(
        name="open_responses",
        root=ResponseCreateParamsNonStreaming,
        expand={},
        skip=_SKIP["open_responses"],
    )


def _messages_schema() -> FormatSchema:
    """The Messages request root (the non-streaming variant)."""
    from anthropic.types.message_create_params import (
        MessageCreateParamsNonStreaming,
    )

    return FormatSchema(
        name="messages",
        root=MessageCreateParamsNonStreaming,
        expand={},
        skip=_SKIP["messages"],
    )


def _google_schema() -> FormatSchema:
    """The generateContent request root, plus the ``contents`` the SDK splits out."""
    from google.genai.types import Content, GenerateContentConfig

    # ``contents`` is a call argument, not a config field, so the walker is run
    # twice: the config at the root, and Content under ``contents[]``.
    return FormatSchema(
        name="google_generate",
        root=GenerateContentConfig,
        expand={},
        skip=_SKIP["google_generate"],
        extra_roots=(("contents[]", Content),),
    )


#: The Interactions request surface, hand-authored: the API ships no Python
#: type.  Every entry is a ``(path, leaf)`` pair, so the walker's contract —
#: one leaf name per path — holds for this format too.
_INTERACTIONS_FIELDS: dict[str, str] = {
    "agent": "agent",
    "agent_config": "agent_config",
    "background": "background",
    "cached_content": "cached_content",
    "environment": "environment",
    "environment_id": "environment_id",
    "generation_config": "generation_config",
    "generation_config.image_config": "image_config",
    "generation_config.max_output_tokens": "max_output_tokens",
    "generation_config.seed": "seed",
    "generation_config.stop_sequences": "stop_sequences",
    "generation_config.temperature": "temperature",
    "generation_config.thinking_level": "thinking_level",
    "generation_config.thinking_summaries": "thinking_summaries",
    "generation_config.tool_choice": "tool_choice",
    "generation_config.top_p": "top_p",
    "generation_config.transcription_config": "transcription_config",
    "generation_config.video_config": "video_config",
    "input": "input",
    "labels": "labels",
    "model": "model",
    "previous_interaction_id": "previous_interaction_id",
    "response_format": "response_format",
    "response_mime_type": "response_mime_type",
    "response_modalities": "response_modalities",
    "safety_settings": "safety_settings",
    "service_tier": "service_tier",
    "store": "store",
    "stream": "stream",
    "system_instruction": "system_instruction",
    "tools": "tools",
    "webhook_config": "webhook_config",
}


def _load_schemas() -> tuple[dict[str, FormatSchema], dict[str, str]]:
    """Load the request roots; a missing SDK is a skip, not an error."""
    builders = {
        "chat_completions": _chat_schema,
        "open_responses": _responses_schema,
        "messages": _messages_schema,
        "google_generate": _google_schema,
    }
    schemas: dict[str, FormatSchema] = {}
    missing: dict[str, str] = {}
    for name, builder in builders.items():
        try:
            schemas[name] = builder()
        except ModuleNotFoundError as exc:
            missing[name] = exc.name or "the provider SDK"
    return schemas, missing


def inventory_and_missing() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build the request inventory and name the SDKs that are absent."""
    schemas, missing = _load_schemas()
    formats: dict[str, dict[str, Any]] = {}
    for name, schema in schemas.items():
        fields, _variants = _walk(schema.root, expand=schema.expand, skip=schema.skip)
        for prefix, root in schema.extra_roots:
            extra_fields, _extra_variants = _walk(
                root, expand=schema.expand, skip=schema.skip, prefix=prefix
            )
            fields.update(extra_fields)
        formats[name] = {"fields": fields}
    formats["google_interactions"] = {
        "fields": dict(_INTERACTIONS_FIELDS),
        "source": "no_sdk",
    }
    return formats, missing


def inventory() -> dict[str, dict[str, Any]]:
    """Build the request inventory for every format whose SDK is installed."""
    formats, _missing = inventory_and_missing()
    return formats


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
    return json.loads(_BASELINE.read_text())


class _RequestFieldsProbe:
    """One format's request schema fields against the ledger."""

    def __init__(
        self,
        name: str,
        current: dict[str, Any],
        baseline: dict[str, Any],
    ) -> None:
        self.name = name
        self._current = current
        self._baseline = baseline

    def run(self) -> dict[str, Any]:
        if not self._baseline:
            raise ProbeFailure(
                "no ledger committed; run write_baseline() and classify the diff"
            )
        recorded = self._baseline.get("formats", {}).get(self.name)
        if recorded is None:
            raise ProbeFailure("format missing from the ledger; regenerate it")

        current_fields = set(self._current["fields"])
        recorded_fields = set(recorded.get("fields", {}))
        added = sorted(current_fields - recorded_fields)
        removed = sorted(recorded_fields - current_fields)
        if added:
            raise ProbeFailure(
                f"new request fields not in the ledger: {added}; regenerate with "
                "write_baseline() and classify the diff"
            )

        for path, entry in recorded.get("fields", {}).items():
            classification = entry.get("class")
            if classification not in _CLASSIFICATIONS:
                raise ProbeFailure(f"{path}: unclassified or unknown class")
            if classification in _REASON_REQUIRED and not entry.get("reason"):
                raise ProbeFailure(f"{path}: {classification} needs a reason")

        return {
            "fields": len(current_fields),
            "added": added,
            "removed": removed,
        }


class _MissingSdkProbe:
    """A format whose provider SDK is not installed here."""

    def __init__(self, name: str, package: str) -> None:
        self.name = name
        self._package = package

    def run(self) -> None:
        raise ProbeSkip(f"SDK not installed: {self._package}")


class _BrokenProbe:
    name = "inventory"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def run(self) -> None:
        raise self._error


def build() -> tuple[list[Probe], dict[str, Any]]:
    try:
        current, missing = inventory_and_missing()
    except Exception as exc:  # noqa: BLE001 - reported as an error case
        return [_BrokenProbe(exc)], {"sdk_versions": {}}
    baseline = read_baseline()
    probes: list[Probe] = [
        _RequestFieldsProbe(name, data, baseline) for name, data in current.items()
    ]
    probes.extend(
        _MissingSdkProbe(name, package) for name, package in sorted(missing.items())
    )
    return probes, {"sdk_versions": _sdk_versions(), "missing": missing}


def write_baseline() -> Path:
    """Regenerate the ledger.

    Existing classifications are preserved.  A new path is written
    ``unclassified`` (unless the walker can decide it on its own) so the suite
    fails until a human reads the reader and records the answer.
    """
    current, missing = inventory_and_missing()
    old = read_baseline()
    old_formats = old.get("formats", {})
    # A format whose SDK is absent keeps its committed ledger.
    formats: dict[str, Any] = {
        name: data for name, data in old_formats.items() if name in missing
    }
    for name, data in current.items():
        old_fields = old_formats.get(name, {}).get("fields", {})
        fields: dict[str, Any] = {}
        for path, leaf in data["fields"].items():
            if path in old_fields:
                fields[path] = old_fields[path]
                continue
            classification, reason = _classify(path, leaf)
            fields[path] = {"class": classification, "reason": reason}
        formats[name] = {"fields": fields}

    payload = {
        "schema_version": 1,
        "suite": "request-fields",
        "sdk_versions": _sdk_versions(),
        "formats": formats,
    }
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _BASELINE

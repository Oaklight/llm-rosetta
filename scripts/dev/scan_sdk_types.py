#!/usr/bin/env python3
"""Scan upstream LLM SDK types against our TypedDict replicas.

Compares field-level coverage of types we replicate and flags potentially
relevant new types from upstream SDKs.  A declared module that cannot be
imported is an error, not a skip: a renamed replica module used to make the
whole provider disappear from the report while the run exited 0.

Types whose name means something different upstream are listed under
``name_collisions`` and excluded from field drift.  Those entries are
reviewed by hand; the report names them so a new collision is visible.

Usage:
    python scripts/dev/scan_sdk_types.py             # Full Markdown report
    python scripts/dev/scan_sdk_types.py --json      # Machine-readable report
    python scripts/dev/scan_sdk_types.py messages    # Single provider
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any, get_type_hints

# Date-stamped type names (e.g., CodeExecutionTool20250522Param) are versioned
# SDK internals — not worth flagging individually.
_DATE_STAMPED_RE = re.compile(r"\d{8}")


class ScanError(RuntimeError):
    """A declared module could not be imported, or a provider is unknown."""


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "messages": {
        "our_modules": [
            "llm_rosetta.types.messages.request_types",
            "llm_rosetta.types.messages.response_types",
        ],
        "sdk_modules": [
            "anthropic.types",
            "anthropic.types.message_create_params",
        ],
        "relevance_patterns": [
            "Block",
            "Param",
            "Message",
            "Content",
            "Tool",
            "Usage",
            "Citation",
            "Metadata",
            "Config",
        ],
        "exclude_patterns": [
            "Beta",
            "Deprecated",
            "Raw",
            "Parsed",
            "Event",
            "Delta",
            "ResultError",
            "Encrypted",
        ],
        "name_collisions": [],
        "sdk_package": "anthropic",
    },
    "openai_chat": {
        "our_modules": [
            "llm_rosetta.types.openai.chat",
        ],
        "sdk_modules": [
            "openai.types.chat",
            "openai.types.chat.completion_create_params",
        ],
        "relevance_patterns": [
            "ChatCompletion",
            "Completion",
            "Message",
            "Tool",
            "Function",
            "Choice",
            "Usage",
            "Content",
            "Param",
            "Format",
            "Logprob",
        ],
        "exclude_patterns": ["Deprecated", "Parsed", "Chunk"],
        # Ours is the function *call* (arguments, name); the SDK's is the
        # function *definition* (description, name, parameters).
        "name_collisions": ["Function"],
        "sdk_package": "openai",
    },
    "openai_responses": {
        "our_modules": [
            "llm_rosetta.types.openai.responses.request_types",
            "llm_rosetta.types.openai.responses.response_types",
        ],
        "sdk_modules": [
            "openai.types.responses",
        ],
        "relevance_patterns": [
            "Response",
            "Tool",
            "Message",
            "Output",
            "Input",
            "Item",
            "Call",
            "Usage",
            "Search",
            "Reasoning",
            "Compaction",
        ],
        "exclude_patterns": ["Deprecated", "Parsed", "Event"],
        # Ours is the tool wrapper ({type, function}); the SDK's is the inner
        # definition (name, description, parameters, strict).
        "name_collisions": ["FunctionToolParam"],
        "sdk_package": "openai",
    },
    "google_genai": {
        "our_modules": [
            "llm_rosetta.types.google.content_types",
            "llm_rosetta.types.google.request_types",
            "llm_rosetta.types.google.response_types",
        ],
        "sdk_modules": [
            "google.genai.types",
        ],
        "relevance_patterns": [
            "Content",
            "Part",
            "Candidate",
            "GenerateContent",
            "Tool",
            "Function",
            "Safety",
            "Citation",
            "Schema",
            "Blob",
            "Thinking",
            "Grounding",
            "Usage",
        ],
        "exclude_patterns": [
            "Deprecated",
            "Dict",
            "Embed",
            "Tuning",
            "Batch",
            "Cached",
            "Corpus",
            "Chunk",
            "Document",
            "File",
            "Live",
            "Model",
            "Permission",
            "Token",
            "Upload",
            "Count",
            "Edit",
            "Image",
            "Predict",
            "Retrieval",
            "Semantic",
            "Video",
            "Vqa",
        ],
        # Ours is the part nested inside a function response; the SDK's is a
        # part in the function response's own parts list (files, inline data).
        "name_collisions": ["FunctionResponsePart"],
        "sdk_package": "google-genai",
    },
}

#: Accepted aliases for CLI selection, keyed by their canonical provider key.
ALIASES: dict[str, str] = {"anthropic": "messages"}


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def get_sdk_version(package_name: str) -> str:
    try:
        meta = importlib.metadata.metadata(package_name)
        return meta["Version"]
    except Exception:
        return "unknown"


def is_typeddict(obj: object) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, dict)
        and hasattr(obj, "__annotations__")
    )


def is_pydantic_model(obj: object) -> bool:
    return isinstance(obj, type) and hasattr(obj, "model_fields")


def get_fields(obj: object) -> set[str] | None:
    """Extract field names from a type (TypedDict or Pydantic model)."""
    if is_pydantic_model(obj):
        fields = getattr(obj, "model_fields", None)
        return set(fields.keys()) if fields else None
    if is_typeddict(obj):
        try:
            hints = get_type_hints(obj)
            return set(hints.keys())
        except Exception:
            annotations = getattr(obj, "__annotations__", None)
            return set(annotations.keys()) if annotations else None
    return None


def _import(module_name: str, *, what: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ScanError(
            f"{what} {module_name!r} cannot be imported: {exc}. "
            "A renamed or deleted module must be fixed in PROVIDERS, not skipped."
        ) from exc


def collect_our_types(module_names: list[str]) -> dict[str, object]:
    """Collect all TypedDict/type-alias names from our modules."""
    types: dict[str, object] = {}
    for mod_name in module_names:
        mod = _import(mod_name, what="replica module")

        all_names = getattr(mod, "__all__", None)
        if all_names is None:
            all_names = [n for n in dir(mod) if not n.startswith("_")]

        for name in all_names:
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            if isinstance(obj, type) or hasattr(obj, "__origin__"):
                types[name] = obj
    return types


def collect_sdk_types(module_names: list[str]) -> dict[str, object]:
    """Collect all public types from SDK modules."""
    types: dict[str, object] = {}
    for mod_name in module_names:
        mod = _import(mod_name, what="SDK module")

        for name in dir(mod):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            if is_pydantic_model(obj) or is_typeddict(obj):
                types[name] = obj
    return types


def is_relevant(name: str, patterns: list[str], excludes: list[str]) -> bool:
    if _DATE_STAMPED_RE.search(name):
        return False
    for excl in excludes:
        if excl.lower() in name.lower():
            return False
    for pat in patterns:
        if pat.lower() in name.lower():
            return True
    return False


def split_collisions(
    drift: dict[str, list[str]], collisions: Iterable[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Return the drift with reviewed name collisions removed.

    The collisions are returned as a sorted list so the report can name what
    it excluded — an entry that stops colliding is then visible as a change.
    """
    excluded = sorted(set(collisions) & set(drift))
    kept = {name: fields for name, fields in drift.items() if name not in excluded}
    return kept, excluded


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def scan_provider(provider_key: str, config: dict[str, Any]) -> dict[str, Any]:
    """Scan one provider and return its machine-readable result."""
    our_types = collect_our_types(config["our_modules"])
    sdk_types = collect_sdk_types(config["sdk_modules"])

    our_type_classes = {
        name: obj for name, obj in our_types.items() if isinstance(obj, type)
    }
    our_names = set(our_types.keys())

    drift: dict[str, list[str]] = {}
    for name, our_obj in sorted(our_type_classes.items()):
        sdk_obj = sdk_types.get(name)
        if sdk_obj is None:
            continue

        our_fields = get_fields(our_obj)
        sdk_fields = get_fields(sdk_obj)
        if our_fields is None or sdk_fields is None:
            continue

        missing = sdk_fields - our_fields
        if missing:
            drift[name] = sorted(missing)

    drift, excluded = split_collisions(drift, config.get("name_collisions", []))

    sdk_only = set(sdk_types.keys()) - our_names
    relevant_new = sorted(
        name
        for name in sdk_only
        if is_relevant(name, config["relevance_patterns"], config["exclude_patterns"])
    )

    return {
        "drift": drift,
        "relevant_new": relevant_new,
        "collisions": excluded,
        "coverage": {
            "matched": sum(1 for name in our_type_classes if name in sdk_types),
            "replicas": len(our_type_classes),
            "sdk_types": len(sdk_types),
        },
    }


def build_report(providers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the full report for the selected providers."""
    versions: dict[str, str] = {}
    for cfg in providers.values():
        package = cfg["sdk_package"]
        versions.setdefault(package, get_sdk_version(package))

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sdk_versions": versions,
        "providers": {key: scan_provider(key, cfg) for key, cfg in providers.items()},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# SDK Type Coverage Report",
        "",
        "Generated: {generated} | {versions}".format(
            generated=report["generated"],
            versions=", ".join(
                f"{pkg}=={version}" for pkg, version in report["sdk_versions"].items()
            ),
        ),
        "",
    ]

    for key, data in report["providers"].items():
        lines.extend([f"## {key}", ""])

        drift = data["drift"]
        lines.append(f"### Field drift in covered types ({len(drift)} issues)")
        lines.append("")
        if drift:
            for name, fields in drift.items():
                lines.append(f"- `{name}`: missing fields `{chr(96).join(fields)}`")
        else:
            lines.append("(none)")
        if data["collisions"]:
            lines.append("")
            lines.append(
                "Excluded as name collisions: "
                + ", ".join(f"`{name}`" for name in data["collisions"])
            )
        lines.append("")

        relevant_new = data["relevant_new"]
        lines.append(f"### Potentially relevant new types ({len(relevant_new)} found)")
        lines.append("")
        if relevant_new:
            lines.extend(f"- `{name}`" for name in relevant_new)
        else:
            lines.append("(none)")
        lines.append("")

        coverage = data["coverage"]
        lines.append(
            f"*Coverage: {coverage['matched']} types matched against SDK, "
            f"{coverage['replicas']} total replicas, "
            f"{coverage['sdk_types']} SDK types scanned.*"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_providers(requested: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve CLI arguments to provider configs, or raise ScanError."""
    if not requested:
        return dict(PROVIDERS)

    selected: dict[str, dict[str, Any]] = {}
    for arg in requested:
        key = arg.lower().replace("-", "_")
        key = ALIASES.get(key, key)
        if key in PROVIDERS:
            selected[key] = PROVIDERS[key]
            continue
        matches = [k for k in PROVIDERS if key in k]
        if not matches:
            raise ScanError(
                f"unknown provider {arg!r}; available: {', '.join(PROVIDERS)}"
            )
        for match in matches:
            selected[match] = PROVIDERS[match]
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    requested = [arg for arg in args if not arg.startswith("-")]

    try:
        providers = select_providers(requested)
        report = build_report(providers)
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

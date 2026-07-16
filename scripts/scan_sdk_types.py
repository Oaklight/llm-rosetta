#!/usr/bin/env python3
"""Scan upstream LLM SDK types against our TypedDict replicas.

Compares field-level coverage of types we replicate and flags potentially
relevant new types from upstream SDKs.

Usage:
    python scripts/scan_sdk_types.py          # Full report
    python scripts/scan_sdk_types.py anthropic # Single provider
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
from datetime import datetime, timezone
from typing import get_type_hints

# Date-stamped type names (e.g., CodeExecutionTool20250522Param) are versioned
# SDK internals — not worth flagging individually.
_DATE_STAMPED_RE = re.compile(r"\d{8}")

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "our_modules": [
            "llm_rosetta.types.anthropic.request_types",
            "llm_rosetta.types.anthropic.response_types",
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
        "sdk_package": "google-genai",
    },
}


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
        return set(obj.model_fields.keys())  # type: ignore[union-attr]
    if is_typeddict(obj):
        try:
            hints = get_type_hints(obj)
            return set(hints.keys())
        except Exception:
            if hasattr(obj, "__annotations__"):
                return set(obj.__annotations__.keys())
    return None


def collect_our_types(module_names: list[str]) -> dict[str, object]:
    """Collect all TypedDict/type-alias names from our modules."""
    types: dict[str, object] = {}
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

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
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

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


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def scan_provider(provider_key: str, config: dict) -> list[str]:
    """Scan a single provider and return report lines."""
    lines: list[str] = []
    lines.append(f"## {provider_key}")
    lines.append("")

    our_types = collect_our_types(config["our_modules"])
    sdk_types = collect_sdk_types(config["sdk_modules"])

    our_type_classes = {
        name: obj for name, obj in our_types.items() if isinstance(obj, type)
    }
    our_names = set(our_types.keys())

    # --- Field drift ---
    drift_lines: list[str] = []
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
            drift_lines.append(
                f"- `{name}`: missing fields `{'`, `'.join(sorted(missing))}`"
            )

    if drift_lines:
        lines.append(f"### Field drift in covered types ({len(drift_lines)} issues)")
        lines.append("")
        lines.extend(drift_lines)
    else:
        lines.append("### Field drift in covered types")
        lines.append("")
        lines.append("(none)")
    lines.append("")

    # --- New potentially relevant types ---
    sdk_only = set(sdk_types.keys()) - our_names
    relevant_new = sorted(
        name
        for name in sdk_only
        if is_relevant(name, config["relevance_patterns"], config["exclude_patterns"])
    )

    if relevant_new:
        lines.append(f"### Potentially relevant new types ({len(relevant_new)} found)")
        lines.append("")
        for name in relevant_new:
            lines.append(f"- `{name}`")
    else:
        lines.append("### Potentially relevant new types")
        lines.append("")
        lines.append("(none)")
    lines.append("")

    # --- Coverage summary ---
    matched = sum(1 for name in our_type_classes if name in sdk_types)
    lines.append(
        f"*Coverage: {matched} types matched against SDK, "
        f"{len(our_type_classes)} total replicas, "
        f"{len(sdk_types)} SDK types scanned.*"
    )
    lines.append("")
    return lines


def main() -> None:
    # Optional: scan only specific providers
    requested = set()
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            key = arg.lower().replace("-", "_")
            if key in PROVIDERS:
                requested.add(key)
            else:
                matches = [k for k in PROVIDERS if key in k]
                if matches:
                    requested.update(matches)
                else:
                    print(f"Unknown provider: {arg}", file=sys.stderr)
                    print(f"Available: {', '.join(PROVIDERS.keys())}", file=sys.stderr)
                    sys.exit(1)

    providers = {k: v for k, v in PROVIDERS.items() if not requested or k in requested}

    # Header
    versions = []
    seen_packages: set[str] = set()
    for cfg in providers.values():
        pkg = cfg["sdk_package"]
        if pkg not in seen_packages:
            seen_packages.add(pkg)
            versions.append(f"{pkg}=={get_sdk_version(pkg)}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("# SDK Type Coverage Report")
    print("")
    print(f"Generated: {now} | {', '.join(versions)}")
    print("")

    # Per-provider reports
    for key, config in providers.items():
        try:
            lines = scan_provider(key, config)
            print("\n".join(lines))
        except ImportError as e:
            print(f"## {key}")
            print("")
            print(f"**Skipped** — SDK not installed: {e}")
            print("")


if __name__ == "__main__":
    main()

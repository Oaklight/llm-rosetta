#!/usr/bin/env python3
"""Unified ARGO API probe tool.

Auto-fetches the model list from ARGO, groups by family, and probes each
endpoint for parameter acceptance.  Results are printed as markdown tables
to stdout; progress goes to stderr.

Usage examples::

    # Probe everything
    python scripts/dev/probe/probe_argo.py

    # Only OAI Chat endpoint, only OpenAI models
    python scripts/dev/probe/probe_argo.py --endpoint chat --family openai

    # Save raw results as JSON
    python scripts/dev/probe/probe_argo.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://apps.inside.anl.gov/argoapi"
MODELS_URL = "https://apps.inside.anl.gov/argoapi/api/v1/models"
OAI_CHAT_URL = f"{BASE_URL}/v1/chat/completions"
ANTHROPIC_URL = f"{BASE_URL}/v1/messages"
RESPONSES_URL = f"{BASE_URL}/v1/responses"

TIMEOUT = 60

# Patterns for models that should be excluded from chat probes.
_EMBEDDING_PATTERNS = re.compile(
    r"embedding|text-embedding|ada|v3small|v3large", re.IGNORECASE
)

# Family classification by owned_by field.
_FAMILY_MAP = {
    "openai": "openai",
    "google": "gemini",
    "anthropic": "claude",
}

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def fetch_models(user: str) -> list[dict[str, Any]]:
    """Fetch the model list from the ARGO API."""
    headers = {"Authorization": f"Bearer {user}", "Content-Type": "application/json"}
    resp = requests.get(MODELS_URL, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def classify_family(owned_by: str) -> str:
    """Map owned_by to a family name."""
    key = owned_by.lower().strip()
    for prefix, family in _FAMILY_MAP.items():
        if prefix in key:
            return family
    return key  # unknown family, use raw value


def is_embedding_model(model: dict[str, Any]) -> bool:
    """Return True if the model is an embedding model."""
    model_id = model.get("id", "")
    owned_by = model.get("owned_by", "")
    return bool(_EMBEDDING_PATTERNS.search(model_id) or "embedding" in owned_by.lower())


def discover_models(user: str) -> dict[str, list[str]]:
    """Fetch models from ARGO and return {family: [model_id, ...]}."""
    raw = fetch_models(user)
    # The response can be a list of dicts or a dict with a "data" key.
    if isinstance(raw, dict) and "data" in raw:
        models = raw["data"]
    elif isinstance(raw, list):
        models = raw
    else:
        print(f"Unexpected model list format: {type(raw)}", file=sys.stderr)
        return {}

    families: dict[str, list[str]] = {}
    for m in models:
        if is_embedding_model(m):
            continue
        model_id = m.get("id", "")
        if not model_id:
            continue
        family = classify_family(m.get("owned_by", "unknown"))
        families.setdefault(family, []).append(model_id)

    # Sort models within each family for stable output.
    for family in families:
        families[family].sort()
    return families


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _headers_for_url(url: str, user: str) -> dict[str, str]:
    """Return the auth headers appropriate for the given endpoint URL."""
    base = {"Content-Type": "application/json"}
    if "/v1/messages" in url:
        base["x-api-key"] = user
    else:
        base["Authorization"] = f"Bearer {user}"
    return base


def post(url: str, body: dict[str, Any], user: str) -> tuple[int, dict[str, Any]]:
    """POST a JSON body and return (status_code, response_json)."""
    headers = _headers_for_url(url, user)
    try:
        r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
        try:
            data: dict[str, Any] = r.json()
        except Exception:
            data = {"_raw": r.text[:300]}
        return r.status_code, data
    except Exception as e:
        return -1, {"error": str(e)[:200]}


def err_msg(data: dict[str, Any]) -> str:
    """Extract a short error message from the response body."""
    if "error" in data:
        e = data["error"]
        if isinstance(e, dict):
            return e.get("message", str(e))[:120]
        return str(e)[:120]
    if "detail" in data:
        return str(data["detail"])[:120]
    return ""


def status_icon(code: int) -> str:
    """Return a short emoji+code label for the HTTP status."""
    if 200 <= code < 300:
        return "pass"
    if code == 400:
        return "FAIL(400)"
    if code == 401:
        return "AUTH(401)"
    if code == 429:
        return "RATE(429)"
    if code >= 500:
        return f"ERR({code})"
    if code == -1:
        return "TIMEOUT"
    return f"?({code})"


def fmt(code: int, data: dict[str, Any], note_len: int = 60) -> str:
    """Format a probe result as icon + optional error snippet."""
    s = status_icon(code)
    if code < 200 or code >= 300:
        msg = err_msg(data)[:note_len]
        if msg:
            s += f" {msg}"
    return s


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------


def oai_body(model: str, **overrides: Any) -> dict[str, Any]:
    """Minimal OpenAI Chat Completions request body."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say OK"}],
    }
    body.update(overrides)
    return body


def anth_body(model: str, **overrides: Any) -> dict[str, Any]:
    """Minimal Anthropic Messages request body."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say OK"}],
    }
    body.update(overrides)
    return body


def responses_body(model: str, **overrides: Any) -> dict[str, Any]:
    """Minimal OpenAI Responses API request body."""
    body: dict[str, Any] = {
        "model": model,
        "input": "Say OK",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

_CLEAN_TOOL_OAI = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "test",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    },
}

_COMPLEX_TOOL_OAI = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "test",
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["x"],
            "additionalProperties": False,
            "propertyNames": {"pattern": "^[a-z]"},
        },
    },
}

_CLEAN_TOOL_ANTH = {
    "name": "f",
    "description": "test",
    "input_schema": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
}

_COMPLEX_TOOL_ANTH = {
    "name": "f",
    "description": "test",
    "input_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["x"],
        "additionalProperties": False,
        "propertyNames": {"pattern": "^[a-z]"},
    },
}

_CLEAN_TOOL_RESPONSES = {
    "type": "function",
    "name": "f",
    "description": "test",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
}


# ---------------------------------------------------------------------------
# Probe functions — OAI Chat Completions
# ---------------------------------------------------------------------------


def probe_oai_chat(models: list[str], user: str) -> dict[str, dict[str, Any]]:
    """Probe all dimensions for the OAI Chat Completions endpoint.

    Returns {model: {dimension: result_string}}.
    """
    results: dict[str, dict[str, Any]] = {}
    total = len(models)

    for i, m in enumerate(models, 1):
        print(f"  [{i}/{total}] {m} ...", file=sys.stderr)
        r: dict[str, Any] = {}

        # developer_role
        c, d = post(
            OAI_CHAT_URL,
            oai_body(
                m,
                messages=[
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": "Say OK"},
                ],
            ),
            user,
        )
        r["developer_role"] = fmt(c, d)

        # null_content
        c, d = post(
            OAI_CHAT_URL,
            oai_body(
                m,
                messages=[
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": None},
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["null_content"] = fmt(c, d)

        # max_tokens variants
        b_mt = oai_body(m)  # has max_tokens=16
        c1, d1 = post(OAI_CHAT_URL, b_mt, user)

        b_mct = oai_body(m)
        del b_mct["max_tokens"]
        b_mct["max_completion_tokens"] = 16
        c2, d2 = post(OAI_CHAT_URL, b_mct, user)

        b_both = oai_body(m, max_completion_tokens=16)
        c3, d3 = post(OAI_CHAT_URL, b_both, user)

        r["max_tokens"] = fmt(c1, d1)
        r["max_completion_tokens"] = fmt(c2, d2)
        r["max_tokens_both"] = fmt(c3, d3)

        # orphaned_tools
        c, d = post(
            OAI_CHAT_URL,
            oai_body(
                m,
                messages=[
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["orphaned_call"] = fmt(c, d)

        c, d = post(
            OAI_CHAT_URL,
            oai_body(
                m,
                messages=[
                    {"role": "user", "content": "Hi"},
                    {"role": "tool", "tool_call_id": "call_999", "content": "result"},
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["orphaned_result"] = fmt(c, d)

        # tool_schema
        c1, d1 = post(OAI_CHAT_URL, oai_body(m, tools=[_CLEAN_TOOL_OAI]), user)
        c2, d2 = post(OAI_CHAT_URL, oai_body(m, tools=[_COMPLEX_TOOL_OAI]), user)
        r["clean_schema"] = fmt(c1, d1)
        r["complex_schema"] = fmt(c2, d2)

        # sampling
        c1, d1 = post(OAI_CHAT_URL, oai_body(m, temperature=0.7), user)
        c2, d2 = post(OAI_CHAT_URL, oai_body(m, top_p=0.9), user)
        c3, d3 = post(OAI_CHAT_URL, oai_body(m, temperature=0.7, top_p=0.9), user)
        r["temperature"] = fmt(c1, d1)
        r["top_p"] = fmt(c2, d2)
        r["temp_and_top_p"] = fmt(c3, d3)

        results[m] = r
        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Probe functions — Anthropic Messages
# ---------------------------------------------------------------------------


def probe_anthropic(models: list[str], user: str) -> dict[str, dict[str, Any]]:
    """Probe all dimensions for the Anthropic Messages endpoint.

    Returns {model: {dimension: result_string}}.
    """
    results: dict[str, dict[str, Any]] = {}
    total = len(models)

    for i, m in enumerate(models, 1):
        print(f"  [{i}/{total}] {m} ...", file=sys.stderr)
        r: dict[str, Any] = {}

        # developer_role
        c, d = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                messages=[
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": "Say OK"},
                ],
            ),
            user,
        )
        r["developer_role"] = fmt(c, d)

        # null_content
        c, d = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                messages=[
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": None},
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["null_content"] = fmt(c, d)

        # orphaned_tools
        c, d = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                messages=[
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_001",
                                "name": "f",
                                "input": {},
                            },
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["orphaned_use"] = fmt(c, d)

        c, d = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_999",
                                "content": "r",
                            },
                        ],
                    },
                    {"role": "user", "content": "Continue"},
                ],
            ),
            user,
        )
        r["orphaned_result"] = fmt(c, d)

        # tool_schema
        c1, d1 = post(ANTHROPIC_URL, anth_body(m, tools=[_CLEAN_TOOL_ANTH]), user)
        c2, d2 = post(ANTHROPIC_URL, anth_body(m, tools=[_COMPLEX_TOOL_ANTH]), user)
        r["clean_schema"] = fmt(c1, d1)
        r["complex_schema"] = fmt(c2, d2)

        # sampling
        c1, d1 = post(ANTHROPIC_URL, anth_body(m, temperature=0.7), user)
        c2, d2 = post(ANTHROPIC_URL, anth_body(m, top_p=0.9), user)
        c3, d3 = post(ANTHROPIC_URL, anth_body(m, temperature=0.7, top_p=0.9), user)
        r["temperature"] = fmt(c1, d1)
        r["top_p"] = fmt(c2, d2)
        r["temp_and_top_p"] = fmt(c3, d3)

        # thinking_modes
        c1, d1 = post(
            ANTHROPIC_URL,
            anth_body(m, max_tokens=2048, thinking={"type": "enabled"}),
            user,
        )
        c2, d2 = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                max_tokens=2048,
                thinking={"type": "enabled", "budget_tokens": 1024},
            ),
            user,
        )
        c3, d3 = post(
            ANTHROPIC_URL,
            anth_body(m, max_tokens=2048, thinking={"type": "adaptive"}),
            user,
        )
        c4, d4 = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                max_tokens=2048,
                thinking={"type": "adaptive", "budget_tokens": 1024},
            ),
            user,
        )
        c5, d5 = post(
            ANTHROPIC_URL,
            anth_body(m, thinking={"type": "disabled"}),
            user,
        )
        r["think_enabled"] = fmt(c1, d1, 45)
        r["think_enabled_budget"] = fmt(c2, d2, 45)
        r["think_adaptive"] = fmt(c3, d3, 45)
        r["think_adaptive_budget"] = fmt(c4, d4, 45)
        r["think_disabled"] = fmt(c5, d5, 45)

        # effort (output_config)
        c1, d1 = post(
            ANTHROPIC_URL,
            anth_body(m, max_tokens=2048, output_config={"effort": "high"}),
            user,
        )
        c2, d2 = post(
            ANTHROPIC_URL,
            anth_body(
                m,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            ),
            user,
        )
        c3, d3 = post(
            ANTHROPIC_URL,
            anth_body(m, max_tokens=2048, output_config={"effort": "low"}),
            user,
        )
        r["effort_high"] = fmt(c1, d1, 45)
        r["effort_adaptive_high"] = fmt(c2, d2, 45)
        r["effort_low"] = fmt(c3, d3, 45)

        results[m] = r
        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Probe functions — OpenAI Responses
# ---------------------------------------------------------------------------


def probe_responses(models: list[str], user: str) -> dict[str, dict[str, Any]]:
    """Probe all dimensions for the OpenAI Responses endpoint.

    Returns {model: {dimension: result_string}}.
    """
    results: dict[str, dict[str, Any]] = {}
    total = len(models)

    for i, m in enumerate(models, 1):
        print(f"  [{i}/{total}] {m} ...", file=sys.stderr)
        r: dict[str, Any] = {}

        # baseline
        c, d = post(RESPONSES_URL, responses_body(m), user)
        r["baseline"] = fmt(c, d)

        # developer_role (system-level instruction via "developer" role input)
        c, d = post(
            RESPONSES_URL,
            responses_body(
                m,
                input=[
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": "Say OK"},
                ],
            ),
            user,
        )
        r["developer_role"] = fmt(c, d)

        # instructions (top-level system prompt)
        c, d = post(
            RESPONSES_URL,
            responses_body(m, instructions="Be brief."),
            user,
        )
        r["instructions"] = fmt(c, d)

        # max_output_tokens
        c, d = post(
            RESPONSES_URL,
            responses_body(m, max_output_tokens=16),
            user,
        )
        r["max_output_tokens"] = fmt(c, d)

        # sampling
        c1, d1 = post(RESPONSES_URL, responses_body(m, temperature=0.7), user)
        c2, d2 = post(RESPONSES_URL, responses_body(m, top_p=0.9), user)
        r["temperature"] = fmt(c1, d1)
        r["top_p"] = fmt(c2, d2)

        # tool_schema
        c, d = post(
            RESPONSES_URL,
            responses_body(m, tools=[_CLEAN_TOOL_RESPONSES]),
            user,
        )
        r["tool_schema"] = fmt(c, d)

        # reasoning_effort
        for effort in ("low", "medium", "high"):
            c, d = post(
                RESPONSES_URL,
                responses_body(m, reasoning={"effort": effort}),
                user,
            )
            r[f"reasoning_{effort}"] = fmt(c, d)

        results[m] = r
        time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a markdown table to stdout."""
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def render_oai_chat(results: dict[str, dict[str, Any]]) -> None:
    """Render OAI Chat probe results as markdown tables."""
    models = list(results.keys())
    if not models:
        return

    print("\n## OAI Chat Completions (`/v1/chat/completions`)\n")
    print(f"Models tested: {len(models)}")

    print_table(
        "Developer role & null content",
        ["Model", "developer role", "null content"],
        [[m, results[m]["developer_role"], results[m]["null_content"]] for m in models],
    )

    print_table(
        "max_tokens variants",
        ["Model", "max_tokens", "max_completion_tokens", "both"],
        [
            [
                m,
                results[m]["max_tokens"],
                results[m]["max_completion_tokens"],
                results[m]["max_tokens_both"],
            ]
            for m in models
        ],
    )

    print_table(
        "Orphaned tool calls",
        ["Model", "orphaned call", "orphaned result"],
        [
            [m, results[m]["orphaned_call"], results[m]["orphaned_result"]]
            for m in models
        ],
    )

    print_table(
        "Tool schema",
        ["Model", "clean schema", "complex schema"],
        [[m, results[m]["clean_schema"], results[m]["complex_schema"]] for m in models],
    )

    print_table(
        "Sampling parameters",
        ["Model", "temperature", "top_p", "T + P"],
        [
            [
                m,
                results[m]["temperature"],
                results[m]["top_p"],
                results[m]["temp_and_top_p"],
            ]
            for m in models
        ],
    )


def render_anthropic(results: dict[str, dict[str, Any]]) -> None:
    """Render Anthropic probe results as markdown tables."""
    models = list(results.keys())
    if not models:
        return

    print("\n## Anthropic Messages (`/v1/messages`)\n")
    print(f"Models tested: {len(models)}")

    print_table(
        "Developer role & null content",
        ["Model", "developer role", "null content"],
        [[m, results[m]["developer_role"], results[m]["null_content"]] for m in models],
    )

    print_table(
        "Orphaned tool calls",
        ["Model", "orphaned use", "orphaned result"],
        [
            [m, results[m]["orphaned_use"], results[m]["orphaned_result"]]
            for m in models
        ],
    )

    print_table(
        "Tool schema",
        ["Model", "clean schema", "complex schema"],
        [[m, results[m]["clean_schema"], results[m]["complex_schema"]] for m in models],
    )

    print_table(
        "Sampling parameters",
        ["Model", "temperature", "top_p", "T + P"],
        [
            [
                m,
                results[m]["temperature"],
                results[m]["top_p"],
                results[m]["temp_and_top_p"],
            ]
            for m in models
        ],
    )

    print_table(
        "Thinking modes",
        [
            "Model",
            "enabled",
            "enabled+budget",
            "adaptive",
            "adaptive+budget",
            "disabled",
        ],
        [
            [
                m,
                results[m]["think_enabled"],
                results[m]["think_enabled_budget"],
                results[m]["think_adaptive"],
                results[m]["think_adaptive_budget"],
                results[m]["think_disabled"],
            ]
            for m in models
        ],
    )

    print_table(
        "output_config.effort",
        ["Model", "effort=high", "adaptive+effort", "effort=low"],
        [
            [
                m,
                results[m]["effort_high"],
                results[m]["effort_adaptive_high"],
                results[m]["effort_low"],
            ]
            for m in models
        ],
    )


def render_responses(results: dict[str, dict[str, Any]]) -> None:
    """Render Responses probe results as markdown tables."""
    models = list(results.keys())
    if not models:
        return

    print("\n## OpenAI Responses (`/v1/responses`)\n")
    print(f"Models tested: {len(models)}")

    print_table(
        "Baseline & instructions",
        ["Model", "baseline", "developer role", "instructions", "max_output_tokens"],
        [
            [
                m,
                results[m]["baseline"],
                results[m]["developer_role"],
                results[m]["instructions"],
                results[m]["max_output_tokens"],
            ]
            for m in models
        ],
    )

    print_table(
        "Sampling & tools",
        ["Model", "temperature", "top_p", "tool schema"],
        [
            [
                m,
                results[m]["temperature"],
                results[m]["top_p"],
                results[m]["tool_schema"],
            ]
            for m in models
        ],
    )

    print_table(
        "Reasoning effort",
        ["Model", "low", "medium", "high"],
        [
            [
                m,
                results[m]["reasoning_low"],
                results[m]["reasoning_medium"],
                results[m]["reasoning_high"],
            ]
            for m in models
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe ARGO API endpoints for per-model parameter acceptance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                              # probe all endpoints and families\n"
            "  %(prog)s --endpoint chat --family openai\n"
            "  %(prog)s --endpoint anthropic --json out.json\n"
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=["chat", "anthropic", "responses", "all"],
        default="all",
        help="which endpoint(s) to probe (default: all)",
    )
    parser.add_argument(
        "--family",
        choices=["openai", "gemini", "claude", "all"],
        default="all",
        help="which model family to probe (default: all)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("ARGO_USER", "pding"),
        help="ARGO username (default: $ARGO_USER or 'pding')",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        dest="json_file",
        help="save raw results as JSON to FILE",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Discover models.
    print("Fetching model list from ARGO ...", file=sys.stderr)
    families = discover_models(args.user)
    if not families:
        print("ERROR: no models discovered from ARGO API.", file=sys.stderr)
        sys.exit(1)

    for fam, mods in sorted(families.items()):
        print(f"  {fam}: {len(mods)} models ({', '.join(mods)})", file=sys.stderr)

    # Filter by family.
    if args.family != "all":
        families = {k: v for k, v in families.items() if k == args.family}
        if not families:
            print(
                f"ERROR: no models found for family '{args.family}'.", file=sys.stderr
            )
            sys.exit(1)

    all_models = []
    for mods in families.values():
        all_models.extend(mods)
    claude_models = families.get("claude", [])

    # Print header.
    print("# ARGO Per-Model Probe Results\n")
    print(f"- **Target**: `{BASE_URL}`")
    print(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- **User**: `{args.user}`")
    print(f"- **Models**: {len(all_models)} (families: {', '.join(sorted(families))})")
    print()

    t0 = time.time()
    raw_results: dict[str, Any] = {}

    # --- OAI Chat ---
    if args.endpoint in ("chat", "all"):
        print("\n--- Probing OAI Chat Completions ---", file=sys.stderr)
        oai_results = probe_oai_chat(all_models, args.user)
        render_oai_chat(oai_results)
        raw_results["oai_chat"] = oai_results

    # --- Anthropic ---
    if args.endpoint in ("anthropic", "all"):
        # Anthropic endpoint only makes sense for claude models.
        anth_models = claude_models if claude_models else all_models
        print("\n--- Probing Anthropic Messages ---", file=sys.stderr)
        anth_results = probe_anthropic(anth_models, args.user)
        render_anthropic(anth_results)
        raw_results["anthropic"] = anth_results

    # --- Responses ---
    if args.endpoint in ("responses", "all"):
        print("\n--- Probing OpenAI Responses ---", file=sys.stderr)
        resp_results = probe_responses(all_models, args.user)
        render_responses(resp_results)
        raw_results["responses"] = resp_results

    elapsed = time.time() - t0
    print(f"\n---\n_Completed in {elapsed:.0f}s, {time.strftime('%Y-%m-%d %H:%M:%S')}_")

    # Optionally save JSON.
    if args.json_file:
        with open(args.json_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        print(f"Raw results saved to {args.json_file}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test streaming round-trip with REAL provider SSE flows.

Captures a real streaming response from each provider, then runs it
through the round-trip pipeline to verify no event inflation occurs.

Supports two test modes:
  - text:  longer text generation (levels out chunk counts across providers)
  - tools: tool-call flow (each provider invokes get_weather)

Requires API keys in .env (or environment variables).

Usage:
    python dev_scripts/test_roundtrip_live.py                    # both modes
    python dev_scripts/test_roundtrip_live.py --mode text         # text only
    python dev_scripts/test_roundtrip_live.py --mode tools        # tools only
    python dev_scripts/test_roundtrip_live.py --provider google
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.base.context import StreamContext

# ============================================================
# Prompts & tool schema
# ============================================================

TEXT_PROMPT = "List 5 fun facts about the number 42, one per line. Be concise."
TOOL_PROMPT = (
    "What is the current weather in New York City? "
    "Use the get_weather tool to find out."
)

TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
    },
    "required": ["city"],
}


def load_env() -> None:
    """Load .env file from project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


# ============================================================
# Provider-specific capture functions
# ============================================================


def capture_anthropic(
    prompt: str, use_tools: bool = False
) -> tuple[list[dict[str, Any]], str]:
    """Capture real Anthropic SSE events."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4.5")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_tools:
        body["tools"] = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": TOOL_SCHEMA,
            }
        ]

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/messages",
        headers=headers,
        json=body,
        timeout=60.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {err[:300]}")
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                continue
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

    return events, model


def capture_openai_chat(
    prompt: str, use_tools: bool = False
) -> tuple[list[dict[str, Any]], str]:
    """Capture real OpenAI Chat SSE events."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": 2048,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": TOOL_SCHEMA,
                },
            }
        ]

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        headers=headers,
        json=body,
        timeout=60.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"OpenAI Chat HTTP {resp.status_code}: {err[:300]}")
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass

    return events, model


def capture_openai_responses(
    prompt: str, use_tools: bool = False
) -> tuple[list[dict[str, Any]], str]:
    """Capture real OpenAI Responses SSE events."""
    api_key = os.environ.get("OPENAI_RESPONSES_API_KEY", "")
    base_url = os.environ.get("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "max_output_tokens": 2048,
        "stream": True,
        "input": [{"role": "user", "content": prompt}],
    }
    if use_tools:
        body["tools"] = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": TOOL_SCHEMA,
            }
        ]

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/responses",
        headers=headers,
        json=body,
        timeout=60.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"OpenAI Responses HTTP {resp.status_code}: {err[:300]}")
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass

    return events, model


def capture_google(
    prompt: str, use_tools: bool = False
) -> tuple[list[dict[str, Any]], str]:
    """Capture real Google GenAI SSE events."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    base_url = os.environ.get(
        "GOOGLE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    model = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2048},
    }
    if use_tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": TOOL_SCHEMA,
                    }
                ]
            }
        ]

    events: list[dict[str, Any]] = []
    url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    with httpx.stream(
        "POST",
        url,
        json=body,
        timeout=60.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"Google HTTP {resp.status_code}: {err[:300]}")
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    return events, model


# ============================================================
# Round-trip logic
# ============================================================


def run_roundtrip(
    provider: str,
    input_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run round-trip and return (ir_events, output_events)."""
    converter = get_converter_for_provider(provider)
    from_ctx = StreamContext()
    to_ctx = StreamContext()

    all_ir: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []

    for inp in input_events:
        ir_events = converter.stream_response_from_provider(inp, context=from_ctx)
        for ir_event in ir_events:
            all_ir.append(ir_event)
            result = converter.stream_response_to_provider(ir_event, context=to_ctx)
            if isinstance(result, list):
                output_events.extend(e for e in result if e)
            elif result:
                output_events.append(result)

    return all_ir, output_events


def event_type_label(provider: str, e: dict[str, Any]) -> str:
    """Get a short type label for a provider event."""
    t = e.get("type")
    if t:
        return t
    if "choices" in e:
        choices = e["choices"]
        if not choices:
            return "usage_chunk" if "usage" in e else "empty_choices"
        delta = choices[0].get("delta", {})
        fr = choices[0].get("finish_reason")
        if fr:
            return f"finish({fr})"
        if "role" in delta:
            return "role_delta"
        if "content" in delta:
            return "content_delta"
        if "reasoning_content" in delta:
            return "reasoning_delta"
        if "tool_calls" in delta:
            tc = delta["tool_calls"][0]
            if tc.get("id"):
                return "tool_call_start"
            return "tool_call_args"
        return "choice_chunk"
    if "candidates" in e:
        cand = e["candidates"][0] if e.get("candidates") else {}
        if cand.get("finishReason"):
            return f"finish({cand['finishReason']})"
        parts = cand.get("content", {}).get("parts", [])
        if parts:
            p0 = parts[0]
            if p0.get("thought"):
                return "thought_chunk"
            if "functionCall" in p0:
                return "func_call_chunk"
            return "text_chunk"
        return "candidate_chunk"
    return "unknown"


def print_result(
    label: str,
    provider: str,
    model: str,
    input_events: list[dict[str, Any]],
    ir_events: list[dict[str, Any]],
    output_events: list[dict[str, Any]],
) -> bool:
    """Print detailed result and return True if inflated."""
    in_types = [event_type_label(provider, e) for e in input_events]
    out_types = [event_type_label(provider, e) for e in output_events]
    ir_types = [e.get("type", "?") for e in ir_events]

    inflated = len(out_types) > len(in_types)
    if len(out_types) == len(in_types):
        status = "OK (exact)"
    elif len(out_types) < len(in_types):
        status = "OK (deflated)"
    else:
        status = "INFLATED"

    print(f"\n{'=' * 70}")
    print(f"  {label} — {provider} ({model})")
    print(
        f"  {len(in_types)} input → {len(ir_types)} IR → {len(out_types)} output"
        f"  [{status}]"
    )
    print(f"{'=' * 70}")
    print(f"  INPUT  ({len(in_types):>2}): {in_types}")
    print(f"  IR     ({len(ir_types):>2}): {ir_types}")
    print(f"  OUTPUT ({len(out_types):>2}): {out_types}")

    if inflated:
        max_len = max(len(in_types), len(out_types))
        print(f"\n  {'#':>3}  {'INPUT':<30} {'OUTPUT':<30} {'DIFF'}")
        print(f"  {'---':>3}  {'-' * 30} {'-' * 30} {'----'}")
        for i in range(max_len):
            inp = in_types[i] if i < len(in_types) else "(none)"
            out = out_types[i] if i < len(out_types) else "(none)"
            diff = "<<<" if inp != out else ""
            print(f"  {i:>3}  {inp:<30} {out:<30} {diff}")

    return inflated


PROVIDERS = {
    "anthropic": capture_anthropic,
    "openai_chat": capture_openai_chat,
    "openai_responses": capture_openai_responses,
    "google": capture_google,
}


def run_pass(
    pass_label: str,
    prompt: str,
    use_tools: bool,
    providers: list[str],
    save_dir: Path | None,
    results: dict[str, bool],
) -> None:
    """Run one test pass (text or tools) across selected providers."""
    print(f"\n{'#' * 70}")
    print(f"  {pass_label}")
    print(f"{'#' * 70}")

    for provider in providers:
        capture_fn = PROVIDERS[provider]
        key = f"{pass_label}/{provider}"
        try:
            print(f"\n  Capturing {provider} ({pass_label})...", end="", flush=True)
            raw_events, model = capture_fn(prompt, use_tools=use_tools)
            print(f" {len(raw_events)} events captured.")

            if save_dir:
                suffix = "tools" if use_tools else "text"
                out_file = save_dir / f"{provider}_{suffix}_raw.jsonl"
                with open(out_file, "w") as f:
                    for e in raw_events:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"  Saved to {out_file}")

            ir_events, output_events = run_roundtrip(provider, raw_events)
            inflated = print_result(
                pass_label, provider, model, raw_events, ir_events, output_events
            )
            results[key] = inflated

        except Exception as exc:
            print(" ERROR")
            print(f"\n{'=' * 70}")
            print(f"  {key}: ERROR — {exc}")
            print(f"{'=' * 70}")
            import traceback

            traceback.print_exc()
            results[key] = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SSE round-trip test")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=None,
        help="Test only this provider (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "tools", "all"],
        default="all",
        help="Test mode: text-only, tools-only, or both (default: all)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override default prompt (applies to selected mode)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save captured events to directory as JSONL files",
    )
    args = parser.parse_args()

    load_env()

    providers = [args.provider] if args.provider else list(PROVIDERS.keys())
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    if args.mode in ("text", "all"):
        run_pass(
            pass_label="text",
            prompt=args.prompt or TEXT_PROMPT,
            use_tools=False,
            providers=providers,
            save_dir=save_dir,
            results=results,
        )

    if args.mode in ("tools", "all"):
        run_pass(
            pass_label="tools",
            prompt=args.prompt or TOOL_PROMPT,
            use_tools=True,
            providers=providers,
            save_dir=save_dir,
            results=results,
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    any_failed = False
    for key, inflated in results.items():
        status = "INFLATED" if inflated else "OK"
        if inflated:
            any_failed = True
        print(f"  {key:<40} {status}")

    if any_failed:
        print("\n  *** SOME TESTS FAILED ***")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()

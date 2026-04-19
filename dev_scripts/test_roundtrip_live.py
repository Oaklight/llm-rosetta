#!/usr/bin/env python3
"""Test streaming round-trip with REAL provider SSE flows.

Captures a real streaming response from each provider, then runs it
through the round-trip pipeline to verify no event inflation occurs.

Requires API keys in .env (or environment variables).

Usage:
    python dev_scripts/test_roundtrip_live.py
    python dev_scripts/test_roundtrip_live.py --provider anthropic
    python dev_scripts/test_roundtrip_live.py --prompt "Use a tool to get the weather in NYC"
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


def capture_anthropic(prompt: str) -> tuple[list[dict[str, Any]], str]:
    """Capture real Anthropic SSE events."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4.5")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 300,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/messages",
        headers=headers,
        json=body,
        timeout=30.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {err[:200]}")
        current_event = None
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    events.append(data)
                except json.JSONDecodeError:
                    pass

    return events, model


def capture_openai_chat(prompt: str) -> tuple[list[dict[str, Any]], str]:
    """Capture real OpenAI Chat SSE events."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_completion_tokens": 300,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        headers=headers,
        json=body,
        timeout=30.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"OpenAI Chat HTTP {resp.status_code}: {err[:200]}")
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


def capture_openai_responses(prompt: str) -> tuple[list[dict[str, Any]], str]:
    """Capture real OpenAI Responses SSE events."""
    api_key = os.environ.get("OPENAI_RESPONSES_API_KEY", "")
    base_url = os.environ.get(
        "OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1"
    )
    model = os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_output_tokens": 300,
        "stream": True,
        "input": [{"role": "user", "content": prompt}],
    }

    events: list[dict[str, Any]] = []
    with httpx.stream(
        "POST",
        f"{base_url}/responses",
        headers=headers,
        json=body,
        timeout=30.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"OpenAI Responses HTTP {resp.status_code}: {err[:200]}")
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


def capture_google(prompt: str) -> tuple[list[dict[str, Any]], str]:
    """Capture real Google GenAI SSE events."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    base_url = os.environ.get(
        "GOOGLE_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    model = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 300},
    }

    events: list[dict[str, Any]] = []
    url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    with httpx.stream(
        "POST",
        url,
        json=body,
        timeout=30.0,
    ) as resp:
        if resp.status_code != 200:
            err = resp.read().decode()
            raise RuntimeError(f"Google HTTP {resp.status_code}: {err[:200]}")
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
            return "tool_calls_delta"
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

    # "Inflated" means output has MORE events than input.
    # output < input is OK (compound chunks decompose legitimately).
    # output == input is ideal (perfect round-trip).
    # output > input is the bug we're fixing.
    inflated = len(out_types) > len(in_types)
    if len(out_types) == len(in_types):
        status = "OK (exact)"
    elif len(out_types) < len(in_types):
        status = "OK (deflated)"
    else:
        status = "INFLATED"

    print(f"\n{'='*70}")
    print(f"  {provider} ({model})")
    print(f"  {len(in_types)} input → {len(ir_types)} IR → {len(out_types)} output  [{status}]")
    print(f"{'='*70}")
    print(f"  INPUT  ({len(in_types):>2}): {in_types}")
    print(f"  IR     ({len(ir_types):>2}): {ir_types}")
    print(f"  OUTPUT ({len(out_types):>2}): {out_types}")

    if inflated:
        max_len = max(len(in_types), len(out_types))
        print(f"\n  {'#':>3}  {'INPUT':<30} {'OUTPUT':<30} {'DIFF'}")
        print(f"  {'---':>3}  {'-'*30} {'-'*30} {'----'}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SSE round-trip test")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=None,
        help="Test only this provider (default: all)",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2+2? Answer in one sentence.",
        help="Prompt to send",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save captured events to directory as JSONL files",
    )
    args = parser.parse_args()

    load_env()

    providers = [args.provider] if args.provider else list(PROVIDERS.keys())
    results: dict[str, bool] = {}

    for provider in providers:
        capture_fn = PROVIDERS[provider]
        try:
            print(f"\n  Capturing {provider}...", end="", flush=True)
            raw_events, model = capture_fn(args.prompt)
            print(f" {len(raw_events)} events captured.")

            if args.save:
                save_dir = Path(args.save)
                save_dir.mkdir(parents=True, exist_ok=True)
                out_file = save_dir / f"{provider}_raw.jsonl"
                with open(out_file, "w") as f:
                    for e in raw_events:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"  Saved to {out_file}")

            ir_events, output_events = run_roundtrip(provider, raw_events)
            inflated = print_result(
                provider, model, raw_events, ir_events, output_events
            )
            results[provider] = inflated

        except Exception as exc:
            print(f" ERROR")
            print(f"\n{'='*70}")
            print(f"  {provider}: ERROR — {exc}")
            print(f"{'='*70}")
            import traceback

            traceback.print_exc()
            results[provider] = True

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    any_failed = False
    for provider, inflated in results.items():
        status = "INFLATED" if inflated else "OK"
        if inflated:
            any_failed = True
        print(f"  {provider:<25} {status}")

    if any_failed:
        print(f"\n  *** SOME TESTS FAILED ***")
        sys.exit(1)
    else:
        print(f"\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()

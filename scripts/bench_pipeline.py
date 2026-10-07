#!/usr/bin/env python3
"""Conversion pipeline benchmark suite.

Generates synthetic payloads matching production size profiles and
benchmarks all conversion paths with per-phase timing.

Usage:
    python scripts/bench_pipeline.py [--rounds N] [--json]
    make bench

Issue: #831 (parent: #828)
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
import warnings
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic payload generators
# ---------------------------------------------------------------------------


def _make_tool(i: int) -> dict[str, Any]:
    """Generate a synthetic Anthropic tool definition."""
    n_params = 3 + (i % 5)
    properties = {}
    required = []
    for j in range(n_params):
        name = f"param_{j}"
        properties[name] = {
            "type": "string",
            "description": f"Parameter {j} for tool_{i}. " + "x" * (20 + j * 3),
        }
        if j < n_params // 2:
            required.append(name)
    return {
        "name": f"tool_{i}",
        "description": f"Tool number {i} that performs operation {i}. " + "y" * 40,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _make_anthropic_messages(
    n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    """Generate properly paired Anthropic messages.

    Produces a realistic conversation flow: user → assistant (text or
    tool_use) → user (tool_result if needed) → ...
    Tool calls and results are always properly paired.
    """
    msgs: list[dict[str, Any]] = []
    tool_turn_interval = max(6, n // 15)
    tc_counter = 0

    i = 0
    while i < n:
        if i % 2 == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"User message {i}. " + "u" * text_size,
                        }
                    ],
                }
            )
            i += 1
        elif i % tool_turn_interval == tool_turn_interval - 1 and i + 2 < n:
            tc_id = f"toolu_{tc_counter:06d}"
            tc_counter += 1
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": f"tool_{tc_counter % n_tools}",
                            "input": {"param_0": "val_" + "a" * (text_size // 4)},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc_id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Result {tc_counter}: " + "r" * text_size,
                                }
                            ],
                        }
                    ],
                }
            )
            i += 2
        else:
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Assistant message {i}. " + "a" * text_size,
                        }
                    ],
                }
            )
            i += 1

    return msgs[:n]


def _make_openai_messages(
    n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    """Generate properly paired OpenAI Chat messages."""
    msgs: list[dict[str, Any]] = []
    tool_turn_interval = max(6, n // 15)
    tc_counter = 0

    i = 0
    while i < n:
        if i % 2 == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": f"User message {i}. " + "u" * text_size,
                }
            )
            i += 1
        elif i % tool_turn_interval == tool_turn_interval - 1 and i + 2 < n:
            tc_id = f"call_{tc_counter:06d}"
            tc_counter += 1
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": f"tool_{tc_counter % n_tools}",
                                "arguments": json.dumps(
                                    {"param_0": "val_" + "a" * (text_size // 4)}
                                ),
                            },
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"Result {tc_counter}: " + "r" * text_size,
                }
            )
            i += 2
        else:
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"Assistant message {i}. " + "a" * text_size,
                }
            )
            i += 1

    return msgs[:n]


def _make_openai_tool(i: int) -> dict[str, Any]:
    """Generate a synthetic OpenAI Chat tool definition."""
    t = _make_tool(i)
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }


PROFILES: dict[str, dict[str, int]] = {
    "small": {"n_messages": 170, "n_tools": 87, "text_size": 2700},
    "medium": {"n_messages": 220, "n_tools": 104, "text_size": 8000},
    "large": {"n_messages": 480, "n_tools": 154, "text_size": 12500},
}


def generate_anthropic_request(profile: str) -> dict[str, Any]:
    cfg = PROFILES[profile]
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are a helpful coding assistant.",
        "messages": _make_anthropic_messages(
            cfg["n_messages"],
            n_tools=cfg["n_tools"],
            text_size=cfg["text_size"],
        ),
        "tools": [_make_tool(i) for i in range(cfg["n_tools"])],
    }


def generate_openai_request(profile: str) -> dict[str, Any]:
    cfg = PROFILES[profile]
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
        ]
        + _make_openai_messages(
            cfg["n_messages"],
            n_tools=cfg["n_tools"],
            text_size=cfg["text_size"],
        ),
        "tools": [_make_openai_tool(i) for i in range(cfg["n_tools"])],
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _payload_size_kb(payload: dict[str, Any]) -> float:
    return len(json.dumps(payload).encode()) / 1024


def bench_pipeline(
    source_format: str,
    target_format: str,
    profile: str,
    rounds: int = 5,
) -> dict[str, Any]:
    from llm_rosetta.converters.base.helpers.cache import clear_all_caches
    from llm_rosetta.pipeline import ConversionPipeline

    if source_format == "anthropic":
        payload = generate_anthropic_request(profile)
    else:
        payload = generate_openai_request(profile)

    size_kb = _payload_size_kb(payload)

    timings: dict[str, list[float]] = {
        "source_to_ir_ms": [],
        "ir_transforms_ms": [],
        "ir_to_target_ms": [],
        "body_transforms_ms": [],
        "request_conversion_ms": [],
    }

    for i in range(rounds + 1):
        if i == 1:
            clear_all_caches()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = ConversionPipeline(
                source_provider=source_format,
                target_provider=target_format,
            )
            pipeline.convert_request(copy.deepcopy(payload))

        prof = pipeline.profile
        if i == 0:
            continue

        for key in timings:
            timings[key].append(prof.get(key, 0.0))

    result: dict[str, Any] = {
        "path": f"{source_format} → {target_format}",
        "profile": profile,
        "payload_kb": round(size_kb, 1),
        "rounds": rounds,
    }
    for key, values in timings.items():
        result[key] = {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return result


def run_all(rounds: int = 5) -> list[dict[str, Any]]:
    paths = [
        ("anthropic", "anthropic"),
        ("anthropic", "openai_chat"),
        ("openai_chat", "openai_chat"),
    ]
    results = []
    for source, target in paths:
        for profile in ["small", "medium", "large"]:
            results.append(bench_pipeline(source, target, profile, rounds))
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(
        f"{'Path':<28} {'Size':>6} {'Payload':>10} "
        f"{'src→ir':>10} {'ir_xform':>10} {'ir→tgt':>10} {'total':>10}"
    )
    lines.append("-" * 96)

    for r in results:
        src_ir = r["source_to_ir_ms"]["mean"]
        ir_xf = r["ir_transforms_ms"]["mean"]
        ir_tgt = r["ir_to_target_ms"]["mean"]
        total = r["request_conversion_ms"]["mean"]
        payload = f"{r['payload_kb']:.0f}KB"
        lines.append(
            f"{r['path']:<28} {r['profile']:>6} {payload:>10} "
            f"{src_ir:>9.2f}ms {ir_xf:>9.2f}ms {ir_tgt:>9.2f}ms {total:>9.2f}ms"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion pipeline benchmark")
    parser.add_argument(
        "--rounds", type=int, default=5, help="Benchmark rounds (default: 5)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"Running pipeline benchmarks ({args.rounds} rounds each)...\n")
    t0 = time.perf_counter()
    results = run_all(rounds=args.rounds)
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_table(results))
        print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

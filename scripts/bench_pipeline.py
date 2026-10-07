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


def _make_tool_anthropic(i: int) -> dict[str, Any]:
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


def _make_tool_openai(i: int) -> dict[str, Any]:
    t = _make_tool_anthropic(i)
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }


def _make_tool_openai_responses(i: int) -> dict[str, Any]:
    t = _make_tool_anthropic(i)
    return {
        "type": "function",
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }


def _make_tool_google(i: int) -> dict[str, Any]:
    t = _make_tool_anthropic(i)
    params = dict(t["input_schema"])
    params.pop("required", None)
    return {
        "name": t["name"],
        "description": t["description"],
        "parameters": params,
    }


def _make_anthropic_messages(
    n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    tool_interval = max(6, n // 15)
    tc = 0
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
        elif i % tool_interval == tool_interval - 1 and i + 2 < n:
            tc_id = f"toolu_{tc:06d}"
            tc += 1
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": f"tool_{tc % n_tools}",
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
                                    "text": f"Result {tc}: " + "r" * text_size,
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
    msgs: list[dict[str, Any]] = []
    tool_interval = max(6, n // 15)
    tc = 0
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
        elif i % tool_interval == tool_interval - 1 and i + 2 < n:
            tc_id = f"call_{tc:06d}"
            tc += 1
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": f"tool_{tc % n_tools}",
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
                    "content": f"Result {tc}: " + "r" * text_size,
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


def _make_openai_responses_messages(
    n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    tool_interval = max(6, n // 15)
    tc = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            msgs.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"User message {i}. " + "u" * text_size,
                        }
                    ],
                }
            )
            i += 1
        elif i % tool_interval == tool_interval - 1 and i + 2 < n:
            tc_id = f"call_{tc:06d}"
            tc += 1
            msgs.append(
                {
                    "type": "function_call",
                    "id": tc_id,
                    "call_id": tc_id,
                    "name": f"tool_{tc % n_tools}",
                    "arguments": json.dumps(
                        {"param_0": "val_" + "a" * (text_size // 4)}
                    ),
                }
            )
            msgs.append(
                {
                    "type": "function_call_output",
                    "call_id": tc_id,
                    "output": f"Result {tc}: " + "r" * text_size,
                }
            )
            i += 2
        else:
            msgs.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Assistant message {i}. " + "a" * text_size,
                        }
                    ],
                }
            )
            i += 1
    return msgs[:n]


def _make_google_messages(
    n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    tool_interval = max(6, n // 15)
    tc = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            msgs.append(
                {
                    "role": "user",
                    "parts": [{"text": f"User message {i}. " + "u" * text_size}],
                }
            )
            i += 1
        elif i % tool_interval == tool_interval - 1 and i + 2 < n:
            tc += 1
            fn_name = f"tool_{tc % n_tools}"
            msgs.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": fn_name,
                                "args": {"param_0": "val_" + "a" * (text_size // 4)},
                            }
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": fn_name,
                                "response": {
                                    "result": f"Result {tc}: " + "r" * text_size
                                },
                            }
                        }
                    ],
                }
            )
            i += 2
        else:
            msgs.append(
                {
                    "role": "model",
                    "parts": [{"text": f"Assistant message {i}. " + "a" * text_size}],
                }
            )
            i += 1
    return msgs[:n]


PROFILES: dict[str, dict[str, int]] = {
    "small": {"n_messages": 170, "n_tools": 87, "text_size": 2700},
    "medium": {"n_messages": 220, "n_tools": 104, "text_size": 8000},
    "large": {"n_messages": 480, "n_tools": 154, "text_size": 12500},
}


def _generate_request(provider: str, profile: str) -> dict[str, Any]:
    cfg = PROFILES[profile]
    n_msgs = cfg["n_messages"]
    n_tools = cfg["n_tools"]
    text_size = cfg["text_size"]

    if provider == "anthropic":
        return {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": "You are a helpful coding assistant.",
            "messages": _make_anthropic_messages(
                n_msgs, n_tools=n_tools, text_size=text_size
            ),
            "tools": [_make_tool_anthropic(i) for i in range(n_tools)],
        }

    if provider == "openai_chat":
        return {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
            ]
            + _make_openai_messages(n_msgs, n_tools=n_tools, text_size=text_size),
            "tools": [_make_tool_openai(i) for i in range(n_tools)],
        }

    if provider == "openai_responses":
        return {
            "model": "gpt-4",
            "input": _make_openai_responses_messages(
                n_msgs, n_tools=n_tools, text_size=text_size
            ),
            "tools": [_make_tool_openai_responses(i) for i in range(n_tools)],
        }

    if provider == "google":
        return {
            "model": "gemini-2.0-flash",
            "contents": _make_google_messages(
                n_msgs, n_tools=n_tools, text_size=text_size
            ),
            "config": {
                "tools": [
                    {
                        "function_declarations": [
                            _make_tool_google(i) for i in range(n_tools)
                        ]
                    }
                ]
            },
        }

    if provider == "google_interactions":
        return {
            "model": "gemini-2.0-flash",
            "contents": _make_google_messages(
                n_msgs, n_tools=n_tools, text_size=text_size
            ),
            "config": {
                "tools": [
                    {
                        "function_declarations": [
                            _make_tool_google(i) for i in range(n_tools)
                        ]
                    }
                ]
            },
        }

    raise ValueError(f"Unknown provider: {provider}")


# Keep backward-compatible public API
def generate_anthropic_request(profile: str) -> dict[str, Any]:
    return _generate_request("anthropic", profile)


def generate_openai_request(profile: str) -> dict[str, Any]:
    return _generate_request("openai_chat", profile)


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

    payload = _generate_request(source_format, profile)
    size_kb = _payload_size_kb(payload)

    phase_keys = [
        "source_to_ir_ms",
        "ir_transforms_ms",
        "ir_to_target_ms",
        "request_conversion_ms",
    ]
    timings: dict[str, list[float]] = {k: [] for k in phase_keys}

    # Clear caches before this path to prevent cross-path leakage
    clear_all_caches()

    for i in range(rounds + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = ConversionPipeline(
                source_provider=source_format,
                target_provider=target_format,
            )
            pipeline.convert_request(copy.deepcopy(payload))

        prof = pipeline.profile
        if i == 0:
            continue  # warmup round

        for key in phase_keys:
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


BENCH_PATHS = [
    ("anthropic", "anthropic"),
    ("anthropic", "openai_chat"),
    ("openai_chat", "openai_chat"),
    ("openai_chat", "anthropic"),
    ("openai_responses", "openai_responses"),
    ("openai_responses", "anthropic"),
    ("google", "google"),
    ("google", "openai_chat"),
    ("google_interactions", "google_interactions"),
]


def run_all(rounds: int = 5) -> list[dict[str, Any]]:
    results = []
    for source, target in BENCH_PATHS:
        for profile in ["small", "medium", "large"]:
            results.append(bench_pipeline(source, target, profile, rounds))
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(
        f"{'Path':<40} {'Size':>6} {'Payload':>10} "
        f"{'src→ir':>10} {'ir_xform':>10} {'ir→tgt':>10} {'total':>10}"
    )
    lines.append("-" * 108)

    prev_path = None
    for r in results:
        path = r["path"]
        if prev_path is not None and path != prev_path:
            lines.append("")
        prev_path = path

        src_ir = r["source_to_ir_ms"]["mean"]
        ir_xf = r["ir_transforms_ms"]["mean"]
        ir_tgt = r["ir_to_target_ms"]["mean"]
        total = r["request_conversion_ms"]["mean"]
        payload = f"{r['payload_kb']:.0f}KB"
        lines.append(
            f"{path:<40} {r['profile']:>6} {payload:>10} "
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

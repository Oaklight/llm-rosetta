#!/usr/bin/env python3
"""Per-request CPU benchmark for the gateway conversion hot path (issue #1410).

The production gateway spends its CPU on one shape of traffic: an Anthropic
Messages request arrives with a long history and many tool definitions, it is
converted to Chat Completions, and the upstream answer comes back as a long
SSE stream that is converted back to Messages one chunk at a time.

This script reproduces that shape locally so a change can be measured before
and after on the same payloads:

1. ``json.loads`` of the raw request body            (request decode)
2. ``ConversionPipeline.convert_request``            (source → IR → target)
3. ``_strip_internal_metadata`` on the target body   (gateway boundary)
4. ``json.dumps`` of the target body                 (upstream encode)
5. per upstream chunk: ``json.loads`` → ``process_chunk`` → ``json.dumps``
   per emitted event                                 (stream decode, convert, encode)

Reported per stage: CPU time (``time.process_time``, the metric the change is
judged on) and wall time.  ``--profile`` adds a cProfile self-time table in
the same shape as the production profile from the issue.

Usage::

    python scripts/dev/bench_hot_path.py --rounds 5
    python scripts/dev/bench_hot_path.py --rounds 5 --profile
    python scripts/dev/bench_hot_path.py --rounds 5 --json
    python scripts/dev/bench_hot_path.py --rounds 5 --cache-stats

Issue: #1410
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import random
import statistics
import sys
import time
from typing import Any

#: The production shape from issue #1410: ~149 messages, ~93 tool
#: definitions, a stream of 400+ chunks.
PRODUCTION_SHAPE: dict[str, int] = {
    "n_messages": 149,
    "n_tools": 93,
    "text_size": 2700,
    "n_chunks": 400,
}

_STAGES = (
    "request_decode",
    "convert",
    "gateway_strip",
    "upstream_encode",
    "chunk_decode",
    "process_chunk",
    "event_encode",
)


# ---------------------------------------------------------------------------
# Synthetic payload generators
# ---------------------------------------------------------------------------


def _make_tool(rng: random.Random, i: int) -> dict[str, Any]:
    n_params = 3 + (i % 5)
    properties: dict[str, Any] = {}
    required: list[str] = []
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


def _make_messages(
    rng: random.Random, n: int, *, n_tools: int, text_size: int
) -> list[dict[str, Any]]:
    """Build a history with the same mix production carries: text turns and
    tool-use/tool-result pairs."""
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
                            "type": "text",
                            "text": "Checking with a tool. " + "c" * (text_size // 4),
                        },
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": f"tool_{tc % n_tools}",
                            "input": {"param_0": "val_" + "a" * (text_size // 4)},
                        },
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


def build_request(shape: dict[str, int], *, round_index: int = 0) -> dict[str, Any]:
    """Build the Messages request for one round.

    Each round is the next turn of the same conversation: identical tools,
    identical history except for the final user message.  That is the
    production steady state — the caches see a list that changed in one
    entry, so their epoch guard misses while most per-entry lookups hit.
    """
    rng = random.Random(1410)
    body: dict[str, Any] = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8192,
        "stream": True,
        "system": [
            {
                "type": "text",
                "text": "You are a coding assistant. " + "s" * 400,
            }
        ],
        "messages": _make_messages(
            rng,
            shape["n_messages"],
            n_tools=shape["n_tools"],
            text_size=shape["text_size"],
        ),
        "tools": [_make_tool(rng, i) for i in range(shape["n_tools"])],
    }
    if round_index:
        last = body["messages"][-1]
        last["content"] = [
            {
                "type": "text",
                "text": f"Turn {round_index}: " + "q" * shape["text_size"],
            }
        ]
    return body


def build_chunks(n: int) -> list[str]:
    """Build the upstream Chat Completions SSE payloads, pre-encoded.

    The transport hands the gateway decoded JSON per line, so the cost the
    gateway pays is ``json.loads`` of exactly these strings.
    """
    chunks: list[str] = []
    base = {
        "id": "chatcmpl-1410",
        "object": "chat.completion.chunk",
        "model": "gpt-5",
        "created": 1700000000,
    }

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return json.dumps(
            {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            },
            ensure_ascii=False,
        )

    chunks.append(chunk({"role": "assistant"}))
    # n - 3 text deltas, then finish + usage.
    for i in range(n - 3):
        chunks.append(chunk({"content": f"token{i} "}))
    chunks.append(chunk({}, "stop"))
    chunks.append(
        json.dumps(
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 42000,
                    "completion_tokens": 398,
                    "total_tokens": 42398,
                },
            },
            ensure_ascii=False,
        )
    )
    return chunks


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_one(
    raw: str, chunk_texts: list[str], *, gateway_strip: bool = True
) -> dict[str, float]:
    """Run one request through the full local hot path, timing each stage.

    Payload construction happens outside the timed region so the numbers are
    the gateway's work, not the generator's.  *gateway_strip* mirrors the
    engine's boundary pass (``prepare_request``), which the benchmark can
    switch off to measure its cost.
    """
    from llm_rosetta.pipeline import ConversionPipeline
    from llm_rosetta_gateway.proxy.metadata import _strip_internal_metadata
    from llm_rosetta_gateway.transport.sse_format import SSE_FORMATTERS

    format_sse = SSE_FORMATTERS["messages"]

    t = time.perf_counter()
    parsed = json.loads(raw)
    request_decode = time.perf_counter() - t

    t = time.perf_counter()
    pipeline = ConversionPipeline(
        source_provider="messages",
        target_provider="chat_completions",
    )
    target_body = pipeline.convert_request(parsed)
    convert = time.perf_counter() - t

    t = time.perf_counter()
    if gateway_strip:
        _strip_internal_metadata(target_body)
    gateway_strip_seconds = time.perf_counter() - t

    t = time.perf_counter()
    json.dumps(target_body)
    upstream_encode = time.perf_counter() - t

    t = time.perf_counter()
    processor = pipeline.create_stream_processor()
    chunk_decode = 0.0
    process_chunk = 0.0
    event_encode = 0.0
    for text in chunk_texts:
        t = time.perf_counter()
        chunk = json.loads(text)
        chunk_decode += time.perf_counter() - t
        t = time.perf_counter()
        events = processor.process_chunk(chunk)
        process_chunk += time.perf_counter() - t
        t = time.perf_counter()
        for event in events:
            format_sse(event)
        event_encode += time.perf_counter() - t
    # EOF flush, as the gateway's `_flush_and_finalize` does.
    t = time.perf_counter()
    for event in processor.process_chunk(None):
        format_sse(event)
    process_chunk += time.perf_counter() - t

    return {
        "request_decode": request_decode,
        "convert": convert,
        "gateway_strip": gateway_strip_seconds,
        "upstream_encode": upstream_encode,
        "chunk_decode": chunk_decode,
        "process_chunk": process_chunk,
        "event_encode": event_encode,
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def benchmark(
    shape: dict[str, int],
    rounds: int,
    *,
    profile: bool = False,
    gateway_strip: bool = True,
) -> dict[str, Any]:
    """Run the benchmark and return per-stage means in milliseconds."""
    # Payloads are built once, outside every timed region.
    raws = [json.dumps(build_request(shape, round_index=i + 1)) for i in range(rounds)]
    chunk_texts = build_chunks(shape["n_chunks"])

    # Warmup: import cost, cache fill, allocator warm-up.
    _run_one(raws[0], chunk_texts, gateway_strip=gateway_strip)

    samples: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    cpu_samples: list[float] = []
    wall_samples: list[float] = []

    profiler = cProfile.Profile() if profile else None
    for i in range(rounds):
        if profiler is not None:
            profiler.enable()
        cpu0, wall0 = time.process_time(), time.perf_counter()
        stage_times = _run_one(raws[i], chunk_texts, gateway_strip=gateway_strip)
        wall_samples.append(time.perf_counter() - wall0)
        cpu_samples.append(time.process_time() - cpu0)
        if profiler is not None:
            profiler.disable()
        for stage, seconds in stage_times.items():
            samples[stage].append(seconds * 1000.0)

    result: dict[str, Any] = {
        "shape": dict(shape),
        "rounds": rounds,
        "gateway_strip": gateway_strip,
        # Median is the headline: the machine is shared, and a single
        # contended round moves the mean without moving the code.
        "cpu_ms_median": round(statistics.median(cpu_samples) * 1000.0, 2),
        "cpu_ms_mean": round(_mean(cpu_samples) * 1000.0, 2),
        "cpu_ms_min": round(min(cpu_samples) * 1000.0, 2),
        "wall_ms_median": round(statistics.median(wall_samples) * 1000.0, 2),
        "stages_ms_median": {
            stage: round(statistics.median(values), 2)
            for stage, values in samples.items()
        },
        "stages_ms_mean": {
            stage: round(_mean(values), 2) for stage, values in samples.items()
        },
        "body_kb": round(len(json.dumps(build_request(shape))) / 1024, 1),
    }
    if profiler is not None:
        result["profile"] = _format_profile(profiler)
    return result


def _format_profile(profiler: cProfile.Profile, limit: int = 30) -> str:
    """Render the cProfile self-time table the way the production profile did."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime").print_stats(limit)
    return stream.getvalue()


def _cache_report(shape: dict[str, int], rounds: int) -> dict[str, Any]:
    """Run *rounds* requests and report the conversion caches' hit rates."""
    from llm_rosetta.converters.base.helpers.cache import cache_info, clear_all_caches

    raws = [json.dumps(build_request(shape, round_index=i + 1)) for i in range(rounds)]
    chunk_texts = build_chunks(shape["n_chunks"])
    # Warm up imports and the allocator, then measure from a clean cache.
    _run_one(raws[0], chunk_texts)
    clear_all_caches()
    before = cache_info()
    for i in range(rounds):
        _run_one(raws[i], chunk_texts)
    after = cache_info()
    report: dict[str, Any] = {"rounds": rounds}
    for name in ("tool_entry", "sanitize", "ir_validation"):
        hits = after[name]["hits"] - before[name]["hits"]
        misses = after[name]["misses"] - before[name]["misses"]
        total = hits + misses
        report[name] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total else None,
        }
    return report


def _print_table(result: dict[str, Any]) -> None:
    stages = result["stages_ms_median"]
    print(f"shape: {result['shape']}  body={result['body_kb']}KB")
    print(f"rounds: {result['rounds']}")
    print()
    print(f"{'stage':<18} {'ms/request':>10}")
    print("-" * 30)
    for stage in _STAGES:
        print(f"{stage:<18} {stages[stage]:>10.2f}")
    print("-" * 30)
    print(f"{'stage sum':<18} {sum(stages.values()):>10.2f}")
    print(f"{'CPU/request':<18} {result['cpu_ms_median']:>10.2f}")
    print(f"{'wall/request':<18} {result['wall_ms_median']:>10.2f}")
    if "profile" in result:
        print()
        print(result["profile"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--profile", action="store_true", help="Add a cProfile self-time table"
    )
    parser.add_argument(
        "--cache-stats", action="store_true", help="Report conversion cache hit rates"
    )
    parser.add_argument(
        "--no-gateway-strip",
        action="store_true",
        help="Skip the engine's boundary _strip_internal_metadata pass",
    )
    parser.add_argument(
        "--n-messages", type=int, default=PRODUCTION_SHAPE["n_messages"]
    )
    parser.add_argument("--n-tools", type=int, default=PRODUCTION_SHAPE["n_tools"])
    parser.add_argument("--text-size", type=int, default=PRODUCTION_SHAPE["text_size"])
    parser.add_argument("--n-chunks", type=int, default=PRODUCTION_SHAPE["n_chunks"])
    args = parser.parse_args()

    shape = {
        "n_messages": args.n_messages,
        "n_tools": args.n_tools,
        "text_size": args.text_size,
        "n_chunks": args.n_chunks,
    }
    result = benchmark(
        shape,
        args.rounds,
        profile=args.profile,
        gateway_strip=not args.no_gateway_strip,
    )
    if args.cache_stats:
        result["caches"] = _cache_report(shape, args.rounds)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_table(result)
        if "caches" in result:
            print()
            print(json.dumps(result["caches"], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

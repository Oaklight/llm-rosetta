"""
Benchmark using real-world Claude Code / Codex CLI payloads.

Tests conversion latency with production-scale tool definitions (31-41 tools,
21KB+ schemas) and multi-turn conversations (up to 218 messages).

Payloads sourced from argo-proxy error dumps.
"""

import copy
import json
import os
import statistics
import sys
import time

# ── Load real payloads ──────────────────────────────────────────────────

DUMP_DIR_LOCAL = "/home/pding/projects/argo-proxy/error_dumps"
DUMP_DIR_LAMBDA = "/tmp/lambda11_dumps"

PAYLOADS: dict[str, dict] = {}


def _load_dump(path: str) -> dict | None:
    """Extract request_body from an argo-proxy error dump."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        dump = json.load(f)
    return dump.get("request_body", {})


def _strip_base64_images(obj):
    """Remove large base64 image data to keep benchmarks focused on conversion logic."""
    if isinstance(obj, dict):
        if obj.get("type") in ("image", "input_image") and "image_url" in obj:
            url = obj["image_url"]
            if isinstance(url, str) and url.startswith("data:"):
                obj = {**obj, "image_url": "data:image/png;base64,AAAA"}
            elif isinstance(url, dict) and isinstance(url.get("url", ""), str):
                if url["url"].startswith("data:"):
                    obj = {**obj, "image_url": {**url, "url": "data:image/png;base64,AAAA"}}
        if obj.get("type") == "image" and "source" in obj:
            src = obj["source"]
            if isinstance(src, dict) and src.get("type") == "base64":
                obj = {**obj, "source": {**src, "data": "AAAA"}}
        return {k: _strip_base64_images(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_base64_images(item) for item in obj]
    return obj


# Codex CLI payload (OpenAI Responses format, 31 tools, 6 items)
_codex = _load_dump(os.path.join(DUMP_DIR_LOCAL, "error_20260407_134812_648826.json"))
if _codex:
    PAYLOADS["codex_responses_6msg"] = _strip_base64_images(_codex)

# Claude Code payload (Anthropic format, 41 tools, 64 messages)
_cc_medium = _load_dump(os.path.join(DUMP_DIR_LAMBDA, "error_20260410_004259_967174.json"))
if _cc_medium:
    PAYLOADS["claude_code_64msg"] = _strip_base64_images(_cc_medium)

# Claude Code payload (Anthropic format, 41 tools, 218 messages)
_cc_large = _load_dump(os.path.join(DUMP_DIR_LAMBDA, "error_20260410_020434_355345.json"))
if _cc_large:
    PAYLOADS["claude_code_218msg"] = _strip_base64_images(_cc_large)

ITERATIONS = 500  # fewer iterations since payloads are much larger


# ── Benchmark functions ────────────────────────────────────────────────


def bench_rosetta_anthropic_roundtrip(payload: dict, iterations: int) -> list[float]:
    """Benchmark Anthropic → IR → Anthropic roundtrip."""
    from llm_rosetta.converters.anthropic import AnthropicConverter

    conv = AnthropicConverter()

    # Warmup
    for _ in range(5):
        ir_req = conv.request_from_provider(copy.deepcopy(payload))
        conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = conv.request_from_provider(p)
        conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000  # ns → μs
        timings.append(elapsed)
    return timings


def bench_rosetta_responses_to_anthropic(payload: dict, iterations: int) -> list[float]:
    """Benchmark OpenAI Responses → IR → Anthropic conversion."""
    from llm_rosetta.converters.anthropic import AnthropicConverter
    from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter

    resp_conv = OpenAIResponsesConverter()
    anth_conv = AnthropicConverter()

    # Warmup
    for _ in range(5):
        ir_req = resp_conv.request_from_provider(copy.deepcopy(payload))
        anth_conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = resp_conv.request_from_provider(p)
        anth_conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


def bench_rosetta_to_openai_chat(payload: dict, iterations: int) -> list[float]:
    """Benchmark Anthropic → IR → OpenAI Chat conversion."""
    from llm_rosetta.converters.anthropic import AnthropicConverter
    from llm_rosetta.converters.openai_chat import OpenAIChatConverter

    anth_conv = AnthropicConverter()
    chat_conv = OpenAIChatConverter()

    # Warmup
    for _ in range(5):
        ir_req = anth_conv.request_from_provider(copy.deepcopy(payload))
        chat_conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = anth_conv.request_from_provider(p)
        chat_conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


def bench_rosetta_to_google(payload: dict, iterations: int) -> list[float]:
    """Benchmark Anthropic → IR → Google GenAI conversion."""
    from llm_rosetta.converters.anthropic import AnthropicConverter
    from llm_rosetta.converters.google_genai import GoogleGenAIConverter

    anth_conv = AnthropicConverter()
    google_conv = GoogleGenAIConverter()

    # Warmup
    for _ in range(5):
        ir_req = anth_conv.request_from_provider(copy.deepcopy(payload))
        google_conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = anth_conv.request_from_provider(p)
        google_conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


# ── Main ───────────────────────────────────────────────────────────────


def compute_stats(timings: list[float]) -> dict:
    return {
        "median_us": round(statistics.median(timings), 1),
        "p95_us": round(sorted(timings)[int(len(timings) * 0.95)], 1),
        "mean_us": round(statistics.mean(timings), 1),
        "stdev_us": round(statistics.stdev(timings), 1),
    }


def main():
    if not PAYLOADS:
        print("ERROR: No payloads found. Check dump file paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(PAYLOADS)} payloads:")
    for name, payload in PAYLOADS.items():
        n_msgs = len(payload.get("messages", payload.get("input", [])))
        n_tools = len(payload.get("tools", []))
        fmt = "anthropic" if "messages" in payload else "responses"
        print(f"  {name}: {fmt} format, {n_msgs} messages, {n_tools} tools")
    print()

    results = {}

    for payload_name, payload in PAYLOADS.items():
        results[payload_name] = {}
        is_responses = "input" in payload

        if is_responses:
            # Codex CLI payload: Responses → Anthropic
            targets = [
                ("responses_to_anthropic", bench_rosetta_responses_to_anthropic),
            ]
        else:
            # Claude Code payload: Anthropic → various targets
            targets = [
                ("anthropic_roundtrip", bench_rosetta_anthropic_roundtrip),
                ("anthropic_to_openai_chat", bench_rosetta_to_openai_chat),
                ("anthropic_to_google", bench_rosetta_to_google),
            ]

        for target_name, bench_fn in targets:
            print(f"  Benchmarking {payload_name} / {target_name}...", flush=True)
            try:
                timings = bench_fn(payload, ITERATIONS)
                stats = compute_stats(timings)
                results[payload_name][target_name] = stats
                print(
                    f"    median={stats['median_us']:>10.1f} μs  "
                    f"p95={stats['p95_us']:>10.1f} μs  "
                    f"stdev={stats['stdev_us']:>8.1f} μs"
                )
            except Exception as e:
                print(f"    FAILED: {e}")
                results[payload_name][target_name] = {"error": str(e)}

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "real_payload_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

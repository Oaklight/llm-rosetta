"""
Benchmark comparing LLM-Rosetta vs LiteLLM using real-world Claude Code payloads.

Uses OpenAI Chat format payloads (converted from real Anthropic Claude Code dumps)
as the common input, then benchmarks OpenAI Chat → Anthropic conversion for both.
"""

import copy
import json
import logging
import os
import statistics
import sys
import time
import warnings

# Suppress LiteLLM startup noise
warnings.filterwarnings("ignore")
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

ITERATIONS = 200  # large payloads, fewer iterations

PAYLOADS: dict[str, dict] = {}

for label in ["64msg", "218msg"]:
    path = f"/tmp/openai_chat_payload_{label}.json"
    if os.path.exists(path):
        with open(path) as f:
            PAYLOADS[label] = json.load(f)


def bench_litellm(payload: dict, iterations: int) -> list[float]:
    """Benchmark LiteLLM's OpenAI Chat → Anthropic transform_request."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    cfg = AnthropicConfig()
    messages = payload["messages"]
    optional_params = {k: v for k, v in payload.items() if k not in ("model", "messages")}
    optional_params.setdefault("max_tokens", 4096)

    # Warmup
    for _ in range(3):
        cfg.transform_request(
            model="claude-sonnet-4-20250514",
            messages=copy.deepcopy(messages),
            optional_params=copy.deepcopy(optional_params),
            litellm_params={},
            headers={},
        )

    timings = []
    for _ in range(iterations):
        msgs = copy.deepcopy(messages)
        opts = copy.deepcopy(optional_params)
        start = time.perf_counter_ns()
        cfg.transform_request(
            model="claude-sonnet-4-20250514",
            messages=msgs,
            optional_params=opts,
            litellm_params={},
            headers={},
        )
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


def bench_rosetta(payload: dict, iterations: int) -> list[float]:
    """Benchmark Rosetta's OpenAI Chat → IR → Anthropic conversion."""
    from llm_rosetta.converters.anthropic import AnthropicConverter
    from llm_rosetta.converters.openai_chat import OpenAIChatConverter

    oc = OpenAIChatConverter()
    an = AnthropicConverter()

    # Warmup
    for _ in range(3):
        ir_req = oc.request_from_provider(copy.deepcopy(payload))
        an.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = oc.request_from_provider(p)
        an.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


def compute_stats(timings: list[float]) -> dict:
    return {
        "median_us": round(statistics.median(timings), 1),
        "p95_us": round(sorted(timings)[int(len(timings) * 0.95)], 1),
        "mean_us": round(statistics.mean(timings), 1),
        "stdev_us": round(statistics.stdev(timings), 1),
    }


def main():
    if not PAYLOADS:
        print("ERROR: No payloads found. Run the conversion step first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(PAYLOADS)} real-world payloads (OpenAI Chat format):")
    for name, p in PAYLOADS.items():
        print(f"  {name}: {len(p['messages'])} messages, {len(p.get('tools', []))} tools, "
              f"{len(json.dumps(p)) // 1024}KB")
    print()

    results = {}

    for name, payload in PAYLOADS.items():
        print(f"=== {name} ===")

        print("  LiteLLM...", flush=True)
        try:
            lt = bench_litellm(payload, ITERATIONS)
            lt_stats = compute_stats(lt)
            print(f"    median={lt_stats['median_us']:>10.1f} μs  "
                  f"p95={lt_stats['p95_us']:>10.1f} μs")
        except Exception as e:
            print(f"    FAILED: {e}")
            lt_stats = {"error": str(e)}

        print("  Rosetta...", flush=True)
        try:
            rt = bench_rosetta(payload, ITERATIONS)
            rt_stats = compute_stats(rt)
            print(f"    median={rt_stats['median_us']:>10.1f} μs  "
                  f"p95={rt_stats['p95_us']:>10.1f} μs")
        except Exception as e:
            print(f"    FAILED: {e}")
            rt_stats = {"error": str(e)}

        if "error" not in lt_stats and "error" not in rt_stats:
            ratio = rt_stats["median_us"] / lt_stats["median_us"]
            print(f"    Ratio (Rosetta/LiteLLM): {ratio:.2f}x")

        results[name] = {"litellm": lt_stats, "rosetta": rt_stats}
        print()

    output_path = os.path.join(os.path.dirname(__file__), "real_litellm_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

"""
Benchmark comparing LLM-Rosetta vs LiteLLM request conversion latency.

Compares one-directional conversion: OpenAI Chat format → provider-native format.
LiteLLM: transform_request() (OpenAI → Anthropic/Google)
LLM-Rosetta: request_from_provider() then request_to_provider() (OpenAI → IR → target)

This is an apples-to-apples comparison of the format translation step,
excluding HTTP calls, SDK initialization, and provider communication.
"""

import json
import logging
import os
import statistics
import time
import warnings

# Suppress LiteLLM startup noise
warnings.filterwarnings("ignore")
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

# ── Payloads (OpenAI Chat format, the common input) ─────────────────────

SIMPLE_TEXT = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 1024,
}

MULTI_TURN = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what about Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."},
        {"role": "user", "content": "Which one has more people?"},
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
}

TOOL_CALLS = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "max_tokens": 1024,
}

PAYLOADS = {
    "simple_text": SIMPLE_TEXT,
    "multi_turn": MULTI_TURN,
    "tool_calls": TOOL_CALLS,
}

ITERATIONS = 1000


# ── LiteLLM benchmark ───────────────────────────────────────────────────


def bench_litellm_anthropic(payload: dict, iterations: int) -> list[float]:
    """Benchmark LiteLLM's OpenAI → Anthropic transform_request."""
    import copy

    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    cfg = AnthropicConfig()
    messages = copy.deepcopy(payload["messages"])
    optional_params = {
        k: v for k, v in payload.items() if k not in ("model", "messages")
    }
    # Ensure max_tokens is set (Anthropic requires it)
    optional_params.setdefault("max_tokens", 1024)

    # Warmup
    for _ in range(10):
        cfg.transform_request(
            model="claude-3-5-sonnet-20241022",
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
            model="claude-3-5-sonnet-20241022",
            messages=msgs,
            optional_params=opts,
            litellm_params={},
            headers={},
        )
        elapsed = (time.perf_counter_ns() - start) / 1000  # ns → μs
        timings.append(elapsed)
    return timings


def bench_litellm_google(payload: dict, iterations: int) -> list[float]:
    """Benchmark LiteLLM's OpenAI → Google (Vertex Gemini) transform_request."""
    import copy

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    cfg = VertexGeminiConfig()
    messages = copy.deepcopy(payload["messages"])
    optional_params = {
        k: v for k, v in payload.items() if k not in ("model", "messages")
    }
    optional_params.setdefault("max_tokens", 1024)

    # Warmup
    for _ in range(10):
        try:
            cfg.transform_request(
                model="gemini-2.0-flash",
                messages=copy.deepcopy(messages),
                optional_params=copy.deepcopy(optional_params),
                litellm_params={},
                headers={},
            )
        except Exception:
            pass

    timings = []
    for _ in range(iterations):
        msgs = copy.deepcopy(messages)
        opts = copy.deepcopy(optional_params)
        start = time.perf_counter_ns()
        try:
            cfg.transform_request(
                model="gemini-2.0-flash",
                messages=msgs,
                optional_params=opts,
                litellm_params={},
                headers={},
            )
        except Exception:
            pass
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


# ── LLM-Rosetta benchmark ───────────────────────────────────────────────


def bench_rosetta_to_anthropic(payload: dict, iterations: int) -> list[float]:
    """Benchmark LLM-Rosetta's OpenAI Chat → IR → Anthropic conversion."""
    import copy

    from llm_rosetta.converters.anthropic import AnthropicConverter
    from llm_rosetta.converters.openai_chat import OpenAIChatConverter

    oc_conv = OpenAIChatConverter()
    an_conv = AnthropicConverter()

    # Warmup
    for _ in range(10):
        ir_req = oc_conv.request_from_provider(copy.deepcopy(payload))
        an_conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = oc_conv.request_from_provider(p)
        an_conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


def bench_rosetta_to_google(payload: dict, iterations: int) -> list[float]:
    """Benchmark LLM-Rosetta's OpenAI Chat → IR → Google conversion."""
    import copy

    from llm_rosetta.converters.google_genai import GoogleGenAIConverter
    from llm_rosetta.converters.openai_chat import OpenAIChatConverter

    oc_conv = OpenAIChatConverter()
    gg_conv = GoogleGenAIConverter()

    # Warmup
    for _ in range(10):
        ir_req = oc_conv.request_from_provider(copy.deepcopy(payload))
        gg_conv.request_to_provider(ir_req)

    timings = []
    for _ in range(iterations):
        p = copy.deepcopy(payload)
        start = time.perf_counter_ns()
        ir_req = oc_conv.request_from_provider(p)
        gg_conv.request_to_provider(ir_req)
        elapsed = (time.perf_counter_ns() - start) / 1000
        timings.append(elapsed)
    return timings


# ── Main ─────────────────────────────────────────────────────────────────


def compute_stats(timings: list[float]) -> dict:
    return {
        "median_us": round(statistics.median(timings), 1),
        "p95_us": round(sorted(timings)[int(len(timings) * 0.95)], 1),
        "mean_us": round(statistics.mean(timings), 1),
        "stdev_us": round(statistics.stdev(timings), 1),
    }


def main():
    results = {}

    # Google/Vertex transform_request is async-only in LiteLLM, cannot be
    # benchmarked synchronously. Anthropic is the representative comparison.
    targets = [
        ("anthropic", bench_litellm_anthropic, bench_rosetta_to_anthropic),
    ]

    for target_name, litellm_fn, rosetta_fn in targets:
        results[target_name] = {}
        for payload_name, payload in PAYLOADS.items():
            print(f"  Benchmarking {payload_name} → {target_name}...", flush=True)

            litellm_timings = litellm_fn(payload, ITERATIONS)
            rosetta_timings = rosetta_fn(payload, ITERATIONS)

            litellm_stats = compute_stats(litellm_timings)
            rosetta_stats = compute_stats(rosetta_timings)

            results[target_name][payload_name] = {
                "litellm": litellm_stats,
                "rosetta": rosetta_stats,
            }

            print(
                f"    LiteLLM:     median={litellm_stats['median_us']:>8.1f} μs  "
                f"p95={litellm_stats['p95_us']:>8.1f} μs"
            )
            print(
                f"    LLM-Rosetta: median={rosetta_stats['median_us']:>8.1f} μs  "
                f"p95={rosetta_stats['p95_us']:>8.1f} μs"
            )
            ratio = litellm_stats["median_us"] / rosetta_stats["median_us"]
            print(f"    Ratio (LiteLLM/Rosetta): {ratio:.2f}x")

    # Save results
    output_path = "benchmarks/litellm_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print LaTeX table
    print("\n% LaTeX table:")
    print("\\begin{tabular}{@{}llrrrr@{}}")
    print("\\toprule")
    print(
        "\\textbf{Target} & \\textbf{Payload} & "
        "\\textbf{LiteLLM} & \\textbf{\\rosetta} & \\textbf{Ratio} \\\\"
    )
    print("\\midrule")
    for target_name in results.keys():
        label = "Anthropic" if target_name == "anthropic" else "Google"
        for payload_name in ["simple_text", "multi_turn", "tool_calls"]:
            d = results[target_name][payload_name]
            lit = d["litellm"]["median_us"]
            ros = d["rosetta"]["median_us"]
            ratio = lit / ros
            nice_payload = payload_name.replace("_", " ").title()
            print(
                f"{label:10s} & {nice_payload:12s} & "
                f"{int(lit):>5d} & {int(ros):>5d} & {ratio:.1f}$\\times$ \\\\"
            )
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()

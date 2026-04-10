"""Benchmark script for LLM-Rosetta conversion overhead.

Measures per-request and per-chunk latency across 4 providers × 6 payload types.
Outputs JSON results suitable for inclusion in the paper's evaluation section.

Usage:
    python benchmarks/bench_conversion.py [--iterations 1000] [--output results.json]
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

from llm_rosetta.auto_detect import get_converter_for_provider
from llm_rosetta.converters.base.context import ConversionContext

# ---------------------------------------------------------------------------
# Test Payloads
# ---------------------------------------------------------------------------

OPENAI_CHAT_PAYLOADS = {
    "simple_text_request": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    },
    "multi_turn_request": {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            {"role": "user", "content": "What about Germany?"},
            {
                "role": "assistant",
                "content": "The capital of Germany is Berlin.",
            },
            {"role": "user", "content": "And what about Italy?"},
        ],
    },
    "tool_calls_request": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What's the weather in NYC and SF?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ],
        "tool_choice": "auto",
    },
    "reasoning_request": {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Solve step by step: If x^2 + 3x - 10 = 0, what are the values of x?",
            }
        ],
    },
    "simple_text_response": {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 18,
            "total_tokens": 30,
        },
        "system_fingerprint": "fp_abc123",
    },
    "tool_calls_response": {
        "id": "chatcmpl-tool456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC", "unit": "fahrenheit"}',
                            },
                        },
                        {
                            "id": "call_def",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF", "unit": "fahrenheit"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "total_tokens": 120,
        },
    },
}

ANTHROPIC_PAYLOADS = {
    "simple_text_request": {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ],
        "max_tokens": 1024,
    },
    "multi_turn_request": {
        "model": "claude-3-5-sonnet-20241022",
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of France?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The capital of France is Paris."}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What about Germany?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The capital of Germany is Berlin.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "And what about Italy?"}],
            },
        ],
        "max_tokens": 1024,
    },
    "tool_calls_request": {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in NYC and SF?",
                    }
                ],
            }
        ],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    },
    "reasoning_request": {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Solve step by step: If x^2 + 3x - 10 = 0, what are the values of x?",
                    }
                ],
            }
        ],
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    },
    "simple_text_response": {
        "id": "msg_01abc",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm doing well, thank you for asking. How can I help you today?",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 18},
    },
    "tool_calls_response": {
        "id": "msg_tool01",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {"city": "NYC", "unit": "fahrenheit"},
            },
            {
                "type": "tool_use",
                "id": "toolu_def",
                "name": "get_weather",
                "input": {"city": "SF", "unit": "fahrenheit"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 80, "output_tokens": 40},
    },
}

GOOGLE_GENAI_PAYLOADS = {
    "simple_text_request": {
        "model": "gemini-2.0-flash",
        "contents": [{"role": "user", "parts": [{"text": "Hello, how are you?"}]}],
        "config": {},
    },
    "multi_turn_request": {
        "model": "gemini-2.0-flash",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "What is the capital of France?"}],
            },
            {
                "role": "model",
                "parts": [{"text": "The capital of France is Paris."}],
            },
            {
                "role": "user",
                "parts": [{"text": "What about Germany?"}],
            },
            {
                "role": "model",
                "parts": [{"text": "The capital of Germany is Berlin."}],
            },
            {
                "role": "user",
                "parts": [{"text": "And what about Italy?"}],
            },
        ],
        "system_instruction": {"parts": [{"text": "You are a helpful assistant."}]},
        "config": {},
    },
    "tool_calls_request": {
        "model": "gemini-2.0-flash",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "What's the weather in NYC and SF?"}],
            }
        ],
        "config": {
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "get_weather",
                            "description": "Get the current weather in a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["city"],
                            },
                        }
                    ]
                }
            ],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        },
    },
    "reasoning_request": {
        "model": "gemini-2.0-flash-thinking",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Solve step by step: If x^2 + 3x - 10 = 0, what are the values of x?"
                    }
                ],
            }
        ],
        "config": {"thinking_config": {"thinking_budget": 4096}},
    },
    "simple_text_response": {
        "response_id": "resp-abc",
        "model_version": "gemini-2.0-flash",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Hello! I'm doing well, thank you for asking. How can I help you today?"
                        }
                    ],
                },
                "finish_reason": "STOP",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 12,
            "candidates_token_count": 18,
            "total_token_count": 30,
        },
    },
    "tool_calls_response": {
        "response_id": "resp-tool",
        "model_version": "gemini-2.0-flash",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"city": "NYC", "unit": "fahrenheit"},
                            }
                        },
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"city": "SF", "unit": "fahrenheit"},
                            }
                        },
                    ],
                },
                "finish_reason": "STOP",
            }
        ],
    },
}

OPENAI_RESPONSES_PAYLOADS = {
    "simple_text_request": {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, how are you?"}],
            }
        ],
    },
    "multi_turn_request": {
        "model": "gpt-4o",
        "instructions": "You are a helpful assistant.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What is the capital of France?",
                    }
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The capital of France is Paris.",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What about Germany?"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The capital of Germany is Berlin.",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "And what about Italy?"}],
            },
        ],
    },
    "tool_calls_request": {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What's the weather in NYC and SF?",
                    }
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": "auto",
    },
    "reasoning_request": {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Solve step by step: If x^2 + 3x - 10 = 0, what are the values of x?",
                    }
                ],
            }
        ],
        "reasoning": {"effort": "high"},
    },
    "simple_text_response": {
        "id": "resp_abc123",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello! I'm doing well, thank you for asking. How can I help you today?",
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 18,
            "total_tokens": 30,
        },
    },
    "tool_calls_response": {
        "id": "resp_tool456",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "NYC", "unit": "fahrenheit"}',
            },
            {
                "type": "function_call",
                "call_id": "call_def",
                "name": "get_weather",
                "arguments": '{"city": "SF", "unit": "fahrenheit"}',
            },
        ],
        "usage": {
            "input_tokens": 80,
            "output_tokens": 40,
            "total_tokens": 120,
        },
    },
}

ALL_PROVIDERS = {
    "openai_chat": OPENAI_CHAT_PAYLOADS,
    "anthropic": ANTHROPIC_PAYLOADS,
    "google": GOOGLE_GENAI_PAYLOADS,
    "openai_responses": OPENAI_RESPONSES_PAYLOADS,
}

# ---------------------------------------------------------------------------
# Benchmark Helpers
# ---------------------------------------------------------------------------


def benchmark_one(func, payload, iterations):
    """Run a conversion function `iterations` times and return timing stats (ns)."""
    times_ns = []
    for _ in range(iterations):
        data = copy.deepcopy(payload)
        t0 = time.perf_counter_ns()
        func(data)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)
    times_ns.sort()
    return {
        "iterations": iterations,
        "min_us": times_ns[0] / 1000,
        "median_us": statistics.median(times_ns) / 1000,
        "mean_us": statistics.mean(times_ns) / 1000,
        "p95_us": times_ns[int(len(times_ns) * 0.95)] / 1000,
        "p99_us": times_ns[int(len(times_ns) * 0.99)] / 1000,
        "max_us": times_ns[-1] / 1000,
        "stdev_us": (statistics.stdev(times_ns) / 1000 if len(times_ns) > 1 else 0),
    }


def run_benchmarks(iterations=1000):
    """Run all benchmarks and return structured results."""
    results = {}

    for provider_name, payloads in ALL_PROVIDERS.items():
        converter = get_converter_for_provider(provider_name)
        results[provider_name] = {}

        # --- request_from_provider (provider → IR) ---
        for payload_key in [
            "simple_text_request",
            "multi_turn_request",
            "tool_calls_request",
            "reasoning_request",
        ]:
            if payload_key not in payloads:
                continue
            label = f"from_provider/{payload_key}"
            payload = payloads[payload_key]
            payload_size = len(json.dumps(payload))

            def convert_from(data, _conv=converter):
                ctx = ConversionContext()
                _conv.request_from_provider(data, context=ctx)

            stats = benchmark_one(convert_from, payload, iterations)
            stats["payload_bytes"] = payload_size
            stats["direction"] = "from_provider"
            results[provider_name][label] = stats

        # --- request round-trip (provider → IR → provider) ---
        for payload_key in [
            "simple_text_request",
            "multi_turn_request",
            "tool_calls_request",
        ]:
            if payload_key not in payloads:
                continue
            label = f"roundtrip/{payload_key}"
            payload = payloads[payload_key]
            payload_size = len(json.dumps(payload))

            def convert_roundtrip(data, _conv=converter):
                ctx = ConversionContext()
                ir = _conv.request_from_provider(data, context=ctx)
                _conv.request_to_provider(ir, context=ConversionContext())

            stats = benchmark_one(convert_roundtrip, payload, iterations)
            stats["payload_bytes"] = payload_size
            stats["direction"] = "roundtrip"
            results[provider_name][label] = stats

        # --- response_from_provider (provider → IR) ---
        for payload_key in ["simple_text_response", "tool_calls_response"]:
            if payload_key not in payloads:
                continue
            label = f"from_provider/{payload_key}"
            payload = payloads[payload_key]
            payload_size = len(json.dumps(payload))

            def convert_resp_from(data, _conv=converter):
                ctx = ConversionContext()
                _conv.response_from_provider(data, context=ctx)

            stats = benchmark_one(convert_resp_from, payload, iterations)
            stats["payload_bytes"] = payload_size
            stats["direction"] = "from_provider"
            results[provider_name][label] = stats

        # --- response round-trip (provider → IR → provider) ---
        for payload_key in ["simple_text_response", "tool_calls_response"]:
            if payload_key not in payloads:
                continue
            label = f"roundtrip/{payload_key}"
            payload = payloads[payload_key]
            payload_size = len(json.dumps(payload))

            def convert_resp_roundtrip(data, _conv=converter):
                ctx = ConversionContext()
                ir = _conv.response_from_provider(data, context=ctx)
                _conv.response_to_provider(ir, context=ConversionContext())

            stats = benchmark_one(convert_resp_roundtrip, payload, iterations)
            stats["payload_bytes"] = payload_size
            stats["direction"] = "roundtrip"
            results[provider_name][label] = stats

    return results


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


def print_summary(results):
    """Print a human-readable summary table."""
    print(f"\n{'=' * 100}")
    print(
        f"{'Provider':<20} {'Benchmark':<40} {'Median(us)':>10} {'P95(us)':>10} {'P99(us)':>10} {'Bytes':>8}"
    )
    print(f"{'=' * 100}")

    for provider, benchmarks in results.items():
        for label, stats in benchmarks.items():
            print(
                f"{provider:<20} {label:<40} "
                f"{stats['median_us']:>10.1f} "
                f"{stats['p95_us']:>10.1f} "
                f"{stats['p99_us']:>10.1f} "
                f"{stats['payload_bytes']:>8}"
            )
        print(f"{'-' * 100}")


def export_latex_table(results):
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{tabular}{@{}llrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Provider} & \textbf{Payload} & "
        r"\textbf{Median ($\mu$s)} & \textbf{P95 ($\mu$s)} & "
        r"\textbf{P99 ($\mu$s)} & \textbf{Size (B)} \\"
    )
    lines.append(r"\midrule")

    for provider, benchmarks in results.items():
        provider_label = {
            "openai_chat": "OpenAI Chat",
            "anthropic": "Anthropic",
            "google": "Google GenAI",
            "openai_responses": "OAI Responses",
        }.get(provider, provider)

        first = True
        for label, stats in benchmarks.items():
            p = provider_label if first else ""
            first = False
            short_label = (
                label.replace("from_provider/", "from: ")
                .replace("roundtrip/", "rt: ")
                .replace("_request", " req")
                .replace("_response", " resp")
                .replace("simple_text", "text")
                .replace("multi_turn", "multi")
                .replace("tool_calls", "tools")
                .replace("reasoning", "reason")
            )
            lines.append(
                f"  {p} & \\code{{{short_label}}} & "
                f"{stats['median_us']:.1f} & {stats['p95_us']:.1f} & "
                f"{stats['p99_us']:.1f} & {stats['payload_bytes']} \\\\"
            )
        lines.append(r"\midrule")

    # remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LLM-Rosetta conversion benchmarks")
    parser.add_argument(
        "--iterations", "-n", type=int, default=1000, help="Iterations per benchmark"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    args = parser.parse_args()

    print(f"Running benchmarks with {args.iterations} iterations per case...")
    results = run_benchmarks(iterations=args.iterations)

    print_summary(results)

    if args.latex:
        print("\n% LaTeX table:")
        print(export_latex_table(results))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

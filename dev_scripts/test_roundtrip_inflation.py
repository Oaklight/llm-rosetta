#!/usr/bin/env python3
"""Test streaming round-trip event inflation for all providers.

For each provider, sends a realistic SSE event sequence through:
  stream_response_from_provider → IR events → stream_response_to_provider

Compares input event count vs output event count to detect inflation.

Usage:
    python dev_scripts/test_roundtrip_inflation.py
"""

from __future__ import annotations

import json
from typing import Any

from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.base.context import StreamContext


def run_roundtrip(
    provider: str,
    input_events: list[dict[str, Any]],
    label: str = "",
) -> list[dict[str, Any]]:
    """Run a streaming round-trip and return output events."""
    converter = get_converter_for_provider(provider)
    from_ctx = StreamContext()
    to_ctx = StreamContext()

    output_events: list[dict[str, Any]] = []

    for inp in input_events:
        ir_events = converter.stream_response_from_provider(inp, context=from_ctx)
        for ir_event in ir_events:
            result = converter.stream_response_to_provider(ir_event, context=to_ctx)
            if isinstance(result, list):
                output_events.extend(e for e in result if e)
            elif result:
                output_events.append(result)

    return output_events


def _event_label(provider: str, e: dict[str, Any]) -> str:
    """Extract a meaningful label from a provider event."""
    t = e.get("type")
    if t:
        return t
    # OpenAI Chat: describe by choices/usage structure
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
            return f"content_delta"
        if "tool_calls" in delta:
            return "tool_calls_delta"
        return "choice_chunk"
    # Google: describe by candidates structure
    if "candidates" in e:
        cand = e["candidates"][0] if e.get("candidates") else {}
        if cand.get("finishReason"):
            return f"finish({cand['finishReason']})"
        parts = cand.get("content", {}).get("parts", [])
        if parts:
            return "text_chunk"
        return "candidate_chunk"
    return "unknown"


def print_comparison(
    provider: str,
    input_events: list[dict[str, Any]],
    output_events: list[dict[str, Any]],
) -> bool:
    """Print comparison and return True if inflated."""
    in_types = [_event_label(provider, e) for e in input_events]
    out_types = [_event_label(provider, e) for e in output_events]

    inflated = len(out_types) != len(in_types) or in_types != out_types

    status = "INFLATED" if inflated else "OK"
    print(f"\n{'='*60}")
    print(f"  {provider}: {len(in_types)} → {len(out_types)} events  [{status}]")
    print(f"{'='*60}")
    print(f"  INPUT  ({len(in_types)}): {in_types}")
    print(f"  OUTPUT ({len(out_types)}): {out_types}")

    if inflated:
        # Show detailed diff
        max_len = max(len(in_types), len(out_types))
        print(f"\n  {'#':>3}  {'INPUT':<30} {'OUTPUT':<30} {'DIFF'}")
        print(f"  {'---':>3}  {'-'*30} {'-'*30} {'----'}")
        for i in range(max_len):
            inp = in_types[i] if i < len(in_types) else "(none)"
            out = out_types[i] if i < len(out_types) else "(none)"
            diff = "<<<" if inp != out else ""
            print(f"  {i:>3}  {inp:<30} {out:<30} {diff}")

    return inflated


# ============================================================
# Anthropic
# ============================================================
ANTHROPIC_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_001",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " world!"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 5},
    },
    {"type": "message_stop"},
]

# ============================================================
# OpenAI Chat
# ============================================================
OPENAI_CHAT_EVENTS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"content": " world!"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
        },
    },
]

# ============================================================
# OpenAI Responses
# ============================================================
OPENAI_RESPONSES_EVENTS: list[dict[str, Any]] = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_001",
            "object": "response",
            "model": "gpt-4o",
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [],
        },
    },
    {
        "type": "response.content_part.added",
        "item_id": "msg_001",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_001",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello",
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_001",
        "output_index": 0,
        "content_index": 0,
        "delta": " world!",
    },
    {
        "type": "response.output_text.done",
        "item_id": "msg_001",
        "output_index": 0,
        "content_index": 0,
        "text": "Hello world!",
    },
    {
        "type": "response.content_part.done",
        "item_id": "msg_001",
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "Hello world!",
            "annotations": [],
        },
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello world!",
                    "annotations": [],
                }
            ],
        },
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_001",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_001",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello world!",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
                "total_tokens": 17,
            },
        },
    },
]

# ============================================================
# Google GenAI
# ============================================================
GOOGLE_EVENTS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
                "index": 0,
            }
        ],
        "modelVersion": "gemini-2.0-flash",
    },
    {
        "candidates": [
            {
                "content": {"parts": [{"text": " world!"}], "role": "model"},
                "index": 0,
            }
        ],
    },
    {
        "candidates": [
            {
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 5,
            "totalTokenCount": 17,
        },
        "modelVersion": "gemini-2.0-flash",
    },
]


# ============================================================
# Edge Cases: Anthropic with thinking + tool call
# ============================================================
ANTHROPIC_THINKING_TOOL_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_002",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 50, "output_tokens": 1},
        },
    },
    # Block 0: thinking
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Let me search for"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": " the answer."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig_abc123"},
    },
    {"type": "content_block_stop", "index": 0},
    # Block 1: tool_use
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "web_search",
            "input": {},
        },
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '"hello"}'},
    },
    {"type": "content_block_stop", "index": 1},
    # Finish
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 42},
    },
    {"type": "message_stop"},
]

# ============================================================
# Edge Cases: Anthropic with no usage in message_start
# ============================================================
ANTHROPIC_NO_INITIAL_USAGE_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_003",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": None,
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]

# ============================================================
# Edge Cases: Anthropic finish without usage (stop only)
# ============================================================
ANTHROPIC_FINISH_NO_USAGE_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_004",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 5, "output_tokens": 0},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "OK"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
    },
    {"type": "message_stop"},
]

# ============================================================
# Edge Cases: OpenAI Chat with tool calls
# ============================================================
OPENAI_CHAT_TOOL_EVENTS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"city":"NYC"}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        },
    },
]

# ============================================================
# Edge Cases: OpenAI Chat with reasoning_content
# ============================================================
OPENAI_CHAT_REASONING_EVENTS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-003",
        "object": "chat.completion.chunk",
        "model": "o3-mini",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-003",
        "object": "chat.completion.chunk",
        "model": "o3-mini",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": "Let me think..."},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-003",
        "object": "chat.completion.chunk",
        "model": "o3-mini",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "The answer is 42."},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-003",
        "object": "chat.completion.chunk",
        "model": "o3-mini",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-003",
        "object": "chat.completion.chunk",
        "model": "o3-mini",
        "created": 1700000000,
        "choices": [],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50,
        },
    },
]

# ============================================================
# Edge Cases: Google with tool call (functionCall)
# ============================================================
GOOGLE_TOOL_EVENTS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "NYC"},
                            }
                        }
                    ],
                    "role": "model",
                },
                "index": 0,
            }
        ],
        "modelVersion": "gemini-2.0-flash",
    },
    {
        "candidates": [
            {
                "content": {"parts": [], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 25,
            "candidatesTokenCount": 10,
            "totalTokenCount": 35,
        },
        "modelVersion": "gemini-2.0-flash",
    },
]

# ============================================================
# Edge Cases: Google with thinking (thought parts)
# ============================================================
GOOGLE_THINKING_EVENTS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Analyzing...", "thought": True}],
                    "role": "model",
                },
                "index": 0,
            }
        ],
        "modelVersion": "gemini-2.5-flash",
    },
    {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "The answer is 42."}],
                    "role": "model",
                },
                "index": 0,
            }
        ],
    },
    {
        "candidates": [
            {
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 20,
            "totalTokenCount": 35,
            "thoughtsTokenCount": 10,
        },
        "modelVersion": "gemini-2.5-flash",
    },
]

# ============================================================
# Edge Cases: Google finish without empty text padding
# ============================================================
GOOGLE_NO_EMPTY_TEXT_EVENTS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                "index": 0,
            }
        ],
        "modelVersion": "gemini-2.0-flash",
    },
    {
        "candidates": [
            {
                "content": {"parts": [], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
        },
    },
]

# ============================================================
# Edge Cases: OpenAI Responses with tool call
# ============================================================
OPENAI_RESPONSES_TOOL_EVENTS: list[dict[str, Any]] = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_002",
            "object": "response",
            "model": "gpt-4o",
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_001",
            "call_id": "call_xyz",
            "name": "get_weather",
            "arguments": "",
        },
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_001",
        "output_index": 0,
        "delta": '{"city":',
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_001",
        "output_index": 0,
        "delta": '"NYC"}',
    },
    {
        "type": "response.function_call_arguments.done",
        "item_id": "fc_001",
        "output_index": 0,
        "arguments": '{"city":"NYC"}',
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_001",
            "call_id": "call_xyz",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
        },
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_002",
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_xyz",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
            },
        },
    },
]


# ============================================================
# Edge Case: OpenAI Chat role chunk with empty content
# (gpt-5-nano sends delta: {role: "assistant", content: "", refusal: null})
# ============================================================
OPENAI_CHAT_ROLE_EMPTY_CONTENT_EVENTS: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "", "refusal": None},
                "finish_reason": None,
            }
        ],
        "usage": None,
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"content": "2 + 2"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"content": " = 4."}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    },
]

# ============================================================
# Edge Case: Google compound text+finish chunk
# (gemini-2.5-flash returns text AND finishReason in the same chunk)
# ============================================================
GOOGLE_COMPOUND_TEXT_FINISH_EVENTS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "2 plus "}], "role": "model"},
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 3,
            "totalTokenCount": 9,
        },
        "modelVersion": "gemini-2.5-flash",
    },
    {
        "candidates": [
            {
                "content": {"parts": [{"text": "2 equals 4."}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 8,
            "totalTokenCount": 14,
        },
        "modelVersion": "gemini-2.5-flash",
    },
]


def run_test_case(
    label: str,
    provider: str,
    events: list[dict[str, Any]],
    results: dict[str, bool],
) -> None:
    """Run a single test case and record the result."""
    try:
        output = run_roundtrip(provider, events)
        inflated = print_comparison(f"{provider} ({label})", events, output)
        key = f"{provider}/{label}"
        results[key] = inflated
    except Exception as exc:
        key = f"{provider}/{label}"
        print(f"\n{'='*60}")
        print(f"  {key}: ERROR — {exc}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        results[key] = True


def main() -> None:
    results: dict[str, bool] = {}

    # Basic cases
    basic_cases = [
        ("basic", "anthropic", ANTHROPIC_EVENTS),
        ("basic", "openai_chat", OPENAI_CHAT_EVENTS),
        ("basic", "openai_responses", OPENAI_RESPONSES_EVENTS),
        ("basic", "google", GOOGLE_EVENTS),
    ]

    # Edge cases
    edge_cases = [
        ("thinking+tool", "anthropic", ANTHROPIC_THINKING_TOOL_EVENTS),
        ("no-initial-usage", "anthropic", ANTHROPIC_NO_INITIAL_USAGE_EVENTS),
        ("finish-no-usage", "anthropic", ANTHROPIC_FINISH_NO_USAGE_EVENTS),
        ("tool-calls", "openai_chat", OPENAI_CHAT_TOOL_EVENTS),
        ("reasoning", "openai_chat", OPENAI_CHAT_REASONING_EVENTS),
        ("tool-call", "google", GOOGLE_TOOL_EVENTS),
        ("thinking", "google", GOOGLE_THINKING_EVENTS),
        ("no-empty-text", "google", GOOGLE_NO_EMPTY_TEXT_EVENTS),
        ("role-empty-content", "openai_chat", OPENAI_CHAT_ROLE_EMPTY_CONTENT_EVENTS),
        ("compound-text-finish", "google", GOOGLE_COMPOUND_TEXT_FINISH_EVENTS),
        ("tool-call", "openai_responses", OPENAI_RESPONSES_TOOL_EVENTS),
    ]

    print("\n" + "=" * 60)
    print("  BASIC CASES")
    print("=" * 60)
    for label, provider, events in basic_cases:
        run_test_case(label, provider, events, results)

    print("\n\n" + "=" * 60)
    print("  EDGE CASES")
    print("=" * 60)
    for label, provider, events in edge_cases:
        run_test_case(label, provider, events, results)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    any_failed = False
    for key, inflated in results.items():
        status = "INFLATED" if inflated else "OK"
        if inflated:
            any_failed = True
        print(f"  {key:<40} {status}")

    if any_failed:
        print(f"\n  *** SOME TESTS FAILED ***")
    else:
        print(f"\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()

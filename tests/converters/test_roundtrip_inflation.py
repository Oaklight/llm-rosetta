"""Streaming round-trip inflation regression tests.

For each provider, sends a realistic SSE event sequence through:
  stream_response_from_provider → IR events → stream_response_to_provider

Verifies that the output event count does NOT exceed the input event count
(no inflation).  Deflation (output < input) is acceptable because compound
chunks may legitimately merge during round-trip.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta import get_converter_for_provider
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.converters.base.context import StreamContext


# ============================================================
# Helper
# ============================================================


def run_roundtrip(
    provider: ProviderType,
    input_events: list[dict[str, Any]],
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


# ============================================================
# Fixtures — Anthropic
# ============================================================

ANTHROPIC_BASIC: list[dict[str, Any]] = [
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

ANTHROPIC_THINKING_TOOL: list[dict[str, Any]] = [
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
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 42},
    },
    {"type": "message_stop"},
]

ANTHROPIC_NO_INITIAL_USAGE: list[dict[str, Any]] = [
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

ANTHROPIC_FINISH_NO_USAGE: list[dict[str, Any]] = [
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
# Fixtures — OpenAI Chat
# ============================================================

OPENAI_CHAT_BASIC: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-001",
        "object": "chat.completion.chunk",
        "model": "gpt-4o",
        "created": 1700000000,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
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

OPENAI_CHAT_TOOL_CALLS: list[dict[str, Any]] = [
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
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
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

OPENAI_CHAT_REASONING: list[dict[str, Any]] = [
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

OPENAI_CHAT_ROLE_EMPTY_CONTENT: list[dict[str, Any]] = [
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
        "choices": [{"index": 0, "delta": {"content": "2 + 2"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-002",
        "object": "chat.completion.chunk",
        "model": "gpt-5-nano",
        "created": 1700000000,
        "choices": [{"index": 0, "delta": {"content": " = 4."}, "finish_reason": None}],
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
# Fixtures — OpenAI Responses
# ============================================================

OPENAI_RESPONSES_BASIC: list[dict[str, Any]] = [
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

OPENAI_RESPONSES_TOOL_CALL: list[dict[str, Any]] = [
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
# Fixtures — Google GenAI
# ============================================================

GOOGLE_BASIC: list[dict[str, Any]] = [
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

GOOGLE_TOOL_CALL: list[dict[str, Any]] = [
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

GOOGLE_THINKING: list[dict[str, Any]] = [
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

GOOGLE_NO_EMPTY_TEXT: list[dict[str, Any]] = [
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

GOOGLE_COMPOUND_TEXT_FINISH: list[dict[str, Any]] = [
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


# ============================================================
# Parametrized test
# ============================================================

ROUNDTRIP_CASES = [
    # Basic cases
    ("anthropic", "basic", ANTHROPIC_BASIC),
    ("openai_chat", "basic", OPENAI_CHAT_BASIC),
    ("openai_responses", "basic", OPENAI_RESPONSES_BASIC),
    ("google", "basic", GOOGLE_BASIC),
    # Anthropic edge cases
    ("anthropic", "thinking+tool", ANTHROPIC_THINKING_TOOL),
    ("anthropic", "no-initial-usage", ANTHROPIC_NO_INITIAL_USAGE),
    ("anthropic", "finish-no-usage", ANTHROPIC_FINISH_NO_USAGE),
    # OpenAI Chat edge cases
    ("openai_chat", "tool-calls", OPENAI_CHAT_TOOL_CALLS),
    ("openai_chat", "reasoning", OPENAI_CHAT_REASONING),
    ("openai_chat", "role-empty-content", OPENAI_CHAT_ROLE_EMPTY_CONTENT),
    # Google edge cases
    ("google", "tool-call", GOOGLE_TOOL_CALL),
    ("google", "thinking", GOOGLE_THINKING),
    ("google", "no-empty-text", GOOGLE_NO_EMPTY_TEXT),
    ("google", "compound-text-finish", GOOGLE_COMPOUND_TEXT_FINISH),
    # OpenAI Responses edge cases
    ("openai_responses", "tool-call", OPENAI_RESPONSES_TOOL_CALL),
]


@pytest.mark.parametrize(
    ("provider", "label", "input_events"),
    ROUNDTRIP_CASES,
    ids=[f"{p}/{label}" for p, label, _ in ROUNDTRIP_CASES],
)
def test_no_roundtrip_inflation(
    provider: ProviderType,
    label: str,
    input_events: list[dict[str, Any]],
) -> None:
    """Output event count must not exceed input event count."""
    output_events = run_roundtrip(provider, input_events)
    assert len(output_events) <= len(input_events), (
        f"{provider}/{label}: inflated {len(input_events)} → {len(output_events)} events"
    )

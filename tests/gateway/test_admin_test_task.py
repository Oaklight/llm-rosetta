"""The admin model test stops at the tool call.

An ACP lane ends a turn with a tool call; the client's next request carries
the result and resumes the same conversation.  The test is that first
request, so it declares a conversation and returns the call instead of
inventing a result.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from llm_rosetta_gateway._vendor import httpclient
from llm_rosetta_gateway.admin.routes import testing
from llm_rosetta_gateway._vendor.httpclient import (
    CaseInsensitiveDict,
    Response as HttpResponse,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def _payload() -> dict[str, Any]:
    return {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": TOOLS,
    }


def _tool_call_body() -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    ).encode()


def _text_body(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


@pytest.fixture(autouse=True)
def _clean_tasks():
    testing._test_tasks.clear()
    yield
    testing._test_tasks.clear()


def _client(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[Any]:
    """Install a fake self-call client and return its recorded requests."""
    calls: list[Any] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def post(
            self, url: str, json: Any = None, headers: Any = None
        ) -> HttpResponse:
            calls.append((url, json, dict(headers or {})))
            reply = responses[min(len(calls), len(responses)) - 1]
            return reply

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(httpclient, "AsyncClient", _Client)
    return calls


async def _run(payload: dict[str, Any]) -> None:
    testing._test_tasks["task-1"] = {
        "started": time.monotonic(),
        "_base_url": "http://127.0.0.1:1",
    }
    await testing._run_test_task(
        "task-1", "/v1/chat/completions", payload, "internal", None
    )


@pytest.mark.asyncio
async def test_a_tool_call_is_shown_and_the_turn_is_released(monkeypatch):
    calls = _client(
        monkeypatch,
        [
            HttpResponse(200, CaseInsensitiveDict(), _tool_call_body(), "http://x"),
            HttpResponse(
                200,
                CaseInsensitiveDict(),
                _text_body("It is 18C and sunny."),
                "http://x",
            ),
        ],
    )

    await _run(_payload())

    assert len(calls) == 2
    _, first, headers = calls[0]
    assert headers["x-claude-code-session-id"] == "admin-test-task-1"
    assert first["tools"] == TOOLS
    _, follow, _ = calls[1]
    assert follow["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": testing._TEST_TOOL_RESULT,
    }
    task = testing._test_tasks["task-1"]
    assert task["status"] == "done"
    assert task["status_code"] == 200
    assert task["body"]["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
    ]
    assert task["body"]["choices"][0]["message"]["content"] is None


@pytest.mark.asyncio
async def test_a_request_without_tools_declares_no_session(monkeypatch):
    calls = _client(
        monkeypatch,
        [HttpResponse(200, CaseInsensitiveDict(), _text_body("ok"), "http://x")],
    )

    await _run({"model": "m", "messages": [{"role": "user", "content": "hi"}]})

    assert len(calls) == 1
    assert "x-claude-code-session-id" not in calls[0][2]


@pytest.mark.asyncio
async def test_a_failed_turn_is_reported_as_the_result(monkeypatch):
    error = json.dumps({"error": "the agent is not signed in"}).encode()
    calls = _client(
        monkeypatch, [HttpResponse(400, CaseInsensitiveDict(), error, "http://x")]
    )

    await _run(_payload())

    assert len(calls) == 1
    assert testing._test_tasks["task-1"]["status"] == "done"
    assert testing._test_tasks["task-1"]["status_code"] == 400
    assert testing._test_tasks["task-1"]["body"] == {
        "error": "the agent is not signed in"
    }


@pytest.mark.asyncio
async def test_the_follow_up_loop_is_bounded(monkeypatch):
    calls = _client(
        monkeypatch,
        [HttpResponse(200, CaseInsensitiveDict(), _tool_call_body(), "http://x")],
    )

    await _run(_payload())

    assert len(calls) == testing._MAX_TEST_TOOL_TURNS
    assert testing._test_tasks["task-1"]["body"]["choices"][0]["message"]["tool_calls"]


@pytest.mark.asyncio
async def test_a_follow_up_error_outranks_the_call(monkeypatch):
    """A lane that cannot carry the result back must not look healthy."""
    error = json.dumps({"error": "the tool result could not be carried"}).encode()
    calls = _client(
        monkeypatch,
        [
            HttpResponse(200, CaseInsensitiveDict(), _tool_call_body(), "http://x"),
            HttpResponse(400, CaseInsensitiveDict(), error, "http://x"),
        ],
    )

    await _run(_payload())

    assert len(calls) == 2
    assert testing._test_tasks["task-1"]["status_code"] == 400
    assert testing._test_tasks["task-1"]["body"] == {
        "error": "the tool result could not be carried"
    }

"""The admin model test answers a client tool call instead of suspending.

An ACP lane keeps the agent's turn open while a client tool call is pending,
so a test that stopped at the call would hold the session for its whole TTL.
The task follows the call with a synthetic result and lets the turn finish.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

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

    monkeypatch.setattr(testing, "AsyncClient", _Client)
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
async def test_a_tool_call_is_answered_and_the_turn_finishes(monkeypatch):
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
    _, follow, _ = calls[1]
    assert follow["messages"][-2] == {
        "role": "assistant",
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
    assert follow["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "18C and sunny",
    }
    assert testing._test_tasks["task-1"]["status"] == "done"
    assert (
        testing._test_tasks["task-1"]["body"]["choices"][0]["message"]["content"]
        == "It is 18C and sunny."
    )


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
async def test_a_failed_turn_is_not_followed(monkeypatch):
    error = json.dumps({"error": "the agent is not signed in"}).encode()
    calls = _client(
        monkeypatch, [HttpResponse(400, CaseInsensitiveDict(), error, "http://x")]
    )

    await _run(_payload())

    assert len(calls) == 1
    assert testing._test_tasks["task-1"]["status"] == "done"
    assert testing._test_tasks["task-1"]["status_code"] == 400


@pytest.mark.asyncio
async def test_the_follow_up_loop_is_bounded(monkeypatch):
    calls = _client(
        monkeypatch,
        [HttpResponse(200, CaseInsensitiveDict(), _tool_call_body(), "http://x")],
    )

    await _run(_payload())

    assert len(calls) == testing._MAX_TEST_TOOL_TURNS

"""Tests for the ``live`` conformance suite.

The network is never touched: a lane probe takes its transport as an
argument, so every assertion here is about the request the suite builds, the
status it turns into which result, and what it is willing to call an answer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tools.conformance import ProbeFailure
from tools.conformance.model import ProbeSkip
from tools.conformance.suites import SUITES
from tools.conformance.suites.live import (
    LANES,
    MAX_TOKENS,
    PROMPT,
    _LaneProbe,
    build,
    configured_lanes,
)

_BY_NAME = {lane.name: lane for lane in LANES}


@pytest.fixture(autouse=True)
def _clear_lane_env(monkeypatch: pytest.MonkeyPatch):
    """Start every test with no lane configured, whatever the shell holds."""
    for lane in LANES:
        for name in (lane.key_env, lane.model_env, lane.base_env):
            if name:
                monkeypatch.delenv(name, raising=False)
    yield


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    lane_name: str,
    *,
    key: str = "test-secret",
    model: str = "test-model",
    base: str | None = None,
) -> None:
    lane = _BY_NAME[lane_name]
    monkeypatch.setenv(lane.key_env, key)
    monkeypatch.setenv(lane.model_env, model)
    if base is not None:
        if lane.base_env is None:
            raise AssertionError(f"{lane_name} has no base env to set")
        monkeypatch.setenv(lane.base_env, base)


class _Transport:
    """A recording stand-in for the HTTP post."""

    def __init__(self, status: int = 200, text: str = "{}") -> None:
        self.status = status
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float,
    ) -> tuple[int, str]:
        self.calls.append(
            {"url": url, "headers": headers, "body": body, "timeout": timeout}
        )
        return self.status, self.text


def _chat_answer(text: str = "ok") -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _messages_answer(text: str = "ok") -> str:
    return json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _google_answer(text: str = "ok") -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
    )


class TestConfiguration:
    def test_every_lane_has_a_catalog_entry(self):
        from llm_rosetta.formats.catalog import BY_PAIR

        for lane in LANES:
            assert ("llm", lane.format) in BY_PAIR, lane.name

    @pytest.mark.parametrize("lane", LANES, ids=lambda lane: lane.name)
    def test_an_unconfigured_lane_is_skipped(self, lane):
        with pytest.raises(ProbeSkip, match=lane.key_env):
            _LaneProbe(lane, post=_Transport()).run()

    def test_a_missing_model_names_its_variable(self, monkeypatch):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        monkeypatch.delenv("ANTHROPIC_MODEL")

        with pytest.raises(ProbeSkip, match="ANTHROPIC_MODEL"):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=_Transport()).run()

    def test_a_missing_base_names_it(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai")

        with pytest.raises(ProbeSkip, match="OPENAI_BASE_URL"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=_Transport()).run()

    def test_a_lane_with_a_default_base_needs_no_base_env(self, monkeypatch):
        _configure(monkeypatch, "google_generate/google")
        transport = _Transport(text=_google_answer())

        data = _LaneProbe(_BY_NAME["google_generate/google"], post=transport).run()

        assert transport.calls[0]["url"].startswith(
            "https://generativelanguage.googleapis.com/v1beta/models/"
        )
        assert data["text_chars"] == 2

    def test_only_complete_lanes_are_listed(self, monkeypatch):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        _configure(monkeypatch, "chat_completions/openai")

        assert configured_lanes() == ["messages/anthropic"]


class TestRequest:
    def test_the_request_asks_for_a_usable_budget(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(text=_chat_answer())

        _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

        body = transport.calls[0]["body"]
        assert body["model"] == "test-model"
        # A single text part is flattened to a string by the chat writer.
        assert body["messages"][0]["content"] == PROMPT
        budget = body.get("max_tokens", body.get("max_completion_tokens"))
        assert budget == MAX_TOKENS

    @pytest.mark.parametrize(
        ("lane_name", "base", "url_tail"),
        [
            (
                "chat_completions/openai",
                "https://o.example/v1",
                "/v1/chat/completions",
            ),
            (
                "open_responses/openai",
                "https://o.example/v1",
                "/v1/responses",
            ),
            (
                "messages/anthropic",
                "https://a.example",
                "/v1/messages",
            ),
            (
                "google_interactions/google",
                "https://g.example",
                "/v1beta/interactions",
            ),
        ],
    )
    def test_the_url_comes_from_the_catalog(
        self, monkeypatch, lane_name, base, url_tail
    ):
        _configure(monkeypatch, lane_name, base=base)
        transport = _Transport(status=500, text="boom")

        with pytest.raises(RuntimeError):
            _LaneProbe(_BY_NAME[lane_name], post=transport).run()

        assert transport.calls[0]["url"].endswith(url_tail)

    @pytest.mark.parametrize(
        ("lane_name", "expected"),
        [
            ("chat_completions/openai", {"Authorization": "Bearer test-secret"}),
            (
                "messages/anthropic",
                {"x-api-key": "test-secret", "anthropic-version": "2023-06-01"},
            ),
            ("google_interactions/google", {"x-goog-api-key": "test-secret"}),
        ],
    )
    def test_the_credential_goes_in_the_headers_the_catalog_names(
        self, monkeypatch, lane_name, expected
    ):
        _configure(monkeypatch, lane_name, base="https://x.example")
        transport = _Transport(status=500, text="boom")

        with pytest.raises(RuntimeError):
            _LaneProbe(_BY_NAME[lane_name], post=transport).run()

        headers = transport.calls[0]["headers"]
        for name, value in expected.items():
            assert headers[name] == value

    def test_google_generate_is_probed_in_the_rest_shape(self, monkeypatch):
        _configure(monkeypatch, "google_generate/google")
        transport = _Transport(text=_google_answer())

        _LaneProbe(_BY_NAME["google_generate/google"], post=transport).run()

        body = transport.calls[0]["body"]
        assert "generationConfig" in body
        assert "config" not in body


class TestOutcome:
    def test_a_good_answer_passes_and_reports_the_echo(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(text=_chat_answer())

        data = _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

        assert data["http_status"] == 200
        assert data["requested_model"] == "test-model"
        assert data["echoed_model"] == "gpt-test"
        assert data["finish_reason"] == "stop"
        assert data["text_chars"] == 2
        assert data["id_prefix"] is None
        assert data["usage_keys"] == [
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
        ]

    def test_a_messages_answer_reads_back(self, monkeypatch):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(text=_messages_answer("ok"))

        data = _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

        assert data["echoed_model"] == "claude-test"
        assert data["id_prefix"] == "msg"

    def test_the_credential_never_reaches_the_report(self, monkeypatch):
        _configure(
            monkeypatch,
            "chat_completions/openai",
            key="half-secret-value",
            base="https://o.example/v1",
        )
        transport = _Transport(text=_chat_answer())

        data = _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

        assert "half-secret-value" not in json.dumps(data)

    def test_a_400_that_names_a_body_field_is_a_failure(self, monkeypatch):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(
            status=400, text='{"error": {"message": "Invalid value for max_tokens"}}'
        )

        with pytest.raises(ProbeFailure, match="HTTP 400"):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

    def test_a_400_that_names_the_credential_is_an_error(self, monkeypatch):
        """Google answers a revoked key with 400, not 401."""
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(status=400, text='{"error": "API key not valid"}')

        with pytest.raises(RuntimeError, match="HTTP 400"):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

    def test_a_400_that_names_no_body_field_is_an_error(self, monkeypatch):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(status=400, text='{"error": "Request blocked"}')

        with pytest.raises(RuntimeError):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

    def test_a_400_that_names_the_model_is_an_error(self, monkeypatch):
        """A stale `*_MODEL` reaches the marker list, not the body-key match."""
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(
            status=400, text='{"error": "Invalid model: gpt-9 is not available"}'
        )

        with pytest.raises(RuntimeError, match="HTTP 400"):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

    def test_a_5xx_that_echoes_a_field_is_an_error(self, monkeypatch):
        """The 400 test is what may fail a case; an outage never is."""
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(
            status=500, text='{"error": "max_completion_tokens must be positive"}'
        )

        with pytest.raises(RuntimeError, match="HTTP 500"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

    def test_a_404_for_a_moved_model_is_an_error(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(
            status=404, text='{"error": {"message": "The model `nope` does not exist"}}'
        )

        with pytest.raises(RuntimeError, match="HTTP 404"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

    @pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
    def test_a_run_the_vendor_could_not_serve_is_an_error(self, monkeypatch, status):
        _configure(monkeypatch, "messages/anthropic", base="https://a.example")
        transport = _Transport(status=status, text="nope")

        with pytest.raises(RuntimeError):
            _LaneProbe(_BY_NAME["messages/anthropic"], post=transport).run()

    def test_an_answer_without_text_is_a_failure(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(text=_chat_answer(""))

        with pytest.raises(ProbeFailure, match="no text"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

    def test_an_answer_without_choices_is_a_failure(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(text=json.dumps({"error": "not a completion"}))

        with pytest.raises(ProbeFailure, match="no text"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()

    def test_a_non_json_answer_is_a_failure(self, monkeypatch):
        _configure(monkeypatch, "chat_completions/openai", base="https://o.example/v1")
        transport = _Transport(text="<html>gateway</html>")

        with pytest.raises(ProbeFailure, match="not JSON"):
            _LaneProbe(_BY_NAME["chat_completions/openai"], post=transport).run()


class TestBuild:
    def test_the_suite_is_registered(self):
        assert "live" in SUITES

    def test_the_suite_names_every_lane(self):
        probes, environment = build()

        assert [probe.name for probe in probes] == [lane.name for lane in LANES]
        assert environment["lanes"] == [lane.name for lane in LANES]
        assert environment["configured"] == configured_lanes()

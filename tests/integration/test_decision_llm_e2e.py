"""
Decision Converter End-to-End Integration Test

Tests the LLMChatDecisionConverter against real LLM APIs across multiple
providers and output formats, verifying that decision requests produce
valid typed answers (noul/choice/score) with proper probability distributions.

Requires API keys in .env (see project root).

Usage:
    conda activate llm-rosetta
    pytest tests/integration/test_decision_llm_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import dotenv
import pytest

dotenv.load_dotenv(override=True)

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.llm_chat import LLMChatDecisionConverter
from llm_rosetta.converters.decision.schema_ops import extract_json
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)

# ============================================================================
# Shared test request
# ============================================================================

DECISION_REQUEST: IRDecisionRequest = {
    "model": "",  # filled per test
    "state": (
        "Customer email: 'I've been trying to get a refund for order #4521 "
        "for two weeks now. Nobody responds to my emails. This is ridiculous. "
        "I want my money back immediately or I'm disputing the charge.'"
    ),
    "questions": {
        "wants_refund": NoulQuestion(
            type="noul",
            instructions="Is the customer requesting a refund?",
            criteria={"true": "Explicitly asks for money back", "false": "No refund request"},
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team should handle this ticket?",
            criteria={
                "billing": "Payments, refunds, charges",
                "support": "General customer help",
                "escalation": "Urgent complaints, legal threats",
            },
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="How frustrated is the customer?",
            criteria=["Calm and factual", "Annoyed but civil", "Very frustrated and angry"],
        ),
    },
}


def _make_request(model: str) -> IRDecisionRequest:
    return {**DECISION_REQUEST, "model": model}


def _send_chat_request(
    base_url: str,
    api_key: str,
    wire: dict[str, Any],
) -> dict[str, Any]:
    """Send a chat completion request via httpx."""
    import httpx

    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, json=wire, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _send_anthropic_request(
    base_url: str,
    api_key: str,
    wire: dict[str, Any],
) -> dict[str, Any]:
    """Send an Anthropic Messages API request."""
    import httpx

    system = ""
    messages = []
    for msg in wire.get("messages", []):
        if msg["role"] == "system":
            system = msg["content"]
        else:
            messages.append(msg)

    body: dict[str, Any] = {
        "model": wire["model"],
        "max_tokens": 4096,
        "system": system,
        "messages": messages,
    }
    if "output_config" in wire:
        body["output_config"] = wire["output_config"]

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/v1/messages",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


def _strip_additional_properties(schema: Any) -> Any:
    """Google API doesn't accept additionalProperties in schemas."""
    if isinstance(schema, dict):
        return {
            k: _strip_additional_properties(v)
            for k, v in schema.items()
            if k != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_strip_additional_properties(v) for v in schema]
    return schema


def _send_google_request(
    api_key: str,
    wire: dict[str, Any],
) -> dict[str, Any]:
    """Send a Google generateContent request."""
    import httpx

    model = wire["model"]
    system_text = ""
    user_text = ""
    for msg in wire.get("messages", []):
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {},
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if "response_mime_type" in wire:
        body["generationConfig"]["responseMimeType"] = wire["response_mime_type"]
    if "response_schema" in wire:
        body["generationConfig"]["responseSchema"] = _strip_additional_properties(
            wire["response_schema"]
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()


def _validate_decision_response(ir_response: dict[str, Any]) -> None:
    """Validate that a decision response has proper structure."""
    assert ir_response["object"] == "decision"
    answers = ir_response["answers"]

    # Noul
    noul_a = answers["wants_refund"]
    assert noul_a["type"] == "noul"
    assert 0.0 <= noul_a["noul"] <= 1.0
    assert noul_a["noul"] > 0.5, "Customer clearly wants a refund"

    # Choice
    choice_a = answers["department"]
    assert choice_a["type"] == "choice"
    assert choice_a["choice"] in ("billing", "support", "escalation")
    assert abs(sum(choice_a["probabilities"].values()) - 1.0) < 0.05
    assert 0.0 <= choice_a["confidence"] <= 1.0

    # Score
    score_a = answers["frustration"]
    assert score_a["type"] == "score"
    assert 0.0 <= score_a["score"] <= 2.0
    assert score_a["score"] > 1.0, "Customer is clearly frustrated"
    assert abs(sum(score_a["probabilities"].values()) - 1.0) < 0.05
    assert score_a["legend"] == {
        "0": "Calm and factual",
        "1": "Annoyed but civil",
        "2": "Very frustrated and angry",
    }


# ============================================================================
# OpenAI Chat Completions
# ============================================================================


class TestOpenAIChatDecision:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_gpt4o_mini(self):
        converter = LLMChatDecisionConverter(output_format="openai_chat")
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("gpt-4o-mini"), context=ctx
        )

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        resp = _send_chat_request(base_url, os.environ["OPENAI_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        _validate_decision_response(ir)
        print(f"\n  gpt-4o-mini: refund={ir['answers']['wants_refund']['noul']:.2f}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']:.2f}")


# ============================================================================
# Anthropic Messages
# ============================================================================


class TestAnthropicDecision:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_claude_haiku(self):
        converter = LLMChatDecisionConverter(output_format="anthropic")
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("claude-haiku-4-5-20251001"), context=ctx
        )

        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        resp = _send_anthropic_request(base_url, os.environ["ANTHROPIC_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        _validate_decision_response(ir)
        print(f"\n  claude-haiku: refund={ir['answers']['wants_refund']['noul']:.2f}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']:.2f}")


# ============================================================================
# Google Gemini
# ============================================================================


class TestGoogleDecision:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

    def test_gemini_flash(self):
        converter = LLMChatDecisionConverter(output_format="google_generate")
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("gemini-3.6-flash"), context=ctx
        )

        resp = _send_google_request(os.environ["GOOGLE_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        _validate_decision_response(ir)
        print(f"\n  gemini-flash: refund={ir['answers']['wants_refund']['noul']:.2f}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']:.2f}")


# ============================================================================
# DeepSeek (OpenAI-compatible)
# ============================================================================


class TestDeepSeekDecision:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

    def test_deepseek_chat_prompted(self):
        """DeepSeek doesn't support strict json_schema — use prompted mode."""
        converter = LLMChatDecisionConverter(output_format="prompted")
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("deepseek-chat"), context=ctx
        )

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        resp = _send_chat_request(base_url, os.environ["DEEPSEEK_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        _validate_decision_response(ir)
        print(f"\n  deepseek-chat: refund={ir['answers']['wants_refund']['noul']:.2f}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']:.2f}")


# ============================================================================
# xAI (OpenAI-compatible)
# ============================================================================


class TestXAIDecision:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("XAI_API_KEY"):
            pytest.skip("XAI_API_KEY not set")

    def test_grok_prompted(self):
        """xAI may not support strict json_schema — use prompted mode."""
        converter = LLMChatDecisionConverter(output_format="prompted")
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("grok-3-mini-fast"), context=ctx
        )

        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        resp = _send_chat_request(base_url, os.environ["XAI_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        _validate_decision_response(ir)
        print(f"\n  grok: refund={ir['answers']['wants_refund']['noul']:.2f}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']:.2f}")


# ============================================================================
# Discrete mode (OpenAI)
# ============================================================================


class TestDiscreteMode:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_discrete_gpt4o_mini(self):
        converter = LLMChatDecisionConverter(
            output_format="openai_chat", answer_mode="discrete"
        )
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("gpt-4o-mini"), context=ctx
        )

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        resp = _send_chat_request(base_url, os.environ["OPENAI_API_KEY"], wire)

        ir = converter.response_from_provider(resp, context=ctx)
        assert ir["object"] == "decision"
        assert ir["answers"]["wants_refund"]["noul"] in (0.0, 1.0)
        assert ir["answers"]["department"]["confidence"] == 1.0
        print(f"\n  discrete gpt-4o-mini: refund={ir['answers']['wants_refund']['noul']}, "
              f"dept={ir['answers']['department']['choice']}, "
              f"frustration={ir['answers']['frustration']['score']}")

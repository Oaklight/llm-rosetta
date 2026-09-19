"""
LLM-Rosetta - LLM Chat Decision Converter

LLM Chat 后端的 decision 转换器
Decision converter backed by LLM chat completions with structured output

Translates IRDecisionRequest into an OpenAI-compatible chat completion
request with ``response_format: json_schema``, and parses the structured
JSON response back into typed decision answers (noul/choice/score).
"""

from __future__ import annotations

import json
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

from .schema_ops import (
    build_decision_schema,
    build_system_prompt,
    parse_decision_answers,
    serialize_state,
)


class LLMChatDecisionConverter(BaseDecisionConverter):
    """Decision converter backed by LLM chat completions.

    Converts decision requests into chat completion requests with
    structured output (json_schema response_format), then parses the
    LLM's JSON response back into typed decision answers.
    """

    _CONVERTER_TAG = "llm_chat_decision"

    # ==================== Request conversion ====================

    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        questions = ir_request["questions"]
        context.options["_decision_questions"] = questions

        system_prompt = build_system_prompt(questions)
        user_content = serialize_state(ir_request["state"])
        schema = build_decision_schema(questions)

        result: dict[str, Any] = {
            "model": ir_request["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest:
        raise NotImplementedError(
            "LLMChatDecisionConverter does not support chat → IR request "
            "conversion. Use TypeSafeDecisionConverter for native format."
        )

    # ==================== Response conversion ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse:
        questions = context.options.get("_decision_questions", {})

        content = self._extract_content(provider_response)
        raw = json.loads(content) if isinstance(content, str) else content
        answers = parse_decision_answers(raw, questions)

        result: IRDecisionResponse = {
            "object": "decision",
            "model": provider_response.get("model", ""),
            "answers": answers,
        }

        p_usage = provider_response.get("usage")
        if p_usage:
            result["usage"] = self._build_p_usage_to_ir(p_usage)

        return result

    def _do_response_to_provider(
        self,
        ir_response: IRDecisionResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "LLMChatDecisionConverter does not support IR → chat response conversion."
        )

    # ==================== Usage conversion ====================

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo:
        usage: DecisionUsageInfo = {}
        if "prompt_tokens" in p_usage:
            usage["input_tokens"] = p_usage["prompt_tokens"]
        elif "input_tokens" in p_usage:
            usage["input_tokens"] = p_usage["input_tokens"]
        if "completion_tokens" in p_usage:
            usage["output_tokens"] = p_usage["completion_tokens"]
        elif "output_tokens" in p_usage:
            usage["output_tokens"] = p_usage["output_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "input_tokens" in ir_usage:
            result["prompt_tokens"] = ir_usage["input_tokens"]
        if "output_tokens" in ir_usage:
            result["completion_tokens"] = ir_usage["output_tokens"]
        return result

    # ==================== Helpers ====================

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """Extract message content from a chat completion response."""
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return response.get("content", "")

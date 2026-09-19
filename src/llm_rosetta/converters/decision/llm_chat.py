"""
LLM-Rosetta - LLM Chat Decision Converter

Decision converter backed by LLM chat completions with structured output.

Translates IRDecisionRequest into chat completion requests, supporting:
- OpenAI response_format (json_schema, strict mode)
- Anthropic output_config.format (json_schema)
- Prompted fallback (schema embedded in system prompt)
- Probabilities and discrete answer modes
- Corrective retry on malformed output
"""

from __future__ import annotations

import json
from typing import Any, Literal

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

from .schema_ops import (
    AnswerMode,
    build_correction_prompt,
    build_decision_schema,
    build_system_prompt,
    extract_json,
    parse_decision_answers,
    serialize_state,
)

OutputFormat = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic",
    "google_generate",
    "google_interactions",
    "prompted",
]


class LLMChatDecisionConverter(BaseDecisionConverter):
    """Decision converter backed by LLM chat completions.

    Args:
        output_format: How to constrain the LLM's output.
            - "openai_chat": response_format with json_schema (default)
            - "openai_responses": text.format with json_schema
            - "anthropic": output_config.format with json_schema
            - "google_generate": response_mime_type + response_schema
            - "google_interactions": response_format with mime_type + response_schema
            - "prompted": embed schema in system prompt (universal fallback)
        answer_mode: "probabilities" (default) or "discrete".
        max_retries: Corrective retries for malformed output (default 0).
    """

    _CONVERTER_TAG = "llm_chat_decision"

    def __init__(
        self,
        *,
        output_format: OutputFormat = "openai_chat",
        answer_mode: AnswerMode = "probabilities",
        max_retries: int = 0,
    ) -> None:
        self.output_format = output_format
        self.answer_mode = answer_mode
        self.max_retries = max_retries

    # ==================== Request conversion ====================

    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        questions = ir_request["questions"]
        schema = build_decision_schema(questions, answer_mode=self.answer_mode)

        context.options["_decision_questions"] = questions
        context.options["_decision_schema"] = schema
        context.options["_decision_answer_mode"] = self.answer_mode
        context.options["_decision_max_retries"] = self.max_retries

        embed_schema = schema if self.output_format == "prompted" else None
        system_prompt = build_system_prompt(
            questions, answer_mode=self.answer_mode, schema=embed_schema
        )
        user_content = serialize_state(ir_request["state"])

        result: dict[str, Any] = {
            "model": ir_request["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        if self.output_format == "openai_chat":
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "schema": schema,
                    "strict": True,
                },
            }
        elif self.output_format == "openai_responses":
            result["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "decision",
                    "schema": schema,
                    "strict": True,
                }
            }
        elif self.output_format == "anthropic":
            result["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        elif self.output_format == "google_generate":
            result["response_mime_type"] = "application/json"
            result["response_schema"] = schema
        elif self.output_format == "google_interactions":
            result["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "response_schema": schema,
            }
        # prompted: no format constraint — schema is in the system prompt

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
        questions = context.options.get("_decision_questions")
        if questions is None:
            raise ValueError(
                "No _decision_questions in context — was request_to_provider "
                "called with the same ConversionContext?"
            )
        answer_mode = context.options.get("_decision_answer_mode", "probabilities")

        content = self._extract_content(provider_response)
        if not content:
            raise ValueError(
                "LLM response contained no message content to parse as decision JSON"
            )
        try:
            raw = json.loads(extract_json(content))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM decision response as JSON: {e}"
            ) from e
        answers = parse_decision_answers(raw, questions, answer_mode=answer_mode)

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

    # ==================== Retry support ====================

    def build_retry_messages(
        self,
        failed_content: str,
        error: str,
    ) -> list[dict[str, str]]:
        """Build messages for a corrective retry attempt.

        Returns assistant + user messages to append to the conversation
        for another attempt.
        """
        return [
            {"role": "assistant", "content": failed_content},
            {"role": "user", "content": build_correction_prompt(error)},
        ]

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
        return (
            _extract_openai_chat(response)
            or _extract_openai_responses(response)
            or _extract_google(response)
            or _extract_anthropic(response)
            or ""
        )


def _extract_openai_chat(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def _extract_openai_responses(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        return part.get("text", "")
    return ""


def _extract_google(response: dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        if texts:
            return "".join(texts)
    return ""


def _extract_anthropic(response: dict[str, Any]) -> str:
    content_blocks = response.get("content", [])
    if isinstance(content_blocks, list) and content_blocks:
        texts = [
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(texts)
    if isinstance(content_blocks, str):
        return content_blocks
    return ""

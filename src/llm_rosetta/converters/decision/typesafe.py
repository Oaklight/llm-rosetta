"""
LLM-Rosetta - TypeSafe Decision Converter

TypeSafe System One (Jev) API 的 decision 转换器
Decision converter for the TypeSafe System One (Jev) API

Near-passthrough converter: the TypeSafe wire format and the IR decision
types share the same question/answer primitives (noul, choice, score),
so conversion is mostly structural (adding/removing ``object`` field,
copying questions/answers).
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)


class TypeSafeDecisionConverter(BaseDecisionConverter):
    """Converter for the TypeSafe System One (Jev) API."""

    _CONVERTER_TAG = "typesafe_decision"

    # ==================== Request conversion ====================

    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "state": ir_request["state"],
            "questions": dict(ir_request["questions"]),
        }
        if "provider_extensions" in ir_request:
            result.update(ir_request["provider_extensions"])
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest:
        return {
            "model": provider_request["model"],
            "state": provider_request["state"],
            "questions": dict(provider_request["questions"]),
        }

    # ==================== Response conversion ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse:
        result: IRDecisionResponse = {
            "object": "decision",
            "model": provider_response["model"],
            "answers": dict(provider_response.get("answers", {})),
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
        result: dict[str, Any] = {
            "model": ir_response["model"],
            "answers": dict(ir_response["answers"]),
        }
        if "usage" in ir_response:
            result["usage"] = self._build_ir_usage_to_p(ir_response["usage"])
        return result

    # ==================== Usage conversion ====================

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo:
        usage: DecisionUsageInfo = {}
        if "input_tokens" in p_usage:
            usage["input_tokens"] = p_usage["input_tokens"]
        if "output_tokens" in p_usage:
            usage["output_tokens"] = p_usage["output_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "input_tokens" in ir_usage:
            result["input_tokens"] = ir_usage["input_tokens"]
        if "output_tokens" in ir_usage:
            result["output_tokens"] = ir_usage["output_tokens"]
        return result

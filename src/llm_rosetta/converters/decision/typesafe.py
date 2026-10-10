"""
LLM-Rosetta - TypeSafe Decision Converter

TypeSafe System One (Jev) API 的 decision 转换器
Decision converter for the TypeSafe System One (Jev) API

The IR names the probabilistic yes/no proposition ``assertion``, while the
TypeSafe wire format calls the same primitive ``noul`` (and names its answer
value field ``noul`` as well).  This converter translates the assertion
primitive in both directions.  The ``choice`` and ``score`` primitives are
identical between IR and wire and pass through unchanged.
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

# TypeSafe wire name for the assertion primitive.
_WIRE_ASSERTION = "noul"


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
            "questions": self._questions_to_wire(ir_request["questions"]),
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
            "questions": self._questions_to_ir(provider_request["questions"]),
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
            "answers": self._answers_to_ir(provider_response.get("answers", {})),
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
            "answers": self._answers_to_wire(ir_response["answers"]),
        }
        if "usage" in ir_response:
            result["usage"] = self._build_ir_usage_to_p(ir_response["usage"])
        return result

    # ==================== Assertion primitive translation ====================

    def _questions_to_wire(self, questions: dict[str, Any]) -> dict[str, Any]:
        return {qid: self._question_to_wire(q) for qid, q in questions.items()}

    def _questions_to_ir(self, questions: dict[str, Any]) -> dict[str, Any]:
        return {qid: self._question_to_ir(q) for qid, q in questions.items()}

    def _answers_to_ir(self, answers: dict[str, Any]) -> dict[str, Any]:
        return {aid: self._answer_to_ir(a) for aid, a in answers.items()}

    def _answers_to_wire(self, answers: dict[str, Any]) -> dict[str, Any]:
        return {aid: self._answer_to_wire(a) for aid, a in answers.items()}

    @staticmethod
    def _question_to_wire(question: dict[str, Any]) -> dict[str, Any]:
        if question.get("type") == "assertion":
            return {**question, "type": _WIRE_ASSERTION}
        return dict(question)

    @staticmethod
    def _question_to_ir(question: dict[str, Any]) -> dict[str, Any]:
        if question.get("type") == _WIRE_ASSERTION:
            return {**question, "type": "assertion"}
        return dict(question)

    @staticmethod
    def _answer_to_ir(answer: dict[str, Any]) -> dict[str, Any]:
        if answer.get("type") == _WIRE_ASSERTION:
            result = {**answer, "type": "assertion"}
            if _WIRE_ASSERTION in result:
                result["probability"] = result.pop(_WIRE_ASSERTION)
            return result
        return dict(answer)

    @staticmethod
    def _answer_to_wire(answer: dict[str, Any]) -> dict[str, Any]:
        if answer.get("type") == "assertion":
            result = {**answer, "type": _WIRE_ASSERTION}
            if "probability" in result:
                result[_WIRE_ASSERTION] = result.pop("probability")
            return result
        return dict(answer)

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

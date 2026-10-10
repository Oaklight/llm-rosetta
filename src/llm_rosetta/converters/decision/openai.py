"""
LLM-Rosetta - OpenAI Decisions Converter

OpenAI Decisions API (`POST /v1/decisions`, model ``gpt-6-luna``) 的
decision 转换器 / Decision converter for OpenAI's Decisions API.

Maps the IR to OpenAI's wire format:

- ``state``  → ``input``  (string, or user messages with ``input_text`` /
  ``input_image`` parts; a structured record is JSON-stringified)
- ``questions`` (a map) → ``questions`` (an **array** with ``name``)
- ``assertion`` → ``predicate`` (no criteria field; the IR entries are carried
  in a marked, reversible JSON envelope inside ``instructions``)
- ``choice``    → ``choices: [{value, description}]``
- ``score``     → ``levels: [{label, description}]``
- answers (an array keyed by ``name``) → IR answers, incl. ``refusal``
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

_MARK = "<<<rosetta:assertion>>>"
"""Marks the reversible JSON envelope for a predicate's folded criteria."""

_IR_TO_WIRE_TYPE = {"assertion": "predicate", "choice": "choice", "score": "score"}
_WIRE_TO_IR_TYPE = {"predicate": "assertion", "choice": "choice", "score": "score"}


def _image_data_url(part: dict[str, Any]) -> str:
    if "image_url" in part:
        return part["image_url"]
    data = part.get("image_data", {})
    media = data.get("media_type", "image/jpeg")
    return f"data:{media};base64,{data.get('data', '')}"


class OpenAIDecisionsConverter(BaseDecisionConverter):
    """Converter for OpenAI's Decisions API."""

    _CONVERTER_TAG = "openai_decisions"

    # ==================== Request conversion ====================

    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "input": _state_to_input(ir_request["state"]),
            "questions": [
                _question_to_wire(name, q)
                for name, q in ir_request["questions"].items()
            ],
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
        questions: dict[str, Any] = {}
        for i, q in enumerate(provider_request.get("questions", [])):
            name = q.get("name") or f"q{i}"
            questions[name] = _question_from_wire(q)
        return {
            "model": provider_request["model"],
            "state": _state_from_input(provider_request.get("input")),
            "questions": questions,
        }

    # ==================== Response conversion ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse:
        answers: dict[str, Any] = {}
        for i, a in enumerate(provider_response.get("answers", [])):
            name = a.get("name") or f"q{i}"
            answers[name] = _answer_from_wire(a)
        result: IRDecisionResponse = {
            "object": "decision",
            "model": provider_response["model"],
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
        result: dict[str, Any] = {
            "model": ir_response["model"],
            "answers": [
                _answer_to_wire(name, a) for name, a in ir_response["answers"].items()
            ],
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


# ============================================================================
# State <-> input
# ============================================================================


def _state_to_input(state: Any) -> Any:
    """IR state → OpenAI ``input`` (string or user-message array)."""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        parts: list[dict[str, Any]] = []
        for part in state:
            if isinstance(part, dict) and part.get("type") == "image":
                parts.append(
                    {"type": "input_image", "image_url": _image_data_url(part)}
                )
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
        return [{"role": "user", "content": parts}]
    # structured record: JSON-stringified (OpenAI input rejects a bare object)
    return json.dumps(state, ensure_ascii=False)


def _state_from_input(input_value: Any) -> Any:
    """OpenAI ``input`` → IR state."""
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        parts: list[dict[str, Any]] = []
        for msg in input_value:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    if p.get("type") == "input_text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "input_image":
                        parts.append(
                            {"type": "image", "image_url": p.get("image_url", "")}
                        )
        return parts
    return input_value


# ============================================================================
# Question <-> wire
# ============================================================================


def _encode_assertion(q: Mapping[str, Any]) -> str:
    """Fold an assertion's entries into a reversible marked JSON envelope."""
    instructions = q["instructions"]
    criteria = q.get("criteria")
    if criteria is None and isinstance(instructions, str) and _MARK not in instructions:
        return instructions
    return _MARK + json.dumps(
        {"instructions": instructions, "criteria": criteria}, ensure_ascii=False
    )


def _decode_assertion(text: str) -> dict[str, Any]:
    if isinstance(text, str) and text.startswith(_MARK):
        payload = json.loads(text[len(_MARK) :])
        result: dict[str, Any] = {
            "type": "assertion",
            "instructions": payload["instructions"],
        }
        if payload.get("criteria") is not None:
            result["criteria"] = payload["criteria"]
        return result
    return {"type": "assertion", "instructions": text}


def _with_desc(key: str, value: Any, description: Any) -> dict[str, Any]:
    """Build an OpenAI option object, omitting a null description."""
    option: dict[str, Any] = {key: value}
    if description is not None:
        option["description"] = description
    return option


def _question_to_wire(name: str, q: Mapping[str, Any]) -> dict[str, Any]:
    ir_type = q["type"]
    wire: dict[str, Any] = {
        "type": _IR_TO_WIRE_TYPE.get(ir_type, ir_type),
        "name": name,
    }
    if ir_type == "assertion":
        wire["instructions"] = _encode_assertion(q)
    elif ir_type == "choice":
        wire["instructions"] = q["instructions"]
        wire["choices"] = [
            _with_desc("value", e["label"], e.get("description")) for e in q["criteria"]
        ]
    elif ir_type == "score":
        wire["instructions"] = q["instructions"]
        wire["levels"] = [
            _with_desc("label", str(e["label"]), e.get("description"))
            for e in q["criteria"]
        ]
    return wire


def _entry(label: Any, description: Any) -> dict[str, Any]:
    """Build an IR entry, omitting a null description."""
    entry: dict[str, Any] = {"label": label}
    if description is not None:
        entry["description"] = description
    return entry


def _question_from_wire(q: dict[str, Any]) -> dict[str, Any]:
    wire_type = q.get("type", "")
    ir_type = _WIRE_TO_IR_TYPE.get(wire_type, wire_type)
    if ir_type == "assertion":
        return _decode_assertion(q["instructions"])
    result: dict[str, Any] = {"type": ir_type, "instructions": q["instructions"]}
    if ir_type == "choice":
        result["criteria"] = [
            _entry(c["value"], c.get("description")) for c in q.get("choices", [])
        ]
    elif ir_type == "score":
        result["criteria"] = [
            _entry(lv["label"], lv.get("description")) for lv in q.get("levels", [])
        ]
    return result


# ============================================================================
# Answer <-> wire
# ============================================================================


def _answer_from_wire(a: dict[str, Any]) -> dict[str, Any]:
    wire_type = a.get("type", "")
    ir_type = _WIRE_TO_IR_TYPE.get(wire_type, wire_type)
    if wire_type == "refusal":
        result: dict[str, Any] = {"type": "refusal"}
        if a.get("reason"):
            result["reason"] = a["reason"]
        return result
    if ir_type == "assertion":
        return {"type": "assertion", "probability": a["probability"]}
    if ir_type == "choice":
        result = {
            "type": "choice",
            "choice": a["choice"],
            "probabilities": {
                str(p["value"]): p["probability"] for p in a.get("probabilities", [])
            },
        }
        _maybe_set(result, a, "confidence")
        return result
    if ir_type == "score":
        result = {
            "type": "score",
            "score": a["score"],
            "probabilities": {
                p["label"]: p["probability"] for p in a.get("probabilities", [])
            },
        }
        _maybe_set(result, a, "confidence")
        return result
    return {"type": "refusal"}


def _answer_to_wire(name: str, a: Mapping[str, Any]) -> dict[str, Any]:
    ir_type = a.get("type", "")
    if ir_type == "assertion":
        return {"type": "predicate", "name": name, "probability": a["probability"]}
    if ir_type == "choice":
        result: dict[str, Any] = {
            "type": "choice",
            "name": name,
            "choice": a["choice"],
            "probabilities": [
                {"value": k, "probability": v} for k, v in a["probabilities"].items()
            ],
        }
        _maybe_set(result, a, "confidence")
        return result
    if ir_type == "score":
        result = {
            "type": "score",
            "name": name,
            "score": a["score"],
            "probabilities": [
                {"value": i, "label": k, "probability": v}
                for i, (k, v) in enumerate(a["probabilities"].items())
            ],
        }
        _maybe_set(result, a, "confidence")
        return result
    result = {"type": "refusal", "name": name}
    if a.get("reason"):
        result["reason"] = a["reason"]
    return result


def _maybe_set(target: dict[str, Any], source: Mapping[str, Any], key: str) -> None:
    if source.get(key) is not None:
        target[key] = source[key]

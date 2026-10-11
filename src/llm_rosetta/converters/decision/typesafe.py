"""
LLM-Rosetta - TypeSafe Decision Converter

TypeSafe System One (Jev) API 的 decision 转换器
Decision converter for the TypeSafe System One (Jev) API

The IR unifies all three question types on ``criteria: list[DecisionEntry]``.
TypeSafe names the assertion primitive ``noul`` and uses per-type criteria
shapes (assertion: ``{true,false}``; choice: ``{value: description}``;
score: ``[description]``).  This converter translates the entries to and from
those shapes; ``choice``/``score`` answers pass through, the assertion answer
is ``noul`` ↔ ``probability``.

The System One family exposes an optional top-level ``images`` array (Cloudflare
Clef, classifier.dev); the converter maps it to and from the IR ``state``
content parts.  An image embedded in a structured ``state`` dict is dropped
with a warning, and there is no refusal answer type (``RefusalAnswer`` is
dropped).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionEntry,
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

_WIRE_ASSERTION = "noul"
_IMAGE_PLACEHOLDER = "[image omitted: provider does not support image input]"

_IR_TO_WIRE_TYPE = {"assertion": "noul", "choice": "choice", "score": "score"}
_WIRE_TO_IR_TYPE = {"noul": "assertion", "choice": "choice", "score": "score"}


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
        warnings = context.warnings
        wire_state, images = _state_to_wire(ir_request["state"], warnings)
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "state": wire_state,
            "questions": {
                qid: _question_to_wire(q, warnings)
                for qid, q in ir_request["questions"].items()
            },
        }
        if images:
            result["images"] = images
        if "provider_extensions" in ir_request:
            result.update(ir_request["provider_extensions"])
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest:
        questions: dict[str, Any] = {
            qid: _question_from_wire(q)
            for qid, q in provider_request["questions"].items()
        }
        return {
            "model": provider_request["model"],
            "state": _state_from_wire(
                provider_request.get("state"),
                provider_request.get("images"),
                context.warnings,
            ),
            "questions": questions,
        }

    # ==================== Response conversion ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse:
        answers: dict[str, Any] = {
            aid: _answer_from_wire(a, context.warnings)
            for aid, a in provider_response.get("answers", {}).items()
        }
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
            "answers": {
                aid: _answer_to_wire(a, context.warnings)
                for aid, a in ir_response["answers"].items()
            },
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
# State translation
# ============================================================================


def _is_image_part(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "image"


def _image_to_data_url(part: Mapping[str, Any]) -> str | None:
    """IR image part → an inline base64 data URL, or ``None`` if it has no data."""
    url = part.get("image_url")
    if url:
        return url
    data = part.get("image_data")
    if not data or not data.get("data"):
        return None
    media = data.get("media_type", "image/jpeg")
    return f"data:{media};base64,{data['data']}"


def _state_to_wire(state: Any, warnings: list[str]) -> tuple[Any, list[str]]:
    """IR state → (TypeSafe ``state``, ``images``).

    The System One ``state`` is ``string | object | array``.  The Cloudflare
    Clef / classifier.dev extension adds a separate top-level ``images`` array
    (inline base64 data URLs), so a content-part list is split: text parts join
    into ``state`` and image parts become ``images``.  An image embedded in a
    structured dict is dropped with a warning (no positional key to re-attach).
    """
    if isinstance(state, str):
        return state, []
    if isinstance(state, list):
        texts: list[str] = []
        images: list[str] = []
        for part in state:
            if _is_image_part(part):
                url = _image_to_data_url(part)
                if url is None:
                    warnings.append(
                        "Image part has neither image_url nor image_data; omitted"
                    )
                else:
                    images.append(url)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            else:
                texts.append(str(part))
        return "\n".join(texts), images
    if isinstance(state, dict):
        # Unwrap the ``{"items": [...]}`` array form produced by _state_from_wire.
        # A content-part list is split (text + images); a structured array of
        # arbitrary values is passed through as a bare array.
        if set(state) == {"items"} and isinstance(state["items"], list):
            items = state["items"]
            if items and all(
                isinstance(p, dict) and p.get("type") in ("text", "image")
                for p in items
            ):
                return _state_to_wire(items, warnings)
            return items, []
        return _strip_images(state, warnings), []
    return state, []


def _strip_images(value: Any, warnings: list[str]) -> Any:
    if _is_image_part(value):
        warnings.append(
            "Image embedded in a structured state; not carried (no positional "
            "key). Move it to a top-level content-part list to use the images[] "
            "extension."
        )
        return _IMAGE_PLACEHOLDER
    if isinstance(value, dict):
        return {k: _strip_images(v, warnings) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_images(v, warnings) for v in value]
    return value


def _state_from_wire(state: Any, images: Any, warnings: list[str]) -> Any:
    """TypeSafe ``state`` (+ optional ``images``) → IR state.

    With ``images`` present the IR state is a content-part list (a text part for
    the wire ``state`` followed by an image part per entry).  A bare array
    without images is wrapped as ``{"items": [...]}`` because the IR reserves
    ``list`` for content parts.
    """
    if images:
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list):
            warnings.append(
                f"Unsupported 'images' value ({type(images).__name__}); ignored"
            )
            images = []
        parts: list[dict[str, Any]] = []
        if state not in (None, "", {}, []):
            text = (
                state
                if isinstance(state, str)
                else json.dumps(state, ensure_ascii=False)
            )
            parts.append({"type": "text", "text": text})
        parts.extend({"type": "image", "image_url": str(img)} for img in images)
        if parts:
            return parts
    if isinstance(state, list):
        return {"items": state}
    return state


# ============================================================================
# Question translation
# ============================================================================


def _question_to_wire(q: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    ir_type = q["type"]
    wire_type = _IR_TO_WIRE_TYPE.get(ir_type, ir_type)
    result: dict[str, Any] = {"type": wire_type, "instructions": q["instructions"]}
    if ir_type == "assertion":
        criteria = q.get("criteria")
        if criteria:
            result["criteria"] = {
                ("true" if entry["label"] else "false"): entry.get("description")
                for entry in criteria
            }
    elif ir_type == "choice":
        result["criteria"] = {
            str(entry["label"]): entry.get("description") for entry in q["criteria"]
        }
    elif ir_type == "score":
        result["criteria"] = [
            entry.get("description") or str(entry["label"]) for entry in q["criteria"]
        ]
    return result


def _question_from_wire(q: dict[str, Any]) -> dict[str, Any]:
    wire_type = q.get("type", "")
    ir_type = _WIRE_TO_IR_TYPE.get(wire_type, wire_type)
    result: dict[str, Any] = {"type": ir_type, "instructions": q["instructions"]}
    criteria = q.get("criteria")
    if ir_type == "assertion":
        if criteria:
            entries: list[DecisionEntry] = []
            for side in ("false", "true"):
                if side in criteria:
                    entries.append(
                        DecisionEntry(
                            label=(side == "true"),
                            description=criteria[side],
                        )
                    )
            result["criteria"] = entries
    elif ir_type == "choice":
        if criteria:
            result["criteria"] = [
                _entry(label=k, description=v) for k, v in criteria.items()
            ]
    elif ir_type == "score":
        if criteria:
            result["criteria"] = [
                _entry(label=v)
                if isinstance(v, str)
                else _entry(label=str(i), description=v)
                for i, v in enumerate(criteria)
            ]
    return result


def _entry(label: Any, description: Any = None) -> DecisionEntry:
    entry = DecisionEntry(label=label)
    if description is not None:
        entry["description"] = description
    return entry


# ============================================================================
# Answer translation
# ============================================================================


def _answer_from_wire(a: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    wire_type = a.get("type", "")
    if wire_type == "noul":
        result: dict[str, Any] = {"type": "assertion", "probability": a["noul"]}
        _maybe_set(result, a, "unknown_probability")
        return result
    if wire_type == "choice":
        return _copy(a, "choice", "probabilities", "confidence", "unknown_probability")
    if wire_type == "score":
        # Re-key probabilities from index keys to level labels via ``legend``.
        legend = a.get("legend", {})
        probs = {
            legend.get(str(i), str(i)): p for i, p in a.get("probabilities", {}).items()
        }
        result = {"type": "score", "score": a.get("score", 0.0), "probabilities": probs}
        _maybe_set(result, a, "confidence")
        _maybe_set(result, a, "unknown_probability")
        return result
    # Unknown / malformed type: warn and pass the payload through rather than
    # silently upgrading it to a (stronger) refusal claim.
    warnings.append(
        f"Unrecognized TypeSafe answer type {wire_type!r}; passed through unchanged"
    )
    return dict(a)


def _answer_to_wire(a: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    ir_type = a.get("type", "")
    if ir_type == "assertion":
        result = {"type": _WIRE_ASSERTION, "noul": a["probability"]}
        _maybe_set(result, a, "unknown_probability")
        return result
    if ir_type == "choice":
        return _copy(a, "choice", "probabilities", "confidence", "unknown_probability")
    if ir_type == "score":
        probs = a.get("probabilities", {})
        # Level order comes from the IR ``probabilities`` key order; the response
        # carries no question, so the true ordinal cannot be re-derived here.
        result = {
            "type": "score",
            "score": a.get("score", 0.0),
            "legend": {str(i): label for i, label in enumerate(probs)},
            "probabilities": {str(i): p for i, p in enumerate(probs.values())},
        }
        _maybe_set(result, a, "confidence")
        _maybe_set(result, a, "unknown_probability")
        return result
    # TypeSafe has no refusal type. Emit a maximally-uncertain noul rather than
    # a fabricated confident 0/1, which downstream would misread as a real answer.
    warnings.append("TypeSafe has no refusal answer type; emitted noul=0.5")
    return {"type": "noul", "noul": 0.5}


_ANSWER_KEYS = ("type",)


def _copy(a: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    result = {k: a[k] for k in _ANSWER_KEYS if k in a}
    for key in keys:
        if key in a:
            result[key] = a[key]
    return result


def _maybe_set(target: dict[str, Any], source: Mapping[str, Any], key: str) -> None:
    if key in source:
        target[key] = source[key]

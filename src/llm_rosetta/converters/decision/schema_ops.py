"""
LLM-Rosetta - Decision Schema Operations

Schema generation and answer parsing for LLM-backed decision evaluation.

Translates typed decision questions into JSON schemas suitable for
structured output (response_format), and parses the LLM's JSON response
back into typed decision answers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

from llm_rosetta.types.ir.decision import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionQuestion,
    DecisionState,
    NoulAnswer,
    ScoreAnswer,
)

_SYSTEM_PROMPT = """\
You are a structured decision model. Evaluate the provided state against \
each question and return calibrated probability estimates.

For each question:
- noul (binary probability): return a single number in [0, 1] representing P(true).
- choice: return an object mapping each option to its probability. \
Values must be non-negative and sum to 1.
- score: return an object mapping each level index ("0", "1", ...) to its \
probability. Values must be non-negative and sum to 1.

Be calibrated: your probabilities should reflect genuine uncertainty. \
Do not default to extremes unless the evidence is overwhelming."""


def build_system_prompt(
    questions: Mapping[str, DecisionQuestion],
) -> str:
    """Build the system prompt describing the questions to evaluate."""
    parts = [_SYSTEM_PROMPT, "", "Questions:"]
    for qid, q in questions.items():
        qtype = q["type"]
        instructions = _serialize_value(q["instructions"])
        if qtype == "noul":
            desc = f"  {qid} (noul): {instructions}"
            noul_criteria: Any = q.get("criteria")
            if isinstance(noul_criteria, dict):
                true_desc = noul_criteria.get("true", "")
                false_desc = noul_criteria.get("false", "")
                if true_desc or false_desc:
                    desc += f" [true={true_desc}, false={false_desc}]"
        elif qtype == "choice":
            options: dict[str, Any] = cast(Any, q).get("criteria", {})
            opts_str = ", ".join(f"{k}: {v}" if v else k for k, v in options.items())
            desc = f"  {qid} (choice): {instructions} [{opts_str}]"
        elif qtype == "score":
            levels: list[str] = cast(Any, q).get("criteria", [])
            levels_str = ", ".join(f"{i}={lv}" for i, lv in enumerate(levels))
            desc = f"  {qid} (score): {instructions} [{levels_str}]"
        else:
            desc = f"  {qid} ({qtype}): {instructions}"
        parts.append(desc)
    return "\n".join(parts)


def build_decision_schema(
    questions: Mapping[str, DecisionQuestion],
) -> dict[str, Any]:
    """Build a JSON schema for structured output from typed questions."""
    answer_properties: dict[str, Any] = {}

    for qid, q in questions.items():
        qtype = q["type"]
        if qtype == "noul":
            answer_properties[qid] = {
                "type": "number",
                "description": (
                    f"P(true) in [0,1]. {_serialize_value(q['instructions'])}"
                ),
            }
        elif qtype == "choice":
            criteria_dict: dict[str, Any] = cast(Any, q).get("criteria", {})
            props: dict[str, Any] = {}
            for label, rubric in criteria_dict.items():
                prop: dict[str, Any] = {"type": "number"}
                if rubric:
                    prop["description"] = rubric
                props[label] = prop
            answer_properties[qid] = {
                "type": "object",
                "properties": props,
                "required": list(criteria_dict.keys()),
                "additionalProperties": False,
                "description": _serialize_value(q["instructions"]),
            }
        elif qtype == "score":
            criteria_list: list[str] = cast(Any, q).get("criteria", [])
            props = {}
            for i, level_desc in enumerate(criteria_list):
                props[str(i)] = {
                    "type": "number",
                    "description": level_desc,
                }
            answer_properties[qid] = {
                "type": "object",
                "properties": props,
                "required": [str(i) for i in range(len(criteria_list))],
                "additionalProperties": False,
                "description": _serialize_value(q["instructions"]),
            }

    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "object",
                "properties": answer_properties,
                "required": list(questions.keys()),
                "additionalProperties": False,
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def serialize_state(state: DecisionState) -> str:
    """Serialize decision state to a string for the user message."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def parse_decision_answers(
    raw: dict[str, Any],
    questions: Mapping[str, DecisionQuestion],
) -> dict[str, DecisionAnswer]:
    """Parse raw LLM JSON output into typed decision answers."""
    raw_answers = raw.get("answers", raw)
    answers: dict[str, DecisionAnswer] = {}

    for qid, q in questions.items():
        raw_answer = raw_answers.get(qid)
        if raw_answer is None:
            continue
        answers[qid] = _parse_single_answer(raw_answer, q)

    return answers


def compute_confidence(probabilities: dict[str, float]) -> float:
    """Compute confidence as 1 - normalized Shannon entropy.

    Returns 1.0 for a peaked distribution (one option has all probability),
    0.0 for a uniform distribution (maximum uncertainty).
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0

    values = list(probabilities.values())
    total = sum(values)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for p in values:
        if p > 0:
            p_norm = p / total
            entropy -= p_norm * math.log2(p_norm)

    max_entropy = math.log2(n)
    if max_entropy <= 0:
        return 1.0

    return max(0.0, min(1.0, 1.0 - entropy / max_entropy))


# ==================== Internal helpers ====================


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Rescale probabilities to sum to 1.0."""
    total = sum(probs.values())
    if total <= 0 or abs(total - 1.0) < 1e-9:
        return probs
    return {k: v / total for k, v in probs.items()}


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_single_answer(
    raw_answer: Any,
    question: DecisionQuestion,
) -> DecisionAnswer:
    qtype = question["type"]

    if qtype == "noul":
        noul_val = (
            float(raw_answer) if not isinstance(raw_answer, float) else raw_answer
        )
        return NoulAnswer(type="noul", noul=noul_val)

    if qtype == "choice":
        probs = _normalize_probs({str(k): float(v) for k, v in raw_answer.items()})
        choice_key = max(probs, key=lambda k: probs[k])
        return ChoiceAnswer(
            type="choice",
            choice=choice_key,
            probabilities=probs,
            confidence=compute_confidence(probs),
        )

    if qtype == "score":
        probs = _normalize_probs({str(k): float(v) for k, v in raw_answer.items()})
        criteria_list: list[str] = cast(Any, question).get("criteria", [])
        legend = {str(i): desc for i, desc in enumerate(criteria_list)}
        score_val = sum(int(k) * v for k, v in probs.items())
        return ScoreAnswer(
            type="score",
            score=score_val,
            legend=legend,
            probabilities=probs,
            confidence=compute_confidence(probs),
        )

    raise ValueError(f"Unknown question type: {qtype}")

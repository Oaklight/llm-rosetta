"""
LLM-Rosetta - Decision Score Operations

Shared scoring logic for reranker and embedding-backed decision converters.

Converts raw per-option scores (relevance scores, cosine similarities)
into typed decision answers via softmax normalization.  This is the
pure-math layer — no API calls, no model access.

Reference: oaklight/jev-explore model/src/reranker_scorer.py
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from llm_rosetta.converters.decision.schema_ops import compute_confidence
from llm_rosetta.types.ir.decision import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionQuestion,
    DecisionState,
    DecisionUsageInfo,
    NoulAnswer,
    ScoreAnswer,
)


def softmax(scores: list[float], *, temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax over a list of scores.

    Args:
        scores: Raw scores (relevance scores, cosine similarities, etc.)
        temperature: Controls distribution sharpness.  Lower values
            produce more peaked distributions.  Useful when raw scores
            are in a narrow range (e.g. cosine similarities).
    """
    if not scores:
        return []
    if temperature != 1.0 and temperature > 0:
        scores = [s / temperature for s in scores]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def build_context(state: DecisionState, instructions: Any) -> str:
    """Build a context string from state and question instructions."""
    state_str = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )
    instr_str = (
        instructions
        if isinstance(instructions, str)
        else json.dumps(instructions, ensure_ascii=False)
    )
    return f"{state_str} {instr_str}"


def get_option_texts(question: DecisionQuestion) -> list[str]:
    """Extract option texts from a question for scoring.

    Returns a list of option descriptions that can be scored against
    the context.  For noul: ["true description", "false description"].
    For choice: one description per label.  For score: one description
    per level.
    """
    qtype = question["type"]
    if qtype == "noul":
        criteria: Any = question.get("criteria")
        if isinstance(criteria, dict):
            return [criteria.get("true", "yes"), criteria.get("false", "no")]
        return ["yes", "no"]
    if qtype == "choice":
        criteria_dict: dict[str, Any] = cast(Any, question).get("criteria", {})
        return [desc or key for key, desc in criteria_dict.items()]
    if qtype == "score":
        criteria_list: list[str] = cast(Any, question).get("criteria", [])
        return list(criteria_list)
    return []


def scores_to_answer(
    scores: list[float],
    question: DecisionQuestion,
    *,
    temperature: float = 1.0,
) -> DecisionAnswer:
    """Convert raw per-option scores into a typed decision answer.

    Applies softmax to the raw scores, then maps probabilities to the
    appropriate answer type (noul/choice/score).

    Args:
        scores: Raw per-option scores.
        question: The question being answered.
        temperature: Softmax temperature.  Lower values produce more
            peaked distributions (useful for narrow-range scores like
            cosine similarities).
    """
    probs = softmax(scores, temperature=temperature)
    qtype = question["type"]

    if qtype == "noul":
        noul_val = max(0.01, min(0.99, probs[0])) if len(probs) >= 2 else 0.5
        return NoulAnswer(type="noul", noul=noul_val)

    if qtype == "choice":
        criteria_dict: dict[str, Any] = cast(Any, question).get("criteria", {})
        keys = list(criteria_dict.keys())
        prob_dict = {k: p for k, p in zip(keys, probs, strict=True)}
        choice = max(prob_dict, key=lambda k: prob_dict[k])
        return ChoiceAnswer(
            type="choice",
            choice=choice,
            probabilities=prob_dict,
            confidence=compute_confidence(prob_dict),
        )

    if qtype == "score":
        criteria_list: list[str] = cast(Any, question).get("criteria", [])
        prob_dict = {str(i): p for i, p in enumerate(probs)}
        legend = {str(i): desc for i, desc in enumerate(criteria_list)}
        score_val = sum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            type="score",
            score=score_val,
            legend=legend,
            probabilities=prob_dict,
            confidence=compute_confidence(prob_dict),
        )

    raise ValueError(f"Unknown question type: {qtype}")


# ============================================================================
# Shared usage helpers (reranker + embedding converters)
# ============================================================================


def build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo:
    """Convert provider usage to IR usage (reranker/embedding style)."""
    usage: DecisionUsageInfo = {}
    if "total_tokens" in p_usage:
        usage["input_tokens"] = p_usage["total_tokens"]
    elif "prompt_tokens" in p_usage:
        usage["input_tokens"] = p_usage["prompt_tokens"]
    return usage


def build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
    """Convert IR usage to provider usage (reranker/embedding style)."""
    result: dict[str, Any] = {}
    if "input_tokens" in ir_usage:
        result["total_tokens"] = ir_usage["input_tokens"]
    return result

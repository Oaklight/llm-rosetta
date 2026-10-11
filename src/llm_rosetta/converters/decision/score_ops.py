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
    AssertionAnswer,
    ChoiceAnswer,
    DecisionAnswer,
    DecisionEntry,
    DecisionQuestion,
    DecisionState,
    DecisionUsageInfo,
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


def _entry_text(entry: DecisionEntry) -> str:
    """Display text for an entry: its description, else its label."""
    description = entry.get("description")
    if description is None:
        return str(entry["label"])
    if isinstance(description, str):
        return description
    return json.dumps(description, ensure_ascii=False)


def _entries(question: DecisionQuestion) -> list[DecisionEntry]:
    return list(cast(Any, question).get("criteria", []) or [])


def get_option_texts(question: DecisionQuestion) -> list[str]:
    """Extract option texts from a question for scoring.

    Returns a list of option descriptions (falling back to the label) that
    can be scored against the context.  For assertion the two entries are
    ordered ``[false, true]``; without criteria it defaults to
    ``["no", "yes"]``.
    """
    qtype = question["type"]
    if qtype == "assertion":
        entries = _entries(question)
        if not entries:
            return ["no", "yes"]
        ordered = sorted(entries, key=lambda e: bool(e["label"]))  # False < True
        return [_entry_text(e) for e in ordered]
    if qtype in ("choice", "score"):
        return [_entry_text(e) for e in _entries(question)]
    return []


def scores_to_answer(
    scores: list[float],
    question: DecisionQuestion,
    *,
    temperature: float = 1.0,
) -> DecisionAnswer:
    """Convert raw per-option scores into a typed decision answer.

    Applies softmax to the raw scores, then maps probabilities to the
    appropriate answer type (assertion/choice/score).  Probabilities are
    keyed by ``str(label)`` for choice/score.

    Args:
        scores: Raw per-option scores.
        question: The question being answered.
        temperature: Softmax temperature.  Lower values produce more
            peaked distributions (useful for narrow-range scores like
            cosine similarities).
    """
    probs = softmax(scores, temperature=temperature)
    qtype = question["type"]

    if qtype == "assertion":
        entries = _entries(question)
        if entries:
            ordered = sorted(entries, key=lambda e: bool(e["label"]))
            true_idx = 1 if bool(ordered[-1]["label"]) else 0
        else:
            true_idx = len(probs) - 1 if probs else 0
        p_true = probs[true_idx] if true_idx < len(probs) else 0.5
        return AssertionAnswer(
            type="assertion", probability=max(0.01, min(0.99, p_true))
        )

    if qtype == "choice":
        labels = [str(e["label"]) for e in _entries(question)]
        prob_dict = {k: p for k, p in zip(labels, probs, strict=True)}
        choice = max(prob_dict, key=lambda k: prob_dict[k]) if prob_dict else ""
        return ChoiceAnswer(
            type="choice",
            choice=choice,
            probabilities=prob_dict,
            confidence=compute_confidence(prob_dict),
        )

    if qtype == "score":
        labels = [str(e["label"]) for e in _entries(question)]
        prob_dict = {k: p for k, p in zip(labels, probs, strict=True)}
        score_val = sum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            type="score",
            score=score_val,
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

"""
LLM-Rosetta - Decision Score Operations

Shared scoring logic for reranker and embedding-backed decision converters.

Converts raw per-option scores (relevance scores, cosine similarities)
into typed decision answers via softmax normalization.  This is the
pure-math layer — no API calls, no model access.

Reference: oaklight/jev-explore model/src/reranker_scorer.py
"""

from __future__ import annotations

import math
from typing import Any, cast

from llm_rosetta.converters.decision.schema_ops import compute_confidence
from llm_rosetta.types.ir.decision import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionQuestion,
    DecisionState,
    NoulAnswer,
    ScoreAnswer,
)


def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax over a list of scores."""
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def build_context(state: DecisionState, instructions: Any) -> str:
    """Build a context string from state and question instructions."""
    if isinstance(state, str):
        state_str = state
    else:
        import json

        state_str = json.dumps(state, ensure_ascii=False)
    if isinstance(instructions, str):
        return f"{state_str} {instructions}"
    import json

    return f"{state_str} {json.dumps(instructions, ensure_ascii=False)}"


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
) -> DecisionAnswer:
    """Convert raw per-option scores into a typed decision answer.

    Applies softmax to the raw scores, then maps probabilities to the
    appropriate answer type (noul/choice/score).
    """
    probs = softmax(scores)
    qtype = question["type"]

    if qtype == "noul":
        noul_val = max(0.01, min(0.99, probs[0])) if len(probs) >= 2 else 0.5
        return NoulAnswer(type="noul", noul=noul_val)

    if qtype == "choice":
        criteria_dict: dict[str, Any] = cast(Any, question).get("criteria", {})
        keys = list(criteria_dict.keys())
        prob_dict = {k: p for k, p in zip(keys, probs)}
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

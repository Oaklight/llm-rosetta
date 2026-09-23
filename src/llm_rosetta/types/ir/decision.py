"""
LLM-Rosetta - IR Decision Types

Decision 中间表示类型定义
Decision intermediate representation type definitions

Decision is a distinct model paradigm alongside chat, embedding, and rerank.
Decision models evaluate a state (context) against typed questions and return
structured probabilistic answers — no text generation involved.

Three question/answer primitives:

- Noul: yes/no proposition → P(true) ∈ [0, 1]
- Choice: pick one from a labeled set → categorical probability distribution
- Score: rate on ordered levels → ordinal probability distribution + E[X]

Reference implementation: TypeSafe.ai System One (Jev)
API docs: https://docs.typesafe.ai/api
"""

import sys
from typing import Any, Literal, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict


# ============================================================================
# Question types
# ============================================================================

DecisionQuestionType = Literal["noul", "choice", "score"]


class NoulCriteria(TypedDict, total=False):
    """Optional descriptions clarifying what yes and no mean."""

    true: str
    false: str


class NoulQuestion(TypedDict):
    """A yes/no question returning P(true) ∈ [0, 1].

    Named "noul" after the middle of "ber-noul-li" — a probabilistic
    counterpart to bool, representing calibrated credence rather than
    a binary value.
    """

    type: Required[Literal["noul"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: NotRequired[NoulCriteria]


class ChoiceQuestion(TypedDict):
    """Pick one option from a labeled set with probability distribution.

    criteria maps option labels to their descriptions.  A null/None
    description means the label is self-explanatory.
    """

    type: Required[Literal["choice"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: Required[dict[str, str | None]]


class ScoreQuestion(TypedDict):
    """Rate on ordered levels with probability distribution.

    criteria is an ordered list of level descriptions (minimum 2).
    The score answer is the probability-weighted expected value
    across levels.
    """

    type: Required[Literal["score"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: Required[list[str]]


DecisionQuestion = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


# ============================================================================
# Answer types
# ============================================================================


class NoulAnswer(TypedDict):
    """Answer to a Noul question: P(true) ∈ [0, 1]."""

    type: Required[Literal["noul"]]
    noul: Required[float]


class ChoiceAnswer(TypedDict):
    """Answer to a Choice question: selected option + full distribution."""

    type: Required[Literal["choice"]]
    choice: Required[str]
    probabilities: Required[dict[str, float]]
    confidence: Required[float]


class ScoreAnswer(TypedDict):
    """Answer to a Score question: weighted score + full distribution."""

    type: Required[Literal["score"]]
    score: Required[float]
    legend: Required[dict[str, str]]
    probabilities: Required[dict[str, float]]
    confidence: Required[float]


DecisionAnswer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


# ============================================================================
# Usage
# ============================================================================


class DecisionUsageInfo(TypedDict, total=False):
    """Decision-specific token usage statistics."""

    input_tokens: int
    output_tokens: int


# ============================================================================
# Request
# ============================================================================

DecisionState = Union[str, dict[str, Any], list[Any]]


class IRDecisionRequest(TypedDict):
    """Unified IR decision request type.

    Required fields:
    - model: model identifier
    - state: context to evaluate (string, object, or array)
    - questions: map of question ID → typed question

    Optional fields:
    - provider_extensions: provider-specific parameters
    """

    model: Required[str]
    state: Required[DecisionState]
    questions: Required[dict[str, DecisionQuestion]]

    provider_extensions: NotRequired[dict[str, Any]]


# ============================================================================
# Response
# ============================================================================


class IRDecisionResponse(TypedDict):
    """Unified IR decision response type.

    answers is keyed by the same question IDs provided in the request.
    """

    object: Required[Literal["decision"]]
    model: Required[str]
    answers: Required[dict[str, DecisionAnswer]]

    usage: NotRequired[DecisionUsageInfo]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DecisionQuestionType",
    "NoulCriteria",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "DecisionQuestion",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "DecisionAnswer",
    "DecisionUsageInfo",
    "DecisionState",
    "IRDecisionRequest",
    "IRDecisionResponse",
]

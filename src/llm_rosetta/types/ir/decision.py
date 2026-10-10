"""
LLM-Rosetta - IR Decision Types

Decision 中间表示类型定义
Decision intermediate representation type definitions

Decision is a distinct model paradigm alongside chat, embedding, and rerank.
Decision models evaluate a state (context) against typed questions and return
structured probabilistic answers — no text generation involved.

Three question/answer primitives, each named for the kind of proposition it
evaluates:

- Assertion: a proposition (yes/no claim) → P(true) ∈ [0, 1]
- Choice: a categorical proposition (unordered set) → categorical distribution
- Score: an ordinal proposition (ordered levels) → ordinal distribution + E[X]

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

DecisionQuestionType = Literal["assertion", "choice", "score"]


class AssertionCriteria(TypedDict, total=False):
    """Optional descriptions clarifying what true and false mean."""

    true: str
    false: str


class AssertionQuestion(TypedDict):
    """A proposition: a yes/no claim → P(true) ∈ [0, 1].

    The probabilistic counterpart to bool — the answer is a calibrated
    credence rather than a binary value.
    """

    type: Required[Literal["assertion"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: NotRequired[AssertionCriteria]


class ChoiceQuestion(TypedDict):
    """A categorical proposition: pick one option from an unordered set.

    The answer is a categorical probability distribution over the labels.
    criteria maps option labels to their descriptions.  A null/None
    description means the label is self-explanatory.
    """

    type: Required[Literal["choice"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: Required[dict[str, str | None]]


class ScoreQuestion(TypedDict):
    """An ordinal proposition: rate on ordered levels.

    The answer is an ordinal probability distribution; the score is the
    probability-weighted expected value across levels.
    criteria is an ordered list of level descriptions (minimum 2).
    """

    type: Required[Literal["score"]]
    instructions: Required[str | dict[str, Any] | list[Any]]
    criteria: Required[list[str]]


DecisionQuestion = Union[AssertionQuestion, ChoiceQuestion, ScoreQuestion]


# ============================================================================
# Answer types
# ============================================================================


class AssertionAnswer(TypedDict):
    """Answer to an assertion question: P(true) ∈ [0, 1]."""

    type: Required[Literal["assertion"]]
    probability: Required[float]


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


DecisionAnswer = Union[AssertionAnswer, ChoiceAnswer, ScoreAnswer]


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
    "AssertionCriteria",
    "AssertionQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "DecisionQuestion",
    "AssertionAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "DecisionAnswer",
    "DecisionUsageInfo",
    "DecisionState",
    "IRDecisionRequest",
    "IRDecisionResponse",
]

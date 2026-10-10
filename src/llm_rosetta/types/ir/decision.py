"""
LLM-Rosetta - IR Decision Types

Decision 中间表示类型定义
Decision intermediate representation type definitions

Decision is a distinct model paradigm alongside chat, embedding, and rerank.
Decision models evaluate a state (context) against typed questions and return
structured probabilistic answers — no text generation involved.

Every question is a set of one or more :class:`DecisionEntry` items, and the
answer is a probability distribution over those entries. The three primitives
are named for the kind of proposition they evaluate:

- Assertion: a proposition (yes/no claim) → P(true) ∈ [0, 1]
- Choice: a categorical proposition (unordered set) → categorical distribution
- Score: an ordinal proposition (ordered levels) → ordinal distribution + E[X]

Reference implementation: TypeSafe.ai System One (Jev)
API docs: https://docs.typesafe.ai/api
"""

import sys
from typing import Any, Literal, Union

from .parts import ImagePart, TextPart

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict


# ============================================================================
# Shared value / entry types
# ============================================================================

# A plain JSON value (no binary media).
JSONValue = Union[
    str, int, float, bool, None, list["JSONValue"], dict[str, "JSONValue"]
]

# A description slot: plain text or structured data. Mirrors TypeSafe's
# ``string | object | array`` for ``instructions`` and ``criteria`` values.
Description = Union[str, dict[str, Any], list[Any]]

# A multimodal evidence part (reuses the chat IR part types).
DecisionInputPart = Union[TextPart, ImagePart]


class DecisionEntry(TypedDict):
    """A single entry a question judges: a labeled option / level / proposition.

    - ``assertion``: exactly **one** entry — the claim itself.
    - ``choice``: N entries, unordered — the option values.
    - ``score``: N entries, ordered (position = ordinal) — the levels.

    ``label`` doubles as the machine value (the key used in the answer's
    ``probabilities``) and the display name. ``description`` is an optional
    fuller rubric/meaning; for a Noul assertion it may hold the two-sided
    ``{"true": ..., "false": ...}`` clarifications.
    """

    label: Required[str | bool]
    description: NotRequired[Description]


# ============================================================================
# Question types
# ============================================================================

DecisionQuestionType = Literal["assertion", "choice", "score"]


class AssertionQuestion(TypedDict):
    """A proposition: a yes/no claim → P(true) ∈ [0, 1].

    The probabilistic counterpart to bool — the answer is a calibrated
    credence rather than a binary value. Holds 0 or 2 entries; TypeSafe's
    two-sided ``criteria{true,false}`` maps to entries labeled ``False`` /
    ``True`` whose ``description`` carries each side's meaning.
    """

    type: Required[Literal["assertion"]]
    instructions: Required[Description]
    criteria: NotRequired[list[DecisionEntry]]


class ChoiceQuestion(TypedDict):
    """A categorical proposition: pick one option from an unordered set.

    The answer is a categorical probability distribution over the entries'
    labels.
    """

    type: Required[Literal["choice"]]
    instructions: Required[Description]
    criteria: Required[list[DecisionEntry]]


class ScoreQuestion(TypedDict):
    """An ordinal proposition: rate on ordered levels.

    The answer is an ordinal probability distribution over the entries
    (position = ordinal); the score is the probability-weighted expected
    value.
    """

    type: Required[Literal["score"]]
    instructions: Required[Description]
    criteria: Required[list[DecisionEntry]]


DecisionQuestion = Union[AssertionQuestion, ChoiceQuestion, ScoreQuestion]


# ============================================================================
# Answer types
# ============================================================================


class AssertionAnswer(TypedDict):
    """Answer to an assertion question: P(true) ∈ [0, 1]."""

    type: Required[Literal["assertion"]]
    probability: Required[float]

    unknown_probability: NotRequired[float]


class ChoiceAnswer(TypedDict):
    """Answer to a choice question: selected option + full distribution."""

    type: Required[Literal["choice"]]
    choice: Required[str | bool]
    probabilities: Required[dict[str, float]]

    confidence: NotRequired[float]
    unknown_probability: NotRequired[float]


class ScoreAnswer(TypedDict):
    """Answer to a score question: weighted score + full distribution.

    ``probabilities`` is keyed by the level label (``str(entry.label)``);
    the level labels/descriptions live on the question's ``criteria``.
    """

    type: Required[Literal["score"]]
    score: Required[float]
    probabilities: Required[dict[str, float]]

    confidence: NotRequired[float]
    unknown_probability: NotRequired[float]


class RefusalAnswer(TypedDict):
    """The model declined to answer a question."""

    type: Required[Literal["refusal"]]
    reason: NotRequired[str]


DecisionAnswer = Union[AssertionAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer]


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

# Shared evidence for every question. A plain string, a structured record
# (values may be an ``ImagePart`` for image-as-a-field evidence), or an
# explicit list of content parts.
DecisionState = Union[str, dict[str, JSONValue | ImagePart], list[DecisionInputPart]]


class IRDecisionRequest(TypedDict):
    """Unified IR decision request type.

    Required fields:
    - model: model identifier
    - state: shared evidence (string, structured record, or content-part list)
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
    "JSONValue",
    "Description",
    "DecisionInputPart",
    "DecisionEntry",
    "DecisionQuestionType",
    "AssertionQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "DecisionQuestion",
    "AssertionAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "RefusalAnswer",
    "DecisionAnswer",
    "DecisionUsageInfo",
    "DecisionState",
    "IRDecisionRequest",
    "IRDecisionResponse",
]

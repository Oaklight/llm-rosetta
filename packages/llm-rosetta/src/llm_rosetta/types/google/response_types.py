"""Google GenAI response types (TypedDict replicas).

This module contains TypedDict replicas of Google GenAI SDK response types.
These are used for type hints and validation in the LLM-Rosetta conversion layer.

Reference: google.genai.types (GenerateContentResponse, Candidate, SafetyRating, etc.)
SDK Source: <python_env>/lib/python3.10/site-packages/google/genai/types.py
"""

from __future__ import annotations

import datetime
from typing import Any, Literal, TypedDict

from .content_types import Content

__all__ = [
    # Finish reason
    "FinishReason",
    # Safety
    "SafetyRating",
    # Citation
    "Citation",
    "CitationMetadata",
    # Grounding
    "GroundingAttribution",
    # Token count
    "ModalityTokenCount",
    # Candidate
    "Candidate",
    # Usage metadata
    "GenerateContentResponseUsageMetadata",
    # Prompt feedback
    "GenerateContentResponsePromptFeedback",
    # Top-level response
    "GenerateContentResponse",
]


# ============================================================================
# FinishReason (Literal type alias)
# ============================================================================

FinishReason = Literal[
    "FINISH_REASON_UNSPECIFIED",
    "STOP",
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "OTHER",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "IMAGE_SAFETY",
    "UNEXPECTED_TOOL_CALL",
    "IMAGE_PROHIBITED_CONTENT",
    "NO_IMAGE",
    "IMAGE_RECITATION",
    "IMAGE_OTHER",
]
"""The reason why the model stopped generating tokens.

Reference: google.genai.types.FinishReason
"""


# ============================================================================
# SafetyRating
# ============================================================================


class SafetyRating(TypedDict, total=False):
    """Safety rating corresponding to the generated content.

    Reference: google.genai.types.SafetyRating
    """

    blocked: bool | None
    """Indicates whether the content was filtered out because of this rating."""

    category: str | None
    """Harm category (e.g., 'HARM_CATEGORY_HARASSMENT')."""

    probability: str | None
    """Harm probability levels in the content."""

    probability_score: float | None
    """Harm probability score."""

    severity: str | None
    """Harm severity levels in the content."""

    severity_score: float | None
    """Harm severity score."""


# ============================================================================
# Citation types
# ============================================================================


class Citation(TypedDict, total=False):
    """Source attribution for content.

    Reference: google.genai.types.Citation
    """

    end_index: int | None
    """End index into the content."""

    license: str | None
    """License of the attribution."""

    publication_date: dict[str, Any] | None
    """Publication date of the attribution."""

    start_index: int | None
    """Start index into the content."""

    title: str | None
    """Title of the attribution."""

    uri: str | None
    """URL reference of the attribution."""


class CitationMetadata(TypedDict, total=False):
    """Citation information when the model quotes another source.

    Reference: google.genai.types.CitationMetadata
    """

    citations: list[Citation] | None
    """Contains citation information when the model directly quotes from
    another source."""


# ============================================================================
# Grounding Attribution
# ============================================================================


class GroundingAttribution(TypedDict, total=False):
    """Grounding attribution information.

    Reference: google.genai.types.GroundingAttribution (via GroundingMetadata)
    """

    source_id: str | None
    """Source identifier."""

    content: dict[str, Any] | None
    """Attribution content."""


# ============================================================================
# ModalityTokenCount
# ============================================================================


class ModalityTokenCount(TypedDict, total=False):
    """Represents token counting info for a single modality.

    Reference: google.genai.types.ModalityTokenCount
    """

    modality: str | None
    """The modality associated with this token count."""

    token_count: int | None
    """Number of tokens."""


# ============================================================================
# Candidate
# ============================================================================


class Candidate(TypedDict, total=False):
    """A response candidate generated from the model.

    Reference: google.genai.types.Candidate
    """

    content: Content | None
    """Contains the multi-part content of the response."""

    citation_metadata: CitationMetadata | None
    """Source attribution of the generated content."""

    finish_message: str | None
    """Describes the reason the model stopped generating tokens."""

    token_count: int | None
    """Number of tokens for this candidate."""

    finish_reason: FinishReason | None
    """The reason why the model stopped generating tokens."""

    avg_logprobs: float | None
    """Average log probability score of the candidate."""

    grounding_metadata: dict[str, Any] | None
    """Metadata specifying sources used to ground generated content."""

    index: int | None
    """Index of the candidate."""

    logprobs_result: dict[str, Any] | None
    """Log-likelihood scores for the response tokens and top tokens."""

    safety_ratings: list[SafetyRating] | None
    """List of ratings for the safety of a response candidate."""


# ============================================================================
# GenerateContentResponseUsageMetadata
# ============================================================================


class GenerateContentResponseUsageMetadata(TypedDict, total=False):
    """Usage metadata about the content generation request and response.

    Reference: google.genai.types.GenerateContentResponseUsageMetadata
    """

    cache_tokens_details: list[ModalityTokenCount] | None
    """A detailed breakdown of the token count for each modality in the
    cached content."""

    cached_content_token_count: int | None
    """The number of tokens in the cached content."""

    candidates_token_count: int | None
    """The total number of tokens in the generated candidates."""

    candidates_tokens_details: list[ModalityTokenCount] | None
    """A detailed breakdown of the token count for each modality in the
    generated candidates."""

    prompt_token_count: int | None
    """The total number of tokens in the prompt."""

    prompt_tokens_details: list[ModalityTokenCount] | None
    """A detailed breakdown of the token count for each modality in the
    prompt."""

    thoughts_token_count: int | None
    """The number of tokens that were part of the model's generated
    'thoughts' output."""

    tool_use_prompt_token_count: int | None
    """The number of tokens in the results from tool executions."""

    tool_use_prompt_tokens_details: list[ModalityTokenCount] | None
    """A detailed breakdown by modality of the token counts from the results
    of tool executions."""

    total_token_count: int | None
    """The total number of tokens for the entire request."""

    traffic_type: str | None
    """The traffic type for this request."""


# ============================================================================
# GenerateContentResponsePromptFeedback
# ============================================================================


class GenerateContentResponsePromptFeedback(TypedDict, total=False):
    """Content filter results for a prompt sent in the request.

    Reference: google.genai.types.GenerateContentResponsePromptFeedback
    """

    block_reason: str | None
    """The reason why the prompt was blocked."""

    block_reason_message: str | None
    """A readable message that explains the reason why the prompt was
    blocked."""

    safety_ratings: list[SafetyRating] | None
    """A list of safety ratings for the prompt."""


# ============================================================================
# GenerateContentResponse
# ============================================================================


class GenerateContentResponse(TypedDict, total=False):
    """Response message for GenerateContent.

    Reference: google.genai.types.GenerateContentResponse
    """

    candidates: list[Candidate] | None
    """Response variations returned by the model."""

    create_time: datetime.datetime | None
    """Timestamp when the request is made to the server."""

    model_version: str | None
    """The model version used to generate the response."""

    prompt_feedback: GenerateContentResponsePromptFeedback | None
    """Content filter results for a prompt sent in the request."""

    response_id: str | None
    """Response identifier."""

    usage_metadata: GenerateContentResponseUsageMetadata | None
    """Usage metadata about the response(s)."""

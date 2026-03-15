"""Google GenAI request parameter types (TypedDict replicas).

This module contains TypedDict replicas of Google GenAI SDK request types.
These are used for type hints and validation in the LLM-Rosetta conversion layer.

Reference: google.genai.types (GenerateContentConfig, FunctionDeclaration, Tool, etc.)
SDK Source: <python_env>/lib/python3.10/site-packages/google/genai/types.py
"""

from __future__ import annotations

import sys
from typing import Any, TypedDict

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

from .content_types import Content, Part

__all__ = [
    # Schema
    "Schema",
    # Function declaration
    "FunctionDeclaration",
    # Tool
    "Tool",
    # Safety setting
    "SafetySetting",
    # Thinking config
    "ThinkingConfig",
    # Generation config (flat config object)
    "GenerateContentConfig",
    # Top-level request
    "GenerateContentRequest",
]


# ============================================================================
# Schema (simplified JSON Schema representation)
# ============================================================================


class Schema(TypedDict, total=False):
    """Schema object for structured output definitions.

    Reference: google.genai.types.Schema

    A select subset of an OpenAPI 3.0 schema object.
    """

    type: str | None
    """Required. Data type."""

    description: str | None
    """Optional. A brief description of the parameter."""

    enum: list[str] | None
    """Optional. Possible values of the element."""

    example: Any | None
    """Optional. Example of the object."""

    format: str | None
    """Optional. The format of the data."""

    items: Schema | None
    """Optional. Schema of the elements of Type.ARRAY."""

    max_items: int | None
    """Optional. Maximum number of the elements for Type.ARRAY."""

    min_items: int | None
    """Optional. Minimum number of the elements for Type.ARRAY."""

    maximum: float | None
    """Optional. Maximum value of the Type.INTEGER and Type.NUMBER."""

    minimum: float | None
    """Optional. Minimum value of the Type.INTEGER and Type.NUMBER."""

    nullable: bool | None
    """Optional. Indicates if the value may be null."""

    properties: dict[str, Schema] | None
    """Optional. Properties of Type.OBJECT."""

    required: list[str] | None
    """Optional. Required properties of Type.OBJECT."""

    title: str | None
    """Optional. The title of the Schema."""


# ============================================================================
# FunctionDeclaration
# ============================================================================


class FunctionDeclaration(TypedDict, total=False):
    """Defines a function that the model can generate JSON inputs for.

    Reference: google.genai.types.FunctionDeclaration
    """

    name: str | None
    """Required. The name of the function to call."""

    description: str | None
    """Optional. Description and purpose of the function."""

    parameters: Schema | None
    """Optional. Describes the parameters to this function in JSON Schema
    Object format."""

    parameters_json_schema: Any | None
    """Optional. Describes the parameters in JSON Schema format.
    Mutually exclusive with ``parameters``."""

    response: Schema | None
    """Optional. Describes the output from this function in JSON Schema format."""

    response_json_schema: Any | None
    """Optional. Describes the output in JSON Schema format.
    Mutually exclusive with ``response``."""


# ============================================================================
# Tool
# ============================================================================


class Tool(TypedDict, total=False):
    """Tool details that the model may use to generate a response.

    Reference: google.genai.types.Tool
    """

    function_declarations: list[FunctionDeclaration] | None
    """List of function declarations that the tool supports."""

    code_execution: dict[str, Any] | None
    """Optional. CodeExecution tool type."""

    google_search: dict[str, Any] | None
    """Optional. GoogleSearch tool type."""

    google_search_retrieval: dict[str, Any] | None
    """Optional. Specialized retrieval tool powered by Google Search."""


# ============================================================================
# SafetySetting
# ============================================================================


class SafetySetting(TypedDict, total=False):
    """Safety settings for content generation.

    Reference: google.genai.types.SafetySetting
    """

    category: str | None
    """Required. Harm category (e.g., 'HARM_CATEGORY_HARASSMENT')."""

    method: str | None
    """Optional. Specify if the threshold is used for probability or severity
    score."""

    threshold: str | None
    """Required. The harm block threshold
    (e.g., 'BLOCK_MEDIUM_AND_ABOVE')."""


# ============================================================================
# ThinkingConfig
# ============================================================================


class ThinkingConfig(TypedDict, total=False):
    """The thinking features configuration.

    Reference: google.genai.types.ThinkingConfig
    """

    include_thoughts: bool | None
    """Indicates whether to include thoughts in the response."""

    thinking_budget: int | None
    """Indicates the thinking budget in tokens.
    0 is DISABLED. -1 is AUTOMATIC."""

    thinking_level: str | None
    """Optional. The thinking level for the model."""


# ============================================================================
# GenerateContentConfig (the ``config`` parameter)
# ============================================================================


class GenerateContentConfig(TypedDict, total=False):
    """Optional model configuration parameters.

    This corresponds to the ``config`` parameter of
    ``google_client.models.generate_content()``.

    Reference: google.genai.types.GenerateContentConfig
    """

    # System instruction
    system_instruction: Content | str | None
    """Instructions for the model to steer it toward better performance."""

    # Generation parameters
    temperature: float | None
    """Controls the degree of randomness in token selection."""

    top_p: float | None
    """Nucleus sampling parameter."""

    top_k: float | None
    """Top-k sampling parameter."""

    candidate_count: int | None
    """Number of response variations to return."""

    max_output_tokens: int | None
    """Maximum number of tokens that can be generated in the response."""

    stop_sequences: list[str] | None
    """List of strings that tells the model to stop generating text."""

    response_logprobs: bool | None
    """Whether to return the log probabilities of the tokens."""

    logprobs: int | None
    """Number of top candidate tokens to return the log probabilities for."""

    presence_penalty: float | None
    """Positive values penalize tokens that already appear in the generated
    text."""

    frequency_penalty: float | None
    """Positive values penalize tokens that repeatedly appear in the generated
    text."""

    seed: int | None
    """Random seed for reproducible outputs."""

    response_mime_type: str | None
    """Output response MIME type of the generated candidate text."""

    response_schema: Schema | dict[str, Any] | None
    """The Schema object for structured output definitions."""

    response_json_schema: Any | None
    """Optional. Output schema in JSON Schema format.
    Mutually exclusive with ``response_schema``."""

    response_modalities: list[str] | None
    """The requested modalities of the response."""

    # Safety and tools
    safety_settings: list[SafetySetting] | None
    """Safety settings to block unsafe content in the response."""

    tools: list[Tool] | None
    """Tools that the model may use to generate a response."""

    tool_config: dict[str, Any] | None
    """Associates model output to a specific function call."""

    # Caching
    cached_content: str | None
    """Resource name of a context cache."""

    # Thinking
    thinking_config: ThinkingConfig | None
    """The thinking features configuration."""

    # Media
    media_resolution: str | None
    """If specified, the media resolution to use."""

    speech_config: str | dict[str, Any] | None
    """The speech generation configuration."""

    audio_timestamp: bool | None
    """If enabled, audio timestamp will be included in the request."""


# ============================================================================
# GenerateContentRequest (top-level request structure)
# ============================================================================


class GenerateContentRequest(TypedDict, total=False):
    """Google GenerativeAI top-level request structure.

    This represents the complete request for
    ``google_client.models.generate_content()``.

    Reference: google.genai.models.generate_content() signature
    """

    model: Required[str]
    """Required. Model ID, e.g. 'gemini-2.0-flash'."""

    contents: Required[list[Content] | list[Part] | str]
    """Required. Input content list."""

    config: GenerateContentConfig | None
    """Optional. Model configuration parameters."""

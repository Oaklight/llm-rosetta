"""Google GenAI request parameter types (TypedDict replicas).

This module contains TypedDict replicas of Google GenAI SDK request types.
These are used for type hints and validation in the LLM-Rosetta conversion layer.

Reference: google.genai.types (GenerateContentConfig, FunctionDeclaration, Tool, etc.)
SDK Source: <python_env>/lib/python3.10/site-packages/google/genai/types.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, TypedDict, Union

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

    type: Optional[str]
    """Required. Data type."""

    description: Optional[str]
    """Optional. A brief description of the parameter."""

    enum: Optional[List[str]]
    """Optional. Possible values of the element."""

    example: Optional[Any]
    """Optional. Example of the object."""

    format: Optional[str]
    """Optional. The format of the data."""

    items: Optional[Schema]
    """Optional. Schema of the elements of Type.ARRAY."""

    max_items: Optional[int]
    """Optional. Maximum number of the elements for Type.ARRAY."""

    min_items: Optional[int]
    """Optional. Minimum number of the elements for Type.ARRAY."""

    maximum: Optional[float]
    """Optional. Maximum value of the Type.INTEGER and Type.NUMBER."""

    minimum: Optional[float]
    """Optional. Minimum value of the Type.INTEGER and Type.NUMBER."""

    nullable: Optional[bool]
    """Optional. Indicates if the value may be null."""

    properties: Optional[Dict[str, Schema]]
    """Optional. Properties of Type.OBJECT."""

    required: Optional[List[str]]
    """Optional. Required properties of Type.OBJECT."""

    title: Optional[str]
    """Optional. The title of the Schema."""


# ============================================================================
# FunctionDeclaration
# ============================================================================


class FunctionDeclaration(TypedDict, total=False):
    """Defines a function that the model can generate JSON inputs for.

    Reference: google.genai.types.FunctionDeclaration
    """

    name: Optional[str]
    """Required. The name of the function to call."""

    description: Optional[str]
    """Optional. Description and purpose of the function."""

    parameters: Optional[Schema]
    """Optional. Describes the parameters to this function in JSON Schema
    Object format."""

    parameters_json_schema: Optional[Any]
    """Optional. Describes the parameters in JSON Schema format.
    Mutually exclusive with ``parameters``."""

    response: Optional[Schema]
    """Optional. Describes the output from this function in JSON Schema format."""

    response_json_schema: Optional[Any]
    """Optional. Describes the output in JSON Schema format.
    Mutually exclusive with ``response``."""


# ============================================================================
# Tool
# ============================================================================


class Tool(TypedDict, total=False):
    """Tool details that the model may use to generate a response.

    Reference: google.genai.types.Tool
    """

    function_declarations: Optional[List[FunctionDeclaration]]
    """List of function declarations that the tool supports."""

    code_execution: Optional[Dict[str, Any]]
    """Optional. CodeExecution tool type."""

    google_search: Optional[Dict[str, Any]]
    """Optional. GoogleSearch tool type."""

    google_search_retrieval: Optional[Dict[str, Any]]
    """Optional. Specialized retrieval tool powered by Google Search."""


# ============================================================================
# SafetySetting
# ============================================================================


class SafetySetting(TypedDict, total=False):
    """Safety settings for content generation.

    Reference: google.genai.types.SafetySetting
    """

    category: Optional[str]
    """Required. Harm category (e.g., 'HARM_CATEGORY_HARASSMENT')."""

    method: Optional[str]
    """Optional. Specify if the threshold is used for probability or severity
    score."""

    threshold: Optional[str]
    """Required. The harm block threshold
    (e.g., 'BLOCK_MEDIUM_AND_ABOVE')."""


# ============================================================================
# ThinkingConfig
# ============================================================================


class ThinkingConfig(TypedDict, total=False):
    """The thinking features configuration.

    Reference: google.genai.types.ThinkingConfig
    """

    include_thoughts: Optional[bool]
    """Indicates whether to include thoughts in the response."""

    thinking_budget: Optional[int]
    """Indicates the thinking budget in tokens.
    0 is DISABLED. -1 is AUTOMATIC."""

    thinking_level: Optional[str]
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
    system_instruction: Optional[Union[Content, str]]
    """Instructions for the model to steer it toward better performance."""

    # Generation parameters
    temperature: Optional[float]
    """Controls the degree of randomness in token selection."""

    top_p: Optional[float]
    """Nucleus sampling parameter."""

    top_k: Optional[float]
    """Top-k sampling parameter."""

    candidate_count: Optional[int]
    """Number of response variations to return."""

    max_output_tokens: Optional[int]
    """Maximum number of tokens that can be generated in the response."""

    stop_sequences: Optional[List[str]]
    """List of strings that tells the model to stop generating text."""

    response_logprobs: Optional[bool]
    """Whether to return the log probabilities of the tokens."""

    logprobs: Optional[int]
    """Number of top candidate tokens to return the log probabilities for."""

    presence_penalty: Optional[float]
    """Positive values penalize tokens that already appear in the generated
    text."""

    frequency_penalty: Optional[float]
    """Positive values penalize tokens that repeatedly appear in the generated
    text."""

    seed: Optional[int]
    """Random seed for reproducible outputs."""

    response_mime_type: Optional[str]
    """Output response MIME type of the generated candidate text."""

    response_schema: Optional[Union[Schema, Dict[str, Any]]]
    """The Schema object for structured output definitions."""

    response_json_schema: Optional[Any]
    """Optional. Output schema in JSON Schema format.
    Mutually exclusive with ``response_schema``."""

    response_modalities: Optional[List[str]]
    """The requested modalities of the response."""

    # Safety and tools
    safety_settings: Optional[List[SafetySetting]]
    """Safety settings to block unsafe content in the response."""

    tools: Optional[List[Tool]]
    """Tools that the model may use to generate a response."""

    tool_config: Optional[Dict[str, Any]]
    """Associates model output to a specific function call."""

    # Caching
    cached_content: Optional[str]
    """Resource name of a context cache."""

    # Thinking
    thinking_config: Optional[ThinkingConfig]
    """The thinking features configuration."""

    # Media
    media_resolution: Optional[str]
    """If specified, the media resolution to use."""

    speech_config: Optional[Union[str, Dict[str, Any]]]
    """The speech generation configuration."""

    audio_timestamp: Optional[bool]
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

    contents: Required[Union[List[Content], List[Part], str]]
    """Required. Input content list."""

    config: Optional[GenerateContentConfig]
    """Optional. Model configuration parameters."""

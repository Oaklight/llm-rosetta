"""Google GenAI types (TypedDict replicas).

This package contains TypedDict replicas of Google GenAI SDK types for use
in LLMIR. These types are used for type hints and validation in the
conversion layer.

The types are organized into three modules:

- content_types: Content and Part types (Blob, FileData, FunctionCall, etc.)
- request_types: Request parameter types (GenerateContentConfig, Tool, etc.)
- response_types: Response types (Candidate, GenerateContentResponse, etc.)
"""

# Content types
from .content_types import (
    Blob,
    CodeExecutionResult,
    CodeExecutionResultPart,
    Content,
    ExecutableCode,
    ExecutableCodePart,
    FileData,
    FileDataPart,
    FunctionCall,
    FunctionCallPart,
    FunctionResponse,
    FunctionResponsePart,
    InlineDataPart,
    Part,
    PartUnion,
    TextPart,
)

# Request types
from .request_types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentRequest,
    SafetySetting,
    Schema,
    ThinkingConfig,
    Tool,
)

# Response types
from .response_types import (
    Candidate,
    Citation,
    CitationMetadata,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    GenerateContentResponseUsageMetadata,
    GroundingAttribution,
    ModalityTokenCount,
    SafetyRating,
)

__all__ = [
    # Content types
    "Blob",
    "CodeExecutionResult",
    "CodeExecutionResultPart",
    "Content",
    "ExecutableCode",
    "ExecutableCodePart",
    "FileData",
    "FileDataPart",
    "FunctionCall",
    "FunctionCallPart",
    "FunctionResponse",
    "FunctionResponsePart",
    "InlineDataPart",
    "Part",
    "PartUnion",
    "TextPart",
    # Request types
    "FunctionDeclaration",
    "GenerateContentConfig",
    "GenerateContentRequest",
    "SafetySetting",
    "Schema",
    "ThinkingConfig",
    "Tool",
    # Response types
    "Candidate",
    "Citation",
    "CitationMetadata",
    "FinishReason",
    "GenerateContentResponse",
    "GenerateContentResponsePromptFeedback",
    "GenerateContentResponseUsageMetadata",
    "GroundingAttribution",
    "ModalityTokenCount",
    "SafetyRating",
]

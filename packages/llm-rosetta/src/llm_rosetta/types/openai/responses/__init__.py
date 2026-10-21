"""OpenAI Responses API types (TypedDict replicas).

This package contains TypedDict replicas of OpenAI SDK's Responses API types
for use in LLM-Rosetta. These types are used for type hints and validation in the
conversion layer.

The types are organized into two modules:
- request_types: Request parameter types (input, tools, config)
- response_types: Response types (18 output item types, usage, status)
"""

# Request types
from .request_types import (
    AudioInputParam,
    Conversation,
    FunctionToolParam,
    ImageInputParam,
    Metadata,
    Reasoning,
    ResponseCreateParams,
    ResponseIncludable,
    ResponseInputParam,
    ResponsePromptParam,
    ResponseTextConfigParam,
    StreamOptions,
    TextInputParam,
    ToolChoice,
)

# Response types
from .response_types import (
    Action,
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    CodeInterpreterOutput,
    ImageGenerationCall,
    IncompleteDetails,
    InputTokensDetails,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
    OutputImage,
    OutputLogs,
    OutputTokensDetails,
    ReasoningContent,
    ReasoningSummary,
    Response,
    ResponseApplyPatchToolCall,
    ResponseApplyPatchToolCallOutput,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseCompactionItem,
    ResponseCustomToolCall,
    ResponseError,
    ResponseFileSearchToolCall,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseStatus,
    ResponseUsage,
)

__all__ = [
    # Request types - Input
    "TextInputParam",
    "ImageInputParam",
    "AudioInputParam",
    "ResponseInputParam",
    # Request types - Config
    "ResponsePromptParam",
    "ResponseTextConfigParam",
    "StreamOptions",
    "Reasoning",
    # Request types - Tools
    "FunctionToolParam",
    "ToolChoice",
    # Request types - Metadata
    "Metadata",
    "ResponseIncludable",
    "Conversation",
    # Request types - Main
    "ResponseCreateParams",
    # Response types - Status and error
    "ResponseStatus",
    "ResponseError",
    "IncompleteDetails",
    # Response types - Usage
    "InputTokensDetails",
    "OutputTokensDetails",
    "ResponseUsage",
    # Response types - Content output
    "ResponseOutputText",
    "ResponseOutputRefusal",
    # Response types - Message output
    "ResponseOutputMessage",
    # Response types - Reasoning
    "ReasoningSummary",
    "ReasoningContent",
    "ResponseReasoningItem",
    # Response types - Function tool calls
    "ResponseFunctionToolCall",
    "ResponseCustomToolCall",
    # Response types - Web search
    "ActionSearch",
    "ActionOpenPage",
    "ActionFind",
    "Action",
    "ResponseFunctionWebSearch",
    # Response types - Code interpreter
    "OutputLogs",
    "OutputImage",
    "CodeInterpreterOutput",
    "ResponseCodeInterpreterToolCall",
    # Response types - File search
    "ResponseFileSearchToolCall",
    # Response types - Computer tool
    "ResponseComputerToolCall",
    # Response types - Shell and patch
    "ResponseFunctionShellToolCall",
    "ResponseApplyPatchToolCall",
    # Response types - Tool outputs
    "ResponseFunctionShellToolCallOutput",
    "ResponseApplyPatchToolCallOutput",
    # Response types - MCP
    "McpCall",
    "McpListTools",
    "McpApprovalRequest",
    # Response types - Special
    "LocalShellCall",
    "ImageGenerationCall",
    "ResponseCompactionItem",
    # Response types - Union
    "ResponseOutputItem",
    # Response types - Main
    "Response",
]

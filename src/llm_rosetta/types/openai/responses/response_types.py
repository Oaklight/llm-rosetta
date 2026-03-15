"""OpenAI Responses API response types (TypedDict replicas).

This module contains TypedDict replicas of OpenAI SDK's Responses API response types.
These types are used for type hints and validation in the LLM-Rosetta conversion layer.

The OpenAI Responses API has 18 distinct output item types, making it the most
complex response structure among all supported providers.

Supported OpenAI SDK Versions: 1.x.x through 2.14.0

Reference: openai.types.responses
SDK Source: <python_env>/lib/python3.10/site-packages/openai/types/responses/
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

__all__ = [
    # Status and error types
    "ResponseStatus",
    "ResponseError",
    "IncompleteDetails",
    # Usage types
    "InputTokensDetails",
    "OutputTokensDetails",
    "ResponseUsage",
    # Content output types
    "ResponseOutputText",
    "ResponseOutputRefusal",
    # Message output
    "ResponseOutputMessage",
    # Tool call types
    "ResponseFunctionToolCall",
    "ResponseCustomToolCall",
    "ResponseReasoningItem",
    "ActionSearch",
    "ActionOpenPage",
    "ActionFind",
    "Action",
    "ResponseFunctionWebSearch",
    "OutputLogs",
    "OutputImage",
    "CodeInterpreterOutput",
    "ResponseCodeInterpreterToolCall",
    "ResponseFileSearchToolCall",
    "ResponseComputerToolCall",
    "ResponseFunctionShellToolCall",
    "ResponseApplyPatchToolCall",
    # Special types
    "McpCall",
    "McpListTools",
    "McpApprovalRequest",
    "LocalShellCall",
    "ImageGenerationCall",
    "ResponseCompactionItem",
    # Tool output types
    "ResponseFunctionShellToolCallOutput",
    "ResponseApplyPatchToolCallOutput",
    # Union types
    "ResponseOutputItem",
    # Main response
    "Response",
]


# ============================================================================
# Status and Error Types
# ============================================================================

ResponseStatus = Literal["in_progress", "completed", "incomplete", "failed"]
"""Response status.

Reference: openai.types.responses.ResponseStatus
"""


class ResponseError(TypedDict, total=False):
    """Error information in a response.

    Reference: openai.types.responses.ResponseError
    """

    code: Literal[
        "server_error",
        "rate_limit_exceeded",
        "invalid_prompt",
        "vector_store_timeout",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    ]
    """Error code."""

    message: str
    """Human-readable error message."""


class IncompleteDetails(TypedDict, total=False):
    """Details about why a response is incomplete.

    Reference: openai.types.responses.IncompleteDetails
    """

    reason: str
    """The reason the response is incomplete."""


# ============================================================================
# Usage Types
# ============================================================================


class InputTokensDetails(TypedDict, total=False):
    """Detailed breakdown of input token usage.

    Reference: openai.types.responses.response_usage.InputTokensDetails
    """

    cached_tokens: int
    """Number of cached tokens."""


class OutputTokensDetails(TypedDict, total=False):
    """Detailed breakdown of output token usage.

    Reference: openai.types.responses.response_usage.OutputTokensDetails
    """

    reasoning_tokens: int
    """Number of reasoning tokens."""


class ResponseUsage(TypedDict, total=False):
    """Token usage statistics for the Responses API.

    Reference: openai.types.responses.ResponseUsage
    """

    input_tokens: int
    """Number of tokens in the input."""

    input_tokens_details: InputTokensDetails
    """Detailed breakdown of input token usage."""

    output_tokens: int
    """Number of tokens in the output."""

    output_tokens_details: OutputTokensDetails
    """Detailed breakdown of output token usage."""

    total_tokens: int
    """Total number of tokens used."""


# ============================================================================
# Content Output Types
# ============================================================================


class ResponseOutputText(TypedDict, total=False):
    """Text output content in a response message.

    Reference: openai.types.responses.ResponseOutputText
    """

    type: Literal["output_text"]
    """Content type, always 'output_text'."""

    text: str
    """The text content."""

    annotations: list[dict[str, Any]]
    """Annotations on the text (e.g., URL citations, file citations)."""

    logprobs: list[dict[str, Any]] | None
    """Log probability information for the text."""


class ResponseOutputRefusal(TypedDict, total=False):
    """Refusal output content in a response message.

    Reference: openai.types.responses.ResponseOutputRefusal
    """

    type: Literal["refusal"]
    """Content type, always 'refusal'."""

    refusal: str
    """The refusal message."""


# ============================================================================
# Message Output
# ============================================================================


class ResponseOutputMessage(TypedDict, total=False):
    """Assistant message output item.

    Reference: openai.types.responses.ResponseOutputMessage
    """

    id: str
    """Unique ID of the output message."""

    type: Literal["message"]
    """Output item type, always 'message'."""

    role: Literal["assistant"]
    """Message role, always 'assistant'."""

    content: list[ResponseOutputText | ResponseOutputRefusal]
    """Message content list."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Message status."""


# ============================================================================
# Reasoning Item
# ============================================================================


class ReasoningSummary(TypedDict, total=False):
    """Summary entry in a reasoning item.

    Reference: openai.types.responses.response_reasoning_item.Summary
    """

    type: Literal["summary_text"]
    """Summary type, always 'summary_text'."""

    text: str
    """The summary text."""


class ReasoningContent(TypedDict, total=False):
    """Content entry in a reasoning item.

    Reference: openai.types.responses.response_reasoning_item.Content
    """

    type: Literal["reasoning_text"]
    """Content type, always 'reasoning_text'."""

    text: str
    """The reasoning text."""


class ResponseReasoningItem(TypedDict, total=False):
    """Reasoning process item.

    Contains the model's internal reasoning steps, including summaries
    and optionally the full reasoning content.

    Reference: openai.types.responses.ResponseReasoningItem
    """

    type: Literal["reasoning"]
    """Output item type, always 'reasoning'."""

    id: str
    """Unique ID of the reasoning item."""

    summary: list[ReasoningSummary]
    """Reasoning summary content."""

    content: list[ReasoningContent] | None
    """Full reasoning text content (if included)."""

    encrypted_content: str | None
    """Encrypted reasoning content."""

    status: Literal["in_progress", "completed", "incomplete"] | None
    """Reasoning status."""


# ============================================================================
# Function Tool Call Types
# ============================================================================


class ResponseFunctionToolCall(TypedDict, total=False):
    """Function tool call output item.

    Reference: openai.types.responses.ResponseFunctionToolCall
    """

    type: Literal["function_call"]
    """Output item type, always 'function_call'."""

    call_id: str
    """Unique ID for this tool call, generated by the model."""

    name: str
    """Name of the function to call."""

    arguments: str
    """JSON string of function arguments."""

    id: str | None
    """Platform-level unique ID for the tool call."""

    status: Literal["in_progress", "completed", "incomplete"] | None
    """Execution status."""


class ResponseCustomToolCall(TypedDict, total=False):
    """Custom tool call output item.

    Reference: openai.types.responses.ResponseCustomToolCall
    """

    type: Literal["custom_tool_call"]
    """Output item type, always 'custom_tool_call'."""

    call_id: str
    """Unique ID for this custom tool call."""

    name: str
    """Name of the custom tool."""

    input: str
    """Model-generated input data for the tool."""

    id: str | None
    """Platform-level unique ID."""


# ============================================================================
# Web Search Types
# ============================================================================


class ActionSearch(TypedDict, total=False):
    """Web search action - execute a search query.

    Reference: openai.types.responses.response_function_web_search.ActionSearch
    """

    type: Literal["search"]
    """Action type, always 'search'."""

    query: str
    """The search query."""


class ActionOpenPage(TypedDict, total=False):
    """Open page action - open a URL from search results.

    Reference: openai.types.responses.response_function_web_search.ActionOpenPage
    """

    type: Literal["open_page"]
    """Action type, always 'open_page'."""

    url: str
    """The URL to open."""


class ActionFind(TypedDict, total=False):
    """Find action - search for a pattern in a loaded page.

    Reference: openai.types.responses.response_function_web_search.ActionFind
    """

    type: Literal["find"]
    """Action type, always 'find'."""

    pattern: str
    """The pattern to search for."""


Action = Union[ActionSearch, ActionOpenPage, ActionFind]
"""Union of all web search action types.

Reference: openai.types.responses.response_function_web_search.Action
"""


class ResponseFunctionWebSearch(TypedDict, total=False):
    """Web search function call output item.

    Reference: openai.types.responses.ResponseFunctionWebSearch
    """

    type: Literal["web_search_call"]
    """Output item type, always 'web_search_call'."""

    id: str
    """Unique ID for this web search call."""

    action: Action
    """The search action details."""

    status: Literal["in_progress", "searching", "completed", "failed"]
    """Search execution status."""


# ============================================================================
# Code Interpreter Types
# ============================================================================


class OutputLogs(TypedDict, total=False):
    """Code execution log output.

    Reference: openai.types.responses.response_code_interpreter_tool_call.OutputLogs
    """

    type: Literal["logs"]
    """Output type, always 'logs'."""

    logs: str
    """The log output text."""


class OutputImage(TypedDict, total=False):
    """Code-generated image output.

    Reference: openai.types.responses.response_code_interpreter_tool_call.OutputImage
    """

    type: Literal["image"]
    """Output type, always 'image'."""

    image: str
    """Base64-encoded image data."""


CodeInterpreterOutput = Union[OutputLogs, OutputImage]
"""Union of code interpreter output types.

Reference: openai.types.responses.response_code_interpreter_tool_call.Output
"""


class ResponseCodeInterpreterToolCall(TypedDict, total=False):
    """Code interpreter tool call output item.

    Reference: openai.types.responses.ResponseCodeInterpreterToolCall
    """

    type: Literal["code_interpreter_call"]
    """Output item type, always 'code_interpreter_call'."""

    id: str
    """Unique ID for this code interpreter call."""

    container_id: str
    """ID of the container running the code."""

    code: str | None
    """The code to execute."""

    outputs: list[CodeInterpreterOutput] | None
    """Code execution outputs (logs or images)."""

    status: Literal["in_progress", "completed", "incomplete", "interpreting", "failed"]
    """Execution status."""


# ============================================================================
# File Search Types
# ============================================================================


class ResponseFileSearchToolCall(TypedDict, total=False):
    """File search tool call output item.

    Reference: openai.types.responses.ResponseFileSearchToolCall
    """

    type: Literal["file_search_call"]
    """Output item type, always 'file_search_call'."""

    id: str
    """Unique ID for this file search call."""

    query: str
    """The search query."""

    results: list[dict[str, Any]] | None
    """Search results."""

    status: str | None
    """Execution status."""


# ============================================================================
# Computer Tool Types
# ============================================================================


class ResponseComputerToolCall(TypedDict, total=False):
    """Computer tool call output item.

    Reference: openai.types.responses.ResponseComputerToolCall
    """

    type: Literal["computer_tool_call"]
    """Output item type, always 'computer_tool_call'."""

    id: str
    """Unique ID for this computer tool call."""

    action: str
    """The computer action to perform."""

    parameters: dict[str, Any]
    """Action parameters."""

    status: str | None
    """Execution status."""


# ============================================================================
# Shell and Patch Tool Types
# ============================================================================


class ResponseFunctionShellToolCall(TypedDict, total=False):
    """Shell function tool call output item.

    Reference: openai.types.responses.ResponseFunctionShellToolCall
    """

    type: Literal["shell_call"]
    """Output item type, always 'shell_call'."""

    id: str
    """Unique ID for this shell call."""

    command: str
    """The shell command to execute."""

    output: str | None
    """Command output."""

    status: str | None
    """Execution status."""


class ResponseApplyPatchToolCall(TypedDict, total=False):
    """Apply patch tool call output item.

    Reference: openai.types.responses.ResponseApplyPatchToolCall
    """

    type: Literal["apply_patch_call"]
    """Output item type, always 'apply_patch_call'."""

    id: str
    """Unique ID for this patch call."""

    patch: str
    """The patch content to apply."""

    file_path: str
    """Target file path for the patch."""

    status: str | None
    """Execution status."""


# ============================================================================
# Tool Output Types
# ============================================================================


class ResponseFunctionShellToolCallOutput(TypedDict, total=False):
    """Shell function tool call output result.

    Reference: openai.types.responses.ResponseFunctionShellToolCallOutput
    """

    type: Literal["shell_output"]
    """Output type, always 'shell_output'."""

    call_id: str
    """ID of the associated shell call."""

    output: str
    """The shell command output."""


class ResponseApplyPatchToolCallOutput(TypedDict, total=False):
    """Apply patch tool call output result.

    Reference: openai.types.responses.ResponseApplyPatchToolCallOutput
    """

    type: Literal["patch_output"]
    """Output type, always 'patch_output'."""

    call_id: str
    """ID of the associated patch call."""

    result: str
    """The patch application result."""


# ============================================================================
# MCP (Model Context Protocol) Types
# ============================================================================


class McpCall(TypedDict, total=False):
    """MCP (Model Context Protocol) call output item.

    Reference: openai.types.responses.McpCall
    """

    type: Literal["mcp_call"]
    """Output item type, always 'mcp_call'."""

    id: str
    """Unique ID for this MCP call."""

    name: str
    """Name of the MCP tool to run."""

    server_label: str
    """Label of the MCP server."""

    arguments: str
    """JSON string of tool arguments."""

    approval_request_id: str | None
    """ID for approval/rejection of the call."""

    output: str | None
    """Tool output."""

    error: str | None
    """Error information."""

    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"] | None
    )
    """Execution status."""


class McpListTools(TypedDict, total=False):
    """MCP list tools output item.

    Reference: openai.types.responses.McpListTools
    """

    type: Literal["mcp_list_tools"]
    """Output item type, always 'mcp_list_tools'."""

    server_label: str
    """Label of the MCP server."""

    tools: list[dict[str, Any]]
    """List of available tools on the server."""


class McpApprovalRequest(TypedDict, total=False):
    """MCP approval request output item.

    Reference: openai.types.responses.McpApprovalRequest
    """

    type: Literal["mcp_approval_request"]
    """Output item type, always 'mcp_approval_request'."""

    id: str
    """Unique ID for this approval request."""

    tool_name: str
    """Name of the tool requiring approval."""

    server_label: str
    """Label of the MCP server."""

    arguments: str
    """JSON string of tool arguments."""


# ============================================================================
# Special Types
# ============================================================================


class LocalShellCall(TypedDict, total=False):
    """Local shell call output item.

    Reference: openai.types.responses.LocalShellCall
    """

    type: Literal["local_shell_call"]
    """Output item type, always 'local_shell_call'."""

    id: str
    """Unique ID for this local shell call."""

    command: str
    """The shell command to execute."""

    output: str | None
    """Command output."""


class ImageGenerationCall(TypedDict, total=False):
    """Image generation call output item.

    Reference: openai.types.responses.ImageGenerationCall
    """

    type: Literal["image_generation_call"]
    """Output item type, always 'image_generation_call'."""

    id: str
    """Unique ID for this image generation call."""

    prompt: str
    """The image generation prompt."""

    image_url: str | None
    """URL of the generated image."""

    status: str | None
    """Generation status."""


class ResponseCompactionItem(TypedDict, total=False):
    """Compaction item output.

    Reference: openai.types.responses.ResponseCompactionItem
    """

    type: Literal["compaction"]
    """Output item type, always 'compaction'."""

    id: str
    """Unique ID for this compaction item."""

    content: str
    """The compacted content."""


# ============================================================================
# Union Types
# ============================================================================

ResponseOutputItem = Union[
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseCompactionItem,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseComputerToolCall,
    ResponseCodeInterpreterToolCall,
    ResponseFunctionShellToolCall,
    ResponseApplyPatchToolCall,
    ResponseCustomToolCall,
    LocalShellCall,
    ResponseFunctionShellToolCallOutput,
    ResponseApplyPatchToolCallOutput,
    ImageGenerationCall,
    McpCall,
    McpListTools,
    McpApprovalRequest,
]
"""Union of all 18 output item types in the Responses API.

Reference: openai.types.responses.ResponseOutputItem
"""


# ============================================================================
# Main Response Type
# ============================================================================


class Response(TypedDict, total=False):
    """OpenAI Responses API response.

    This is the main response type returned by the Responses API.
    Contains 20+ fields covering the complete response structure.

    Reference: openai.types.responses.Response
    """

    # Required fields
    id: str
    """Unique identifier for this response."""

    created_at: float
    """Unix timestamp (seconds) when this response was created."""

    model: str
    """The model ID used for generation."""

    object: Literal["response"]
    """Object type, always 'response'."""

    output: list[ResponseOutputItem]
    """Core output content list - contains all output items."""

    parallel_tool_calls: bool
    """Whether parallel tool calls are enabled."""

    tool_choice: Any
    """Tool choice configuration."""

    tools: list[dict[str, Any]]
    """Available tools list."""

    # Optional fields
    background: bool | None
    """Whether the response runs in the background."""

    conversation: dict[str, Any] | None
    """Conversation information."""

    error: ResponseError | None
    """Error information, if any."""

    incomplete_details: IncompleteDetails | None
    """Details about why the response is incomplete."""

    instructions: Any | None
    """System instructions used for this response."""

    max_output_tokens: int | None
    """Maximum number of output tokens."""

    max_tool_calls: int | None
    """Maximum number of tool calls."""

    metadata: dict[str, str] | None
    """Metadata key-value pairs."""

    previous_response_id: str | None
    """ID of the previous response."""

    prompt: dict[str, Any] | None
    """Prompt template information."""

    prompt_cache_key: str | None
    """Key for prompt caching."""

    prompt_cache_retention: Literal["in-memory", "24h"] | None
    """Prompt cache retention policy."""

    reasoning: dict[str, Any] | None
    """Reasoning configuration and output."""

    safety_identifier: str | None
    """Safety identifier."""

    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    """Service tier for request routing."""

    status: (
        Literal[
            "completed", "failed", "in_progress", "cancelled", "queued", "incomplete"
        ]
        | None
    )
    """Response status."""

    temperature: float | None
    """Sampling temperature."""

    text: dict[str, Any] | None
    """Text configuration."""

    top_logprobs: int | None
    """Number of top log probabilities returned."""

    top_p: float | None
    """Nucleus sampling parameter."""

    truncation: Literal["auto", "disabled"] | None
    """Truncation strategy."""

    usage: ResponseUsage | None
    """Token usage details."""

    user: str | None
    """User identifier."""

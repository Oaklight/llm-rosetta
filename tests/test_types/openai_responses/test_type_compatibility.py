"""Test compatibility between LLMIR OpenAI Responses type replicas and OpenAI SDK types.

This module tests:
- All TypedDict replicas can be correctly instantiated
- Required and optional fields work as expected
- If the OpenAI SDK is available, validates compatibility with SDK types

Reference: tests/test_types/openai/chat/test_type_compatibility.py
Reference: tests/test_types/google_genai/test_type_compatibility.py
"""

import pytest

from llmir.types.openai.responses import (
    Action,
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    AudioInputParam,
    Conversation,
    FunctionToolParam,
    ImageGenerationCall,
    ImageInputParam,
    IncompleteDetails,
    InputTokensDetails,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
    Metadata,
    OutputImage,
    OutputLogs,
    OutputTokensDetails,
    Reasoning,
    ReasoningContent,
    ReasoningSummary,
    Response,
    ResponseApplyPatchToolCall,
    ResponseApplyPatchToolCallOutput,
    ResponseCodeInterpreterToolCall,
    ResponseCompactionItem,
    ResponseComputerToolCall,
    ResponseCreateParams,
    ResponseCustomToolCall,
    ResponseError,
    ResponseFileSearchToolCall,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponsePromptParam,
    ResponseReasoningItem,
    ResponseTextConfigParam,
    ResponseUsage,
    StreamOptions,
    TextInputParam,
    ToolChoice,
)


# ============================================================================
# Request type instantiation tests
# ============================================================================


class TestInputTypes:
    """Test input content type instantiation."""

    def test_text_input_param(self):
        """Test creating a TextInputParam."""
        text_input: TextInputParam = {"type": "text", "text": "Hello, world!"}
        assert text_input["type"] == "text"
        assert text_input["text"] == "Hello, world!"

    def test_image_input_param(self):
        """Test creating an ImageInputParam."""
        image_input: ImageInputParam = {
            "type": "image",
            "image": "https://example.com/image.jpg",
        }
        assert image_input["type"] == "image"
        assert image_input["image"] == "https://example.com/image.jpg"

    def test_audio_input_param(self):
        """Test creating an AudioInputParam."""
        audio_input: AudioInputParam = {
            "type": "audio",
            "audio": "base64_encoded_audio_data",
        }
        assert audio_input["type"] == "audio"
        assert audio_input["audio"] == "base64_encoded_audio_data"

    def test_response_input_param_string(self):
        """Test ResponseInputParam with simple string."""
        input_param: ResponseInputParam = "Hello, world!"
        assert isinstance(input_param, str)

    def test_response_input_param_typed(self):
        """Test ResponseInputParam with typed input."""
        input_param: ResponseInputParam = {"type": "text", "text": "Hello"}
        assert input_param["type"] == "text"

    def test_response_input_param_list(self):
        """Test ResponseInputParam with list of inputs."""
        input_param: ResponseInputParam = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "image": "https://example.com/img.jpg"},
        ]
        assert len(input_param) == 2


class TestConfigTypes:
    """Test configuration type instantiation."""

    def test_response_prompt_param(self):
        """Test creating a ResponsePromptParam."""
        prompt: ResponsePromptParam = {
            "id": "prompt_123",
            "name": "my_prompt",
            "version": 1,
        }
        assert prompt["id"] == "prompt_123"
        assert prompt["name"] == "my_prompt"
        assert prompt["version"] == 1

    def test_response_prompt_param_minimal(self):
        """Test creating a minimal ResponsePromptParam."""
        prompt: ResponsePromptParam = {"id": "prompt_456"}
        assert prompt["id"] == "prompt_456"

    def test_response_text_config_param(self):
        """Test creating a ResponseTextConfigParam."""
        text_config: ResponseTextConfigParam = {
            "type": "text",
            "text": "Configuration text",
        }
        assert text_config["type"] == "text"

    def test_stream_options(self):
        """Test creating StreamOptions."""
        opts: StreamOptions = {"include_usage": True}
        assert opts["include_usage"] is True

    def test_reasoning(self):
        """Test creating Reasoning config."""
        reasoning: Reasoning = {"enabled": True, "max_tokens": 1000}
        assert reasoning["enabled"] is True
        assert reasoning["max_tokens"] == 1000


class TestToolTypes:
    """Test tool type instantiation."""

    def test_function_tool_param(self):
        """Test creating a FunctionToolParam."""
        tool: FunctionToolParam = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"

    def test_tool_choice_string(self):
        """Test ToolChoice with string literal."""
        choice: ToolChoice = "auto"
        assert choice == "auto"

    def test_tool_choice_none(self):
        """Test ToolChoice with 'none'."""
        choice: ToolChoice = "none"
        assert choice == "none"

    def test_tool_choice_required(self):
        """Test ToolChoice with 'required'."""
        choice: ToolChoice = "required"
        assert choice == "required"

    def test_tool_choice_dict(self):
        """Test ToolChoice with dict specifying a tool."""
        choice: ToolChoice = {
            "type": "function",
            "function": {"name": "specific_tool"},
        }
        assert choice["type"] == "function"


class TestMetadataTypes:
    """Test metadata type instantiation."""

    def test_metadata(self):
        """Test creating Metadata."""
        metadata: Metadata = {"key1": "value1", "key2": "value2"}
        assert metadata["key1"] == "value1"

    def test_conversation(self):
        """Test creating a Conversation."""
        conv: Conversation = {"id": "conv_123"}
        assert conv["id"] == "conv_123"

    def test_conversation_with_messages(self):
        """Test creating a Conversation with messages."""
        conv: Conversation = {
            "id": "conv_456",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert conv["id"] == "conv_456"
        assert len(conv["messages"]) == 1


class TestRequestParams:
    """Test main request parameter type instantiation."""

    def test_minimal_request(self):
        """Test creating a minimal ResponseCreateParams."""
        request: ResponseCreateParams = {
            "input": "Hello, world!",
            "model": "gpt-4o",
        }
        assert request["input"] == "Hello, world!"
        assert request["model"] == "gpt-4o"

    def test_request_with_tools(self):
        """Test creating a request with tools."""
        request: ResponseCreateParams = {
            "input": "What's the weather?",
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        assert request["model"] == "gpt-4o"

    def test_request_with_generation_params(self):
        """Test creating a request with generation control parameters."""
        request: ResponseCreateParams = {
            "input": "Test",
            "model": "gpt-4o",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 1024,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
        }
        assert request["temperature"] == 0.7
        assert request["max_output_tokens"] == 1024

    def test_request_with_all_optional_params(self):
        """Test creating a request with many optional parameters."""
        request: ResponseCreateParams = {
            "input": "Test",
            "model": "gpt-4o",
            "instructions": "You are a helpful assistant.",
            "temperature": 0.7,
            "metadata": {"session": "test_123"},
            "truncation": "auto",
            "user": "user_abc",
            "store": True,
            "service_tier": "auto",
            "reasoning": {"enabled": True, "max_tokens": 500},
            "reasoning_effort": "high",
        }
        assert request["instructions"] == "You are a helpful assistant."
        assert request["metadata"]["session"] == "test_123"


# ============================================================================
# Response type instantiation tests
# ============================================================================


class TestStatusAndErrorTypes:
    """Test status and error type instantiation."""

    def test_response_error(self):
        """Test creating a ResponseError."""
        error: ResponseError = {
            "code": "server_error",
            "message": "Invalid input",
        }
        assert error["code"] == "server_error"
        assert error["message"] == "Invalid input"

    def test_incomplete_details(self):
        """Test creating IncompleteDetails."""
        details: IncompleteDetails = {"reason": "max_tokens"}
        assert details["reason"] == "max_tokens"


class TestUsageTypes:
    """Test usage type instantiation."""

    def test_response_usage(self):
        """Test creating ResponseUsage."""
        usage: ResponseUsage = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 30,
        }
        assert usage["total_tokens"] == 30

    def test_response_usage_with_details(self):
        """Test creating ResponseUsage with token details."""
        usage: ResponseUsage = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 50},
            "output_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 80},
            "total_tokens": 300,
        }
        assert usage["input_tokens_details"]["cached_tokens"] == 50

    def test_input_tokens_details(self):
        """Test creating InputTokensDetails."""
        details: InputTokensDetails = {"cached_tokens": 50}
        assert details["cached_tokens"] == 50

    def test_output_tokens_details(self):
        """Test creating OutputTokensDetails."""
        details: OutputTokensDetails = {"reasoning_tokens": 80}
        assert details["reasoning_tokens"] == 80


class TestContentOutputTypes:
    """Test content output type instantiation."""

    def test_response_output_text(self):
        """Test creating ResponseOutputText."""
        text: ResponseOutputText = {
            "type": "output_text",
            "text": "Hello! How can I help?",
            "annotations": [],
            "logprobs": None,
        }
        assert text["type"] == "output_text"
        assert text["text"] == "Hello! How can I help?"

    def test_response_output_text_with_annotations(self):
        """Test creating ResponseOutputText with annotations."""
        text: ResponseOutputText = {
            "type": "output_text",
            "text": "According to sources...",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "start_index": 0,
                    "end_index": 25,
                }
            ],
        }
        assert len(text["annotations"]) == 1

    def test_response_output_refusal(self):
        """Test creating ResponseOutputRefusal."""
        refusal: ResponseOutputRefusal = {
            "type": "refusal",
            "refusal": "I cannot assist with that request.",
        }
        assert refusal["type"] == "refusal"
        assert refusal["refusal"] == "I cannot assist with that request."


class TestMessageOutputTypes:
    """Test message output type instantiation."""

    def test_response_output_message(self):
        """Test creating ResponseOutputMessage."""
        message: ResponseOutputMessage = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello!",
                    "annotations": [],
                }
            ],
            "status": "completed",
        }
        assert message["id"] == "msg_123"
        assert message["role"] == "assistant"
        assert message["status"] == "completed"
        assert len(message["content"]) == 1

    def test_response_output_message_with_refusal(self):
        """Test creating ResponseOutputMessage with refusal content."""
        message: ResponseOutputMessage = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "refusal",
                    "refusal": "I cannot do that.",
                }
            ],
            "status": "completed",
        }
        assert message["content"][0]["type"] == "refusal"


class TestReasoningTypes:
    """Test reasoning type instantiation."""

    def test_reasoning_summary(self):
        """Test creating ReasoningSummary."""
        summary: ReasoningSummary = {
            "type": "summary_text",
            "text": "The model considered multiple approaches.",
        }
        assert summary["type"] == "summary_text"

    def test_reasoning_content(self):
        """Test creating ReasoningContent."""
        content: ReasoningContent = {
            "type": "reasoning_text",
            "text": "First, let me analyze the problem...",
        }
        assert content["type"] == "reasoning_text"

    def test_response_reasoning_item(self):
        """Test creating ResponseReasoningItem."""
        reasoning: ResponseReasoningItem = {
            "type": "reasoning",
            "id": "reason_123",
            "summary": [{"type": "summary_text", "text": "Summary of reasoning"}],
            "content": None,
            "encrypted_content": None,
            "status": "completed",
        }
        assert reasoning["type"] == "reasoning"
        assert reasoning["id"] == "reason_123"
        assert len(reasoning["summary"]) == 1

    def test_response_reasoning_item_with_content(self):
        """Test creating ResponseReasoningItem with full content."""
        reasoning: ResponseReasoningItem = {
            "type": "reasoning",
            "id": "reason_456",
            "summary": [{"type": "summary_text", "text": "Summary"}],
            "content": [
                {"type": "reasoning_text", "text": "Step 1: Analyze..."},
                {"type": "reasoning_text", "text": "Step 2: Conclude..."},
            ],
            "status": "completed",
        }
        assert len(reasoning["content"]) == 2


class TestFunctionToolCallTypes:
    """Test function tool call type instantiation."""

    def test_response_function_tool_call(self):
        """Test creating ResponseFunctionToolCall."""
        func_call: ResponseFunctionToolCall = {
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
            "id": "tool_123",
            "status": "completed",
        }
        assert func_call["type"] == "function_call"
        assert func_call["call_id"] == "call_123"
        assert func_call["name"] == "get_weather"

    def test_response_function_tool_call_minimal(self):
        """Test creating a minimal ResponseFunctionToolCall."""
        func_call: ResponseFunctionToolCall = {
            "type": "function_call",
            "call_id": "call_456",
            "name": "search",
            "arguments": "{}",
        }
        assert func_call["name"] == "search"

    def test_response_custom_tool_call(self):
        """Test creating ResponseCustomToolCall."""
        custom_call: ResponseCustomToolCall = {
            "type": "custom_tool_call",
            "call_id": "custom_123",
            "name": "custom_tool",
            "input": "input data",
            "id": "tool_456",
        }
        assert custom_call["type"] == "custom_tool_call"
        assert custom_call["name"] == "custom_tool"


class TestWebSearchTypes:
    """Test web search type instantiation."""

    def test_action_search(self):
        """Test creating ActionSearch."""
        action: ActionSearch = {"type": "search", "query": "weather in NYC"}
        assert action["type"] == "search"
        assert action["query"] == "weather in NYC"

    def test_action_open_page(self):
        """Test creating ActionOpenPage."""
        action: ActionOpenPage = {
            "type": "open_page",
            "url": "https://example.com",
        }
        assert action["type"] == "open_page"

    def test_action_find(self):
        """Test creating ActionFind."""
        action: ActionFind = {"type": "find", "pattern": "temperature"}
        assert action["type"] == "find"

    def test_action_union(self):
        """Test Action union type."""
        action: Action = {"type": "search", "query": "test"}
        assert action["type"] == "search"

    def test_response_function_web_search(self):
        """Test creating ResponseFunctionWebSearch."""
        web_search: ResponseFunctionWebSearch = {
            "type": "web_search_call",
            "id": "search_123",
            "action": {"type": "search", "query": "test query"},
            "status": "completed",
        }
        assert web_search["type"] == "web_search_call"
        assert web_search["action"]["type"] == "search"


class TestCodeInterpreterTypes:
    """Test code interpreter type instantiation."""

    def test_output_logs(self):
        """Test creating OutputLogs."""
        logs: OutputLogs = {"type": "logs", "logs": "Hello, world!\n"}
        assert logs["type"] == "logs"
        assert logs["logs"] == "Hello, world!\n"

    def test_output_image(self):
        """Test creating OutputImage."""
        image: OutputImage = {"type": "image", "image": "base64_data"}
        assert image["type"] == "image"

    def test_response_code_interpreter_tool_call(self):
        """Test creating ResponseCodeInterpreterToolCall."""
        code_interp: ResponseCodeInterpreterToolCall = {
            "type": "code_interpreter_call",
            "id": "code_123",
            "container_id": "container_456",
            "code": "print('hello')",
            "outputs": [{"type": "logs", "logs": "hello\n"}],
            "status": "completed",
        }
        assert code_interp["type"] == "code_interpreter_call"
        assert code_interp["code"] == "print('hello')"
        assert len(code_interp["outputs"]) == 1

    def test_response_code_interpreter_no_outputs(self):
        """Test creating ResponseCodeInterpreterToolCall without outputs."""
        code_interp: ResponseCodeInterpreterToolCall = {
            "type": "code_interpreter_call",
            "id": "code_789",
            "container_id": "container_abc",
            "code": None,
            "outputs": None,
            "status": "in_progress",
        }
        assert code_interp["outputs"] is None


class TestFileSearchTypes:
    """Test file search type instantiation."""

    def test_response_file_search_tool_call(self):
        """Test creating ResponseFileSearchToolCall."""
        file_search: ResponseFileSearchToolCall = {
            "type": "file_search_call",
            "id": "fs_123",
            "query": "project documentation",
            "results": [{"file_id": "file_1", "score": 0.95}],
            "status": "completed",
        }
        assert file_search["type"] == "file_search_call"
        assert file_search["query"] == "project documentation"


class TestComputerToolTypes:
    """Test computer tool type instantiation."""

    def test_response_computer_tool_call(self):
        """Test creating ResponseComputerToolCall."""
        computer_call: ResponseComputerToolCall = {
            "type": "computer_tool_call",
            "id": "comp_123",
            "action": "click",
            "parameters": {"x": 100, "y": 200},
            "status": "completed",
        }
        assert computer_call["type"] == "computer_tool_call"
        assert computer_call["action"] == "click"


class TestShellAndPatchTypes:
    """Test shell and patch tool type instantiation."""

    def test_response_function_shell_tool_call(self):
        """Test creating ResponseFunctionShellToolCall."""
        shell_call: ResponseFunctionShellToolCall = {
            "type": "shell_call",
            "id": "shell_123",
            "command": "ls -la",
            "output": "total 0\n",
            "status": "completed",
        }
        assert shell_call["type"] == "shell_call"
        assert shell_call["command"] == "ls -la"

    def test_response_apply_patch_tool_call(self):
        """Test creating ResponseApplyPatchToolCall."""
        patch_call: ResponseApplyPatchToolCall = {
            "type": "apply_patch_call",
            "id": "patch_123",
            "patch": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            "file_path": "src/file.py",
            "status": "completed",
        }
        assert patch_call["type"] == "apply_patch_call"
        assert patch_call["file_path"] == "src/file.py"

    def test_response_function_shell_tool_call_output(self):
        """Test creating ResponseFunctionShellToolCallOutput."""
        output: ResponseFunctionShellToolCallOutput = {
            "type": "shell_output",
            "call_id": "shell_123",
            "output": "command output here",
        }
        assert output["type"] == "shell_output"

    def test_response_apply_patch_tool_call_output(self):
        """Test creating ResponseApplyPatchToolCallOutput."""
        output: ResponseApplyPatchToolCallOutput = {
            "type": "patch_output",
            "call_id": "patch_123",
            "result": "Patch applied successfully",
        }
        assert output["type"] == "patch_output"


class TestMcpTypes:
    """Test MCP type instantiation."""

    def test_mcp_call(self):
        """Test creating McpCall."""
        mcp_call: McpCall = {
            "type": "mcp_call",
            "id": "mcp_123",
            "name": "mcp_tool",
            "server_label": "server1",
            "arguments": '{"arg": "value"}',
            "approval_request_id": None,
            "output": "result",
            "error": None,
            "status": "completed",
        }
        assert mcp_call["type"] == "mcp_call"
        assert mcp_call["name"] == "mcp_tool"
        assert mcp_call["server_label"] == "server1"

    def test_mcp_call_minimal(self):
        """Test creating a minimal McpCall."""
        mcp_call: McpCall = {
            "type": "mcp_call",
            "id": "mcp_456",
            "name": "tool",
            "server_label": "srv",
            "arguments": "{}",
        }
        assert mcp_call["id"] == "mcp_456"

    def test_mcp_list_tools(self):
        """Test creating McpListTools."""
        mcp_list: McpListTools = {
            "type": "mcp_list_tools",
            "server_label": "server1",
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"},
            ],
        }
        assert mcp_list["type"] == "mcp_list_tools"
        assert len(mcp_list["tools"]) == 2

    def test_mcp_approval_request(self):
        """Test creating McpApprovalRequest."""
        approval: McpApprovalRequest = {
            "type": "mcp_approval_request",
            "id": "approval_123",
            "tool_name": "dangerous_tool",
            "server_label": "server1",
            "arguments": '{"action": "delete"}',
        }
        assert approval["type"] == "mcp_approval_request"
        assert approval["tool_name"] == "dangerous_tool"


class TestSpecialTypes:
    """Test special output type instantiation."""

    def test_local_shell_call(self):
        """Test creating LocalShellCall."""
        shell: LocalShellCall = {
            "type": "local_shell_call",
            "id": "local_123",
            "command": "echo hello",
            "output": "hello\n",
        }
        assert shell["type"] == "local_shell_call"
        assert shell["command"] == "echo hello"

    def test_image_generation_call(self):
        """Test creating ImageGenerationCall."""
        img_gen: ImageGenerationCall = {
            "type": "image_generation_call",
            "id": "img_123",
            "prompt": "A sunset over the ocean",
            "image_url": "https://example.com/generated.png",
            "status": "completed",
        }
        assert img_gen["type"] == "image_generation_call"
        assert img_gen["prompt"] == "A sunset over the ocean"

    def test_response_compaction_item(self):
        """Test creating ResponseCompactionItem."""
        compaction: ResponseCompactionItem = {
            "type": "compaction",
            "id": "compact_123",
            "content": "Compacted conversation summary...",
        }
        assert compaction["type"] == "compaction"
        assert compaction["id"] == "compact_123"


# ============================================================================
# Complete Response tests
# ============================================================================


class TestCompleteResponse:
    """Test complete Response type instantiation."""

    def test_minimal_response(self):
        """Test creating a minimal Response."""
        response: Response = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        }
        assert response["id"] == "resp_123"
        assert response["object"] == "response"
        assert response["status"] == "completed"

    def test_response_with_text_output(self):
        """Test creating a Response with text output."""
        response: Response = {
            "id": "resp_456",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello! How can I help you today?",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 22,
            },
        }
        assert len(response["output"]) == 1
        assert response["usage"]["total_tokens"] == 22

    def test_response_with_tool_call(self):
        """Test creating a Response with function tool call."""
        response: Response = {
            "id": "resp_789",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        }
        assert response["output"][0]["type"] == "function_call"

    def test_response_with_reasoning(self):
        """Test creating a Response with reasoning output."""
        response: Response = {
            "id": "resp_reason",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "o3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reason_1",
                    "summary": [
                        {"type": "summary_text", "text": "Analyzed the problem"}
                    ],
                    "status": "completed",
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The answer is 42.",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "reasoning": {"enabled": True},
        }
        assert len(response["output"]) == 2
        assert response["output"][0]["type"] == "reasoning"

    def test_response_with_all_optional_fields(self):
        """Test creating a Response with many optional fields."""
        response: Response = {
            "id": "resp_full",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_full",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Response",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 50,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 100,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 150,
            },
            "instructions": "You are a helpful assistant.",
            "metadata": {"session": "test"},
            "error": None,
            "incomplete_details": None,
            "conversation": None,
            "prompt": None,
            "reasoning": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "user": "user_123",
        }
        assert response["instructions"] == "You are a helpful assistant."
        assert response["user"] == "user_123"

    def test_response_failed(self):
        """Test creating a failed Response."""
        response: Response = {
            "id": "resp_fail",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "failed",
            "error": {
                "code": "server_error",
                "message": "Internal server error",
            },
        }
        assert response["status"] == "failed"
        assert response["error"]["code"] == "server_error"

    def test_response_incomplete(self):
        """Test creating an incomplete Response."""
        response: Response = {
            "id": "resp_incomplete",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_inc",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Partial...",
                            "annotations": [],
                        }
                    ],
                    "status": "incomplete",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_tokens"},
        }
        assert response["status"] == "incomplete"
        assert response["incomplete_details"]["reason"] == "max_tokens"


# ============================================================================
# __init__ module exports test
# ============================================================================


class TestResponsesInit:
    """Test the __init__ module exports."""

    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        from llmir.types.openai.responses import __all__

        expected = [
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
        assert set(__all__) == set(expected)

    def test_all_types_importable(self):
        """Test that all types in __all__ are importable."""
        import llmir.types.openai.responses as mod

        for name in mod.__all__:
            obj = getattr(mod, name)
            assert obj is not None, f"{name} should not be None"


# ============================================================================
# SDK compatibility tests (only run if SDK is available)
# ============================================================================


class TestSDKCompatibility:
    """Test compatibility with OpenAI SDK types.

    These tests only run if the OpenAI SDK is installed and has the
    Responses API types available.
    """

    def test_sdk_response_validation(self):
        """Test creating a Response with LLMIR replica and validating with SDK."""
        try:
            from openai.types.responses import Response as SDKResponse
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_response: Response = {
            "id": "resp_sdk_test",
            "object": "response",
            "created_at": 1234567890.0,
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_sdk",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello from LLMIR!",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

        # Validate with SDK Pydantic model
        sdk_validated = SDKResponse.model_validate(llmir_response)
        assert sdk_validated.id == "resp_sdk_test"
        assert sdk_validated.model == "gpt-4o"
        assert sdk_validated.status == "completed"

    def test_sdk_function_tool_call_validation(self):
        """Test creating a function tool call and validating with SDK."""
        try:
            from openai.types.responses import (
                ResponseFunctionToolCall as SDKFunctionToolCall,
            )
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_call: ResponseFunctionToolCall = {
            "type": "function_call",
            "call_id": "call_sdk_test",
            "name": "get_weather",
            "arguments": '{"location": "NYC"}',
            "status": "completed",
        }

        sdk_validated = SDKFunctionToolCall.model_validate(llmir_call)
        assert sdk_validated.name == "get_weather"
        assert sdk_validated.call_id == "call_sdk_test"

    def test_sdk_mcp_call_validation(self):
        """Test creating an MCP call and validating with SDK."""
        try:
            from openai.types.responses.response_output_item import (
                McpCall as SDKMcpCall,
            )
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_mcp: McpCall = {
            "type": "mcp_call",
            "id": "mcp_sdk_test",
            "name": "mcp_tool",
            "server_label": "server1",
            "arguments": '{"key": "value"}',
            "status": "completed",
        }

        sdk_validated = SDKMcpCall.model_validate(llmir_mcp)
        assert sdk_validated.name == "mcp_tool"
        assert sdk_validated.server_label == "server1"

    def test_sdk_reasoning_item_validation(self):
        """Test creating a reasoning item and validating with SDK."""
        try:
            from openai.types.responses import (
                ResponseReasoningItem as SDKReasoningItem,
            )
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_reasoning: ResponseReasoningItem = {
            "type": "reasoning",
            "id": "reason_sdk_test",
            "summary": [{"type": "summary_text", "text": "Analyzed the problem"}],
            "status": "completed",
        }

        sdk_validated = SDKReasoningItem.model_validate(llmir_reasoning)
        assert sdk_validated.id == "reason_sdk_test"

    def test_sdk_web_search_validation(self):
        """Test creating a web search call and validating with SDK."""
        try:
            from openai.types.responses import (
                ResponseFunctionWebSearch as SDKWebSearch,
            )
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_search: ResponseFunctionWebSearch = {
            "type": "web_search_call",
            "id": "search_sdk_test",
            "action": {"type": "search", "query": "test query"},
            "status": "completed",
        }

        sdk_validated = SDKWebSearch.model_validate(llmir_search)
        assert sdk_validated.id == "search_sdk_test"

    def test_sdk_code_interpreter_validation(self):
        """Test creating a code interpreter call and validating with SDK."""
        try:
            from openai.types.responses import (
                ResponseCodeInterpreterToolCall as SDKCodeInterpreter,
            )
        except ImportError:
            pytest.skip("OpenAI SDK not available")

        llmir_code: ResponseCodeInterpreterToolCall = {
            "type": "code_interpreter_call",
            "id": "code_sdk_test",
            "container_id": "container_test",
            "code": "print('hello')",
            "status": "completed",
        }

        sdk_validated = SDKCodeInterpreter.model_validate(llmir_code)
        assert sdk_validated.id == "code_sdk_test"


if __name__ == "__main__":
    pytest.main([__file__])

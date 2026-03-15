"""
Tests for LLM-Rosetta IR Types Module

测试 IR 类型模块的所有组件：
- 类型定义和验证
- 类型守卫函数
- 辅助函数
- 消息创建函数
- 扩展项类型
"""

from typing import cast

import pytest

# 直接从 IR 模块导入，避免通过主模块导入旧的转换器
from llm_rosetta.types.ir import (
    # Content parts
    TextPart,
    ImagePart,
    ImageData,
    FilePart,
    FileData,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    AudioPart,
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    MessageMetadata,
    is_message,
    is_system_message,
    is_user_message,
    is_assistant_message,
    is_tool_message,
    # Message creation functions
    create_system_message,
    create_user_message,
    create_assistant_message,
    create_tool_message,
    # Helper functions
    extract_text_content,
    extract_tool_calls,
    create_tool_result_message,
    # Type guards
    is_part_type,
    isinstance_part,
    get_part_type,
    TYPE_CLASS_MAP,
    # Extensions
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
    is_extension_item,
    # Tools
    ToolDefinition,
    ToolChoice,
    ToolCallConfig,
    # Configs
    GenerationConfig,
    ResponseFormatConfig,
    StreamConfig,
    ReasoningConfig,
    CacheConfig,
    # Request/Response
    IRRequest,
    IRResponse,
    UsageInfo,
    ChoiceInfo,
)


class TestContentParts:
    """测试内容部分类型"""

    def test_text_part_creation(self):
        """测试文本部分创建"""
        text_part: TextPart = {"type": "text", "text": "Hello, world!"}

        assert text_part["type"] == "text"
        assert text_part["text"] == "Hello, world!"
        assert is_part_type(text_part, TextPart)

    def test_image_part_with_url(self):
        """测试带URL的图像部分"""
        image_part: ImagePart = {
            "type": "image",
            "image_url": "https://example.com/image.jpg",
            "detail": "high",
        }

        assert image_part["type"] == "image"
        assert image_part["image_url"] == "https://example.com/image.jpg"
        assert image_part["detail"] == "high"
        assert is_part_type(image_part, ImagePart)

    def test_image_part_with_data(self):
        """测试带base64数据的图像部分"""
        image_data: ImageData = {
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "media_type": "image/png",
        }

        image_part: ImagePart = {
            "type": "image",
            "image_data": image_data,
            "detail": "auto",
        }

        assert image_part["type"] == "image"
        assert image_part["image_data"]["media_type"] == "image/png"
        assert is_part_type(image_part, ImagePart)

    def test_file_part_creation(self):
        """测试文件部分创建"""
        file_data: FileData = {
            "data": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            "media_type": "text/plain",
        }

        file_part: FilePart = {
            "type": "file",
            "file_data": file_data,
            "file_name": "hello.txt",
            "file_type": "text/plain",
        }

        assert file_part["type"] == "file"
        assert file_part["file_name"] == "hello.txt"
        assert is_part_type(file_part, FilePart)

    def test_tool_call_part_creation(self):
        """测试工具调用部分创建"""
        tool_call: ToolCallPart = {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "tool_input": {"location": "Beijing"},
            "tool_type": "function",
        }

        assert tool_call["type"] == "tool_call"
        assert tool_call["tool_name"] == "get_weather"
        assert tool_call["tool_input"]["location"] == "Beijing"
        assert is_part_type(tool_call, ToolCallPart)

    def test_tool_result_part_creation(self):
        """测试工具结果部分创建"""
        tool_result: ToolResultPart = {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "result": {"temperature": 25, "condition": "sunny"},
            "is_error": False,
        }

        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_call_id"] == "call_123"
        assert not tool_result["is_error"]
        assert is_part_type(tool_result, ToolResultPart)

    def test_reasoning_part_creation(self):
        """测试推理部分创建"""
        reasoning: ReasoningPart = {
            "type": "reasoning",
            "reasoning": "Let me think about this step by step...",
            "signature": "reasoning_abc123",
            "status": "completed",
        }

        assert reasoning["type"] == "reasoning"
        assert reasoning["status"] == "completed"
        assert is_part_type(reasoning, ReasoningPart)

    def test_refusal_part_creation(self):
        """测试拒绝部分创建"""
        refusal: RefusalPart = {
            "type": "refusal",
            "refusal": "I cannot provide information about that topic.",
        }

        assert refusal["type"] == "refusal"
        assert "cannot provide" in refusal["refusal"]
        assert is_part_type(refusal, RefusalPart)

    def test_citation_part_creation(self):
        """测试引用部分创建"""
        citation: CitationPart = {
            "type": "citation",
            "url_citation": {
                "start_index": 10,
                "end_index": 25,
                "title": "Example Article",
                "url": "https://example.com/article",
            },
        }

        assert citation["type"] == "citation"
        assert citation["url_citation"]["title"] == "Example Article"
        assert is_part_type(citation, CitationPart)

    def test_audio_part_creation(self):
        """测试音频部分创建"""
        audio: AudioPart = {"type": "audio", "audio_id": "audio_123", "detail": "high"}

        assert audio["type"] == "audio"
        assert audio["audio_id"] == "audio_123"
        assert is_part_type(audio, AudioPart)


class TestTypeGuards:
    """测试类型守卫函数"""

    def test_is_part_type_function(self):
        """测试通用类型检查函数"""
        text_part = {"type": "text", "text": "hello"}
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "func",
            "tool_input": {},
        }

        # 正确的类型检查
        assert is_part_type(text_part, TextPart)
        assert is_part_type(tool_call, ToolCallPart)

        # 错误的类型检查
        assert not is_part_type(text_part, ToolCallPart)
        assert not is_part_type(tool_call, TextPart)

        # 无效输入
        assert not is_part_type("not a dict", TextPart)
        assert not is_part_type(None, TextPart)

    def test_isinstance_part_function(self):
        """测试类似isinstance的函数"""
        text_part = {"type": "text", "text": "hello"}

        # 单个类型检查
        assert isinstance_part(text_part, TextPart)
        assert not isinstance_part(text_part, ToolCallPart)

        # 多个类型检查
        assert isinstance_part(text_part, TextPart, ImagePart)
        assert isinstance_part(text_part, ImagePart, TextPart)
        assert not isinstance_part(text_part, ToolCallPart, ImagePart)

    def test_get_part_type_function(self):
        """测试获取部分类型函数"""
        text_part = {"type": "text", "text": "hello"}
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "func",
            "tool_input": {},
        }

        assert get_part_type(text_part) == TextPart
        assert get_part_type(tool_call) == ToolCallPart
        assert get_part_type({"type": "unknown"}) is None
        assert get_part_type("not a dict") is None

    def test_type_class_map(self):
        """测试类型映射表"""
        assert TYPE_CLASS_MAP["text"] == TextPart
        assert TYPE_CLASS_MAP["image"] == ImagePart
        assert TYPE_CLASS_MAP["tool_call"] == ToolCallPart
        assert TYPE_CLASS_MAP["tool_result"] == ToolResultPart

        # 确保所有基本类型都在映射表中
        expected_types = {
            "text",
            "image",
            "file",
            "tool_call",
            "tool_result",
            "reasoning",
            "refusal",
            "citation",
            "audio",
        }
        assert set(TYPE_CLASS_MAP.keys()) == expected_types


class TestMessages:
    """测试消息类型"""

    def test_system_message_creation(self):
        """测试系统消息创建"""
        system_msg: SystemMessage = {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        }

        assert system_msg["role"] == "system"
        content = cast(list, system_msg["content"])
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert is_system_message(system_msg)
        assert is_message(system_msg)

    def test_user_message_creation(self):
        """测试用户消息创建"""
        user_msg: UserMessage = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image_url": "https://example.com/image.jpg"},
            ],
        }

        assert user_msg["role"] == "user"
        content = cast(list, user_msg["content"])
        assert len(content) == 2
        assert is_user_message(user_msg)
        assert is_message(user_msg)

    def test_assistant_message_creation(self):
        """测试助手消息创建"""
        assistant_msg: AssistantMessage = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "search",
                    "tool_input": {"query": "example"},
                },
            ],
        }

        assert assistant_msg["role"] == "assistant"
        content = cast(list, assistant_msg["content"])
        assert len(content) == 2
        assert is_assistant_message(assistant_msg)
        assert is_message(assistant_msg)

    def test_tool_message_creation(self):
        """测试工具消息创建"""
        tool_msg: ToolMessage = {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": "call_1",
                    "result": "Search completed successfully",
                }
            ],
        }

        assert tool_msg["role"] == "tool"
        content = cast(list, tool_msg["content"])
        assert len(content) == 1
        assert is_tool_message(tool_msg)
        assert is_message(tool_msg)

    def test_message_metadata(self):
        """测试消息元数据"""
        metadata: MessageMetadata = {
            "message_id": "msg_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "streaming": {"is_streaming": True, "is_final": False, "chunk_index": 1},
            "custom": {"priority": "high"},
        }

        user_msg: UserMessage = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "metadata": metadata,
        }

        assert user_msg["metadata"]["message_id"] == "msg_123"
        assert user_msg["metadata"]["streaming"]["is_streaming"]
        assert user_msg["metadata"]["custom"]["priority"] == "high"


class TestMessageCreationFunctions:
    """测试消息创建函数"""

    def test_create_system_message(self):
        """测试创建系统消息函数"""
        msg = create_system_message("You are a helpful assistant.")

        assert msg["role"] == "system"
        content = cast(list, msg["content"])
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "You are a helpful assistant."
        assert is_system_message(msg)

    def test_create_user_message(self):
        """测试创建用户消息函数"""
        msg = create_user_message("Hello, how are you?")

        assert msg["role"] == "user"
        content = cast(list, msg["content"])
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello, how are you?"
        assert is_user_message(msg)

    def test_create_assistant_message(self):
        """测试创建助手消息函数"""
        msg = create_assistant_message("I'm doing well, thank you!")

        assert msg["role"] == "assistant"
        content = cast(list, msg["content"])
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "I'm doing well, thank you!"
        assert is_assistant_message(msg)

    def test_create_tool_message(self):
        """测试创建工具消息函数"""
        msg = create_tool_message("call_123", {"result": "success"})

        assert msg["role"] == "tool"
        content = cast(list, msg["content"])
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_call_id"] == "call_123"
        assert content[0]["result"]["result"] == "success"
        assert not content[0]["is_error"]
        assert is_tool_message(msg)

    def test_create_tool_message_with_error(self):
        """测试创建错误工具消息"""
        msg = create_tool_message("call_456", "Error occurred", is_error=True)

        assert msg["role"] == "tool"
        content = cast(list, msg["content"])
        assert content[0]["tool_call_id"] == "call_456"
        assert content[0]["result"] == "Error occurred"
        assert content[0]["is_error"]

    def test_create_tool_result_message(self):
        """测试创建工具结果消息函数"""
        msg = create_tool_result_message("call_789", {"data": "test"})

        assert msg["role"] == "tool"
        content = cast(list, msg["content"])
        assert content[0]["tool_call_id"] == "call_789"
        assert content[0]["result"]["data"] == "test"
        assert not content[0]["is_error"]


class TestHelperFunctions:
    """测试辅助函数"""

    def test_extract_text_content(self):
        """测试提取文本内容"""
        message: Message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "image", "image_url": "https://example.com/image.jpg"},
                {"type": "text", "text": "world!"},
            ],
        }

        text = extract_text_content(message)
        assert text == "Hello world!"

    def test_extract_text_content_empty(self):
        """测试提取空文本内容"""
        message: Message = {
            "role": "user",
            "content": [
                {"type": "image", "image_url": "https://example.com/image.jpg"}
            ],
        }

        text = extract_text_content(message)
        assert text == ""

    def test_extract_tool_calls(self):
        """测试提取工具调用"""
        message: Message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "search",
                    "tool_input": {"query": "test"},
                },
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "tool_name": "calculate",
                    "tool_input": {"expression": "2+2"},
                },
            ],
        }

        # 提取所有工具调用
        tool_calls = extract_tool_calls(message)
        assert len(tool_calls) == 2
        assert tool_calls[0]["tool_name"] == "search"
        assert tool_calls[1]["tool_name"] == "calculate"

    def test_extract_tool_calls_with_limit(self):
        """测试限制提取工具调用数量"""
        message: Message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "search",
                    "tool_input": {"query": "test"},
                },
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "tool_name": "calculate",
                    "tool_input": {"expression": "2+2"},
                },
            ],
        }

        # 只提取第一个工具调用
        tool_calls = extract_tool_calls(message, limit=1)
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "search"

    def test_extract_tool_calls_empty(self):
        """测试提取空工具调用"""
        message: Message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "No tools needed."}],
        }

        tool_calls = extract_tool_calls(message)
        assert len(tool_calls) == 0


class TestExtensionItems:
    """测试扩展项类型"""

    def test_system_event_creation(self):
        """测试系统事件创建"""
        event: SystemEvent = {
            "type": "system_event",
            "event_type": "session_start",
            "timestamp": "2024-01-01T00:00:00Z",
            "message": "Session started successfully",
        }

        assert event["type"] == "system_event"
        assert event["event_type"] == "session_start"
        assert is_extension_item(event)

    def test_batch_marker_creation(self):
        """测试批次标记创建"""
        marker: BatchMarker = {
            "type": "batch_marker",
            "batch_id": "batch_123",
            "batch_type": "start",
            "total_items": 5,
            "completed_items": 0,
        }

        assert marker["type"] == "batch_marker"
        assert marker["batch_id"] == "batch_123"
        assert is_extension_item(marker)

    def test_session_control_creation(self):
        """测试会话控制创建"""
        control: SessionControl = {
            "type": "session_control",
            "control_type": "cancel_tool",
            "target_id": "call_123",
            "reason": "User requested cancellation",
        }

        assert control["type"] == "session_control"
        assert control["control_type"] == "cancel_tool"
        assert is_extension_item(control)

    def test_tool_chain_node_creation(self):
        """测试工具链节点创建"""
        node: ToolChainNode = {
            "type": "tool_chain_node",
            "node_id": "node_1",
            "tool_call": {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "tool_input": {"query": "test"},
            },
            "depends_on": ["node_0"],
            "auto_execute": True,
        }

        assert node["type"] == "tool_chain_node"
        assert node["node_id"] == "node_1"
        assert node["auto_execute"]
        assert is_extension_item(node)


class TestToolTypes:
    """测试工具类型"""

    def test_tool_definition_creation(self):
        """测试工具定义创建"""
        tool_def: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
            },
            "required_parameters": ["location"],
            "metadata": {"version": "1.0"},
        }

        assert tool_def["type"] == "function"
        assert tool_def["name"] == "get_weather"
        assert "location" in tool_def["required_parameters"]

    def test_tool_choice_creation(self):
        """测试工具选择创建"""
        # Auto choice
        auto_choice: ToolChoice = {"mode": "auto", "tool_name": ""}
        assert auto_choice["mode"] == "auto"

        # Specific tool choice
        specific_choice: ToolChoice = {"mode": "tool", "tool_name": "get_weather"}
        assert specific_choice["mode"] == "tool"
        assert specific_choice["tool_name"] == "get_weather"

    def test_tool_call_config_creation(self):
        """测试工具调用配置创建"""
        config: ToolCallConfig = {"disable_parallel": True, "max_calls": 3}

        assert config["disable_parallel"]
        assert config["max_calls"] == 3


class TestConfigTypes:
    """测试配置类型"""

    def test_generation_config_creation(self):
        """测试生成配置创建"""
        config: GenerationConfig = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 1000,
            "stop_sequences": ["END", "\n\n"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "seed": 42,
        }

        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1000
        assert "END" in config["stop_sequences"]

    def test_response_format_config_creation(self):
        """测试响应格式配置创建"""
        config: ResponseFormatConfig = {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
            "mime_type": "application/json",
        }

        assert config["type"] == "json_schema"
        assert "result" in config["json_schema"]["properties"]

    def test_stream_config_creation(self):
        """测试流式配置创建"""
        config: StreamConfig = {"enabled": True, "include_usage": True}

        assert config["enabled"]
        assert config["include_usage"]

    def test_reasoning_config_creation(self):
        """测试推理配置创建"""
        config: ReasoningConfig = {
            "effort": "medium",
            "type": "enabled",
            "budget_tokens": 1000,
        }

        assert config["effort"] == "medium"
        assert config["type"] == "enabled"
        assert config["budget_tokens"] == 1000

    def test_cache_config_creation(self):
        """测试缓存配置创建"""
        config: CacheConfig = {"key": "cache_key_123", "retention": "24h"}

        assert config["key"] == "cache_key_123"
        assert config["retention"] == "24h"


class TestRequestResponseTypes:
    """测试请求响应类型"""

    def test_ir_request_creation(self):
        """测试IR请求创建"""
        request: IRRequest = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "system_instruction": "You are a helpful assistant.",
            "generation": {"temperature": 0.7, "max_tokens": 1000},
            "stream": {"enabled": True},
        }

        assert request["model"] == "gpt-4o"
        assert len(request["messages"]) == 1
        assert request["generation"]["temperature"] == 0.7

    def test_usage_info_creation(self):
        """测试使用信息创建"""
        usage: UsageInfo = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 10},
        }

        assert usage["total_tokens"] == 150
        assert usage["prompt_tokens_details"]["cached_tokens"] == 20

    def test_choice_info_creation(self):
        """测试选择信息创建"""
        choice: ChoiceInfo = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
            },
            "finish_reason": {"reason": "stop"},
        }

        assert choice["index"] == 0
        assert choice["finish_reason"]["reason"] == "stop"
        assert choice["message"]["role"] == "assistant"

    def test_ir_response_creation(self):
        """测试IR响应创建"""
        response: IRResponse = {
            "id": "resp_123",
            "object": "response",
            "created": 1640995200,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello!"}],
                    },
                    "finish_reason": {"reason": "stop"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        assert response["id"] == "resp_123"
        assert len(response["choices"]) == 1
        assert response["usage"]["total_tokens"] == 15


if __name__ == "__main__":
    pytest.main([__file__])

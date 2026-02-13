"""
Tests for LLMIR Converters Base Module

测试 converters/base 模块的所有组件：
- BaseConverter 抽象基类
- BaseContentOps 内容操作抽象基类
- BaseMessageOps 消息操作抽象基类
- BaseToolOps 工具操作抽象基类
- BaseConfigOps 配置操作抽象基类
"""

import pytest
from abc import ABC
from typing import Any, Dict, List, Tuple, Union, Iterable

from llmir.converters.base import (
    BaseConverter,
    BaseContentOps,
    BaseMessageOps,
    BaseToolOps,
    BaseConfigOps,
)
from llmir.types.ir import (
    # Content parts
    TextPart,
    ImagePart,
    FilePart,
    AudioPart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    # Messages
    Message,
    ExtensionItem,
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
)


# ============================================================================
# Mock implementations for testing
# ============================================================================


class MockContentOps(BaseContentOps):
    """Mock implementation of BaseContentOps for testing"""

    @staticmethod
    def ir_text_to_p(ir_text: TextPart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "text", "content": ir_text["text"]}

    @staticmethod
    def p_text_to_ir(provider_text: Any, **kwargs: Any) -> TextPart:
        return {"type": "text", "text": provider_text.get("content", "")}

    @staticmethod
    def ir_image_to_p(ir_image: ImagePart, **kwargs: Any) -> Dict[str, Any]:
        result = {"type": "image"}
        if "image_url" in ir_image:
            result["url"] = ir_image["image_url"]
        if "image_data" in ir_image:
            result["data"] = ir_image["image_data"]["data"]
            result["media_type"] = ir_image["image_data"]["media_type"]
        return result

    @staticmethod
    def p_image_to_ir(provider_image: Any, **kwargs: Any) -> ImagePart:
        result: ImagePart = {"type": "image"}
        if "url" in provider_image:
            result["image_url"] = provider_image["url"]
        if "data" in provider_image:
            result["image_data"] = {
                "data": provider_image["data"],
                "media_type": provider_image.get("media_type", "image/jpeg"),
            }
        return result

    @staticmethod
    def ir_file_to_p(ir_file: FilePart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "file", "name": ir_file.get("file_name", "unknown")}

    @staticmethod
    def p_file_to_ir(provider_file: Any, **kwargs: Any) -> FilePart:
        return {"type": "file", "file_name": provider_file.get("name", "unknown")}

    @staticmethod
    def ir_audio_to_p(ir_audio: AudioPart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "audio", "id": ir_audio["audio_id"]}

    @staticmethod
    def p_audio_to_ir(provider_audio: Any, **kwargs: Any) -> AudioPart:
        return {"type": "audio", "audio_id": provider_audio.get("id", "unknown")}

    @staticmethod
    def ir_reasoning_to_p(ir_reasoning: ReasoningPart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "reasoning", "content": ir_reasoning.get("reasoning", "")}

    @staticmethod
    def p_reasoning_to_ir(provider_reasoning: Any, **kwargs: Any) -> ReasoningPart:
        return {"type": "reasoning", "reasoning": provider_reasoning.get("content", "")}

    @staticmethod
    def ir_refusal_to_p(ir_refusal: RefusalPart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "refusal", "message": ir_refusal["refusal"]}

    @staticmethod
    def p_refusal_to_ir(provider_refusal: Any, **kwargs: Any) -> RefusalPart:
        return {"type": "refusal", "refusal": provider_refusal.get("message", "")}

    @staticmethod
    def ir_citation_to_p(ir_citation: CitationPart, **kwargs: Any) -> Dict[str, Any]:
        return {"type": "citation", "data": ir_citation}

    @staticmethod
    def p_citation_to_ir(provider_citation: Any, **kwargs: Any) -> CitationPart:
        return {"type": "citation", "url_citation": provider_citation.get("data", {})}


class MockMessageOps(BaseMessageOps):
    """Mock implementation of BaseMessageOps for testing"""

    @staticmethod
    def ir_messages_to_p(
        ir_messages: Iterable[Union[Message, ExtensionItem]], **kwargs: Any
    ) -> Tuple[List[Any], List[str]]:
        provider_messages = []
        warnings = []

        for item in ir_messages:
            if "role" in item:  # Message
                provider_msg = {"role": item["role"], "content": []}

                for part in item.get("content", []):
                    if part.get("type") == "text":
                        provider_msg["content"].append(
                            {"type": "text", "text": part["text"]}
                        )
                    elif part.get("type") == "tool_call":
                        provider_msg["content"].append(
                            {
                                "type": "tool_call",
                                "id": part["tool_call_id"],
                                "name": part["tool_name"],
                                "arguments": part["tool_input"],
                            }
                        )
                    else:
                        warnings.append(f"Unsupported content type: {part.get('type')}")

                provider_messages.append(provider_msg)
            else:  # ExtensionItem
                warnings.append(f"Extension item ignored: {item.get('type')}")

        return provider_messages, warnings

    @staticmethod
    def p_messages_to_ir(
        provider_messages: List[Any], **kwargs: Any
    ) -> List[Union[Message, ExtensionItem]]:
        ir_messages = []

        for msg in provider_messages:
            ir_msg = {"role": msg["role"], "content": []}

            for part in msg.get("content", []):
                if part.get("type") == "text":
                    ir_msg["content"].append({"type": "text", "text": part["text"]})
                elif part.get("type") == "tool_call":
                    ir_msg["content"].append(
                        {
                            "type": "tool_call",
                            "tool_call_id": part["id"],
                            "tool_name": part["name"],
                            "tool_input": part["arguments"],
                        }
                    )

            ir_messages.append(ir_msg)

        return ir_messages


class MockToolOps(BaseToolOps):
    """Mock implementation of BaseToolOps for testing"""

    @staticmethod
    def ir_tool_definition_to_p(
        ir_tool: ToolDefinition, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "name": ir_tool["name"],
            "description": ir_tool["description"],
            "parameters": ir_tool["parameters"],
        }

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> ToolDefinition:
        return {
            "type": "function",
            "name": provider_tool["name"],
            "description": provider_tool["description"],
            "parameters": provider_tool["parameters"],
            "required_parameters": provider_tool.get("required", []),
            "metadata": {},
        }

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: ToolChoice, **kwargs: Any) -> Any:
        if ir_tool_choice["mode"] == "auto":
            return "auto"
        elif ir_tool_choice["mode"] == "none":
            return "none"
        elif ir_tool_choice["mode"] == "tool":
            return {"type": "function", "name": ir_tool_choice["tool_name"]}
        return "auto"

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> ToolChoice:
        if provider_tool_choice == "auto":
            return {"mode": "auto", "tool_name": ""}
        elif provider_tool_choice == "none":
            return {"mode": "none", "tool_name": ""}
        elif isinstance(provider_tool_choice, dict):
            return {"mode": "tool", "tool_name": provider_tool_choice["name"]}
        return {"mode": "auto", "tool_name": ""}

    @staticmethod
    def ir_tool_call_to_p(ir_tool_call: ToolCallPart, **kwargs: Any) -> Dict[str, Any]:
        return {
            "id": ir_tool_call["tool_call_id"],
            "name": ir_tool_call["tool_name"],
            "arguments": ir_tool_call["tool_input"],
        }

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> ToolCallPart:
        return {
            "type": "tool_call",
            "tool_call_id": provider_tool_call["id"],
            "tool_name": provider_tool_call["name"],
            "tool_input": provider_tool_call["arguments"],
        }

    @staticmethod
    def ir_tool_result_to_p(
        ir_tool_result: ToolResultPart, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "call_id": ir_tool_result["tool_call_id"],
            "result": ir_tool_result["result"],
        }

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> ToolResultPart:
        return {
            "type": "tool_result",
            "tool_call_id": provider_tool_result["call_id"],
            "result": provider_tool_result["result"],
        }

    @staticmethod
    def ir_tool_config_to_p(
        ir_tool_config: ToolCallConfig, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "parallel_calls": not ir_tool_config.get("disable_parallel", False),
            "max_calls": ir_tool_config.get("max_calls", 10),
        }

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> ToolCallConfig:
        return {
            "disable_parallel": not provider_tool_config.get("parallel_calls", True),
            "max_calls": provider_tool_config.get("max_calls", 10),
        }


class MockConfigOps(BaseConfigOps):
    """Mock implementation of BaseConfigOps for testing"""

    @staticmethod
    def ir_generation_config_to_p(
        ir_config: GenerationConfig, **kwargs: Any
    ) -> Dict[str, Any]:
        result = {}
        if "temperature" in ir_config:
            result["temperature"] = ir_config["temperature"]
        if "max_tokens" in ir_config:
            result["max_tokens"] = ir_config["max_tokens"]
        if "top_p" in ir_config:
            result["top_p"] = ir_config["top_p"]
        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        result = {}
        if "temperature" in provider_config:
            result["temperature"] = provider_config["temperature"]
        if "max_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_tokens"]
        if "top_p" in provider_config:
            result["top_p"] = provider_config["top_p"]
        return result

    @staticmethod
    def ir_response_format_to_p(
        ir_format: ResponseFormatConfig, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"type": ir_format.get("type", "text")}

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        return {"type": provider_format.get("type", "text")}

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> Dict[str, Any]:
        return {"stream": ir_stream.get("enabled", False)}

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        return {"enabled": provider_stream.get("stream", False)}

    @staticmethod
    def ir_reasoning_config_to_p(
        ir_reasoning: ReasoningConfig, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"reasoning_effort": ir_reasoning.get("effort", "medium")}

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        return {"effort": provider_reasoning.get("reasoning_effort", "medium")}

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> Dict[str, Any]:
        return {"cache_key": ir_cache.get("key", "")}

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        return {"key": provider_cache.get("cache_key", "")}


class MockConverter(BaseConverter):
    """Mock implementation of BaseConverter for testing"""

    # 指定使用的ops类
    content_ops_class = MockContentOps
    message_ops_class = MockMessageOps
    tool_ops_class = MockToolOps
    config_ops_class = MockConfigOps

    def request_to_provider(
        self, ir_request: IRRequest, **kwargs: Any
    ) -> Tuple[Dict[str, Any], List[str]]:
        provider_request = {"model": ir_request["model"]}
        warnings = []

        # 转换消息
        if "messages" in ir_request:
            messages, msg_warnings = self.message_ops_class.ir_messages_to_p(
                ir_request["messages"], **kwargs
            )
            provider_request["messages"] = messages
            warnings.extend(msg_warnings)

        # 转换生成配置
        if "generation" in ir_request:
            provider_request.update(
                self.config_ops_class.ir_generation_config_to_p(
                    ir_request["generation"], **kwargs
                )
            )

        return provider_request, warnings

    def request_from_provider(
        self, provider_request: Dict[str, Any], **kwargs: Any
    ) -> IRRequest:
        ir_request: IRRequest = {"model": provider_request["model"], "messages": []}

        if "messages" in provider_request:
            ir_request["messages"] = self.message_ops_class.p_messages_to_ir(
                provider_request["messages"], **kwargs
            )

        return ir_request

    def response_from_provider(
        self, provider_response: Dict[str, Any], **kwargs: Any
    ) -> IRResponse:
        return {
            "id": provider_response.get("id", ""),
            "object": provider_response.get("object", ""),
            "created": provider_response.get("created", 0),
            "model": provider_response.get("model", ""),
            "choices": provider_response.get("choices", []),
            "usage": provider_response.get("usage", {}),
        }

    def response_to_provider(
        self, ir_response: IRResponse, **kwargs: Any
    ) -> Dict[str, Any]:
        return {
            "id": ir_response["id"],
            "object": ir_response["object"],
            "created": ir_response["created"],
            "model": ir_response["model"],
            "choices": ir_response["choices"],
            "usage": ir_response["usage"],
        }

    def messages_to_provider(
        self, messages: Iterable[Union[Message, ExtensionItem]], **kwargs: Any
    ) -> Tuple[List[Any], List[str]]:
        return self.message_ops_class.ir_messages_to_p(messages, **kwargs)

    def messages_from_provider(
        self, provider_messages: List[Any], **kwargs: Any
    ) -> List[Union[Message, ExtensionItem]]:
        return self.message_ops_class.p_messages_to_ir(provider_messages, **kwargs)


# ============================================================================
# Test classes
# ============================================================================


class TestBaseConverter:
    """测试 BaseConverter 抽象基类"""

    def setup_method(self):
        """设置测试"""
        self.converter = MockConverter()

    def test_converter_is_abstract(self):
        """测试 BaseConverter 是抽象类"""
        assert issubclass(BaseConverter, ABC)

        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseConverter()

    def test_converter_has_required_methods(self):
        """测试转换器有必需的方法"""
        required_methods = [
            "request_to_provider",
            "request_from_provider",
            "response_from_provider",
            "response_to_provider",
            "messages_to_provider",
            "messages_from_provider",
        ]

        for method_name in required_methods:
            assert hasattr(BaseConverter, method_name)
            assert callable(getattr(BaseConverter, method_name))

    def test_converter_ops_class_attributes(self):
        """测试转换器ops类属性"""
        assert hasattr(BaseConverter, "content_ops_class")
        assert hasattr(BaseConverter, "tool_ops_class")
        assert hasattr(BaseConverter, "message_ops_class")
        assert hasattr(BaseConverter, "config_ops_class")

    def test_request_to_provider(self):
        """测试请求转换到provider"""
        ir_request: IRRequest = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "generation": {"temperature": 0.7, "max_tokens": 100},
        }

        provider_request, warnings = self.converter.request_to_provider(ir_request)

        assert provider_request["model"] == "test-model"
        assert "messages" in provider_request
        assert provider_request["temperature"] == 0.7
        assert provider_request["max_tokens"] == 100
        assert isinstance(warnings, list)

    def test_request_from_provider(self):
        """测试从provider转换请求"""
        provider_request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }

        ir_request = self.converter.request_from_provider(provider_request)

        assert ir_request["model"] == "test-model"
        assert len(ir_request["messages"]) == 1
        assert ir_request["messages"][0]["role"] == "user"

    def test_messages_to_provider(self):
        """测试消息转换到provider"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there!"},
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "tool_name": "search",
                        "tool_input": {"query": "test"},
                    },
                ],
            },
        ]

        provider_messages, warnings = self.converter.messages_to_provider(messages)

        assert len(provider_messages) == 2
        assert provider_messages[0]["role"] == "user"
        assert provider_messages[1]["role"] == "assistant"
        assert isinstance(warnings, list)

    def test_messages_from_provider(self):
        """测试从provider转换消息"""
        provider_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]

        ir_messages = self.converter.messages_from_provider(provider_messages)

        assert len(ir_messages) == 1
        assert ir_messages[0]["role"] == "user"
        assert ir_messages[0]["content"][0]["type"] == "text"

    def test_message_to_provider_convenience(self):
        """测试单个消息转换便利方法"""
        message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

        provider_message, warnings = self.converter.message_to_provider(message)

        assert provider_message["role"] == "user"
        assert isinstance(warnings, list)

    def test_message_from_provider_convenience(self):
        """测试从provider转换单个消息便利方法"""
        provider_message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }

        ir_message = self.converter.message_from_provider(provider_message)

        assert ir_message["role"] == "user"
        assert ir_message["content"][0]["type"] == "text"


class TestBaseContentOps:
    """测试 BaseContentOps 抽象基类"""

    def setup_method(self):
        """设置测试"""
        self.content_ops = MockContentOps()

    def test_content_ops_is_abstract(self):
        """测试 BaseContentOps 是抽象类"""
        assert issubclass(BaseContentOps, ABC)

        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseContentOps()

    def test_content_ops_has_required_methods(self):
        """测试内容操作有必需的方法"""
        required_methods = [
            "ir_text_to_p",
            "p_text_to_ir",
            "ir_image_to_p",
            "p_image_to_ir",
            "ir_file_to_p",
            "p_file_to_ir",
            "ir_audio_to_p",
            "p_audio_to_ir",
            "ir_reasoning_to_p",
            "p_reasoning_to_ir",
            "ir_refusal_to_p",
            "p_refusal_to_ir",
            "ir_citation_to_p",
            "p_citation_to_ir",
        ]

        for method_name in required_methods:
            assert hasattr(BaseContentOps, method_name)
            assert callable(getattr(BaseContentOps, method_name))

    def test_text_conversion(self):
        """测试文本转换"""
        ir_text: TextPart = {"type": "text", "text": "Hello, world!"}

        # IR → Provider
        provider_text = self.content_ops.ir_text_to_p(ir_text)
        assert provider_text["type"] == "text"
        assert provider_text["content"] == "Hello, world!"

        # Provider → IR
        converted_back = self.content_ops.p_text_to_ir(provider_text)
        assert converted_back["type"] == "text"
        assert converted_back["text"] == "Hello, world!"

    def test_image_conversion(self):
        """测试图像转换"""
        ir_image: ImagePart = {
            "type": "image",
            "image_url": "https://example.com/image.jpg",
            "detail": "high",
        }

        # IR → Provider
        provider_image = self.content_ops.ir_image_to_p(ir_image)
        assert provider_image["type"] == "image"
        assert provider_image["url"] == "https://example.com/image.jpg"

        # Provider → IR
        converted_back = self.content_ops.p_image_to_ir(provider_image)
        assert converted_back["type"] == "image"
        assert converted_back["image_url"] == "https://example.com/image.jpg"

    def test_reasoning_conversion(self):
        """测试推理转换"""
        ir_reasoning: ReasoningPart = {
            "type": "reasoning",
            "reasoning": "Let me think about this...",
            "status": "completed",
        }

        # IR → Provider
        provider_reasoning = self.content_ops.ir_reasoning_to_p(ir_reasoning)
        assert provider_reasoning["type"] == "reasoning"
        assert provider_reasoning["content"] == "Let me think about this..."

        # Provider → IR
        converted_back = self.content_ops.p_reasoning_to_ir(provider_reasoning)
        assert converted_back["type"] == "reasoning"
        assert converted_back["reasoning"] == "Let me think about this..."


class TestBaseMessageOps:
    """测试 BaseMessageOps 抽象基类"""

    def setup_method(self):
        """设置测试"""
        self.message_ops = MockMessageOps()

    def test_message_ops_is_abstract(self):
        """测试 BaseMessageOps 是抽象类"""
        assert issubclass(BaseMessageOps, ABC)

        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseMessageOps()

    def test_message_ops_has_required_methods(self):
        """测试消息操作有必需的方法"""
        required_methods = ["ir_messages_to_p", "p_messages_to_ir"]

        for method_name in required_methods:
            assert hasattr(BaseMessageOps, method_name)
            assert callable(getattr(BaseMessageOps, method_name))

    def test_messages_conversion(self):
        """测试消息批量转换"""
        ir_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi!"},
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "tool_name": "search",
                        "tool_input": {"query": "test"},
                    },
                ],
            },
        ]

        # IR → Provider
        provider_messages, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(provider_messages) == 2
        assert provider_messages[0]["role"] == "user"
        assert provider_messages[1]["role"] == "assistant"
        assert isinstance(warnings, list)

        # Provider → IR
        converted_back = self.message_ops.p_messages_to_ir(provider_messages)
        assert len(converted_back) == 2
        assert converted_back[0]["role"] == "user"
        assert converted_back[1]["role"] == "assistant"

    def test_extension_item_handling(self):
        """测试扩展项处理"""
        items = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "type": "system_event",
                "event_type": "session_start",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        ]

        provider_messages, warnings = self.message_ops.ir_messages_to_p(items)

        # 应该只有一个消息被转换，扩展项被忽略并产生警告
        assert len(provider_messages) == 1
        assert len(warnings) == 1
        assert "Extension item ignored" in warnings[0]

    def test_validate_messages(self):
        """测试消息验证"""
        # 有效消息
        valid_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        errors = self.message_ops.validate_messages(valid_messages)
        assert len(errors) == 0

        # 无效消息 - 不是列表
        invalid_messages = "not a list"
        errors = self.message_ops.validate_messages(invalid_messages)
        assert len(errors) > 0
        assert "must be an iterable" in errors[0]

        # 无效消息 - 缺少role或type
        invalid_messages = [{"some_field": "value"}]
        errors = self.message_ops.validate_messages(invalid_messages)
        assert len(errors) > 0
        assert "must have either 'role'" in errors[0]


class TestBaseToolOps:
    """测试 BaseToolOps 抽象基类"""

    def setup_method(self):
        """设置测试"""
        self.tool_ops = MockToolOps()

    def test_tool_ops_is_abstract(self):
        """测试 BaseToolOps 是抽象类"""
        assert issubclass(BaseToolOps, ABC)

        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseToolOps()

    def test_tool_ops_has_required_methods(self):
        """测试工具操作有必需的方法"""
        required_methods = [
            "ir_tool_definition_to_p",
            "p_tool_definition_to_ir",
            "ir_tool_choice_to_p",
            "p_tool_choice_to_ir",
            "ir_tool_call_to_p",
            "p_tool_call_to_ir",
            "ir_tool_result_to_p",
            "p_tool_result_to_ir",
            "ir_tool_config_to_p",
            "p_tool_config_to_ir",
        ]

        for method_name in required_methods:
            assert hasattr(BaseToolOps, method_name)
            assert callable(getattr(BaseToolOps, method_name))

    def test_tool_definition_conversion(self):
        """测试工具定义转换"""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {"type": "object", "properties": {}},
            "required_parameters": ["location"],
            "metadata": {},
        }

        # IR → Provider
        provider_tool = self.tool_ops.ir_tool_definition_to_p(ir_tool)
        assert provider_tool["name"] == "get_weather"
        assert provider_tool["description"] == "Get weather information"

        # Provider → IR
        converted_back = self.tool_ops.p_tool_definition_to_ir(provider_tool)
        assert converted_back["type"] == "function"
        assert converted_back["name"] == "get_weather"

    def test_tool_choice_conversion(self):
        """测试工具选择转换"""
        # Auto choice
        ir_choice: ToolChoice = {"mode": "auto", "tool_name": ""}
        provider_choice = self.tool_ops.ir_tool_choice_to_p(ir_choice)
        assert provider_choice == "auto"

        # Specific tool choice
        ir_choice = {"mode": "tool", "tool_name": "get_weather"}
        provider_choice = self.tool_ops.ir_tool_choice_to_p(ir_choice)
        assert provider_choice["type"] == "function"
        assert provider_choice["name"] == "get_weather"

    def test_tool_call_conversion(self):
        """测试工具调用转换"""
        ir_call: ToolCallPart = {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "search",
            "tool_input": {"query": "test"},
        }

        # IR → Provider
        provider_call = self.tool_ops.ir_tool_call_to_p(ir_call)
        assert provider_call["id"] == "call_123"
        assert provider_call["name"] == "search"
        assert provider_call["arguments"]["query"] == "test"

        # Provider → IR
        converted_back = self.tool_ops.p_tool_call_to_ir(provider_call)
        assert converted_back["type"] == "tool_call"
        assert converted_back["tool_call_id"] == "call_123"
        assert converted_back["tool_name"] == "search"


class TestBaseConfigOps:
    """测试 BaseConfigOps 抽象基类"""

    def setup_method(self):
        """设置测试"""
        self.config_ops = MockConfigOps()

    def test_config_ops_is_abstract(self):
        """测试 BaseConfigOps 是抽象类"""
        assert issubclass(BaseConfigOps, ABC)

        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseConfigOps()

    def test_config_ops_has_required_methods(self):
        """测试配置操作有必需的方法"""
        required_methods = [
            "ir_generation_config_to_p",
            "p_generation_config_to_ir",
            "ir_response_format_to_p",
            "p_response_format_to_ir",
            "ir_stream_config_to_p",
            "p_stream_config_to_ir",
            "ir_reasoning_config_to_p",
            "p_reasoning_config_to_ir",
            "ir_cache_config_to_p",
            "p_cache_config_to_ir",
        ]

        for method_name in required_methods:
            assert hasattr(BaseConfigOps, method_name)
            assert callable(getattr(BaseConfigOps, method_name))

    def test_generation_config_conversion(self):
        """测试生成配置转换"""
        ir_config: GenerationConfig = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        }

        # IR → Provider
        provider_config = self.config_ops.ir_generation_config_to_p(ir_config)
        assert provider_config["temperature"] == 0.7
        assert provider_config["max_tokens"] == 1000
        assert provider_config["top_p"] == 0.9

        # Provider → IR
        converted_back = self.config_ops.p_generation_config_to_ir(provider_config)
        assert converted_back["temperature"] == 0.7
        assert converted_back["max_tokens"] == 1000

    def test_response_format_conversion(self):
        """测试响应格式转换"""
        ir_format: ResponseFormatConfig = {"type": "json_object"}

        # IR → Provider
        provider_format = self.config_ops.ir_response_format_to_p(ir_format)
        assert provider_format["type"] == "json_object"

        # Provider → IR
        converted_back = self.config_ops.p_response_format_to_ir(provider_format)
        assert converted_back["type"] == "json_object"

    def test_stream_config_conversion(self):
        """测试流式配置转换"""
        ir_stream: StreamConfig = {"enabled": True, "include_usage": True}

        # IR → Provider
        provider_stream = self.config_ops.ir_stream_config_to_p(ir_stream)
        assert provider_stream["stream"] is True

        # Provider → IR
        converted_back = self.config_ops.p_stream_config_to_ir(provider_stream)
        assert converted_back["enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__])

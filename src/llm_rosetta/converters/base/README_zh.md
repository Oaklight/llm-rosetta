# Base Converter 模块 — 底层向上分层架构（Bottom-Up Ops Pattern）

[English Version](./README_en.md) | [中文版](./README_zh.md)

## 概述

Base Converter 模块为所有 provider 转换器提供抽象基类，按**功能域**组织，采用 **Bottom-Up Ops Pattern** 设计。每个功能域处理转换管道的特定层面，从底层的内容部分到顶层的完整请求/响应对象。

所有 4 个 provider 转换器（OpenAI Chat、OpenAI Responses、Anthropic、Google GenAI）均已完成此模式的重构。

## 文件结构

```
src/llm_rosetta/converters/base/
├── __init__.py      # 模块导出
├── content.py       # BaseContentOps — 内容转换（text, image, file, audio, reasoning, refusal, citation）
├── tools.py         # BaseToolOps — 工具转换（definition, choice, call, result, config）
├── messages.py      # BaseMessageOps — 消息转换（批量/单个消息，验证）
├── configs.py       # BaseConfigOps — 配置转换（generation, response_format, stream, reasoning, cache）
└── converter.py     # BaseConverter — 主转换器（通过类属性组合 ops 类）
```

## 架构：Bottom-Up Ops Pattern

转换管道按层次组织，上层调用下层：

```
L3  Converter       — request_to_provider / request_from_provider / response_*
 │
 ├── L2  ConfigOps   — generation, response_format, stream, reasoning, cache
 ├── L1  MessageOps  — 批量消息转换（调用 ContentOps + ToolOps）
 ├── L0.5 ToolOps    — tool definition, choice, call, result, config
 └── L0  ContentOps  — text, image, file, audio, reasoning, refusal, citation
```

**核心原则：**

- **功能域组织**：每个 ops 类处理一个功能域（content、tools、messages、configs）
- **清晰的转换层次**：content → messages → requests/responses
- **组合模式**：子类通过类属性指定 ops 类，而非继承
- **双向转换**：每个概念都有 `ir_*_to_p()` 和 `p_*_to_ir()` 方法

## 模块说明

### `content.py` — BaseContentOps

处理单个内容部分的转换。所有方法均为 `@staticmethod @abstractmethod`。

| 内容类型 | IR → Provider | Provider → IR |
|---------|---------------|---------------|
| 文本 | `ir_text_to_p(ir_text: TextPart) → Any` | `p_text_to_ir(provider_text) → TextPart` |
| 图像 | `ir_image_to_p(ir_image: ImagePart) → Any` | `p_image_to_ir(provider_image) → ImagePart` |
| 文件 | `ir_file_to_p(ir_file: FilePart) → Any` | `p_file_to_ir(provider_file) → FilePart` |
| 音频 | `ir_audio_to_p(ir_audio: AudioPart) → Any` | `p_audio_to_ir(provider_audio) → AudioPart` |
| 推理 | `ir_reasoning_to_p(ir_reasoning: ReasoningPart) → Any` | `p_reasoning_to_ir(provider_reasoning) → ReasoningPart` |
| 拒绝 | `ir_refusal_to_p(ir_refusal: RefusalPart) → Any` | `p_refusal_to_ir(provider_refusal) → RefusalPart` |
| 引用 | `ir_citation_to_p(ir_citation: CitationPart) → Any` | `p_citation_to_ir(provider_citation) → CitationPart` |

### `tools.py` — BaseToolOps

处理工具的完整生命周期：定义 → 选择 → 调用 → 结果 → 配置。所有方法均为 `@staticmethod @abstractmethod`。

| 工具概念 | IR → Provider | Provider → IR |
|---------|---------------|---------------|
| 定义 | `ir_tool_definition_to_p(ir_tool: ToolDefinition) → Any` | `p_tool_definition_to_ir(provider_tool) → ToolDefinition` |
| 选择 | `ir_tool_choice_to_p(ir_tool_choice: ToolChoice) → Any` | `p_tool_choice_to_ir(provider_tool_choice) → ToolChoice` |
| 调用 | `ir_tool_call_to_p(ir_tool_call: ToolCallPart) → Any` | `p_tool_call_to_ir(provider_tool_call) → ToolCallPart` |
| 结果 | `ir_tool_result_to_p(ir_tool_result: ToolResultPart) → Any` | `p_tool_result_to_ir(provider_tool_result) → ToolResultPart` |
| 配置 | `ir_tool_config_to_p(ir_tool_config: ToolCallConfig) → Any` | `p_tool_config_to_ir(provider_tool_config) → ToolCallConfig` |

### `messages.py` — BaseMessageOps

处理消息级别的转换，是 content/tool 层与 request/response 层之间的桥梁。

| 方法 | 签名 | 说明 |
|------|------|------|
| `ir_messages_to_p` | `(ir_messages: Iterable[Message \| ExtensionItem]) → Tuple[List[Any], List[str]]` | 批量转换 IR 消息到 provider 格式（抽象） |
| `p_messages_to_ir` | `(provider_messages: List[Any]) → List[Message \| ExtensionItem]` | 批量转换 provider 消息到 IR 格式（抽象） |
| `ir_message_to_p` | `(ir_message) → Tuple[Any, List[str]]` | 单个消息便利方法（具体） |
| `p_message_to_ir` | `(provider_message) → Message \| ExtensionItem` | 单个消息便利方法（具体） |
| `validate_messages` | `(messages) → List[str]` | 验证消息列表（具体，可重写） |

### `configs.py` — BaseConfigOps

处理所有配置参数的转换。所有方法均为 `@staticmethod @abstractmethod`。

| 配置类型 | IR → Provider | Provider → IR |
|---------|---------------|---------------|
| 生成配置 | `ir_generation_config_to_p(ir_config: GenerationConfig) → Any` | `p_generation_config_to_ir(provider_config) → GenerationConfig` |
| 响应格式 | `ir_response_format_to_p(ir_format: ResponseFormatConfig) → Any` | `p_response_format_to_ir(provider_format) → ResponseFormatConfig` |
| 流式配置 | `ir_stream_config_to_p(ir_stream: StreamConfig) → Any` | `p_stream_config_to_ir(provider_stream) → StreamConfig` |
| 推理配置 | `ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig) → Any` | `p_reasoning_config_to_ir(provider_reasoning) → ReasoningConfig` |
| 缓存配置 | `ir_cache_config_to_p(ir_cache: CacheConfig) → Any` | `p_cache_config_to_ir(provider_cache) → CacheConfig` |

### `converter.py` — BaseConverter

顶层转换器，组合所有 ops 类。定义 6 个抽象方法 + 2 个便利方法。

**类属性**（由子类设置）：

```python
content_ops_class: Optional[Type] = None   # → BaseContentOps 子类
tool_ops_class: Optional[Type] = None      # → BaseToolOps 子类
message_ops_class: Optional[Type] = None   # → BaseMessageOps 子类
config_ops_class: Optional[Type] = None    # → BaseConfigOps 子类
```

**抽象方法：**

| 方法 | 签名 | 说明 |
|------|------|------|
| `request_to_provider` | `(ir_request: IRRequest, **kwargs) → Tuple[Dict, List[str]]` | IR 请求 → provider 请求。Google 支持 `output_format="rest"` 参数，可直接生成 REST API 格式的请求体。 |
| `request_from_provider` | `(provider_request: Dict) → IRRequest` | Provider 请求 → IR 请求 |
| `response_from_provider` | `(provider_response: Dict) → IRResponse` | Provider 响应 → IR 响应 |
| `response_to_provider` | `(ir_response: IRResponse) → Dict` | IR 响应 → provider 响应 |
| `messages_to_provider` | `(messages: Iterable[Message \| ExtensionItem]) → Tuple[List, List[str]]` | IR 消息 → provider 消息 |
| `messages_from_provider` | `(provider_messages: List) → List[Message \| ExtensionItem]` | Provider 消息 → IR 消息 |

**便利方法**（具体）：

| 方法 | 说明 |
|------|------|
| `message_to_provider` | 单个消息包装，调用 `messages_to_provider` |
| `message_from_provider` | 单个消息包装，调用 `messages_from_provider` |

## 实现指南

### 创建新的 Provider 转换器

1. **创建 provider 目录**：

    ```
    src/llm_rosetta/converters/your_provider/
    ├── __init__.py
    ├── content_ops.py
    ├── tool_ops.py
    ├── message_ops.py
    ├── config_ops.py
    └── converter.py
    ```

2. **实现内容操作**：

    ```python
    from ..base import BaseContentOps

    class YourProviderContentOps(BaseContentOps):
        @staticmethod
        def ir_text_to_p(ir_text, **kwargs):
            return {"type": "text", "value": ir_text["text"]}

        @staticmethod
        def p_text_to_ir(provider_text, **kwargs):
            return {"type": "text", "text": provider_text["value"]}

        # 实现所有其他抽象方法...
    ```

3. **实现工具操作**：

    ```python
    from ..base import BaseToolOps

    class YourProviderToolOps(BaseToolOps):
        @staticmethod
        def ir_tool_definition_to_p(ir_tool, **kwargs):
            return {"name": ir_tool["name"], "params": ir_tool["parameters"]}

        # 实现所有其他抽象方法...
    ```

4. **实现消息操作**：

    ```python
    from ..base import BaseMessageOps

    class YourProviderMessageOps(BaseMessageOps):
        def __init__(self, content_ops, tool_ops):
            self.content_ops = content_ops
            self.tool_ops = tool_ops

        @staticmethod
        def ir_messages_to_p(ir_messages, **kwargs):
            # 使用 content_ops 和 tool_ops 转换消息内容
            ...

        # 实现所有其他抽象方法...
    ```

5. **实现配置操作**：

    ```python
    from ..base import BaseConfigOps

    class YourProviderConfigOps(BaseConfigOps):
        @staticmethod
        def ir_generation_config_to_p(ir_config, **kwargs):
            return {"temp": ir_config.get("temperature", 1.0)}

        # 实现所有其他抽象方法...
    ```

6. **组合转换器**：

    ```python
    from ..base import BaseConverter
    from .content_ops import YourProviderContentOps
    from .tool_ops import YourProviderToolOps
    from .message_ops import YourProviderMessageOps
    from .config_ops import YourProviderConfigOps

    class YourProviderConverter(BaseConverter):
        content_ops_class = YourProviderContentOps
        tool_ops_class = YourProviderToolOps
        message_ops_class = YourProviderMessageOps
        config_ops_class = YourProviderConfigOps

        def __init__(self):
            self.content_ops = self.content_ops_class()
            self.tool_ops = self.tool_ops_class()
            self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
            self.config_ops = self.config_ops_class()

        def request_to_provider(self, ir_request, **kwargs):
            warnings = []
            result = {"model": ir_request["model"]}
            # 使用 self.message_ops, self.config_ops, self.tool_ops
            # 转换请求的各个字段
            ...
            return result, warnings

        # 实现所有其他抽象方法...
    ```

## 现有实现

所有 4 个 provider 转换器均遵循 Bottom-Up Ops Pattern：

| Provider | Converter | ContentOps | ToolOps | MessageOps | ConfigOps |
|----------|-----------|------------|---------|------------|-----------|
| OpenAI Chat | `OpenAIChatConverter` | `OpenAIChatContentOps` | `OpenAIChatToolOps` | `OpenAIChatMessageOps` | `OpenAIChatConfigOps` |
| OpenAI Responses | `OpenAIResponsesConverter` | `OpenAIResponsesContentOps` | `OpenAIResponsesToolOps` | `OpenAIResponsesMessageOps` | `OpenAIResponsesConfigOps` |
| Anthropic | `AnthropicConverter` | `AnthropicContentOps` | `AnthropicToolOps` | `AnthropicMessageOps` | `AnthropicConfigOps` |
| Google GenAI | `GoogleGenAIConverter` | `GoogleGenAIContentOps` | `GoogleGenAIToolOps` | `GoogleGenAIMessageOps` | `GoogleGenAIConfigOps` |

## 方法命名规范

- **`ir_*_to_p()`**：从 IR 格式转换到 Provider 格式
- **`p_*_to_ir()`**：从 Provider 格式转换到 IR 格式
- **`*_text_*`**、**`*_image_*`** 等：内容类型操作
- **`*_tool_definition_*`**、**`*_tool_call_*`** 等：工具生命周期操作
- **`*_messages_*`**：批量消息操作
- **`*_generation_config_*`**、**`*_stream_config_*`** 等：配置操作
- **`*_request_*`**、**`*_response_*`**：顶层请求/响应操作

## 测试策略

每个 ops 类可以独立测试：

- **ContentOps 测试**：测试单个内容部分的转换
- **ToolOps 测试**：测试工具定义、选择、调用、结果的转换
- **MessageOps 测试**：测试包含混合内容类型的批量消息转换
- **ConfigOps 测试**：测试配置参数的转换
- **Converter 测试**：测试完整的请求/响应转换管道

参见 `tests/test_converters_base.py` 获取 base 模块测试，`tests/converters/<provider>/` 获取各 provider 的分层测试。
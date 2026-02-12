# Body-Level Converter (Legacy Design)

> **⚠️ Historical Reference**: This document describes an early design for body-level
> conversion using `UnifiedBody` / `BodyConverter`. This approach has been superseded
> by the current IR (Intermediate Representation) architecture using `IRRequest` /
> `IRResponse` with per-provider converters and the bottom-up Ops pattern.
> Retained here for design evolution context.

LLMIR 的 body 级别转换器提供了在不同 LLM provider SDK 之间进行完整参数转换的功能。与现有的消息级别转换器不同，body 转换器可以处理 SDK 入口函数的完整参数集合，实现真正的 SDK 兼容性。

## 核心概念

### 支持的 SDK 类型

- **OpenAI Chat Completions** (`openai_chat`): `openai_client.chat.completions.create()`
- **OpenAI Responses** (`openai_responses`): `openai_responses_client.responses.create()`
- **Anthropic Messages** (`anthropic`): `anthropic_client.messages.create()`
- **Google GenerativeAI** (`google`): `google_client.models.generate_content()`

### UnifiedBody 中间表示

`UnifiedBody`是所有 SDK 参数的统一中间表示，包含：

- **核心参数**: `model`, `messages`
- **生成控制**: `max_tokens`, `temperature`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `seed`
- **工具相关**: `tools`, `tool_choice`
- **控制参数**: `stop_sequences`, `stream`, `system_instruction`
- **响应格式**: `response_format`, `response_mime_type`, `response_schema`
- **高级参数**: `logprobs`, `top_logprobs`, `n`
- **SDK 特定**: `sdk_specific` (保存无法映射的特定参数)

## 基本使用

### 简单转换

```python
from llmir import convert_body, SDKType

# OpenAI Chat格式
openai_body = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_completion_tokens": 150,
    "temperature": 0.7
}

# 转换为Anthropic格式
anthropic_body = convert_body(
    openai_body,
    SDKType.OPENAI_CHAT,
    SDKType.ANTHROPIC
)

print(anthropic_body)
# {
#     "model": "gpt-4",
#     "messages": [{"role": "user", "content": "What is the capital of France?"}],
#     "max_tokens": 150,
#     "temperature": 0.7,
#     "system": "You are a helpful assistant."
# }
```

### 自动检测转换

```python
from llmir import BodyConverter, SDKType

converter = BodyConverter()

# 自动检测源SDK类型
source_body = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
}

# 自动检测并转换
result = converter.convert_auto(source_body, SDKType.ANTHROPIC)
```

### 工具调用转换

```python
# OpenAI格式的工具调用
openai_with_tools = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    "tool_choice": "auto"
}

# 转换为Anthropic格式
anthropic_result = convert_body(
    openai_with_tools,
    SDKType.OPENAI_CHAT,
    SDKType.ANTHROPIC
)
```

## 高级功能

### SDK 特定参数处理

每个 SDK 都有一些特定的参数，这些参数会被保存在`sdk_specific`字段中：

```python
# OpenAI特定参数
openai_specific = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "logit_bias": {"50256": -100},  # OpenAI特定
    "user": "user123",              # OpenAI特定
    "service_tier": "auto"          # OpenAI特定
}

converter = BodyConverter()
unified = converter.from_openai_chat(openai_specific)

# 特定参数保存在sdk_specific中
print(unified.sdk_specific)
# {"logit_bias": {"50256": -100}, "user": "user123", "service_tier": "auto"}

# 转换回OpenAI格式时会恢复特定参数
converted_back = converter.to_openai_chat(unified)
print(converted_back["logit_bias"])  # {"50256": -100}
```

### SDK 可用性检查

```python
from llmir import is_sdk_available, get_available_providers

# 检查特定SDK是否可用
if is_sdk_available("openai_chat"):
    print("OpenAI SDK可用")

# 获取所有可用的providers
available = get_available_providers()
print(available)
# {"openai_chat": True, "anthropic": False, ...}
```

### 参数验证

```python
from llmir import validate_provider_body

body = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
}

# 验证body是否符合OpenAI Chat格式
is_valid = validate_provider_body("openai_chat", body)
print(is_valid)  # True
```

## 参数映射规则

### 通用参数映射

| UnifiedBody          | OpenAI Chat                 | OpenAI Responses        | Anthropic        | Google                      |
| -------------------- | --------------------------- | ----------------------- | ---------------- | --------------------------- |
| `max_tokens`         | `max_completion_tokens`     | `max_completion_tokens` | `max_tokens`     | `config.max_output_tokens`  |
| `system_instruction` | `messages[0]` (role=system) | `instructions`          | `system`         | `config.system_instruction` |
| `stop_sequences`     | `stop`                      | -                       | `stop_sequences` | `config.stop_sequences`     |
| `tools`              | `tools`                     | `tools`                 | `tools`          | `config.tools`              |
| `tool_choice`        | `tool_choice`               | -                       | `tool_choice`    | `config.tool_config`        |

### SDK 特定参数

#### OpenAI Chat 特定

- `audio`, `logit_bias`, `service_tier`, `user`

#### OpenAI Responses 特定

- `modalities`

#### Anthropic 特定

- `metadata`, `service_tier`, `thinking`

#### Google 特定

- `safety_settings`, `cached_content`, `response_modalities`, `thinking_config`

## 类型支持

### 可选依赖

body 转换器支持可选依赖，即使某些 SDK 未安装也能正常工作：

```python
# 即使没有安装anthropic包，也可以进行类型检查
from llmir.types.anthropic_types import AnthropicBody  # 使用TypedDict fallback
```

### 类型检查

```python
from llmir.types.provider_types import OpenAIChatBody, AnthropicBody

def process_openai_request(body: OpenAIChatBody) -> AnthropicBody:
    return convert_body(body, SDKType.OPENAI_CHAT, SDKType.ANTHROPIC)
```

## 错误处理

```python
from llmir import BodyConverter, SDKType

converter = BodyConverter()

try:
    # 无效的body
    invalid_body = {"model": "gpt-4"}  # 缺少messages

    # 自动检测会返回None
    detected = converter.detect_source_sdk(invalid_body)
    if detected is None:
        print("无法检测SDK类型")

    # 转换可能抛出异常
    result = converter.convert_auto(invalid_body, SDKType.ANTHROPIC)

except ValueError as e:
    print(f"转换失败: {e}")
```

## 性能考虑

- body 转换器复用现有的消息和工具转换器，保持高效
- 转换过程是无状态的，可以安全地并发使用
- `UnifiedBody`使用 dataclass，内存效率高

## 限制和注意事项

1. **参数兼容性**: 不是所有参数都能在 SDK 间完美映射
2. **SDK 特定功能**: 某些 SDK 特有功能在转换后可能丢失
3. **验证限制**: 参数验证是基础的，不能替代完整的 schema 验证
4. **版本兼容**: 依赖于各 SDK 的当前版本，API 变化可能影响转换

## 示例代码

完整的使用示例请参考 `examples/body_converter_example.py`。

## API 参考

### 主要类和函数

- `BodyConverter`: 主要的转换器类
- `UnifiedBody`: 统一的参数表示
- `SDKType`: SDK 类型枚举
- `convert_body()`: 便捷转换函数
- `detect_provider_from_body()`: 自动检测函数
- `validate_provider_body()`: 验证函数
- `is_sdk_available()`: SDK 可用性检查

### 类型定义

- `OpenAIChatBody`, `OpenAIResponsesBody`: OpenAI 类型
- `AnthropicBody`: Anthropic 类型
- `GoogleBody`: Google 类型
- `ProviderBody`: 所有 provider 类型的联合类型

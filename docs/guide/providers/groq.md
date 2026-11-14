# Groq

Groq 在其自有推理硬件上提供开源权重模型，接口兼容 OpenAI。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `groq--chat_completions` | `chat_completions` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Pre-IR | `rename_reasoning()` | 将 `message`/`delta` 上的 `reasoning` 改名为 `reasoning_content` |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

该改名对不含 reasoning 的响应是空操作，因此同一 shim 也服务非推理模型。
`effort_field` 为 `reasoning_effort`。

!!! note
    `reasoning` 字段名依据 Groq 文档中 `openai/gpt-oss-*` 的响应结构，
    尚未实测。

## 速率限制

免费套餐给 chat 模型 30 RPM、1,000 RPD、8,000 TPM、200,000 TPD。
准确额度按组织计算，从账户的限额页面读取，而不是由 shim 决定。

## 网关配置

```jsonc
{
  "providers": {
    "groq": {
      "type": "groq--chat_completions",
      "api_key": "${GROQ_API_KEY}"
    }
  }
}
```

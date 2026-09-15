# DeepSeek

DeepSeek 通过兼容 OpenAI Chat Completions 的 API 提供支持，并针对其 R1 推理模型的思考行为进行了专门处理。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |

## 推理支持

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | `low` → `max` |

**思考模式：**

| IR 模式 | DeepSeek 值 |
|:---|:---|
| `enabled` | `enabled` |
| `disabled` | `disabled` |

!!! warning "无 Auto 模式"
    DeepSeek R1 要求显式启用或禁用思考功能——不支持 `auto` 模式。当 IR 请求 `mode: auto` 时，仅发送 effort 级别，不包含思考块。

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `strip_fields("n", "logit_bias", "seed")` | 移除不支持的请求字段 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

```jsonc
{
  "providers": {
    "deepseek": {
      "shim": "deepseek",
      "api_key": "${DEEPSEEK_API_KEY}"
    }
  }
}
```

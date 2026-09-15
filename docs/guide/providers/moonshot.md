# Moonshot

Moonshot（Kimi）通过兼容 OpenAI Chat Completions 的 API 提供支持。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `moonshot` | `openai_chat` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `strip_fields("logprobs", "top_logprobs", "logit_bias", "seed")` | 移除不支持的请求字段 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

```jsonc
{
  "providers": {
    "moonshot": {
      "shim": "moonshot",
      "api_key": "${MOONSHOT_API_KEY}"
    }
  }
}
```

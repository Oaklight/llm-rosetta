# Qwen

Qwen（阿里云 / DashScope）通过兼容 OpenAI Chat Completions 的 API 提供支持。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `qwen` | `openai_chat` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `strip_fields("frequency_penalty", "logit_bias")` | 移除不支持的请求字段 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

```jsonc
{
  "providers": {
    "qwen": {
      "shim": "qwen",
      "api_key": "${DASHSCOPE_API_KEY}"
    }
  }
}
```

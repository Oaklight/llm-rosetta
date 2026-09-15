# xAI

xAI（Grok）通过兼容 OpenAI Chat Completions 的 API 提供支持。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `xai` | `openai_chat` | `https://api.x.ai/v1` | `XAI_API_KEY` |

## 推理支持

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | `minimal` → `xhigh` |

!!! note "扩展 Effort 范围"
    xAI 支持扩展的 effort 范围，最高可达 `xhigh`，超出了大多数其他提供方使用的标准 `high` 上限。

## 转换规则

| 类型 | 转换 | 用途 |
|:---|:---|:---|
| Post-IR | `strip_fields("logit_bias")` | 移除不支持的请求字段 |
| IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

```jsonc
{
  "providers": {
    "xai": {
      "shim": "xai",
      "api_key": "${XAI_API_KEY}"
    }
  }
}
```

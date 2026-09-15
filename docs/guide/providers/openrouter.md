# OpenRouter

OpenRouter 通过两个 shim 端点提供支持，可以使用 OpenAI Chat Completions 和 Anthropic Messages 两种格式访问其统一 API。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `openrouter--openai_chat` | `openai_chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `openrouter--anthropic` | `anthropic` | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` |

## 推理支持

**Chat Completions (`openrouter--openai_chat`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | `low` → `xhigh` |

**Anthropic (`openrouter--anthropic`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `output_config.effort` |
| Effort 范围 | `low` → `max` |
| 思考模式 | `auto` → `adaptive`、`enabled`、`disabled` |

## 转换规则

| Shim | 类型 | 转换 | 用途 |
|:---|:---|:---|:---|
| `openrouter--openai_chat` | Pre-IR | `rename_reasoning()` | 将 `message.reasoning` 重命名为 `message.reasoning_content` 以匹配 OpenRouter 的字段命名 |
| `openrouter--anthropic` | IR | `auto_cache_breakpoints()` | 插入缓存断点以支持 prompt 缓存 |
| 两者 | IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "openrouter": {
          "shim": "openrouter--openai_chat",
          "api_key": "${OPENROUTER_API_KEY}"
        }
      }
    }
    ```

=== "Anthropic"

    ```jsonc
    {
      "providers": {
        "openrouter-anthropic": {
          "shim": "openrouter--anthropic",
          "api_key": "${OPENROUTER_API_KEY}"
        }
      }
    }
    ```

## 备注

- OpenRouter 使用统一的 API Key 访问所有上游模型。Shim 的选择决定了与 OpenRouter 通信所使用的 API 格式，而非可用的模型。
- Chat Completions 端点支持扩展的 effort 范围，最高可达 `xhigh`。

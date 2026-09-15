# MiniMax

MiniMax 通过两个 shim 端点提供支持，分别覆盖 OpenAI Chat Completions 和 Anthropic Messages API 格式。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `minimax--openai_chat` | `openai_chat` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` |
| `minimax--anthropic` | `anthropic` | `https://api.minimaxi.com/anthropic` | `MINIMAX_API_KEY` |

## 推理支持

**Chat Completions (`minimax--openai_chat`)：**

| 字段 | 值 |
|:---|:---|
| Effort 范围 | `low` → `high` |
| 思考模式 | `auto` → `adaptive`、`enabled`、`disabled` |

**Anthropic (`minimax--anthropic`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `output_config.effort` |
| Effort 范围 | `low` → `max` |
| 思考模式 | `auto` → `adaptive`、`enabled`、`disabled` |

## 转换规则

| Shim | 类型 | 转换 | 用途 |
|:---|:---|:---|:---|
| `minimax--openai_chat` | Pre-IR | `parse_think_tags()` | 将响应中的 `<think>...</think>` 标签解析为结构化的 `reasoning_content` |
| `minimax--openai_chat` | Post-IR | `strip_fields("logprobs", "top_logprobs", "seed", "stop")` | 移除不支持的请求字段 |
| `minimax--openai_chat` | Post-IR | `inject_reasoning_split()` | 当请求中包含思考配置时设置 `reasoning_split: true` |
| `minimax--anthropic` | IR | `auto_cache_breakpoints()` | 插入缓存断点以支持 prompt 缓存 |
| 两者 | IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "minimax": {
          "shim": "minimax--openai_chat",
          "api_key": "${MINIMAX_API_KEY}"
        }
      }
    }
    ```

=== "Anthropic"

    ```jsonc
    {
      "providers": {
        "minimax-anthropic": {
          "shim": "minimax--anthropic",
          "api_key": "${MINIMAX_API_KEY}"
        }
      }
    }
    ```

## 备注

- Chat Completions 端点包含对 MiniMax `<think>` 标签格式的特殊处理。当模型以 `<think>...</think>` 标签返回推理内容而非结构化的 `reasoning_content` 时，shim 会自动将其解析为标准推理格式。
- 当启用思考功能时，`reasoning_split` 标志会自动注入到请求中，通知 MiniMax 将推理过程与最终响应分离。

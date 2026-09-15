# Volcengine

火山引擎（字节跳动 / 豆包）通过两个 shim 端点提供支持，分别覆盖 OpenAI Chat Completions 和 Responses API 格式。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `volcengine--openai_chat` | `openai_chat` | `https://ark.cn-beijing.volces.com/api/v3` | `VOLCENGINE_API_KEY` |
| `volcengine--openai_responses` | `openai_responses` | `https://ark.cn-beijing.volces.com/api/v3` | `VOLCENGINE_API_KEY` |

## 推理支持

**Chat Completions (`volcengine--openai_chat`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | `minimal` → `high` |
| 思考模式 | `auto`、`enabled`、`disabled` |

**Responses API (`volcengine--openai_responses`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning.effort` |
| Effort 范围 | `minimal` → `high` |

## 转换规则

| Shim | 类型 | 转换 | 用途 |
|:---|:---|:---|:---|
| `volcengine--openai_chat` | Post-IR | `strip_fields("logprobs", "top_logprobs")` | 移除不支持的请求字段 |
| 两者 | IR | `hoist_late_system_messages()` | 将靠后的系统消息移至对话开头 |

## 网关配置

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "volcengine": {
          "shim": "volcengine--openai_chat",
          "api_key": "${VOLCENGINE_API_KEY}"
        }
      }
    }
    ```

=== "Responses API"

    ```jsonc
    {
      "providers": {
        "volcengine-responses": {
          "shim": "volcengine--openai_responses",
          "api_key": "${VOLCENGINE_API_KEY}"
        }
      }
    }
    ```

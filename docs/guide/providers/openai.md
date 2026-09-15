# OpenAI

OpenAI 通过两个 shim 端点提供支持，分别覆盖 Chat Completions API 和较新的 Responses API。

## 端点

| Shim 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 |
|:---|:---|:---|:---|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |

## 推理支持

两个端点均支持推理/思考功能，字段名称因 API 标准不同而略有差异。

**Chat Completions (`openai`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning_effort` |
| Effort 范围 | `minimal` → `high` |
| 可见性模式 | `auto`、`concise`、`detailed` |

**Responses API (`openai_responses`)：**

| 字段 | 值 |
|:---|:---|
| Effort 字段 | `reasoning.effort` |
| Effort 范围 | `minimal` → `high` |
| 可见性模式 | `auto`、`concise`、`detailed` |

## 能力标志

| 标志 | `openai` | `openai_responses` |
|:---|:---:|:---:|
| 自定义工具 | ✅ | ✅ |
| 最大工具描述长度 | 1024 字符 | — |
| 原生工具搜索 | — | ✅ |

## 转换规则

两个端点均应用 `hoist_late_system_messages()` 将对话中靠后的系统消息提升到开头。无字段被剥离或重命名。

## 网关配置

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "openai": {
          "shim": "openai",
          "api_key": "${OPENAI_API_KEY}"
        }
      }
    }
    ```

=== "Responses API"

    ```jsonc
    {
      "providers": {
        "openai-responses": {
          "shim": "openai_responses",
          "api_key": "${OPENAI_API_KEY}"
        }
      }
    }
    ```

## 备注

- `openai` shim 使用 Chat Completions 格式（`/v1/chat/completions`），而 `openai_responses` 使用 Responses 格式（`/v1/responses`）。请根据下游客户端所需的格式进行选择。
- 响应 ID 前缀不同：Chat Completions 为 `chatcmpl-`，Responses API 为 `resp_`。

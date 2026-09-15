# Volcengine

Volcengine (ByteDance / Doubao) is supported through two shim endpoints covering both the OpenAI Chat Completions and Responses API formats.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `volcengine--openai_chat` | `openai_chat` | `https://ark.cn-beijing.volces.com/api/v3` | `VOLCENGINE_API_KEY` |
| `volcengine--openai_responses` | `openai_responses` | `https://ark.cn-beijing.volces.com/api/v3` | `VOLCENGINE_API_KEY` |

## Reasoning Support

**Chat Completions (`volcengine--openai_chat`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | `minimal` → `high` |
| Thinking modes | `auto`, `enabled`, `disabled` |

**Responses API (`volcengine--openai_responses`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning.effort` |
| Effort range | `minimal` → `high` |

## Transforms

| Shim | Type | Transform | Purpose |
|:---|:---|:---|:---|
| `volcengine--openai_chat` | Post-IR | `strip_fields("logprobs", "top_logprobs")` | Removes unsupported request fields |
| Both | IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

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

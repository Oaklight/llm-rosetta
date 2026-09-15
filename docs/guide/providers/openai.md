# OpenAI

OpenAI is supported through two shim endpoints covering both the Chat Completions API and the newer Responses API.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |

## Reasoning Support

Both endpoints support reasoning/thinking with slightly different field names matching their respective API standards.

**Chat Completions (`openai`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | `minimal` → `high` |
| Visibility modes | `auto`, `concise`, `detailed` |

**Responses API (`openai_responses`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning.effort` |
| Effort range | `minimal` → `high` |
| Visibility modes | `auto`, `concise`, `detailed` |

## Capability Flags

| Flag | `openai` | `openai_responses` |
|:---|:---:|:---:|
| Custom tools | ✅ | ✅ |
| Max tool description | 1024 chars | — |
| Native tool search | — | ✅ |

## Transforms

Both endpoints apply `hoist_late_system_messages()` to consolidate any late system messages to the beginning of the conversation. No fields are stripped or renamed.

## Gateway Configuration

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

## Notes

- The `openai` shim uses the Chat Completions format (`/v1/chat/completions`), while `openai_responses` uses the Responses format (`/v1/responses`). Choose based on which downstream client format you need.
- Response ID prefixes differ: `chatcmpl-` for Chat Completions, `resp_` for Responses API.

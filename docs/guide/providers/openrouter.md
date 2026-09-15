# OpenRouter

OpenRouter is supported through two shim endpoints, allowing access to its unified API in both OpenAI Chat Completions and Anthropic Messages formats.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `openrouter--openai_chat` | `openai_chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `openrouter--anthropic` | `anthropic` | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` |

## Reasoning Support

**Chat Completions (`openrouter--openai_chat`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | `low` → `xhigh` |

**Anthropic (`openrouter--anthropic`):**

| Field | Value |
|:---|:---|
| Effort field | `output_config.effort` |
| Effort range | `low` → `max` |
| Thinking modes | `auto` → `adaptive`, `enabled`, `disabled` |

## Transforms

| Shim | Type | Transform | Purpose |
|:---|:---|:---|:---|
| `openrouter--openai_chat` | Pre-IR | `rename_reasoning()` | Renames `message.reasoning` → `message.reasoning_content` to match OpenRouter's field naming |
| `openrouter--anthropic` | IR | `auto_cache_breakpoints()` | Inserts cache breakpoints for prompt caching |
| Both | IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

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

## Notes

- OpenRouter uses a unified API key for all upstream models. The shim choice determines which API format is used to communicate with OpenRouter, not which models are available.
- The Chat Completions endpoint supports an extended effort range up to `xhigh`.

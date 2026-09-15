# Argo

Argo is Argonne National Laboratory's internal API gateway, providing access to various LLM models through a unified endpoint. It is supported through two shim endpoints.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `argo--openai_chat` | `openai_chat` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` |
| `argo--anthropic` | `anthropic` | `https://apps.inside.anl.gov/argoapi` | `ARGO_API_KEY` |

!!! info "Model ID Field"
    Both Argo shims use `model_id_field: internal_id`, meaning the model identifier is sent as `internal_id` in the request body rather than the standard `model` field.

## Reasoning Support

**Chat Completions (`argo--openai_chat`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | Full IR ladder (`minimal` → `max`) |

**Anthropic (`argo--anthropic`):**

| Field | Value |
|:---|:---|
| Effort field | `output_config.effort` |
| Effort range | `low` → `max` |
| Budget ratio | 0.8 |
| Thinking default | `auto` |
| Unsigned blocks | `preserve` |
| Visibility modes | `auto`/`concise`/`detailed` → `summarized`, `none` → `omitted` |

**Thinking modes:**

| IR Mode | Argo Anthropic Value |
|:---|:---|
| `auto` | `adaptive` |
| `enabled` | `enabled` |
| `disabled` | `disabled` |

### Model Overrides

| Model | Thinking Modes | Budget Ratio |
|:---|:---|:---:|
| `claudehaiku45` | `enabled`, `disabled` only | 0.8 |
| `claudesonnet4` | `enabled`, `disabled` only | 0.8 |
| `claudeopus47` | `adaptive` only (enabled → adaptive) | — |
| `claudeopus48` | `adaptive` only (enabled → adaptive) | — |

## Capability Flags

| Flag | `argo--openai_chat` | `argo--anthropic` |
|:---|:---:|:---:|
| Custom tools | ✅ | — |
| Max tool description | 1024 chars | — |

## Transforms

**`argo--openai_chat`:**

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `rename max_tokens → max_completion_tokens` | Adapts to Argo's field naming |
| Post-IR | `downgrade developer → system` | Converts developer role to system role |
| Post-IR | `default null content → ""` | Prevents null content errors |
| Post-IR | `strip temperature` (claudeopus47\*) | Removes temperature for Opus 4.7 models |
| Post-IR | `flatten system content arrays` (gemini\*) | Flattens system arrays for Gemini models |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |
| IR | `truncate_images(50)` (gpt/o\*) | Limits images to 50 per request for GPT/o-series |
| IR | `unwind_parallel_tool_calls()` (gemini\*) | Serializes parallel tool calls for Gemini models |

**`argo--anthropic`:**

| Type | Transform | Purpose |
|:---|:---|:---|
| Pre-IR | `normalize_openai_response()` | Converts OpenAI-shaped responses to Anthropic format (handles Argo's inconsistent response shapes) |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |
| IR | `auto_cache_breakpoints()` | Inserts cache breakpoints for prompt caching |

## Gateway Configuration

=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "argo": {
          "shim": "argo--openai_chat",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

=== "Anthropic"

    ```jsonc
    {
      "providers": {
        "argo-anthropic": {
          "shim": "argo--anthropic",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

## Notes

- Argo acts as a proxy to multiple upstream providers (OpenAI, Anthropic, Google). The shim handles provider-specific quirks transparently, including response format normalization and model-specific field adjustments.
- The `unsigned_blocks: preserve` setting on the Anthropic endpoint preserves unsigned reasoning blocks in the response, which is needed because Argo's proxy layer does not sign thinking blocks.
- Both shims export a `model_list_transform` that normalizes the model listing response format.

# Argo

Argo is Argonne National Laboratory's internal API gateway, providing access to various LLM models through a unified endpoint. It is supported through three shim endpoints.

!!! warning "Not built in"
    The three Argo shims ship in the standalone
    [llm-rosetta-shim-argo](https://github.com/Oaklight/llm-rosetta-shim-argo)
    package. Declare it before using the names below:

    ```jsonc
    { "shims": { "presets": ["argo"] } }
    ```

    The package registers the same canonical names, so provider configs do not
    change. **Settings → Shim Sources** offers the source as a preset. See
    [Standalone shim packages](../shims.md#standalone-shim-packages).

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `argo--chat_completions` | `chat_completions` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` |
| `argo--open_responses` | `open_responses` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` |
| `argo--messages` | `messages` | `https://apps.inside.anl.gov/argoapi` | `ARGO_API_KEY` |

!!! info "Model ID Field"
    All three Argo shims use `model_id_field: internal_id`, meaning the model identifier is sent as `internal_id` in the request body rather than the standard `model` field.

## Reasoning Support

**Chat Completions (`argo--chat_completions`):**

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | Full IR ladder (`minimal` → `max`) |

**Anthropic (`argo--messages`):**

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

| Flag | `argo--chat_completions` | `argo--open_responses` | `argo--messages` |
|:---|:---:|:---:|:---:|
| Custom tools | ✅ | ✅ | — |
| Max tool description | 1024 chars | 1024 chars | — |

## Transforms

**`argo--chat_completions`:**

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

**`argo--messages`:**

| Type | Transform | Purpose |
|:---|:---|:---|
| Pre-IR | `normalize_openai_response()` | Converts OpenAI-shaped responses to Anthropic format (handles Argo's inconsistent response shapes) |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |
| IR | `auto_cache_breakpoints()` | Inserts cache breakpoints for prompt caching |

**`argo--open_responses`:**

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip temperature` (gpto\*, gpt5\*) | Removes temperature for the GPT/o-series models that reject it |

## Gateway Configuration

The `presets` declaration above is required — without it these provider entries
fail to load because the shims are not built in.


=== "Chat Completions"

    ```jsonc
    {
      "providers": {
        "argo": {
          "shim": "argo--chat_completions",
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
          "shim": "argo--messages",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

=== "Responses"

    ```jsonc
    {
      "providers": {
        "argo-responses": {
          "shim": "argo--open_responses",
          "api_key": "${ARGO_API_KEY}"
        }
      }
    }
    ```

## Notes

- Argo acts as a proxy to multiple upstream providers (OpenAI, Anthropic, Google). The shim handles provider-specific quirks transparently, including response format normalization and model-specific field adjustments.
- The `unsigned_blocks: preserve` setting on the Anthropic endpoint preserves unsigned reasoning blocks in the response, which is needed because Argo's proxy layer does not sign thinking blocks.
- All three shims export a `model_list_transform` that normalizes the model listing response format.

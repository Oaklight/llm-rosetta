---
title: Migrating to 0.14
---

# Migrating to 0.14

Release 0.14.0 removes the deprecated import paths and class aliases that 0.13.0 kept for one release (epic [#919](https://github.com/Oaklight/llm-rosetta/issues/919)). A removed name fails with a plain `ImportError`, `ModuleNotFoundError` or `AttributeError`; the [retired-names](#retired-names) diagnostic covers provider, shim and model-type names, not Python import paths. This page lists what is gone, what replaces it, and what still works.

Every row below was checked against `origin/master` (commit `1ef424793a`): each removed name fails to import, and each replacement imports.

## Removed converter packages {#removed-converter-packages}

| Removed import path | Use instead |
| --- | --- |
| `llm_rosetta.converters.openai_chat` | `llm_rosetta.converters.llm.chat_completions` |
| `llm_rosetta.converters.openai_responses` | `llm_rosetta.converters.llm.open_responses` |
| `llm_rosetta.converters.anthropic` | `llm_rosetta.converters.llm.messages` |
| `llm_rosetta.converters.google_genai` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_generate` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_interactions` | `llm_rosetta.converters.llm.google_interactions` |
| `llm_rosetta.converters.embedding` | `llm_rosetta.converters.embeddings` |

The `google_generate` and `google_interactions` rows are the old top-level locations; the same names live under `converters.llm` now. Submodule paths follow their package: `converters.openai_chat.converter` becomes `converters.llm.chat_completions.converter`.

## Removed class names {#removed-class-names}

| Removed name | Use instead |
| --- | --- |
| `OpenAIChatConverter` | `ChatCompletionsConverter` |
| `AnthropicConverter` | `MessagesConverter` |
| `GoogleGenAIConverter` | `GoogleGenerateConverter` |
| `GoogleConverter` | `GoogleGenerateConverter` |

The four names are gone from `llm_rosetta`, `llm_rosetta.converters`, `llm_rosetta.converters.llm`, and from the canonical packages (`converters.llm.chat_completions`, `converters.llm.messages`, `converters.llm.google_generate`). Replacements import from the same places. `ChatCompletionsConverter` and `MessagesConverter` are also exported from `llm_rosetta.converters.llm.<format>`.

## Removed ops and stream-context aliases {#removed-ops-aliases}

| Removed name | Use instead |
| --- | --- |
| `OpenAIChatConfigOps`, `OpenAIChatContentOps`, `OpenAIChatMessageOps`, `OpenAIChatToolOps` | `ChatCompletions*Ops` in `converters.llm.chat_completions` |
| `AnthropicConfigOps`, `AnthropicContentOps`, `AnthropicMessageOps`, `AnthropicToolOps` | `Messages*Ops` in `converters.llm.messages` |
| `OpenAIResponsesConfigOps` | `OpenResponsesConfigOps` |
| `OpenAIResponsesContentOps` | `OpenResponsesContentOps` |
| `OpenAIResponsesMessageOps` | `OpenResponsesMessageOps` |
| `OpenAIResponsesToolOps` | `OpenResponsesToolOps` |
| `OpenAIResponsesStreamContext` | `OpenResponsesStreamContext` |

The `OpenResponses*` names live in `llm_rosetta.converters.llm.open_responses`. `OpenAIResponsesConverter` is a real class (the OpenAI profile of the Open Responses converter) and is **not** removed.

## Find the old names in your code {#find-old-names}

```bash
grep -rnE "converters\.(openai_chat|openai_responses|anthropic|google_genai|google_generate|google_interactions|embedding)\b|\b(OpenAIChat|Anthropic)(Converter|(Config|Content|Message|Tool)Ops)\b|\bGoogle(GenAI)?Converter\b|\bOpenAIResponses(Config|Content|Message|Tool)Ops\b|\bOpenAIResponsesStreamContext\b" .
```

## What still works {#still-works}

- **`llm_rosetta.gateway`** is still importable on `origin/master`. It warns and forwards to `llm_rosetta_gateway`. Import `llm_rosetta_gateway` directly now; the alias will be removed in a later step of #919.
- **Retired provider, shim and model-type names** (next section) still resolve, with a warning, while `retired_names.ALIASES_ENABLED` is `True`, which is the default.

## Retired names {#retired-names}

`llm_rosetta.retired_names.RETIRED_NAMES` lists 31 names that earlier releases renamed. Each row says what the name became. With `llm_rosetta.retired_names.ALIASES_ENABLED = False` the alias translation is switched off and a retired name raises `ValueError`, for example:

```text
'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions')
```

For a shim that moved to a standalone package, the message also names the `shims.presets` key to declare.

### Provider types and formats {#provider-types}

The `type:` value of a provider entry, and the `format` argument of `convert()`, `ConversionPipeline` and `get_converter_for_provider`.

| Old name | Replacement |
| --- | --- |
| `google` | `google_generate` |
| `google-genai` | `google_generate` |
| `openai_chat` | `chat_completions` |
| `openai_responses` | `openai--open_responses` |
| `anthropic` | `messages` |
| `openai-chat` | `chat_completions` |
| `openai-responses` | `openai--open_responses` |
| `open-responses` | `open_responses` |
| `google-interactions` | `google_interactions` |

### Shim names {#shim-names}

The `shim:` value of a provider entry. The `shims.presets` column is the preset to declare when the shim now ships as a standalone package.

| Old name | Replacement | `shims.presets` |
| --- | --- | --- |
| `openai` | `openai--chat_completions` |  |
| `google` | `google--google_generate` |  |
| `google_generate` | `google--google_generate` |  |
| `google_interactions` | `google--google_interactions` |  |
| `anthropic` | `anthropic--messages` |  |
| `argo--openai_chat` | `argo--chat_completions` | `argo` |
| `argo--openai_responses` | `argo--open_responses` | `argo` |
| `minimax--openai_chat` | `minimax--chat_completions` |  |
| `openrouter--openai_chat` | `openrouter--chat_completions` |  |
| `volcengine--openai_chat` | `volcengine--chat_completions` |  |
| `volcengine--openai_responses` | `volcengine--open_responses` |  |
| `argo--anthropic` | `argo--messages` | `argo` |
| `minimax--anthropic` | `minimax--messages` |  |
| `openrouter--anthropic` | `openrouter--messages` |  |
| `alcf--metis` | `alcf_metis--chat_completions` | `alcf` |
| `alcf--minerva` | `alcf_minerva--chat_completions` | `alcf` |
| `alcf--sophia` | `alcf_sophia--chat_completions` | `alcf` |
| `asksage--anthropic` | `asksage--messages` | `asksage` |
| `asksage--openai_chat` | `asksage--chat_completions` | `asksage` |
| `asksage--openai_responses` | `asksage--open_responses` | `asksage` |

### Model types {#model-types}

The model-type spelling follows its endpoint.

| Old name | Replacement |
| --- | --- |
| `embedding` | `embeddings` |
| `decision` | `decisions` |

---
title: 迁移到 0.14
---

# 迁移到 0.14

0.14.0 移除了 0.13.0 为过渡保留一个版本的已弃用导入路径与类别名（epic [#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。被移除的名字会以普通的 `ImportError`、`ModuleNotFoundError` 或 `AttributeError` 失败；[retired-names](#retired-names) 诊断只覆盖提供方、shim 与模型类型名称，不覆盖 Python 导入路径。本页列出已移除的内容、替代写法，以及仍然可用的部分。

下表每一行都已针对 `origin/master`（commit `1ef424793a`）核对：每个被移除的名字导入失败，每个替代写法都能导入。

## 已移除的转换器包 {#removed-converter-packages}

| 已移除的导入路径 | 改用 |
| --- | --- |
| `llm_rosetta.converters.openai_chat` | `llm_rosetta.converters.llm.chat_completions` |
| `llm_rosetta.converters.openai_responses` | `llm_rosetta.converters.llm.open_responses` |
| `llm_rosetta.converters.anthropic` | `llm_rosetta.converters.llm.messages` |
| `llm_rosetta.converters.google_genai` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_generate` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_interactions` | `llm_rosetta.converters.llm.google_interactions` |
| `llm_rosetta.converters.embedding` | `llm_rosetta.converters.embeddings` |

`google_generate` 与 `google_interactions` 两行是旧的顶层位置，同名模块现在位于 `converters.llm` 之下。子模块路径跟随其包：`converters.openai_chat.converter` 变为 `converters.llm.chat_completions.converter`。

## 已移除的类名 {#removed-class-names}

| 已移除的名字 | 改用 |
| --- | --- |
| `OpenAIChatConverter` | `ChatCompletionsConverter` |
| `AnthropicConverter` | `MessagesConverter` |
| `GoogleGenAIConverter` | `GoogleGenerateConverter` |
| `GoogleConverter` | `GoogleGenerateConverter` |

这四个名字已从 `llm_rosetta`、`llm_rosetta.converters`、`llm_rosetta.converters.llm` 以及规范包（`converters.llm.chat_completions`、`converters.llm.messages`、`converters.llm.google_generate`）中移除。替代名从相同位置导入。`ChatCompletionsConverter` 与 `MessagesConverter` 也从 `llm_rosetta.converters.llm.<format>` 导出。

## 已移除的 ops 与 stream-context 别名 {#removed-ops-aliases}

| 已移除的名字 | 改用 |
| --- | --- |
| `OpenAIChatConfigOps`、`OpenAIChatContentOps`、`OpenAIChatMessageOps`、`OpenAIChatToolOps` | `converters.llm.chat_completions` 中的 `ChatCompletions*Ops` |
| `AnthropicConfigOps`、`AnthropicContentOps`、`AnthropicMessageOps`、`AnthropicToolOps` | `converters.llm.messages` 中的 `Messages*Ops` |
| `OpenAIResponsesConfigOps` | `OpenResponsesConfigOps` |
| `OpenAIResponsesContentOps` | `OpenResponsesContentOps` |
| `OpenAIResponsesMessageOps` | `OpenResponsesMessageOps` |
| `OpenAIResponsesToolOps` | `OpenResponsesToolOps` |
| `OpenAIResponsesStreamContext` | `OpenResponsesStreamContext` |

`OpenResponses*` 位于 `llm_rosetta.converters.llm.open_responses`。`OpenAIResponsesConverter` 是真实的类（Open Responses 转换器的 OpenAI profile），**没有**被移除。

## 在代码中查找旧名字 {#find-old-names}

```bash
grep -rnE "converters\.(openai_chat|openai_responses|anthropic|google_genai|google_generate|google_interactions|embedding)\b|\b(OpenAIChat|Anthropic)(Converter|(Config|Content|Message|Tool)Ops)\b|\bGoogle(GenAI)?Converter\b|\bOpenAIResponses(Config|Content|Message|Tool)Ops\b|\bOpenAIResponsesStreamContext\b" .
```

## 仍然可用的部分 {#still-works}

- **`llm_rosetta.gateway`** 在 `origin/master` 上仍可导入。它会告警并转发到 `llm_rosetta_gateway`。请现在就直接导入 `llm_rosetta_gateway`；该别名会在 #919 的后续步骤中移除。
- **已退役的提供方、shim 与模型类型名称**（见下节）在 `retired_names.ALIASES_ENABLED` 为 `True`（默认值）时仍会解析，并附带告警。

## 已退役的名字 {#retired-names}

`llm_rosetta.retired_names.RETIRED_NAMES` 列出 31 个早先版本改过名的名字，每一行说明该名字变成了什么。把 `llm_rosetta.retired_names.ALIASES_ENABLED = False` 后，别名翻译被关闭，退役名字会抛 `ValueError`，例如：

```text
'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions')
```

对于已迁入独立包的 shim，消息还会写出需要声明的 `shims.presets` 键。

### 提供方类型与格式 {#provider-types}

提供方条目的 `type:` 值，以及 `convert()`、`ConversionPipeline`、`get_converter_for_provider` 的 `format` 参数。

| 旧名字 | 替代 |
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

### Shim 名称 {#shim-names}

提供方条目的 `shim:` 值。`shims.presets` 一列是该 shim 已作为独立包发布时需要声明的预设。

| 旧名字 | 替代 | `shims.presets` |
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

### 模型类型 {#model-types}

模型类型的写法跟随其端点。

| 旧名字 | 替代 |
| --- | --- |
| `embedding` | `embeddings` |
| `decision` | `decisions` |

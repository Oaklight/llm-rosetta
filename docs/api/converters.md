---
title: Converters
---

# Converters API

!!! info "v0.13 重命名"
    `GoogleGenAIConverter` → **`GoogleGenerateConverter`**。
    `to_google_genai()` / `from_google_genai()` → `to_google_generate()` / `from_google_generate()`。
    旧名称已在 0.14.0 移除；参见 [迁移到 0.14](../guide/migrating-to-0.14.md)。

## Convenience Functions

以下函数都可从 `llm_rosetta` 导入（`llm_rosetta.convert()`、`llm_rosetta.get_converter_for_provider()`）；已弃用的 `llm_rosetta.auto_detect` 只做转发，将在 0.15.0 移除。

::: llm_rosetta.pipeline.convert

::: llm_rosetta.pipeline.convert_response

::: llm_rosetta.converters.registry.get_converter_for_provider

## Conversion Context

::: llm_rosetta.converters.base.context.ConversionContext

::: llm_rosetta.converters.base.context.StreamContext

## 转换警告

::: llm_rosetta.conversion_warnings.ConversionWarning

::: llm_rosetta.conversion_warnings.WarningSink

::: llm_rosetta.conversion_warnings.report_drop

## Conversion Pipeline

::: llm_rosetta.pipeline.ConversionPipeline

## Chat / Completions Converters

::: llm_rosetta.converters.base.converter.BaseConverter

::: llm_rosetta.converters.llm.chat_completions.converter.ChatCompletionsConverter

::: llm_rosetta.converters.llm.open_responses.converter.OpenResponsesConverter

::: llm_rosetta.converters.llm.open_responses.converter.OpenAIResponsesConverter

::: llm_rosetta.converters.llm.messages.converter.MessagesConverter

::: llm_rosetta.converters.llm.google_generate.converter.GoogleGenerateConverter

::: llm_rosetta.converters.llm.google_interactions.converter.GoogleInteractionsConverter

## Embedding Converters

::: llm_rosetta.converters.base.embedding_converter.BaseEmbeddingConverter

::: llm_rosetta.converters.embeddings.openai.OpenAIEmbeddingConverter

::: llm_rosetta.converters.embeddings.jina.JinaEmbeddingConverter

::: llm_rosetta.converters.embeddings.voyage.VoyageEmbeddingConverter

::: llm_rosetta.converters.embeddings.cohere.CohereEmbeddingConverter

::: llm_rosetta.converters.embeddings.google.GoogleEmbeddingConverter

::: llm_rosetta.converters.embeddings.minimax.MiniMaxEmbeddingConverter

## Rerank Converters

::: llm_rosetta.converters.base.rerank_converter.BaseRerankConverter

::: llm_rosetta.converters.rerank.jina.JinaRerankConverter

::: llm_rosetta.converters.rerank.cohere.CohereRerankConverter

::: llm_rosetta.converters.rerank.voyage.VoyageRerankConverter

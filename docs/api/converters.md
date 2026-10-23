---
title: Converters
---

# Converters API

!!! info "v0.13 重命名"
    `GoogleGenAIConverter` → **`GoogleGenerateConverter`**。
    `to_google_genai()` / `from_google_genai()` → `to_google_generate()` / `from_google_generate()`。
    旧名称仍可作为弃用别名使用。

## Convenience Functions

::: llm_rosetta.auto_detect.convert

::: llm_rosetta.auto_detect.convert_response

::: llm_rosetta.auto_detect.get_converter_for_provider

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

::: llm_rosetta.converters.embedding.openai.OpenAIEmbeddingConverter

::: llm_rosetta.converters.embedding.jina.JinaEmbeddingConverter

::: llm_rosetta.converters.embedding.voyage.VoyageEmbeddingConverter

::: llm_rosetta.converters.embedding.cohere.CohereEmbeddingConverter

::: llm_rosetta.converters.embedding.google.GoogleEmbeddingConverter

## Rerank Converters

::: llm_rosetta.converters.base.rerank_converter.BaseRerankConverter

::: llm_rosetta.converters.rerank.jina.JinaRerankConverter

::: llm_rosetta.converters.rerank.cohere.CohereRerankConverter

::: llm_rosetta.converters.rerank.voyage.VoyageRerankConverter

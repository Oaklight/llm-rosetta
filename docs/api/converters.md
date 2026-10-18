---
title: Converters
---

# Converters API

!!! info "Renamed in v0.13"
    `GoogleGenAIConverter` → **`GoogleGenerateConverter`**.
    `to_google_genai()` / `from_google_genai()` → `to_google_generate()` / `from_google_generate()`.
    Old names remain as deprecated aliases.

## Convenience Functions

::: llm_rosetta.auto_detect.convert

::: llm_rosetta.auto_detect.convert_response

::: llm_rosetta.auto_detect.get_converter_for_provider

## Conversion Context

::: llm_rosetta.converters.base.context.ConversionContext

::: llm_rosetta.converters.base.context.StreamContext

## Conversion Pipeline

::: llm_rosetta.pipeline.ConversionPipeline

## Chat / Completions Converters

::: llm_rosetta.converters.base.converter.BaseConverter

::: llm_rosetta.converters.chat_completions.converter.ChatCompletionsConverter

::: llm_rosetta.converters.open_responses.converter.OpenResponsesConverter

::: llm_rosetta.converters.open_responses.converter.OpenAIResponsesConverter

::: llm_rosetta.converters.messages.converter.MessagesConverter

::: llm_rosetta.converters.google_generate.converter.GoogleGenerateConverter

::: llm_rosetta.converters.google_interactions.converter.GoogleInteractionsConverter

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

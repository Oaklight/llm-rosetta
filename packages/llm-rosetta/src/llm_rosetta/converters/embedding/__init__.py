"""
LLM-Rosetta - Embedding Converters

Embedding 格式转换器
Embedding format converters

Provides converters for:
- OpenAI: the baseline format (IR follows OpenAI convention)
- Jina: OpenAI-compatible with task mapping
- Voyage: OpenAI-compatible with input_type mapping
- Cohere: different structure (texts, embeddings.{type}, meta)
- Google GenAI: content.parts + taskType, model in the URL
- MiniMax: texts + type (db|query), vectors + base_resp envelope
"""

from .cohere import CohereEmbeddingConverter
from .google import GoogleEmbeddingConverter
from .jina import JinaEmbeddingConverter
from .minimax import MiniMaxEmbeddingConverter
from .openai import OpenAIEmbeddingConverter
from .pipeline import (
    EMBEDDING_FORMATS,
    EmbeddingConversionPipeline,
    get_embedding_converter,
)
from .voyage import VoyageEmbeddingConverter

__all__ = [
    "OpenAIEmbeddingConverter",
    "JinaEmbeddingConverter",
    "VoyageEmbeddingConverter",
    "CohereEmbeddingConverter",
    "GoogleEmbeddingConverter",
    "MiniMaxEmbeddingConverter",
    "EmbeddingConversionPipeline",
    "EMBEDDING_FORMATS",
    "get_embedding_converter",
]

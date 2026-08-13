"""
LLM-Rosetta - Embedding Converters

Embedding 格式转换器
Embedding format converters

Provides converters for:
- OpenAI: the baseline format (IR follows OpenAI convention)
- Jina: OpenAI-compatible with task mapping
- Voyage: OpenAI-compatible with input_type mapping
- Cohere: different structure (texts, embeddings.{type}, meta)
"""

from .cohere import CohereEmbeddingConverter
from .jina import JinaEmbeddingConverter
from .openai import OpenAIEmbeddingConverter
from .voyage import VoyageEmbeddingConverter

__all__ = [
    "OpenAIEmbeddingConverter",
    "JinaEmbeddingConverter",
    "VoyageEmbeddingConverter",
    "CohereEmbeddingConverter",
]

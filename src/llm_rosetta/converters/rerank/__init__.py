"""
LLM-Rosetta - Rerank Converters

Rerank 格式转换器
Rerank format converters

Provides converters for:
- Jina: results + top-level usage (also used by GPUStack, vLLM, Xinference)
- Cohere: results + meta.billed_units (also used by OpenVINO, Siliconflow variant)
- Voyage: data + top-level usage (OpenAI Embeddings-style)
"""

from .cohere import CohereRerankConverter
from .jina import JinaRerankConverter
from .voyage import VoyageRerankConverter

__all__ = [
    "JinaRerankConverter",
    "CohereRerankConverter",
    "VoyageRerankConverter",
]

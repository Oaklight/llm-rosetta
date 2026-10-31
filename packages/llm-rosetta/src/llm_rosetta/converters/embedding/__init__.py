"""
Deprecated — use ``llm_rosetta.converters.embeddings`` instead.

This package exists only for backward compatibility and will be removed in
0.16.0 (#919).  It re-exports the canonical names and aliases the legacy
submodule paths so that existing imports such as
``from llm_rosetta.converters.embedding.openai import OpenAIEmbeddingConverter``
keep resolving.  The package was renamed to the plural spelling to match the
model type every other surface uses (``embeddings``).
"""

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "llm_rosetta.converters.embedding is deprecated, "
    "use llm_rosetta.converters.embeddings instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..embeddings import (  # noqa: F401, E402
    EMBEDDING_FORMATS,
    CohereEmbeddingConverter,
    EmbeddingConversionPipeline,
    GoogleEmbeddingConverter,
    JinaEmbeddingConverter,
    MiniMaxEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
    get_embedding_converter,
)
from ..embeddings import (  # noqa: F401, E402
    cohere as cohere,
    google as google,
    jina as jina,
    minimax as minimax,
    openai as openai,
    pipeline as pipeline,
    voyage as voyage,
)

# Register the legacy submodule paths (e.g. ``embedding.pipeline``) so that
# ``from llm_rosetta.converters.embedding.pipeline import ...`` keeps working.
# The submodule objects are shared with ``embeddings`` (same identity).
_sys.modules[__name__ + ".cohere"] = cohere
_sys.modules[__name__ + ".google"] = google
_sys.modules[__name__ + ".jina"] = jina
_sys.modules[__name__ + ".minimax"] = minimax
_sys.modules[__name__ + ".openai"] = openai
_sys.modules[__name__ + ".pipeline"] = pipeline
_sys.modules[__name__ + ".voyage"] = voyage

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

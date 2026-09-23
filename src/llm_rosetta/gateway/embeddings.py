"""Backward-compat shim — real module is at gateway.pipelines.embeddings."""

from .pipelines.embeddings import *  # noqa: F401,F403
from .pipelines.embeddings import (  # noqa: F401 — underscore names excluded by star import
    _detect_embedding_source,
    _resolve_embedding_provider,
)

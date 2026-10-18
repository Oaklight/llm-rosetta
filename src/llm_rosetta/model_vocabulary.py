"""The model-type vocabulary: categories and their wire formats.

This is deliberately a leaf module: it holds data only, so the shim loader can
validate a ``(model_type, format)`` pair while ``llm_rosetta.shims`` is still
initialising, without importing the converter package (which imports
``shims.provider_shim`` and would close the cycle).

:mod:`llm_rosetta.converters.registry` binds converter classes to these pairs
and checks at import time that its maps match, so the two cannot drift.
"""

from __future__ import annotations

__all__ = ["FORMATS_BY_TYPE", "MODEL_TYPES"]

#: The capability categories, in display order.
MODEL_TYPES: tuple[str, ...] = ("llm", "embedding", "rerank", "decision")

#: Formats within each category, in display order.  Format names are scoped by
#: the category, so ``google_generate`` may name both a chat and an embedding
#: format; lookup is always by the pair.
FORMATS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "llm": (
        "chat_completions",
        "messages",
        "open_responses",
        "google_generate",
        "google_interactions",
    ),
    "embedding": ("openai", "cohere", "jina", "voyage"),
    "rerank": ("jina", "cohere", "voyage"),
    "decision": ("systemone", "openai_decisions"),
}

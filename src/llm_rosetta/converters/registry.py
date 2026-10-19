"""The one converter registry, keyed by ``(model_type, format)``.

A converter is identified by the capability category it serves (``llm``,
``embedding``, ``rerank``, ``decision``) and the wire dialect within that
category (``chat_completions``, ``openai``, ``jina``, ``systemone``, …).  The
pair matters because format names are namespaced by category: for example
``google_generate`` names both a chat converter and an embedding converter.

The vocabulary itself lives in :mod:`llm_rosetta.model_vocabulary`, a leaf
module the shim loader can import before the converter package is ready.  This
module binds classes to those pairs and checks at import time that the two
cannot drift.

The per-family maps stay in their own pipeline modules — those modules are the
leaf definitions, and importing them from here (rather than the reverse) keeps
the import graph acyclic: a family pipeline importing this module would run
into the parent package's ``__init__``, which imports that same pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from llm_rosetta.converters.llm.chat_completions.converter import (
    ChatCompletionsConverter,
)
from llm_rosetta.converters.decision.pipeline import _DECISION_CONVERTERS
from llm_rosetta.converters.embedding.pipeline import _EMBEDDING_CONVERTERS
from llm_rosetta.converters.llm.google_generate.converter import (
    GoogleGenerateConverter,
)
from llm_rosetta.converters.llm.google_interactions.converter import (
    GoogleInteractionsConverter,
)
from llm_rosetta.converters.llm.messages.converter import MessagesConverter
from llm_rosetta.converters.llm.open_responses.converter import OpenResponsesConverter
from llm_rosetta.converters.rerank.pipeline import _RERANK_CONVERTERS
from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE as _DECLARED_FORMATS
from llm_rosetta.model_vocabulary import MODEL_TYPES

__all__ = [
    "CONVERTER_REGISTRY",
    "FORMATS_BY_TYPE",
    "MODEL_TYPES",
    "formats_for",
    "get_converter_class",
]

#: ``model_type → format → converter class``.  Insertion order is the display
#: order the admin payload and the model-type descriptors use.
CONVERTER_REGISTRY: dict[str, dict[str, type]] = {
    "llm": {
        "chat_completions": ChatCompletionsConverter,
        "messages": MessagesConverter,
        "open_responses": OpenResponsesConverter,
        "google_generate": GoogleGenerateConverter,
        "google_interactions": GoogleInteractionsConverter,
    },
    "embedding": cast(dict[str, type], _EMBEDDING_CONVERTERS),
    "rerank": cast(dict[str, type], _RERANK_CONVERTERS),
    "decision": cast(dict[str, type], _DECISION_CONVERTERS),
}

#: Membership sets, derived from :data:`CONVERTER_REGISTRY`.  Use
#: :func:`formats_for` when the display order matters.
FORMATS_BY_TYPE: dict[str, frozenset[str]] = {
    model_type: frozenset(converters)
    for model_type, converters in CONVERTER_REGISTRY.items()
}

# The vocabulary module is data-only so the shim loader can use it early; the
# binding above is what makes it real.  A converter added without its
# vocabulary entry (or vice versa) must not silently disappear from one side.
for _model_type in MODEL_TYPES:
    _registered = FORMATS_BY_TYPE.get(_model_type, frozenset())
    _declared = frozenset(_DECLARED_FORMATS[_model_type])
    if _registered != _declared:
        raise RuntimeError(
            f"Converter registry out of sync for model_type={_model_type!r}: "
            f"registered {sorted(_registered)}, declared {sorted(_declared)}"
        )
del _model_type, _registered, _declared


def formats_for(model_type: str) -> tuple[str, ...]:
    """Return the formats of *model_type* in registry (display) order.

    Raises:
        ValueError: If *model_type* is not a registered category.
    """
    try:
        return tuple(CONVERTER_REGISTRY[model_type])
    except KeyError:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available: {', '.join(MODEL_TYPES)}"
        ) from None


def get_converter_class(model_type: str, format_name: str) -> type:
    """Return the converter class for ``(model_type, format_name)``.

    Raises:
        ValueError: If the pair is not registered; the message names both the
            category and its available formats.
    """
    try:
        formats = CONVERTER_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available: {', '.join(MODEL_TYPES)}"
        ) from None
    cls = formats.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown {model_type} format: '{format_name}'. "
            f"Available: {', '.join(formats)}"
        )
    return cls


def converter_map(model_type: str) -> Mapping[str, type]:
    """Return the ``format → class`` mapping for *model_type*."""
    try:
        return CONVERTER_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Available: {', '.join(MODEL_TYPES)}"
        ) from None

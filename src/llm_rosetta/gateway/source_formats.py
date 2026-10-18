"""Source-format detection for the non-chat model types.

Each handler serves one route for every dialect of its paradigm (unlike the
LLM routes, which have one route per format), so it has to infer the incoming
dialect from the request before it can build a conversion pipeline.  The
model-type descriptor carries the right detector for its handler; this module
holds the detectors themselves so both the descriptor and the tests can reach
them without importing a handler.

Decision detection is deliberately not here: decision is its own clean-up
track (#926 non-goal) and keeps its handler-local function.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.provider_names import canonical_provider_name


def detect_embedding_source(request: Any, body: dict[str, Any], config: Any) -> str:
    """Infer the source embedding format from the request body.

    Detection order:
    1. ``texts`` field (without ``input``) → Cohere
    2. ``task`` field → Jina
    3. ``input_type`` or ``output_dtype`` beside ``input`` → Voyage
       (Cohere sends ``input_type`` too, but with ``texts``, which rule 1
       already claims; ``output_dtype`` is optional, so ``input_type`` is the
       signal an ordinary Voyage request carries)
    4. Google-shaped body (``requests`` or ``content.parts`` without ``input``
       or ``texts``) → Google GenAI
    5. Fall back to ``config.default_embedding_format``

    *request* is unused for embeddings but keeps the signature uniform with
    :func:`detect_rerank_source`.
    """
    if "texts" in body and "input" not in body:
        return "cohere"
    if "task" in body:
        return "jina"
    if "input_type" in body or "output_dtype" in body:
        return "voyage"
    if "input" not in body and ("requests" in body or "content" in body):
        return "google_generate"
    return canonical_provider_name(config.default_embedding_format)


def detect_rerank_source(request: Any, body: dict[str, Any], config: Any) -> str:
    """Infer the source rerank format from the request path and body.

    Detection order:
    1. ``/v2/rerank`` path → Cohere (only Cohere uses v2)
    2. ``top_k`` in body (without ``top_n``) → Voyage
    3. Fall back to ``config.default_rerank_format``
    """
    path: str = getattr(request, "path", "")
    if path.startswith("/v2/"):
        return "cohere"
    if "top_k" in body and "top_n" not in body:
        return "voyage"
    return canonical_provider_name(config.default_rerank_format)

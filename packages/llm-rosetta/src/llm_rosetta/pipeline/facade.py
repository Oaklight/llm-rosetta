"""The library's one-call facade: ``convert`` and ``convert_response``.

Both build a :class:`~llm_rosetta.pipeline.ConversionPipeline` from provider
names, resolving each name's shim, so a caller converts a body without
constructing a pipeline.  They were in ``llm_rosetta.auto_detect`` until
0.14.0; that module forwards them until 0.15.0.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.contracts.resolver import resolve_shim
from llm_rosetta.contracts.routing import ProviderType
from llm_rosetta.converters.registry import detect_provider

from .chat import ConversionPipeline

__all__ = ["convert", "convert_response"]


def _detect_source(
    source_body: dict[str, Any],
    source_provider: ProviderType | str | None,
) -> str:
    """Detect or validate the source provider, raising on failure."""
    if source_provider is not None:
        return str(source_provider)
    detected = detect_provider(source_body)
    if detected is None:
        raise ValueError(
            "Unable to detect source provider. "
            "Please specify source_provider explicitly."
        )
    return detected


def convert(
    source_body: dict[str, Any],
    target_provider: ProviderType | str,
    source_provider: ProviderType | str | None = None,
    *,
    model: str | None = None,
    baseline: bool = True,
) -> dict[str, Any]:
    """Auto-detect source provider and convert to target provider format.

    Delegates to :class:`~llm_rosetta.pipeline.ConversionPipeline`
    internally, resolving source and target shims from the provider
    names.  For response or streaming conversion, use
    :func:`convert_response` or :class:`ConversionPipeline` directly.

    Args:
        source_body: Source provider request body.
        target_provider: Target provider type or registered shim name.
        source_provider: Optional source provider type or shim name.
            Auto-detected from *source_body* when not provided.
        model: Optional model name passed as ``upstream_model`` to the
            pipeline (used for per-model shim overrides).
        baseline: When ``True`` (default for library helpers), skip the
            IR round-trip when source and target providers are the same.
            Set to ``False`` to force a full IR conversion regardless.

    Returns:
        Target provider format request body.

    Raises:
        ValueError: If source provider cannot be detected.

    Examples:
        >>> openai_body = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> google_body = convert(openai_body, "google_generate")

        >>> anthropic_body = {"messages": [...]}
        >>> openai_body = convert(anthropic_body, "chat_completions", source_provider="messages")

        >>> # With shim transforms
        >>> body = convert(req, "messages", source_provider="deepseek", model="deepseek-r1")

        >>> # Force IR round-trip even for same-provider
        >>> body = {"messages": [...], "max_tokens": 256}
        >>> normalised = convert(body, "chat_completions", baseline=False)
    """
    # A retired name (``openai_chat``, ``volcengine--openai_chat``) is not
    # resolved here: the pipeline below raises a ValueError that names its
    # replacement.
    src = _detect_source(source_body, source_provider)
    tgt = str(target_provider)

    source_shim = (
        resolve_shim(str(source_provider))
        if source_provider is not None
        else resolve_shim(src)
    )
    pipeline = ConversionPipeline(
        src,
        tgt,
        source_shim=source_shim,
        target_shim=resolve_shim(str(target_provider)),
        upstream_model=model,
        baseline=baseline,
        # Library callers expect Google SDK format; gateway uses "rest"
        google_output_format="sdk",
    )
    return pipeline.convert_request(source_body)


def convert_response(
    response_body: dict[str, Any],
    request_body: dict[str, Any],
    source_provider: ProviderType | str,
    target_provider: ProviderType | str,
    *,
    model: str | None = None,
    baseline: bool = True,
) -> dict[str, Any]:
    """Convert a response body from target provider format back to source.

    Creates a :class:`~llm_rosetta.pipeline.ConversionPipeline`,
    replays the request conversion to establish context, then converts
    the response.

    Args:
        response_body: Target-format response body from upstream.
        request_body: The original source-format request body (needed
            to establish conversion context, e.g. custom-tool state).
        source_provider: The client/source provider type or shim name.
        target_provider: The upstream/target provider type or shim name.
        model: Optional model name for per-model shim overrides.
        baseline: When ``True`` (default), skip the IR round-trip for
            same-provider requests.  Set to ``False`` to force full
            IR conversion.

    Returns:
        Source-format response body.

    Raises:
        ValueError: If conversion fails.
    """
    src = str(source_provider)
    tgt = str(target_provider)

    pipeline = ConversionPipeline(
        src,
        tgt,
        source_shim=resolve_shim(str(source_provider)),
        target_shim=resolve_shim(str(target_provider)),
        upstream_model=model,
        baseline=baseline,
        # Library callers expect Google SDK format; gateway uses "rest"
        google_output_format="sdk",
    )
    # Replay request to populate pipeline context (tool state, metadata)
    pipeline.convert_request(request_body)
    return pipeline.convert_response(response_body)

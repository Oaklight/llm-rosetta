"""
LLM Provider Auto-Detection

自动检测 LLM provider 请求体格式的工具函数
Utility functions for auto-detecting LLM provider request body formats
"""

from typing import Any, Literal, cast

from .formats.detect import detect_format as _detect_body

# Canonical provider/format names.  A retired spelling (``openai_chat``,
# ``anthropic``, ``google``, ...) is not accepted: it fails with the
# retired-names message, which names its replacement (see
# :mod:`llm_rosetta.retired_names`).
ProviderType = Literal[
    "chat_completions",
    "open_responses",
    "messages",
    "google_generate",
    "google_interactions",
    "decision",
]


def detect_provider(body: dict[str, Any]) -> ProviderType | None:
    """Auto-detect provider type from request body structure.

    Args:
        body: Provider request body dict.

    Returns:
        Detected provider type, or ``None`` if unrecognised.

    Examples:
        >>> detect_provider({"messages": [{"role": "user", "content": "Hello"}]})
        'chat_completions'
        >>> detect_provider({"input": [{"type": "message", "role": "user"}]})
        'open_responses'
        >>> detect_provider({"messages": [{"role": "user", "content": [{"type": "text"}]}]})
        'messages'
        >>> detect_provider({"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]})
        'google_generate'
    """
    # Detection order (and each predicate) lives in the format catalog; a new
    # format joins it by adding one ``FormatSpec`` there.
    return cast("ProviderType | None", _detect_body(body))


_converter_cache: dict[str, Any] = {}


def get_converter_for_provider(provider: str):
    """Get the corresponding converter for a provider type or shim name.

    Converter instances are cached — the same object is returned for the
    same resolved base provider.  This is safe because converters are
    stateless (all per-request state lives in ``ConversionContext``).

    Args:
        provider: Provider type string or registered shim name.

    Returns:
        Corresponding converter instance (cached).

    Raises:
        ValueError: If the provider is not a known type or shim name.  A
            retired spelling gets the retired-names message, which names its
            replacement.
    """
    # The OpenAI Responses profile shim shares the base converter (its traits
    # come from the shim's ``context.options``, not a subclass).  Fold it onto
    # the base so both spellings return the same cached instance, and so the
    # profile resolves without the shim registry.
    if provider == "openai--open_responses":
        provider = "open_responses"
    if provider in _converter_cache:
        return _converter_cache[provider]

    from .converters.registry import CONVERTER_REGISTRY
    from .contracts.resolver import resolve_format

    converter_map = CONVERTER_REGISTRY["llm"]

    # Direct match against base converter types
    if provider in converter_map:
        instance = converter_map[provider]()
        _converter_cache[provider] = instance
        return instance

    # Resolve through the shim registry.
    base = resolve_format(provider)
    if base in converter_map:
        instance = converter_map[base]()
        _converter_cache[provider] = instance
        _converter_cache[base] = instance
        return instance

    from . import retired_names

    retired_names.check_not_retired(
        provider, retired_names.PROVIDER_TYPE, retired_names.SHIM
    )
    # A shim whose vendor left core (#1660) is not a typo: say where it went.
    moved = retired_names.moved_shim_message(provider)
    if moved is not None:
        raise ValueError(f"Unsupported provider: {provider}; {moved}")
    raise ValueError(f"Unsupported provider: {provider}")


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
    from .pipeline import ConversionPipeline
    from .contracts.resolver import resolve_shim

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
    from .pipeline import ConversionPipeline
    from .contracts.resolver import resolve_shim

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

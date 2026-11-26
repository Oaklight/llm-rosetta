"""
LLM-Rosetta - Google GenAI response helpers

Module-level helpers shared by the converter and its stream mixins: usage
modality lists, system-instruction hoisting, grounding metadata and finish
reasons.
"""

from collections.abc import Mapping
from typing import Any, cast


from llm_rosetta.ir.model import (
    CitationPart,
    is_message,
    is_text_part,
)
from ...base.errors import require_representable_finish_reason
from llm_rosetta.ir.context import (
    StreamContext,
)
from ...base.helpers.citations import (
    citation_end_index,
    citation_provider_fields,
    citation_start_index,
    citation_title,
    citation_type_of,
    citation_url,
    extra_fields,
    make_url_citation,
    report_dropped_citation,
)
from llm_rosetta.ir.primitives.native import native_record
from llm_rosetta.ir.primitives.system_message_hoist import hoist_system_in_converter
from llm_rosetta.ir.primitives.finish_reason import (
    FAILURE_REASONS,
)
from ._constants import (
    GOOGLE_REASON_TO_PROVIDER,
)


def _modality_list_to_dict(modality_list: list[dict]) -> dict[str, int]:
    """Convert Google's ``list[ModalityTokenCount]`` to IR ``dict[str, int]``.

    Example: ``[{"modality": "TEXT", "token_count": 42}]``
    → ``{"text_tokens": 42}``
    """
    result: dict[str, int] = {}
    for item in modality_list:
        modality = (item.get("modality") or "unknown").lower()
        count = item.get("token_count") or item.get("tokenCount") or 0
        result[f"{modality}_tokens"] = count
    return result


_GOOGLE_MODALITIES = frozenset({"TEXT", "IMAGE", "VIDEO", "AUDIO", "DOCUMENT"})


def _dict_to_modality_list(details: dict[str, int]) -> list[dict[str, Any]]:
    """Convert IR ``dict[str, int]`` back to Google's ``list[ModalityTokenCount]``.

    Only emits entries whose modality is in Google's ``Modality`` enum.
    IR keys like ``cached_tokens`` or ``reasoning_tokens`` are handled by
    dedicated fields (``cachedContentTokenCount``, ``thoughtsTokenCount``)
    and must not appear as ``ModalityTokenCount`` entries.

    Example: ``{"text_tokens": 42}``
    → ``[{"modality": "TEXT", "tokenCount": 42}]``
    """
    result: list[dict[str, Any]] = []
    for key, count in details.items():
        modality = key.removesuffix("_tokens").upper()
        if modality not in _GOOGLE_MODALITIES:
            continue
        result.append({"modality": modality, "tokenCount": count})
    return result


def _resolve_system_instruction(
    provider_request: Mapping[str, Any], config: Mapping[str, Any]
) -> Any:
    """Resolve a Google system instruction out of either output shape.

    ``config.system_instruction`` is where the SDK shape keeps it (and where
    ``GenerateContentConfig`` declares it); the top level is where the REST
    shape keeps it.  Both spellings are accepted in either place, and
    ``config`` wins when a hand-built dict carries both — that way reading the
    instruction back into IR and lifting it onto the REST wire agree on one
    precedence instead of resolving the same dict two different ways (#936
    review).

    Args:
        provider_request: The provider request dict.
        config: Its ``config`` sub-dict, already extracted and type-checked.

    Returns:
        The instruction found, or ``None`` when the request carries none.
    """
    return (
        config.get("system_instruction")
        or config.get("systemInstruction")
        or provider_request.get("system_instruction")
        or provider_request.get("systemInstruction")
    )


def _hoist_and_extract_system(
    ir_messages: list,
    ir_request: Mapping[str, Any],
    system_instruction: dict[str, Any] | None,
) -> tuple[list, dict[str, Any] | None]:
    """Hoist late system messages and extract leading ones for system_instruction.

    Defense-in-depth: idempotent with pipeline's intrinsic transform.
    """
    ir_messages, hoisted_si = hoist_system_in_converter(ir_messages, ir_request)
    if hoisted_si and system_instruction is None:
        texts = [
            p["text"]
            for p in hoisted_si
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        if texts:
            system_instruction = {
                "role": "user",
                "parts": [{"text": t} for t in texts],
            }

    # Defensive: hoist_system_in_converter already rewrites all system
    # messages (leading → extracted, late → user envelope), so this loop
    # should not match. Kept as a safety net for edge cases.
    for item in ir_messages:
        if is_message(item) and item.get("role") == "system":
            msg_parts = [
                {"text": part["text"]}
                for part in item.get("content", [])
                if is_text_part(part)
            ]
            if system_instruction is None:
                system_instruction = {"role": "user", "parts": msg_parts}
            else:
                cast(list, system_instruction["parts"]).extend(msg_parts)

    return ir_messages, system_instruction


def _grounding_metadata_from_ir(
    citations: list[CitationPart],
    message_provider_metadata: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Rebuild Gemini's ``groundingMetadata`` from IR citations.

    The inverse of :func:`_grounding_citations`: sources become
    ``groundingChunks`` (deduplicated, since supports point at them by index),
    spans become ``groundingSupports``, and the search-level metadata on the
    message rides the same object.  A citation with no URL — a document
    citation from another dialect — has no chunk to point at and is reported
    as dropped.
    """
    chunks: list[dict[str, Any]] = []
    index_by_key: dict[tuple[Any, ...], int] = {}
    supports: list[dict[str, Any]] = []
    for citation in citations:
        url = citation_url(citation)
        if not url:
            report_dropped_citation(
                citation, warnings=warnings, why="Google grounds citations in a URL"
            )
            continue
        title = citation_title(citation)
        key = (url, title)
        position = index_by_key.get(key)
        if position is None:
            position = len(chunks)
            index_by_key[key] = position
            fields = {
                k: v
                for k, v in citation_provider_fields(citation, "google").items()
                if not k.startswith("_")
            }
            wrapper = (
                "maps"
                if citation_provider_fields(citation, "google").get("_chunk_type")
                == "maps"
                else "web"
            )
            source: dict[str, Any] = {"uri": url}
            if title:
                source["title"] = title
            source.update(fields)
            chunks.append({wrapper: source})
        start = citation_start_index(citation)
        end = citation_end_index(citation)
        if start is not None or end is not None:
            supports.append(
                {
                    "segment": {
                        "startIndex": start or 0,
                        "endIndex": end or 0,
                    },
                    "groundingChunkIndices": [position],
                }
            )
    if not chunks:
        return {}
    metadata: dict[str, Any] = {"groundingChunks": chunks}
    if supports:
        metadata["groundingSupports"] = supports
    google = message_provider_metadata.get("google")
    if isinstance(google, dict):
        # Back to the wire spelling: the IR holds them snake_case under
        # ``provider_metadata``, the API spells them camelCase.
        if google.get("web_search_queries"):
            metadata["webSearchQueries"] = google["web_search_queries"]
        entry_point = google.get("search_entry_point")
        if isinstance(entry_point, dict) and entry_point.get("rendered_content"):
            metadata["searchEntryPoint"] = {
                "renderedContent": entry_point["rendered_content"]
            }
    return metadata


def _search_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Google's search-level metadata: the queries and the rendered entry point.

    Neither is a citation — they describe the search, not a source — and the
    IR has no field for them, so they ride ``provider_metadata.google`` on the
    event or message that closes the turn.  Measured on the live API: both keys
    arrive together with the sources on the final chunk, and
    ``searchEntryPoint`` is independently optional (absent when the tool is
    ``google_maps``).
    """
    queries = metadata.get("web_search_queries") or metadata.get("webSearchQueries")
    entry_point = metadata.get("search_entry_point") or metadata.get("searchEntryPoint")
    rendered = None
    if isinstance(entry_point, dict):
        rendered = entry_point.get("rendered_content") or entry_point.get(
            "renderedContent"
        )
    result: dict[str, Any] = {}
    if queries:
        result["web_search_queries"] = queries
    if rendered:
        result["search_entry_point"] = {"rendered_content": rendered}
    return result


def _grounding_sources(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the sources of the grounding chunks, with their wrapper key.

    A chunk wraps its source under ``web`` (the usual case), ``maps``, or
    ``retrievedContext``; the key is kept on the source as ``_chunk_type`` so a
    write back puts the source under the same wrapper instead of guessing.
    """
    chunks = metadata.get("grounding_chunks") or metadata.get("groundingChunks") or []
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            sources.append({})
            continue
        for wrapper in ("web", "maps", "retrieved_context", "retrievedContext"):
            source = chunk.get(wrapper)
            if isinstance(source, dict):
                source = dict(source)
                source["_chunk_type"] = "maps" if wrapper == "maps" else "web"
                sources.append(source)
                break
        else:
            sources.append({})
    return sources


def _grounding_citation(
    source: dict[str, Any],
    segment: dict[str, Any] | None,
) -> CitationPart | None:
    """Build one IR citation from a grounding source and optional segment.

    A ``maps`` chunk carries ``placeId``/``text`` that have no shared field, so
    they ride ``provider_metadata.google`` — on the citation, where they belong,
    not the search-level metadata.
    """
    url = source.get("uri")
    if not url:
        return None
    start = end = None
    if segment:
        start = segment.get("start_index", segment.get("startIndex"))
        end = segment.get("end_index", segment.get("endIndex"))
    return make_url_citation(
        url,
        title=source.get("title"),
        start_index=start if isinstance(start, int) else None,
        end_index=end if isinstance(end, int) else None,
        provider="google",
        provider_fields=extra_fields(source, ("uri", "title")),
    )


_GROUNDING_CITATIONS_KEY = "_google_grounding_citations"

#: Where the stream holds Google's search-level metadata until the finish.
_GROUNDING_SEARCH_KEY = "_google_grounding_search"

#: Citation fields that identify one citation, so the same source attached to
#: different segments is kept and a repeat of one is not.
_GROUNDING_CITATION_KEY_FIELDS = ("url", "start_index", "end_index")


def _citation_key(citation: CitationPart) -> tuple[Any, ...]:
    """Identify a citation, so an accumulated repeat is not added twice."""
    location = citation.get("url_citation") or citation.get("text_citation") or {}
    return (
        citation_type_of(citation),
        citation.get("url") or location.get("url"),
        citation.get("start_index")
        if citation.get("start_index") is not None
        else location.get("start_index"),
        citation.get("end_index")
        if citation.get("end_index") is not None
        else location.get("end_index"),
    )


def _citation_is_new(context: StreamContext, citation: CitationPart) -> bool:
    """Whether this citation has not been emitted in this stream yet.

    Google may repeat a source across chunks, and a citation is emitted where
    it is reported, so the dedupe lives on the stream context rather than on a
    buffer.
    """
    seen = context.metadata.setdefault(f"{_GROUNDING_CITATIONS_KEY}_keys", set())
    key = _citation_key(citation)
    if key in seen:
        return False
    seen.add(key)
    return True


def _stash_search_metadata(context: StreamContext, search: dict[str, Any]) -> None:
    """Hold the search-level metadata until the finish event carries it."""
    if not search:
        return
    stashed = context.metadata.setdefault(_GROUNDING_SEARCH_KEY, {})
    stashed.update(search)


def _take_search_metadata(context: StreamContext | None) -> dict[str, Any]:
    """The held search-level metadata, cleared so a second finish cannot repeat it."""
    if context is None:
        return {}
    stashed = context.metadata.pop(_GROUNDING_SEARCH_KEY, {})
    return dict(stashed) if isinstance(stashed, dict) else {}


def _grounding_citations(metadata: dict[str, Any]) -> list[CitationPart]:
    """Convert Gemini grounding metadata into IR URL citations."""
    sources = _grounding_sources(metadata)
    supports = (
        metadata.get("grounding_supports") or metadata.get("groundingSupports") or []
    )
    citations: list[CitationPart] = []
    seen: set[tuple[Any, ...]] = set()
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment")
        segment = segment if isinstance(segment, dict) else None
        indices = (
            support.get("grounding_chunk_indices")
            or support.get("groundingChunkIndices")
            or []
        )
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(sources):
                continue
            citation = _grounding_citation(sources[index], segment)
            if citation is None:
                continue
            key = _citation_key(citation)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    if citations:
        return citations
    return [
        citation
        for source in sources
        if (citation := _grounding_citation(source, None)) is not None
    ]


def _google_finish_reason(
    reason: str,
    *,
    native: Any,
    base: str,
    response_id: str,
    metadata: Mapping[str, Any] | None,
) -> str:
    """Map an IR reason to a ``finishReason``, restoring Google's own word.

    Google's vocabulary is finer than the IR's on both sides of the failure
    line.  ``MALFORMED_FUNCTION_CALL``, ``LANGUAGE`` and a blocked ``OTHER``
    collapse into ``error``, and the filtered family (``RECITATION``,
    ``BLOCKLIST``, ``PROHIBITED_CONTENT``, ``SPII``) collapses into
    ``content_filter``.  The reader records the word it read under
    :data:`GOOGLE_NATIVE_FINISH_REASON` whenever the mapping would lose it, so
    a Google→Google round trip answers what Google said rather than a coarser
    word — or, for a failure, a 502.

    What Google cannot express is a *foreign* failure: an upstream that
    reported ``status: "failed"`` or a cancelled generation.  Such a turn
    arrives with no record at all, and ``OTHER`` would be Google's catch-all
    for a turn it produced, not a translation of "the request did not
    succeed", so the writer raises.

    The origin check is what makes the record mean something rather than being
    a lookup by shape.  This converter is the record's only writer today, so it
    is also defence in depth: a reader another dialect grows later cannot make
    Google speak a word Google never said.

    Args:
        reason: The IR finish reason.
        native: The ``provider_metadata`` the source reader left the record
            in.  It is read through :func:`native_record`, so both the current
            record and the two keys it replaced are understood.
        base: This converter's tag, checked against the record's origin.
        response_id: The response the reason belongs to.
        metadata: Upstream metadata recorded with the reason.

    Returns:
        The Google ``finishReason``.

    Raises:
        UnrepresentableFinishReason: A failure reason no Google-origin record
            explains.
    """
    record = native_record(native, origin=base)
    if record:
        native_reason = record.get("finish_reason")
        if isinstance(native_reason, str) and native_reason:
            return native_reason
    require_representable_finish_reason(
        "Google generateContent",
        reason,
        response_id=response_id,
        provider_metadata=metadata,
    )
    return GOOGLE_REASON_TO_PROVIDER.get(reason, "STOP")


def _native_word_lost(reason: str, native: Any) -> bool:
    """Whether the IR mapping would drop the word Google reported.

    Args:
        reason: The IR reason the candidate mapped to.
        native: The raw value Google reported, if any.

    Returns:
        ``True`` when the word must be recorded to survive a Google round
        trip: any value at all for a failure (the record is also what says the
        turn came from Google), and otherwise only a word the mapping would
        replace with a coarser one.
    """
    if not isinstance(native, str) or not native:
        return False
    return reason in FAILURE_REASONS or GOOGLE_REASON_TO_PROVIDER.get(reason) != native


def _native_prompt_feedback(
    metadata: Mapping[str, Any], base: str
) -> dict[str, Any] | None:
    """Return the blocked prompt's ``promptFeedback``, if this is its turn.

    Args:
        metadata: The IR message's (or finish event's) ``provider_metadata``.
        base: This converter's tag, checked against the record's origin.

    Returns:
        The verbatim ``promptFeedback`` dict, or ``None`` when the turn is not
        a Google blocked prompt.
    """
    feedback = native_record(metadata, origin=base).get("prompt_feedback")
    if isinstance(feedback, Mapping):
        return dict(feedback)
    return None

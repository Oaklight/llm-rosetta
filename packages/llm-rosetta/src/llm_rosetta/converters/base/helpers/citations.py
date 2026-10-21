"""Reading and building IR citation parts.

A ``CitationPart`` carries a discriminator (``citation_type``) and the fields
every dialect shares, flat on the part: ``url``, ``title``, ``cited_text``,
``start_index``, ``end_index``.  Provider-private fields live under
``provider_metadata``, keyed by provider.

The pre-2026 spelling nested those fields under ``url_citation`` /
``text_citation`` and had no discriminator.  It is still read here — so
hand-built IR and older callers keep working — and never written, the same
one-spelling rule ``tool_group`` (#999) and ``intrinsic_kind`` (#1010) follow.

Nothing here infers a citation's kind from the shape of its fields: a part with
a ``url`` and no ``citation_type`` is a *legacy* url citation, which the
fallback recognises by the nested key it was written into, not by guessing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ....types.ir import CitationPart

logger = logging.getLogger(__name__)

#: A source on the web, addressed by URL.  OpenAI chat/Responses annotations,
#: Google's grounding and Interactions annotations, and Anthropic's
#: ``web_search_result_location`` all reduce to this.
CITATION_TYPE_URL = "url_citation"

#: A file the provider's own file search matched (OpenAI Responses
#: ``file_citation``: ``file_id``, ``filename``, ``index``).
CITATION_TYPE_FILE = "file_citation"

#: Anthropic's four document citations.  They cite a span of a document the
#: caller supplied, so they carry no URL and only transform natively.
CITATION_TYPE_CHAR_LOCATION = "char_location"
CITATION_TYPE_PAGE_LOCATION = "page_location"
CITATION_TYPE_CONTENT_BLOCK_LOCATION = "content_block_location"
CITATION_TYPE_SEARCH_RESULT_LOCATION = "search_result_location"

#: Anthropic's citation for a result its own server-side web search returned.
CITATION_TYPE_WEB_SEARCH_RESULT_LOCATION = "web_search_result_location"

#: The pre-2026 generic text citation, kept so an untyped legacy part still
#: reads as something.
CITATION_TYPE_TEXT = "text_citation"

#: The nested spellings a legacy part may have written its fields into, in the
#: order they are consulted.
_LEGACY_CONTAINERS = ("url_citation", "text_citation")

#: Which legacy container a discriminator-less part is read as.
_LEGACY_TYPE_BY_CONTAINER = {
    "url_citation": CITATION_TYPE_URL,
    "text_citation": CITATION_TYPE_TEXT,
}

_SHARED_FIELDS = ("url", "title", "cited_text", "start_index", "end_index")


def citation_type_of(part: Any) -> str:
    """The part's citation kind, or ``""`` when it has none to report.

    Reads the ``citation_type`` field first; a part written in the legacy
    spelling is named after the container its fields were nested in.
    """
    if not isinstance(part, dict):
        return ""
    kind = part.get("citation_type")
    if isinstance(kind, str) and kind:
        return kind
    for container in _LEGACY_CONTAINERS:
        if isinstance(part.get(container), dict) and part[container]:
            return _LEGACY_TYPE_BY_CONTAINER[container]
    return ""


def citation_field(part: Any, name: str) -> Any:
    """Read one shared field, flat on the part or from the legacy nesting.

    Returns ``None`` when the field is absent (or empty) in both spellings.
    """
    if not isinstance(part, dict):
        return None
    if name not in _SHARED_FIELDS:
        raise KeyError(f"{name!r} is not a shared citation field")
    value = part.get(name)
    if value is not None and value != "":
        return value
    for container in _LEGACY_CONTAINERS:
        nested = part.get(container)
        if isinstance(nested, dict):
            value = nested.get(name)
            if value is not None and value != "":
                return value
    return None


def citation_url(part: Any) -> str | None:
    """The citation's URL, when it has one."""
    value = citation_field(part, "url")
    return value if isinstance(value, str) else None


def citation_title(part: Any) -> str | None:
    """The title the provider gave the source."""
    value = citation_field(part, "title")
    return value if isinstance(value, str) else None


def citation_cited_text(part: Any) -> str | None:
    """The span of text the provider says it drew on."""
    value = citation_field(part, "cited_text")
    return value if isinstance(value, str) else None


def citation_start_index(part: Any) -> int | None:
    """The start of the span the citation annotates, when given."""
    value = citation_field(part, "start_index")
    return value if isinstance(value, int) else None


def citation_end_index(part: Any) -> int | None:
    """The end of the span the citation annotates, when given."""
    value = citation_field(part, "end_index")
    return value if isinstance(value, int) else None


def citation_span(part: Any) -> tuple[int | None, int | None]:
    """The ``(start_index, end_index)`` the citation annotates, when given."""
    return citation_start_index(part), citation_end_index(part)


def citation_provider_metadata(part: Any) -> dict[str, Any]:
    """The provider-private fields, keyed by provider (``{}`` when absent)."""
    if not isinstance(part, dict):
        return {}
    metadata = part.get("provider_metadata")
    return metadata if isinstance(metadata, dict) else {}


def citation_provider_fields(part: Any, provider: str) -> dict[str, Any]:
    """One provider's private fields from a citation (``{}`` when absent)."""
    fields = citation_provider_metadata(part).get(provider)
    return fields if isinstance(fields, dict) else {}


def extra_fields(payload: Any, known: Iterable[str]) -> dict[str, Any]:
    """The fields of a provider payload that are *not* shared ones.

    Readers keep these under ``provider_metadata`` so a dialect's private
    fields survive the trip through the IR instead of being dropped at the
    boundary; ``known`` names the fields the caller has already mapped.
    """
    if not isinstance(payload, dict):
        return {}
    skip = set(known) | {"type"}
    return {k: v for k, v in payload.items() if k not in skip}


def make_citation(
    citation_type: str,
    *,
    url: str | None = None,
    title: str | None = None,
    cited_text: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    provider: str | None = None,
    provider_fields: dict[str, Any] | None = None,
) -> CitationPart:
    """Build a citation part in the one spelling this IR writes.

    Only the fields that carry a value are written, so a part round-trips as
    the shape it arrived in; ``provider``/``provider_fields`` are set together
    as ``provider_metadata[provider]``.
    """
    part: CitationPart = {"type": "citation", "citation_type": citation_type}
    if url is not None:
        part["url"] = url
    if title is not None:
        part["title"] = title
    if cited_text is not None:
        part["cited_text"] = cited_text
    if start_index is not None:
        part["start_index"] = start_index
    if end_index is not None:
        part["end_index"] = end_index
    if provider and provider_fields:
        part["provider_metadata"] = {provider: dict(provider_fields)}
    return part


def make_url_citation(
    url: str,
    *,
    title: str | None = None,
    cited_text: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    provider: str | None = None,
    provider_fields: dict[str, Any] | None = None,
) -> CitationPart:
    """Build a web-source citation (the one kind every dialect understands)."""
    return make_citation(
        CITATION_TYPE_URL,
        url=url,
        title=title,
        cited_text=cited_text,
        start_index=start_index,
        end_index=end_index,
        provider=provider,
        provider_fields=provider_fields,
    )


def report_dropped_citation(
    part: Any, *, warnings: list[str] | None = None, why: str = ""
) -> str:
    """Report a citation the target has no form for, and return the message.

    Same contract as the intrinsic drop report: a caller's ``warnings`` sink is
    preferred, and the log line is written either way, so a library caller that
    has no sink still sees the loss.
    """
    kind = citation_type_of(part) or "untyped"
    message = f"Citation of type {kind!r} has no form in this target; dropped"
    if why:
        message = f"{message} ({why})"
    if isinstance(warnings, list):
        warnings.append(message)
    else:
        logger.warning("%s", message)
    return message


def citation_warning_sink(
    kwargs: dict[str, Any] | None = None, context: Any = None
) -> list[str] | None:
    """The warnings sink a citation drop should be reported to, when there is one.

    A converter's response-leg writer is reached both from the pipeline (which
    passes ``warnings``) and from the library API (where the conversion context
    carries the sink).
    """
    sink = (kwargs or {}).get("warnings")
    if isinstance(sink, list):
        return sink
    warnings_ = getattr(context, "warnings", None)
    return warnings_ if isinstance(warnings_, list) else None

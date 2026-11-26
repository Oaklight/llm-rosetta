"""One declarative batch record per ``(model_type, format)`` pair.

A batch surface is a fact about a wire format that is orthogonal to its chat
surface: it says how a list of requests is delivered, what the job is called,
how its states are spelled, and what the vendor limits are.  It therefore
lives in its own record, :class:`BatchSpec`, keyed by the same pair as
:class:`~llm_rosetta.formats.spec.FormatSpec` — not as more fields on that
record.

Like the format catalog, this module is a leaf: it reads the catalog and
imports nothing from ``converters`` or ``shims``, so the shim loader can read
it while the package is still initialising.  ``formats/batch`` names its
records by pair, so "a batch record may only describe a format the catalog
knows" is the same rule as "a format has a converter": the registry refuses a
catalog entry with no converter binding, and :func:`check_batch_specs` refuses
a batch record whose pair is not in the catalog.

Absence is meaningful.  A format with no record has no native batch surface
(``google_interactions`` is generateContent-only), and a provider shim that
declares nothing has none either.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from llm_rosetta.formats.catalog import BY_PAIR
from llm_rosetta.formats.spec import FormatSpec
from llm_rosetta.ir.model.batch import BatchJobState

#: How a dialect delivers its request container.  A format with no batch
#: surface has no record, so there is no "none" value.
BatchContainer = Literal["file", "inline"]

#: How results are joined back to requests: a client-supplied string, or the
#: ordinal (the only dialect that preserves order is Gemini's inline route).
BatchItemKey = Literal["custom_id", "index"]

#: The codec ids a record may name.  ``emulated`` is not a codec — it marks a
#: job the gateway runs itself — so it is deliberately absent here and
#: accepted only by the shim declaration.
DIALECTS: tuple[str, ...] = (
    "openai_file",
    "anthropic_inline",
    "gemini_inline",
    "gemini_file",
)


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """What a vendor accepts in one batch.

    Attributes:
        max_items: Request count ceiling, when the vendor states one.
        max_bytes: Container size ceiling in bytes.  For a file dialect this
            is the upload size; for an inline dialect the request body size.
        window_hours: Target turnaround, in hours.  A target, not a deadline:
            an expired job is a first-class state, not an error.
    """

    max_items: int | None = None
    max_bytes: int | None = None
    window_hours: float | None = None


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Everything the repository knows about one pair's batch surface.

    Attributes:
        model_type: The pair's model type (``llm`` today).
        format: The pair's wire format.
        dialect: Codec id — which code builds and parses the containers.
        container: How requests are delivered: ``file`` or ``inline``.  A
            format with no batch surface has no record at all.
        create_path: Upstream path template for job creation.  Placeholders
            (``{model}``, ``{batch_id}``, ``{file_id}``) are filled by the
            codec; the templates are relative to the upstream base URL, as the
            format catalog's ``url_template`` is.
        retrieve_path: Path template for a job's current state.
        cancel_path: Path template for cancellation.
        results_path: Path template for the results.  Absent when results are
            inline on the job resource (Gemini inline) or delivered through a
            separate file download named by ``file_results_path``.
        file_results_path: Path template for downloading one result file, for
            a dialect whose results are two Files API downloads
            (``/v1/files/{file_id}/content``) or a file named on the job.
        item_key: How a result line is joined back to its request.
        state_map: The dialect's state words → the normalized
            :data:`~llm_rosetta.ir.model.batch.BatchJobState` vocabulary.
        supported_endpoints: Request endpoints this batch route accepts, e.g.
            OpenAI's ``/v1/responses`` and ``/v1/chat/completions``.
        limits: Vendor limits.
        result_retention_days: How long results stay downloadable, when the
            vendor states it.  ``None`` means not documented or not stated.
        discount: Whether the vendor bills batch work at a discount.  Display
            only — the gateway never computes prices.
        file_variant: The record for a format's second container, when the
            file variant has facts of its own (``google_generate``: inline
            ``gemini_inline``, file ``gemini_file`` with a different byte
            ceiling and result destination).  ``None`` when the format has one
            container.  A variant is a full record on the same pair; it is
            reached through :attr:`file_dialect` / :func:`batch_specs_for_dialect`,
            not through :data:`BATCH_BY_PAIR`.
    """

    model_type: str
    format: str
    dialect: str
    container: BatchContainer
    create_path: str
    retrieve_path: str
    cancel_path: str
    item_key: BatchItemKey
    state_map: Mapping[str, str]
    limits: BatchLimits = BatchLimits()
    supported_endpoints: tuple[str, ...] = ()
    results_path: str | None = None
    file_results_path: str | None = None
    result_retention_days: float | None = None
    discount: bool = False
    file_variant: BatchSpec | None = None

    @property
    def file_dialect(self) -> str | None:
        """The file variant's codec id, or ``None`` when there is none."""
        return None if self.file_variant is None else self.file_variant.dialect

    @property
    def format_spec(self) -> FormatSpec:
        """The catalog record for this pair, the chat-side facts by value.

        The two records describe one format from two angles, so a codec reads
        the auth key, route or usage shape here rather than restating them;
        the pair is the join, and :func:`check_batch_specs` guarantees it
        resolves.
        """
        return BY_PAIR[(self.model_type, self.format)]


#: State maps, one per dialect vocabulary.  The normalized target vocabulary
#: lives in :mod:`llm_rosetta.ir.model.batch`; a dialect word with no exact
#: normalized word maps to the state it behaves like:
#:
#: * OpenAI's ``cancelling`` is a cancellation in flight — the job still
#:   executes and its counts still move — so it stays ``running``; the job
#:   becomes ``cancelled`` on the vendor's own word.
#: * Anthropic's ``canceling`` is the same transitional state.
#: * ``partially_completed`` and ``finalizing`` are never upstream words for
#:   every dialect: the first is derived from counts, the second only OpenAI
#:   publishes it.
_OPENAI_STATES: Mapping[str, str] = MappingProxyType(
    {
        "validating": "validating",
        "failed": "failed",
        "in_progress": "running",
        "finalizing": "finalizing",
        "completed": "completed",
        "expired": "expired",
        "cancelling": "running",
        "cancelled": "cancelled",
    }
)

_ANTHROPIC_STATES: Mapping[str, str] = MappingProxyType(
    {
        "in_progress": "running",
        "canceling": "running",
        "ended": "completed",
    }
)

_GEMINI_STATES: Mapping[str, str] = MappingProxyType(
    {
        "JOB_STATE_PENDING": "queued",
        "JOB_STATE_RUNNING": "running",
        "JOB_STATE_SUCCEEDED": "completed",
        "JOB_STATE_FAILED": "failed",
        "JOB_STATE_CANCELLED": "cancelled",
        "JOB_STATE_EXPIRED": "expired",
    }
)

#: The Gemini file variant: the same create endpoint, an input JSONL uploaded
#: through the Files API, and results written to a file the job names.  Facts
#: are from the Gemini Batch API guide
#: (https://ai.google.dev/gemini-api/docs/batch-api, fetched 2026-11-19):
#:
#: * "The maximum allowed file size for an input file is 2GB" — recorded as
#:   2 GiB; the guide does not say whether GB is binary (unverified unit).
#: * Each input line carries a ``key`` that the response line repeats, so the
#:   join key is a client string, unlike the inline route's ordinal.
#: * Results are downloaded from the file named on the finished job
#:   (``dest.fileName``) at ``/download/v1beta/{file_id}:download?alt=media``,
#:   where ``{file_id}`` is the full ``files/...`` name.
#: * Result retention and a per-file item cap are not stated there, so both
#:   are ``None`` (unverified), not guessed.
_GEMINI_FILE = BatchSpec(
    model_type="llm",
    format="google_generate",
    dialect="gemini_file",
    container="file",
    create_path="/v1beta/models/{model}:batchGenerateContent",
    retrieve_path="/v1beta/batches/{batch_id}",
    cancel_path="/v1beta/batches/{batch_id}:cancel",
    file_results_path="/download/v1beta/{file_id}:download?alt=media",
    item_key="custom_id",
    state_map=_GEMINI_STATES,
    supported_endpoints=("/v1beta/models/{model}:generateContent",),
    limits=BatchLimits(max_bytes=2 * 1024 * 1024 * 1024, window_hours=24),
    discount=True,
)

#: The measured batch records.  Values are from the vendor documentation as
#: reviewed for epic #1509 (2026-11-13); a fact marked 未验证 there is recorded
#: as ``None`` here rather than guessed.
BATCH_SPECS: tuple[BatchSpec, ...] = (
    BatchSpec(
        model_type="llm",
        format="chat_completions",
        dialect="openai_file",
        container="file",
        create_path="/v1/batches",
        retrieve_path="/v1/batches/{batch_id}",
        cancel_path="/v1/batches/{batch_id}/cancel",
        file_results_path="/v1/files/{file_id}/content",
        item_key="custom_id",
        state_map=_OPENAI_STATES,
        supported_endpoints=("/v1/chat/completions",),
        limits=BatchLimits(
            max_items=50_000, max_bytes=200 * 1024 * 1024, window_hours=24
        ),
        result_retention_days=30,
        discount=True,
    ),
    BatchSpec(
        model_type="llm",
        format="open_responses",
        dialect="openai_file",
        container="file",
        create_path="/v1/batches",
        retrieve_path="/v1/batches/{batch_id}",
        cancel_path="/v1/batches/{batch_id}/cancel",
        file_results_path="/v1/files/{file_id}/content",
        item_key="custom_id",
        state_map=_OPENAI_STATES,
        supported_endpoints=("/v1/responses",),
        limits=BatchLimits(
            max_items=50_000, max_bytes=200 * 1024 * 1024, window_hours=24
        ),
        result_retention_days=30,
        discount=True,
    ),
    BatchSpec(
        model_type="llm",
        format="messages",
        dialect="anthropic_inline",
        container="inline",
        # One create call carries the whole batch; there is no file upload and
        # no endpoint field — the dialect is the endpoint.
        create_path="/v1/messages/batches",
        retrieve_path="/v1/messages/batches/{batch_id}",
        cancel_path="/v1/messages/batches/{batch_id}/cancel",
        results_path="/v1/messages/batches/{batch_id}/results",
        item_key="custom_id",
        state_map=_ANTHROPIC_STATES,
        supported_endpoints=("/v1/messages",),
        limits=BatchLimits(
            max_items=100_000, max_bytes=256 * 1024 * 1024, window_hours=24
        ),
        result_retention_days=29,
        discount=True,
    ),
    BatchSpec(
        model_type="llm",
        format="google_generate",
        dialect="gemini_inline",
        container="inline",
        create_path="/v1beta/models/{model}:batchGenerateContent",
        retrieve_path="/v1beta/batches/{batch_id}",
        cancel_path="/v1beta/batches/{batch_id}:cancel",
        # Inline results come back on the job resource in request order, so
        # the ordinal is the join key.  The file variant is the same endpoint
        # with a different container and its own facts (``_GEMINI_FILE``);
        # this record is the inline one the create call uses by default.
        item_key="index",
        state_map=_GEMINI_STATES,
        supported_endpoints=("/v1beta/models/{model}:generateContent",),
        # "Keep the total request size under 20MB" — the Gemini Batch API
        # guide (https://ai.google.dev/gemini-api/docs/batch-api, fetched
        # 2026-11-19).  Whether MB is binary is not stated; 20 MiB is assumed.
        limits=BatchLimits(
            max_bytes=20 * 1024 * 1024,
            window_hours=24,
        ),
        discount=True,
        file_variant=_GEMINI_FILE,
    ),
)

#: ``(model_type, format) → batch record``.
BATCH_BY_PAIR: dict[tuple[str, str], BatchSpec] = {
    (spec.model_type, spec.format): spec for spec in BATCH_SPECS
}

#: ``dialect id → every record that owns it``.  Two formats may share one codec
#: (``chat_completions`` and ``open_responses`` both use ``openai_file``) and
#: differ in ``supported_endpoints``, so a dialect alone does not name one
#: record; a file variant is listed under its own dialect id.
BY_DIALECT: dict[str, tuple[BatchSpec, ...]] = {}
for _spec in BATCH_SPECS:
    for _owner in (_spec, *((_spec.file_variant,) if _spec.file_variant else ())):
        BY_DIALECT[_owner.dialect] = (*BY_DIALECT.get(_owner.dialect, ()), _owner)
del _spec, _owner

#: The dialect ids a shim may declare, plus the gateway's own fallback.
DECLARABLE_DIALECTS: tuple[str, ...] = (*BY_DIALECT, "emulated")


def check_batch_specs() -> None:
    """Refuse a malformed batch record.

    The catalog already guarantees that every pair has a converter (the
    import-time check in :mod:`llm_rosetta.converters.registry`), so a pair
    that is in the catalog has one.  Beyond the join, each record (and each
    file variant) must name a codec in :data:`DIALECTS`, a known container and
    item key, at least one supported endpoint, and a state map whose values
    are all in the normalized :data:`~llm_rosetta.ir.model.batch.BatchJobState`
    vocabulary.

    Raises:
        RuntimeError: A pair repeats, is absent from the format catalog, or a
            record carries a value outside the vocabularies above.
    """
    seen: set[tuple[str, str]] = set()
    states = set(get_args(BatchJobState))
    containers = set(get_args(BatchContainer))
    item_keys = set(get_args(BatchItemKey))
    for spec in BATCH_SPECS:
        pair = (spec.model_type, spec.format)
        if pair in seen:
            raise RuntimeError(
                f"Duplicate batch spec for {pair!r}; one pair, one record"
            )
        seen.add(pair)
        if pair not in BY_PAIR:
            raise RuntimeError(
                f"Batch spec for {pair!r} has no format catalog entry, so it "
                f"has no converter binding. Add the FormatSpec first."
            )
        variant = spec.file_variant
        if variant is not None and (
            (variant.model_type, variant.format) != pair
            or variant.file_variant is not None
        ):
            raise RuntimeError(
                f"Batch spec for {pair!r}: a file variant must be on the same "
                f"pair and cannot carry a variant of its own"
            )
        for record in (spec, *((variant,) if variant else ())):
            where = f"Batch spec for {pair!r} ({record.dialect!r})"
            if record.dialect not in DIALECTS:
                raise RuntimeError(
                    f"{where}: dialect is not one of {', '.join(DIALECTS)}"
                )
            if record.container not in containers:
                raise RuntimeError(
                    f"{where}: container {record.container!r} is not one of "
                    f"{', '.join(sorted(containers))}"
                )
            if record.item_key not in item_keys:
                raise RuntimeError(
                    f"{where}: item_key {record.item_key!r} is not one of "
                    f"{', '.join(sorted(item_keys))}"
                )
            if not record.supported_endpoints:
                raise RuntimeError(f"{where}: supported_endpoints is empty")
            bad = sorted({v for v in record.state_map.values() if v not in states})
            if bad:
                raise RuntimeError(
                    f"{where}: state_map values {', '.join(bad)} are not in "
                    f"the normalized BatchJobState vocabulary"
                )


check_batch_specs()


def batch_spec_for(model_type: str, format: str) -> BatchSpec | None:
    """The batch record for ``(model_type, format)``, or ``None``.

    ``None`` means "no native batch surface" — the format may have no batch
    API at all, or none this repository has measured.

    Args:
        model_type: The pair's model type.
        format: The pair's wire format.

    Returns:
        The record, or ``None``.
    """
    return BATCH_BY_PAIR.get((model_type, format))


def batch_specs_for_dialect(dialect: str) -> tuple[BatchSpec, ...]:
    """Every batch record that owns *dialect*; empty for an unknown id.

    ``emulated`` resolves to an empty tuple: it is a gateway mode, not a codec.
    A file variant (``gemini_file``) resolves to its own record.

    Args:
        dialect: A codec id (``openai_file``, ``anthropic_inline``, …).

    Returns:
        The owning records, in declaration order.
    """
    return BY_DIALECT.get(dialect, ())


def batch_spec_for_dialect(dialect: str) -> BatchSpec | None:
    """The one batch record that owns *dialect*, or ``None`` for an unknown id.

    A dialect shared by several formats (``openai_file``) does not name one
    record, and picking the first would hand an ``open_responses`` lane the
    ``chat_completions`` endpoints, so the lookup refuses.  Use
    :func:`batch_spec_for` with the pair, or :func:`batch_specs_for_dialect`.

    ``emulated`` resolves to ``None``: it is a gateway mode, not a codec.

    Args:
        dialect: A codec id (``openai_file``, ``anthropic_inline``, …).

    Returns:
        The record, or ``None``.

    Raises:
        ValueError: More than one record owns *dialect*.
    """
    owners = batch_specs_for_dialect(dialect)
    if len(owners) > 1:
        raise ValueError(
            f"Batch dialect {dialect!r} is shared by "
            f"{', '.join(f'{s.model_type}/{s.format}' for s in owners)}; "
            f"look the record up by pair with batch_spec_for(model_type, format) "
            f"or list owners with batch_specs_for_dialect()"
        )
    return owners[0] if owners else None


__all__ = [
    "BATCH_BY_PAIR",
    "BATCH_SPECS",
    "BY_DIALECT",
    "DECLARABLE_DIALECTS",
    "DIALECTS",
    "BatchContainer",
    "BatchItemKey",
    "BatchLimits",
    "BatchSpec",
    "batch_spec_for",
    "batch_spec_for_dialect",
    "batch_specs_for_dialect",
    "check_batch_specs",
]

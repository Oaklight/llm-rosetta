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
from typing import Literal

from llm_rosetta.formats.catalog import BY_PAIR
from llm_rosetta.formats.spec import FormatSpec

#: How a dialect delivers its request container.
BatchContainer = Literal["file", "inline", "none"]

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
        container: How requests are delivered.  ``none`` would mean the
            dialect has no batch surface at all; such a format simply has no
            record, so the value exists only for a record that reserves one.
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
            :data:`~llm_rosetta.types.ir.batch.BatchJobState` vocabulary.
        supported_endpoints: Request endpoints this batch route accepts, e.g.
            OpenAI's ``/v1/responses`` and ``/v1/chat/completions``.
        limits: Vendor limits.
        result_retention_days: How long results stay downloadable, when the
            vendor states it.  ``None`` means not documented or not stated.
        discount: Whether the vendor bills batch work at a discount.  Display
            only — the gateway never computes prices.
        file_dialect: The second codec id for a format whose file variant is a
            distinct container (``google_generate``: inline ``gemini_inline``,
            file ``gemini_file``).  ``None`` when the format has one container.
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
    file_dialect: str | None = None

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
#: lives in :mod:`llm_rosetta.types.ir.batch`; a dialect word with no exact
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
        # with a different input/output container, so it is named by
        # ``file_dialect`` alone; this record's own container is the inline
        # one the create call uses by default.
        item_key="index",
        state_map=_GEMINI_STATES,
        supported_endpoints=("/v1beta/models/{model}:generateContent",),
        limits=BatchLimits(
            max_bytes=20 * 1024 * 1024,
            window_hours=24,
        ),
        discount=True,
        file_dialect="gemini_file",
    ),
)

#: ``(model_type, format) → batch record``.
BATCH_BY_PAIR: dict[tuple[str, str], BatchSpec] = {
    (spec.model_type, spec.format): spec for spec in BATCH_SPECS
}

#: ``dialect id → batch record``.  Two formats may share one codec
#: (``chat_completions`` and ``open_responses`` both use ``openai_file``), and
#: they differ only in ``supported_endpoints``.
BY_DIALECT: dict[str, BatchSpec] = {}
for _spec in BATCH_SPECS:
    BY_DIALECT.setdefault(_spec.dialect, _spec)
    if _spec.file_dialect is not None:
        BY_DIALECT.setdefault(_spec.file_dialect, _spec)
del _spec

#: The dialect ids a shim may declare, plus the gateway's own fallback.
DECLARABLE_DIALECTS: tuple[str, ...] = (*BY_DIALECT, "emulated")


def check_batch_specs() -> None:
    """Refuse a batch record the catalog does not know, or a duplicate pair.

    The catalog already guarantees that every pair has a converter (the
    import-time check in :mod:`llm_rosetta.converters.registry`), so a pair
    that is in the catalog has one; a batch record can only be wrong by
    naming a pair that is not.

    Raises:
        RuntimeError: A pair repeats, or is absent from the format catalog.
    """
    seen: set[tuple[str, str]] = set()
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


def batch_spec_for_dialect(dialect: str) -> BatchSpec | None:
    """The batch record that owns *dialect*, or ``None`` for an unknown id.

    ``emulated`` resolves to ``None``: it is a gateway mode, not a codec.

    Args:
        dialect: A codec id (``openai_file``, ``anthropic_inline``, …).

    Returns:
        The record, or ``None``.
    """
    return BY_DIALECT.get(dialect)


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
    "check_batch_specs",
]

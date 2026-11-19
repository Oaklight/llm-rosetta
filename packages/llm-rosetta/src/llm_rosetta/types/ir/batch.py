"""Batch job IR — one envelope for every batch dialect.

Batch is a job paradigm, not a message paradigm: a dialect accepts a *list* of
requests, returns a job, and the job's results arrive later — in two files, one
results stream, or an inline list depending on the vendor.  The types here
describe that envelope in provider-neutral terms so a codec only has to
translate the container, never the meaning.

Per-item conversion stays in the ordinary converters: :class:`BatchItem`
carries an :class:`~llm_rosetta.types.ir.request.IRRequest` and
:class:`BatchResultItem` carries an
:class:`~llm_rosetta.types.ir.response.IRResponse`, so nothing here duplicates
the chat IR.

Ordering is never part of the contract.  Every consumer joins results on
``client_ref``; Gemini's positional results are one keying strategy (the
ordinal becomes the reference), not a guarantee.

Every value is a plain ``str`` / ``int`` / list / mapping, so a job survives
``json.dumps`` unchanged.
"""

import sys
from typing import Literal, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

from .request import IRRequest
from .response import IRResponse

# ============================================================================
# Job state
# ============================================================================

#: The normalized lifecycle.  The dialects' vocabularies do not line up —
#: OpenAI says ``in_progress`` and ``finalizing``, Anthropic says
#: ``in_progress`` and ``ended``, Gemini says ``JOB_STATE_RUNNING`` — so each
#: dialect's words are translated by the ``state_map`` on its batch record.
#:
#: ``partially_completed`` has no upstream word: it is derived from the counts
#: (some items succeeded, some errored or expired), because an expired OpenAI
#: batch keeps its completed lines and only the unfinished requests become
#: ``batch_expired`` errors.
BatchJobState = Literal[
    "validating",
    "queued",
    "running",
    "finalizing",
    "completed",
    "partially_completed",
    "cancelled",
    "expired",
    "failed",
]

#: How a job is served.  ``emulated`` marks a job the gateway ran itself from
#: ordinary requests: it consumes the interactive rate-limit pool and earns no
#: batch discount, so the label is a billing fact, not just a display one.
BatchMode = Literal["native", "emulated"]


# ============================================================================
# Counts
# ============================================================================


class BatchItemCounts(TypedDict, total=False):
    """How many items a job holds in each outcome, as the upstream reports it.

    Dialects disagree on which keys they publish (Anthropic reports
    ``processing``/``succeeded``/``errored``/``canceled``/``expired``;
    OpenAI reports ``total``/``completed``/``failed``), so every key is
    optional and an absent key means "not reported", never zero.
    """

    total: int
    processing: int
    succeeded: int
    errored: int
    cancelled: int
    expired: int


# ============================================================================
# Items
# ============================================================================


class BatchItem(TypedDict):
    """One request inside a batch.

    ``client_ref`` is the dialect's item key when it has one (``custom_id``)
    or the ordinal when it does not (Gemini inline).  The gateway must never
    encode routing or conversion metadata in it: Anthropic restricts it to
    ``^[A-Za-z0-9_-]{1,64}$``, and clients join results on the value they
    supplied.
    """

    client_ref: Required[str]
    request: Required[IRRequest]


class BatchItemError(TypedDict):
    """One item's failure, in the shape every dialect reduces to.

    OpenAI's error lines, Anthropic's ``result.type`` and Gemini's inline
    status objects all answer the same three questions: what code, what
    message, is a retry worth it.
    """

    code: Required[str]
    message: Required[str]
    retryable: NotRequired[bool]


class BatchResultItem(TypedDict):
    """One item's outcome: exactly one of ``response`` or ``error`` is set.

    ``client_ref`` repeats the value the request carried, which is the only
    reliable join key — result order is not the input order.
    """

    client_ref: Required[str]
    response: NotRequired[IRResponse | None]
    error: NotRequired[BatchItemError | None]


# ============================================================================
# Job
# ============================================================================


class BatchJob(TypedDict):
    """The envelope: identity, lifecycle, counts, expiry and per-item results.

    ``created_at`` and ``expires_at`` are Unix timestamps.  ``expires_at`` is
    ``None`` when the dialect sets no deadline (Gemini's 24-hour target is
    documented as a target, not a deadline).  ``results`` is present only once
    the dialect publishes them: Anthropic serves results after ``ended``, and
    a still-running job reports counts instead.  ``counts`` is absent for
    Gemini, whose job resource carries no per-item counters.
    """

    id: Required[str]
    state: Required[BatchJobState]
    mode: Required[BatchMode]

    counts: NotRequired[BatchItemCounts]
    created_at: NotRequired[int]
    expires_at: NotRequired[int | None]
    results: NotRequired[list[BatchResultItem]]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BatchItem",
    "BatchItemCounts",
    "BatchItemError",
    "BatchJob",
    "BatchJobState",
    "BatchMode",
    "BatchResultItem",
]

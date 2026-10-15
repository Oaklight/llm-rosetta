"""Usage metering — the ledger's columns, totals, semantics and report.

This package owns what "how many tokens did we move, and for whom" means, and it
deliberately owns nothing else: no HTTP, no configuration, no credentials, no
gateway.  A library user records usage and reads a report without any of that;
the gateway is one consumer, supplying credential subjects and SQLite storage.

Three pieces, in the order a figure travels:

- :mod:`~llm_rosetta.observability.usage.semantics` — what each provider's usage
  payload means: where its counters sit, and whether its cache counters add to
  the input or sit inside it.  Declared once, read by both the streaming
  extractor and the repair.
- :mod:`~llm_rosetta.observability.usage.normalize` — the ledger columns, and
  :func:`~llm_rosetta.observability.usage.normalize.token_total`, the canonical
  total every view sums.
- :mod:`~llm_rosetta.observability.usage.report` —
  :class:`~llm_rosetta.observability.usage.report.UsageReport`: windowed totals
  per subject, plus the traffic that no subject owns, kept so the parts add up
  to the whole.

A library user with no credentials to account by::

    from llm_rosetta.observability.usage import UsageReport

    report = UsageReport.build(groups=rows)      # subjects=() -> all unattributed
    report.total("24h")

The gateway adds the two things a library user has none of — which subjects may
be attributed, and where the rows are stored::

    from llm_rosetta.observability.usage import UsageReport

    report = UsageReport.build(
        groups=await persistence.query_usage_groups(),
        subjects=[(bucket, label) for ...],       # live credentials
        internal_subject=INTERNAL_USAGE_BUCKET,
        internal_label=INTERNAL_LABEL,
    )
    report.to_payload({bucket: key_id for ...})
"""

from __future__ import annotations

from .normalize import IR_FIELDS, LEDGER_COLUMNS, token_total, usage_columns
from .report import (
    NO_SUBJECT_LABEL,
    USAGE_WINDOW_LABELS,
    USAGE_WINDOWS,
    SubjectUsage,
    UsageReport,
    add_windows,
    zero_windows,
)
from .semantics import (
    ADDITIVE_CACHE_STANDARDS,
    GOOGLE_METADATA,
    MESSAGES_FIELDS,
    OPENAI_DETAILS,
    STANDARD_SHAPES,
    USAGE_SHAPES,
    UsageShape,
    cache_is_additive,
    shape_for,
)

__all__ = [
    "ADDITIVE_CACHE_STANDARDS",
    "GOOGLE_METADATA",
    "IR_FIELDS",
    "LEDGER_COLUMNS",
    "MESSAGES_FIELDS",
    "NO_SUBJECT_LABEL",
    "OPENAI_DETAILS",
    "STANDARD_SHAPES",
    "USAGE_SHAPES",
    "USAGE_WINDOW_LABELS",
    "USAGE_WINDOWS",
    "SubjectUsage",
    "UsageReport",
    "UsageShape",
    "add_windows",
    "cache_is_additive",
    "shape_for",
    "token_total",
    "usage_columns",
    "zero_windows",
]

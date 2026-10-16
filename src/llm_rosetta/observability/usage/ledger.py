"""The usage ledger without a store: usage for one process, kept in memory.

The gateway's ledger is SQLite (:class:`~llm_rosetta.observability.PersistenceManager`);
this is the same ledger for a process that has no database — a script, a batch
job, a test — and it answers with the same report, so the two are one subsystem
rather than two spellings of one.

::

    from llm_rosetta.observability import UsageLedger

    ledger = UsageLedger()
    ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-job")
    ledger.report().total("24h")

A record carries a **subject** when the caller wants to account by one and
carries none when it does not; a ledger with no subjects still produces a
correct report, with every recorded token in the remainder.  Windows are
measured against the clock as the report is built, so a record ages out of the
narrow ones on its own.

Two defaults are worth knowing before relying on them.
:meth:`UsageLedger.report` attributes **every subject this ledger has
recorded**: there is no keystore here to say which of them are still live, and
a ledger's own records are the whole truth about it, so that is the only useful
reading.  A ledger shared between parties that do not answer for each other
should pass ``subjects=`` explicitly.  The gateway reads the same rows
differently, and deliberately: a subject whose credential no longer exists
lands in the report's remainder there, because it does have a keystore to ask.

The second: a usage dict that names no token count records nothing and returns
``None``.  That is right for a response that carried no usage and wrong for a
misspelled one — from the outside the two are indistinguishable.  Pass a
``warnings`` sink to :meth:`UsageLedger.record` to tell them apart.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .normalize import LEDGER_COLUMNS, usage_columns
from .report import USAGE_WINDOWS, UsageReport, zero_windows

__all__ = ["UsageLedger", "UsageRecord"]


@dataclass(frozen=True)
class UsageRecord:
    """One recorded call."""

    #: What the caller accounts by, or ``None`` for traffic it does not.
    subject: str | None
    #: How that subject is displayed, if it has a name of its own.
    label: str | None
    model: str | None
    #: One of the gateway's canonical model types (``llm``, ``embedding``,
    #: ``rerank``, ``decision``); ``llm`` when the caller does not say.
    model_type: str
    #: Unix time the record is dated at.
    timestamp: float
    #: The ledger columns this call reported.
    columns: Mapping[str, int]


class UsageLedger:
    """Records usage in memory and reports it the way a stored ledger is read."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._records: list[UsageRecord] = []
        self._clock = clock

    def record(
        self,
        usage: Mapping[str, Any] | None,
        *,
        model: str | None = None,
        subject: str | None = None,
        label: str | None = None,
        model_type: str = "llm",
        at: float | None = None,
        warnings: list[str] | None = None,
    ) -> UsageRecord | None:
        """Record one call's usage.

        Args:
            usage: An IR usage dict, from a response or accumulated from a
                stream.  A dict that names no token count records nothing —
                including one whose counts arrived as strings, and one whose
                keys are misspelled.  Explicit zeros are a count: a call that
                reported none is a call that happened.
            model: The model that served it.
            subject: What to account it by; omit to leave it unattributed.
            label: A display name for the subject.  A record without one is
                grouped without one, the way the stored ledger groups it.
            model_type: The gateway's canonical type for the call.
            at: Unix time to date the record at; defaults to now.
            warnings: A list to append to when nothing was recorded and the
                usage was not simply empty, so a batch job can assert it
                dropped nothing.

        Returns:
            The record, or ``None`` when *usage* named no token count.
        """
        columns = usage_columns(usage)
        if not columns:
            if warnings is not None and usage:
                keys = ", ".join(sorted(str(key) for key in usage)) or "none"
                warnings.append(
                    f"usage for {model or 'a call'} named no token count and "
                    f"was not recorded (keys seen: {keys})"
                )
            return None
        record = UsageRecord(
            subject=subject,
            label=label,
            model=model,
            model_type=model_type,
            timestamp=self._clock() if at is None else at,
            columns=columns,
        )
        self._records.append(record)
        return record

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> Sequence[UsageRecord]:
        """Every record, oldest first."""
        return tuple(self._records)

    def clear(self) -> None:
        """Forget every record."""
        self._records.clear()

    def subjects(self) -> list[tuple[str, str]]:
        """Every subject recorded, with its label — what a report may attribute.

        A record that carried no label of its own reports the subject as its
        own label, so a report has something to display.  :meth:`groups` keeps
        the label the record actually carried — ``None`` — because that is what
        the stored ledger's reader reports for the same rows, and the two
        readings of one record have to agree on everything a consumer can
        compare.
        """
        seen: dict[str, str] = {}
        for record in self._records:
            if record.subject is not None:
                seen.setdefault(record.subject, record.label or record.subject)
        return list(seen.items())

    def groups(self) -> list[dict[str, Any]]:
        """The records summed per ``(subject, label)`` and window.

        This is the seam a report is built on: the shape the stored ledger's
        reader returns, so one report serves both.  The key names are the
        stored ledger's, not this class's.
        """
        now = self._clock()
        widest = now - USAGE_WINDOWS[-1][1] * 3600
        aggregated: dict[tuple[str | None, str | None], dict[str, dict[str, int]]] = {}
        for record in self._records:
            # A subject with no traffic in the widest window is absent, the way
            # the stored reader leaves it out.
            if record.timestamp < widest:
                continue
            key = (record.subject, record.label)
            windows = aggregated.get(key)
            if windows is None:
                windows = zero_windows()
                aggregated[key] = windows
            for name, hours in USAGE_WINDOWS:
                if record.timestamp < now - hours * 3600:
                    continue
                slot = windows[name]
                slot["request_count"] += 1
                for column in LEDGER_COLUMNS:
                    value = record.columns.get(column)
                    if value is not None:
                        slot[column] += value

        return [
            {"bucket": subject, "label": label, "windows": windows}
            for (subject, label), windows in aggregated.items()
        ]

    def report(
        self,
        *,
        subjects: Sequence[tuple[str, str] | str] | None = None,
        internal_subject: str | None = None,
        internal_label: str | None = None,
    ) -> UsageReport:
        """Build the report over everything recorded so far.

        Args:
            subjects: What may be attributed.  The default is **every subject
                this ledger has seen**, which is what a caller recording its
                own traffic wants, and the one place this ledger and the
                gateway read the same rows differently: the gateway has a
                keystore and passes only its live credentials, so a subject
                whose credential is gone is attributed here and sits in the
                remainder there.  Pass ``subjects=()`` to attribute nothing,
                or names / ``(subject, label)`` pairs to name them yourself.
            internal_subject: A subject to report as the caller's own traffic.
            internal_label: The label marking it.
        """
        return UsageReport.build(
            groups=self.groups(),
            subjects=self.subjects() if subjects is None else subjects,
            internal_subject=internal_subject,
            internal_label=internal_label,
        )

    def total(self, window: str) -> dict[str, int]:
        """Every recorded token in one window, however it is attributed."""
        return self.report().total(window)

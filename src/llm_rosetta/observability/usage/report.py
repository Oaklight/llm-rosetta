"""The usage report: how the ledger's rows add up, and by whom.

A ledger row carries token columns and a **subject** — whatever the writer
wanted to account by.  This module turns rows into the one report every view
renders, and it is where the classification lives:

* **attributed** — the row's subject is one the caller named (in the gateway,
  the credential buckets of the keys that still exist);
* **internal** — the caller's own traffic (in the gateway, the admin panel's
  token), reported on its own so it is not mistaken for a user's;
* **unattributed** — everything else: a subject whose credential is gone, a row
  written before subjects existed, a request that carried no subject at all.
  Kept, with a per-label split, rather than dropped.

Keeping the third class is what makes the report add up: the dashboard counts
every row, so a report that quietly dropped the ones it could not attribute
would disagree with it.  :meth:`UsageReport.total` sums all three, so no view
has to know that rule.

A caller with no credentials at all — a library user — passes no subjects and
gets every row in the remainder, which is still a correct answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .normalize import LEDGER_COLUMNS

__all__ = [
    "USAGE_WINDOW_LABELS",
    "USAGE_WINDOWS",
    "SUBJECTS_KEY",
    "INTERNAL_KEY",
    "UNATTRIBUTED_KEY",
    "NO_SUBJECT_LABEL",
    "SubjectUsage",
    "UsageReport",
    "add_windows",
    "zero_windows",
]

#: Rolling windows a report is cut into, as (label, hours), narrowest first.
USAGE_WINDOWS: tuple[tuple[str, int], ...] = (
    ("5h", 5),
    ("24h", 24),
    ("7d", 24 * 7),
    ("30d", 24 * 30),
)

#: The window labels, in display order.
USAGE_WINDOW_LABELS: tuple[str, ...] = tuple(name for name, _hours in USAGE_WINDOWS)

#: Payload keys.  Named for the panel that renders them; the model above speaks
#: "subject" because a library user has no keys to speak of.
SUBJECTS_KEY = "keys"
INTERNAL_KEY = "internal"
UNATTRIBUTED_KEY = "unattributed"

#: Label shown for rows that carry no label at all — an anonymous request, or a
#: subject created without one.
NO_SUBJECT_LABEL = "(no credential)"

Windows = dict[str, dict[str, int]]


def zero_windows(windows: Sequence[str] = USAGE_WINDOW_LABELS) -> Windows:
    """A window map with every counter at zero."""
    empty = {"request_count": 0, **dict.fromkeys(LEDGER_COLUMNS, 0)}
    return {name: dict(empty) for name in windows}


def add_windows(target: Windows, source: Mapping[str, Mapping[str, int]]) -> None:
    """Add *source* windows into *target* in place."""
    for name, counts in source.items():
        slot = target.setdefault(
            name, {"request_count": 0, **dict.fromkeys(LEDGER_COLUMNS, 0)}
        )
        for counter, value in counts.items():
            slot[counter] = slot.get(counter, 0) + value


@dataclass(frozen=True)
class SubjectUsage:
    """One subject's usage over the report's windows."""

    subject: str
    label: str
    windows: Windows
    #: True when another live subject carries the same label — the log cannot
    #: say which of them spent the tokens, so this subject's own row reads zero
    #: and its traffic sits in the report's remainder.
    shared_label: bool = False

    def window(self, name: str) -> dict[str, int]:
        """The counters for one window, zero-filled when absent."""
        return self.windows.get(name) or zero_windows()[name]


@dataclass(frozen=True)
class UsageReport:
    """Every logged token, cut into windows and classified by subject."""

    windows: tuple[str, ...]
    by_subject: Mapping[str, SubjectUsage] = field(default_factory=dict)
    internal: Windows | None = None
    unattributed: Windows | None = None
    unattributed_by_label: Mapping[str, Windows] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        groups: Iterable[Mapping[str, Any]],
        subjects: Sequence[tuple[str, str]] = (),
        windows: Sequence[str] = USAGE_WINDOW_LABELS,
        internal_subject: str | None = None,
        internal_label: str | None = None,
    ) -> UsageReport:
        """Classify ledger *groups* into a report.

        Args:
            groups: Ledger rows already summed per ``(subject, label)``, each
                ``{"bucket": str | None, "label": str | None, "windows": {...}}``.
            subjects: ``(subject, label)`` pairs the caller wants attributed.
            windows: The window labels present in *groups*.
            internal_subject: Subject to report as the caller's own traffic.
            internal_label: Label marking that traffic.  Rows labelled with it
                that predate the subject are counted as internal too, unless a
                live subject has claimed the label.

        Returns:
            The report.  Only the classes that saw traffic are populated.
        """
        windows = tuple(windows)
        live = {subject: label for subject, label in subjects}
        # A live subject labelled like the internal token owns that label.
        internal_claimed = (
            internal_label is not None and internal_label in live.values()
        )

        by_bucket: dict[str, Windows] = {}
        internal = zero_windows(windows)
        internal_seen = False
        unattributed = zero_windows(windows)
        by_label: dict[str, Windows] = {}

        for group in groups:
            bucket = group.get("bucket")
            label = group.get("label")
            counts = group.get("windows") or {}
            if bucket is not None and bucket in live:
                add_windows(by_bucket.setdefault(bucket, zero_windows(windows)), counts)
            elif internal_subject is not None and (
                bucket == internal_subject
                or (
                    bucket is None
                    and internal_label is not None
                    and label == internal_label
                    and not internal_claimed
                )
            ):
                add_windows(internal, counts)
                internal_seen = True
            else:
                add_windows(unattributed, counts)
                add_windows(
                    by_label.setdefault(
                        str(label or NO_SUBJECT_LABEL), zero_windows(windows)
                    ),
                    counts,
                )

        label_counts: dict[str, int] = {}
        for label in live.values():
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        by_subject = {
            subject: SubjectUsage(
                subject=subject,
                label=label,
                windows=by_bucket.get(subject) or zero_windows(windows),
                shared_label=bool(label) and label_counts.get(label, 0) > 1,
            )
            for subject, label in live.items()
        }

        return cls(
            windows=windows,
            by_subject=by_subject,
            internal=internal if internal_seen else None,
            unattributed=(
                unattributed
                if any(unattributed[name]["request_count"] for name in windows)
                else None
            ),
            unattributed_by_label=by_label,
        )

    def attributed(self, subject: str, window: str) -> dict[str, int]:
        """One subject's counters for one window."""
        usage = self.by_subject.get(subject)
        return usage.window(window) if usage is not None else zero_windows()[window]

    def window_parts(self, window: str) -> Iterator[Mapping[str, int]]:
        """Every class's counters for one window, one dict per class."""
        for usage in self.by_subject.values():
            yield usage.window(window)
        for extra in (self.internal, self.unattributed):
            if extra is not None:
                counts = extra.get(window)
                if counts:
                    yield counts

    def total(self, window: str) -> dict[str, int]:
        """Every class summed — what the ledger holds for one window.

        This is the reconciliation the report exists for: it must equal the sum
        of the ledger's own rows for the same window.
        """
        slot: dict[str, int] = {"request_count": 0, **dict.fromkeys(LEDGER_COLUMNS, 0)}
        for counts in self.window_parts(window):
            slot["request_count"] += counts.get("request_count", 0)
            for name in LEDGER_COLUMNS:
                slot[name] += counts.get(name, 0)
        return slot

    def to_payload(self, names: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Render the report as the admin API's JSON.

        Args:
            names: Subject -> the name to file it under.  The gateway maps a
                credential bucket to the key's id; without it the subject is
                its own name.

        Returns:
            The payload the API Keys panel reads.
        """
        names = names or {}
        keys: dict[str, Any] = {}
        for subject, usage in self.by_subject.items():
            entry: dict[str, Any] = {
                "label": usage.label,
                "windows": usage.windows,
            }
            if usage.shared_label:
                entry["shared_label"] = True
            keys[names.get(subject, subject)] = entry

        payload: dict[str, Any] = {
            "windows": list(self.windows),
            SUBJECTS_KEY: keys,
        }
        if self.internal is not None:
            payload[INTERNAL_KEY] = {"windows": self.internal}
        if self.unattributed is not None:
            payload[UNATTRIBUTED_KEY] = {
                "windows": self.unattributed,
                "by_label": dict(self.unattributed_by_label),
            }
        return payload

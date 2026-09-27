"""Persistent baselines for round-trip fidelity differences.

:mod:`llm_rosetta.fidelity` returns :class:`~llm_rosetta.fidelity.FidelityDiff`
lists that vanish when the process exits, so converter regressions can only be
spotted by re-reading assertions manually.  This module makes fidelity
differences savable and comparable:

1. A **stable baseline key** is derived from the provider pair
   (``source -> target``), the comparison direction and a canonical fingerprint
   of the request characteristics.
2. Diff entries — field path, diff kind and severity — are written to a JSON
   baseline file that carries a ``format_version``.
3. On subsequent runs the current diffs are compared against the stored
   baseline for that key.  The report distinguishes **new** entries,
   **disappeared** entries and **severity changes**.

Comparisons are read-only by default and never touch the file; the baseline is
only rewritten when ``update=True`` is passed explicitly.  Updates take an
exclusive cross-process file lock (``fcntl`` on Unix, ``msvcrt`` on Windows)
and perform a read-modify-write so concurrent workers cannot clobber each
other's keys.  File contents are fully deterministic — entries are sorted and
writes are atomic — so a baseline survives restarts with stable ordering and
can be committed to version control.

Usage::

    baseline = FidelityBaseline("fidelity-baseline.json")
    report = check_fidelity_against_baseline(
        baseline,
        source_provider="openai_chat",
        target_provider="anthropic",
        original=request_body,
        roundtripped=roundtrip(request_body),
        mode="critical",
        update=False,          # read-only: report, do not rewrite the baseline
    )
    if report.has_changes:
        print(report)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from .fidelity import FidelityChecker, FidelityDiff

# ============================================================================
# Format version
# ============================================================================

#: Baseline format version produced by this module.
BASELINE_FORMAT_VERSION = 1

#: Versions this module can read.  An unknown (e.g. newer) version always
#: raises :class:`BaselineFormatError`; it is never treated as an empty
#: baseline.
SUPPORTED_BASELINE_FORMAT_VERSIONS = frozenset({1})

_BASELINE_TOP_KEY = "format_version"
_ENTRIES_TOP_KEY = "entries"
_DIFFS_RECORD_KEY = "diffs"


class BaselineFormatError(ValueError):
    """Raised when a baseline file is corrupt or uses an incompatible version.

    This error deliberately fails loud: callers must not mistake an
    unreadable / future-version baseline for an empty one, which would make
    every current diff look "new" and silently mask regressions.
    """


# ============================================================================
# Report data model
# ============================================================================


@dataclass(frozen=True)
class BaselineEntry:
    """One persisted fidelity diff, identified by ``(path, kind)``."""

    path: str
    kind: str
    severity: str

    @classmethod
    def from_diff(cls, diff: FidelityDiff) -> BaselineEntry:
        """Build a baseline entry from a :class:`FidelityDiff`."""
        return cls(
            path=diff.path,
            kind=diff.kind,
            severity=diff.effective_severity,
        )

    @classmethod
    def from_dict(cls, raw: Any, *, path_hint: str = "") -> BaselineEntry:
        """Parse an on-disk entry, validating its shape."""
        if not isinstance(raw, dict):
            raise BaselineFormatError(
                f"Baseline entry must be an object{path_hint}, got {type(raw).__name__}"
            )
        try:
            entry_path = raw["path"]
            kind = raw["kind"]
            severity = raw["severity"]
        except KeyError as exc:
            raise BaselineFormatError(
                f"Baseline entry{path_hint} is missing key {exc.args[0]!r}"
            ) from exc
        if not all(isinstance(v, str) for v in (entry_path, kind, severity)):
            raise BaselineFormatError(
                f"Baseline entry{path_hint} fields 'path', 'kind' and 'severity' "
                "must all be strings"
            )
        return cls(path=entry_path, kind=kind, severity=severity)

    def to_dict(self) -> dict[str, str]:
        """Serialize to the on-disk JSON representation."""
        return {"path": self.path, "kind": self.kind, "severity": self.severity}

    @property
    def key(self) -> tuple[str, str]:
        """Stable identity used when matching current entries to the baseline."""
        return (self.path, self.kind)


@dataclass(frozen=True)
class SeverityChange:
    """An entry present in both snapshots whose severity changed."""

    path: str
    kind: str
    old_severity: str
    new_severity: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to the report JSON representation."""
        return {
            "path": self.path,
            "kind": self.kind,
            "old_severity": self.old_severity,
            "new_severity": self.new_severity,
        }

    def __str__(self) -> str:
        return (
            f"{self.path} ({self.kind}): severity "
            f"{self.old_severity} -> {self.new_severity}"
        )


@dataclass
class BaselineComparisonReport:
    """Result of comparing current fidelity diffs against a stored baseline.

    Args:
        key: Stable baseline key the comparison ran under.
        new_entries: Entries in the current run but absent from the baseline.
        disappeared_entries: Entries in the baseline but absent from the
            current run (i.e. previously known fidelity loss is gone).
        severity_changes: Entries present in both snapshots whose severity
            level changed.
        current_count: Number of entries in the current run.
        baseline_count: Number of entries stored in the baseline.
        updated: Whether the baseline was rewritten during this comparison.
    """

    key: str
    new_entries: list[BaselineEntry] = field(default_factory=list)
    disappeared_entries: list[BaselineEntry] = field(default_factory=list)
    severity_changes: list[SeverityChange] = field(default_factory=list)
    current_count: int = 0
    baseline_count: int = 0
    updated: bool = False

    @property
    def has_changes(self) -> bool:
        """Whether anything differs from the baseline."""
        return bool(
            self.new_entries or self.disappeared_entries or self.severity_changes
        )

    @property
    def is_empty(self) -> bool:
        """Inverse of :attr:`has_changes`; a clean, stable comparison."""
        return not self.has_changes

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation; lists stay in stable order."""
        return {
            "key": self.key,
            "has_changes": self.has_changes,
            "updated": self.updated,
            "baseline_count": self.baseline_count,
            "current_count": self.current_count,
            "new_entries": [e.to_dict() for e in self.new_entries],
            "disappeared_entries": [e.to_dict() for e in self.disappeared_entries],
            "severity_changes": [c.to_dict() for c in self.severity_changes],
        }

    def __str__(self) -> str:
        if self.is_empty:
            return (
                f"No fidelity drift for key {self.key!r} "
                f"({self.current_count} entries match the baseline)."
            )
        lines = [f"Fidelity drift for key {self.key!r}:"]
        for entry in self.new_entries:
            lines.append(f"  + new: {entry.path} ({entry.kind}, {entry.severity})")
        for entry in self.disappeared_entries:
            lines.append(
                f"  - disappeared: {entry.path} ({entry.kind}, {entry.severity})"
            )
        for change in self.severity_changes:
            lines.append(f"  ~ {change}")
        return "\n".join(lines)


# ============================================================================
# Stable fingerprint / baseline key
# ============================================================================


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical, whitespace-free JSON string.

    Dict keys are sorted, so two semantically equal payloads that differ only
    in key insertion order hash identically.  Only JSON-native values are
    accepted; anything else raises ``TypeError`` rather than being coerced
    through an unstable ``repr``.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_fingerprint(
    source_provider: str,
    target_provider: str,
    *,
    direction: Literal["request", "response"] = "request",
    body: Mapping[str, Any] | None = None,
    features: Mapping[str, Any] | None = None,
) -> str:
    """Compute the stable SHA-256 fingerprint of a request/response case.

    The fingerprint binds together the provider pair, the comparison
    direction and the request characteristics.  ``features`` selects
    explicit characteristics (e.g. a curated subset of the body); when
    omitted, the whole ``body`` is used.  At least one of them must be given.
    """
    if body is None and features is None:
        raise ValueError("Provide either 'body' or 'features' to fingerprint")
    characteristics = features if features is not None else body
    payload = canonical_json(
        {
            "source_provider": source_provider,
            "target_provider": target_provider,
            "direction": direction,
            "characteristics": characteristics,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_baseline_key(
    source_provider: str,
    target_provider: str,
    fingerprint: str,
    *,
    direction: Literal["request", "response"] = "request",
) -> str:
    """Build the human-readable, unique key under which entries are stored."""
    return f"{source_provider}->{target_provider}:{direction}:{fingerprint}"


# ============================================================================
# Entry normalization and report building
# ============================================================================


def normalize_entries(
    diffs: Sequence[FidelityDiff | BaselineEntry],
) -> list[BaselineEntry]:
    """Convert diffs to baseline entries, de-duplicated and in stable order.

    Entry identity is ``(path, kind)``; sorting by that identity guarantees
    identical output for repeated runs of the same request.
    """
    by_key: dict[tuple[str, str], BaselineEntry] = {}
    for diff in diffs:
        if isinstance(diff, BaselineEntry):
            entry = diff
        elif isinstance(diff, FidelityDiff):
            entry = BaselineEntry.from_diff(diff)
        else:
            raise TypeError(
                "Baseline entries must be FidelityDiff or BaselineEntry, "
                f"got {type(diff).__name__}"
            )
        by_key.setdefault(entry.key, entry)
    return sorted(by_key.values(), key=lambda e: e.key)


def _build_report(
    key: str,
    old_entries: Sequence[BaselineEntry],
    current_entries: Sequence[BaselineEntry],
    *,
    updated: bool,
) -> BaselineComparisonReport:
    old_map = {entry.key: entry for entry in old_entries}
    new_map = {entry.key: entry for entry in current_entries}

    new_entries = [new_map[k] for k in sorted(new_map.keys() - old_map.keys())]
    disappeared_entries = [old_map[k] for k in sorted(old_map.keys() - new_map.keys())]
    severity_changes: list[SeverityChange] = []
    for identity in sorted(old_map.keys() & new_map.keys()):
        old_entry = old_map[identity]
        new_entry = new_map[identity]
        if old_entry.severity != new_entry.severity:
            severity_changes.append(
                SeverityChange(
                    path=new_entry.path,
                    kind=new_entry.kind,
                    old_severity=old_entry.severity,
                    new_severity=new_entry.severity,
                )
            )

    return BaselineComparisonReport(
        key=key,
        new_entries=new_entries,
        disappeared_entries=disappeared_entries,
        severity_changes=severity_changes,
        current_count=len(new_map),
        baseline_count=len(old_map),
        updated=updated,
    )


# ============================================================================
# Cross-process file locking
# ============================================================================


def _lock_exclusive(handle: Any) -> None:
    """Acquire an exclusive cross-process lock on an open sidecar file."""
    if sys.platform == "win32":
        import msvcrt  # Windows-only module.

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX)


def _unlock_exclusive(handle: Any) -> None:
    """Release the lock acquired by :func:`_lock_exclusive`."""
    if sys.platform == "win32":
        import msvcrt  # Windows-only module.

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _baseline_lock(path: str) -> Iterator[None]:
    """Serialize read-modify-write cycles on ``path`` across processes.

    Uses a ``.lock`` sidecar with ``flock`` (Unix) / ``msvcrt.locking``
    (Windows), mirroring the gateway config lock.  Multiple worker processes
    that update the same baseline queue up instead of racing.
    """
    lock_path = os.path.realpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        _lock_exclusive(handle)
        try:
            yield
        finally:
            _unlock_exclusive(handle)


# ============================================================================
# On-disk format loading / writing
# ============================================================================


def _scaffold() -> dict[str, Any]:
    return {
        _BASELINE_TOP_KEY: BASELINE_FORMAT_VERSION,
        _ENTRIES_TOP_KEY: {},
    }


def _load_and_validate(path: str) -> dict[str, Any]:
    """Read and strictly validate a baseline file.

    Raises:
        BaselineFormatError: unreadable JSON, wrong shape, missing
            ``format_version`` or an incompatible version — never silently
            falling back to an empty baseline.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineFormatError(
            f"Could not read fidelity baseline {path!r}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise BaselineFormatError(
            f"Fidelity baseline {path!r} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    if _BASELINE_TOP_KEY not in raw:
        raise BaselineFormatError(
            f"Fidelity baseline {path!r} is missing {_BASELINE_TOP_KEY!r}; "
            "refusing to treat it as an empty baseline"
        )

    version = raw[_BASELINE_TOP_KEY]
    if isinstance(version, bool) or not isinstance(version, int):
        raise BaselineFormatError(
            f"Fidelity baseline {path!r} has non-integer format_version {version!r}"
        )
    if version not in SUPPORTED_BASELINE_FORMAT_VERSIONS:
        supported = ", ".join(
            str(v) for v in sorted(SUPPORTED_BASELINE_FORMAT_VERSIONS)
        )
        raise BaselineFormatError(
            f"Incompatible fidelity baseline format version {version} in "
            f"{path!r}; supported versions: {supported}. Refusing to treat it "
            "as an empty baseline."
        )

    entries = raw.get(_ENTRIES_TOP_KEY, {})
    if not isinstance(entries, dict):
        raise BaselineFormatError(
            f"Fidelity baseline {path!r}: {_ENTRIES_TOP_KEY!r} must be an object"
        )
    return {_BASELINE_TOP_KEY: version, _ENTRIES_TOP_KEY: entries}


def _entries_from_record(record: Any, *, key: str, path: str) -> list[BaselineEntry]:
    """Extract and validate the sorted diff entries of one baseline record."""
    hint = f" in baseline record {key!r} ({path!r})"
    if not isinstance(record, dict):
        raise BaselineFormatError(f"Baseline record{hint} must be an object")
    raw_diffs = record.get(_DIFFS_RECORD_KEY, [])
    if not isinstance(raw_diffs, list):
        raise BaselineFormatError(
            f"Baseline record{hint}: {_DIFFS_RECORD_KEY!r} must be a list"
        )
    entries = [BaselineEntry.from_dict(raw, path_hint=hint) for raw in raw_diffs]
    return sorted(entries, key=lambda e: e.key)


# ============================================================================
# FidelityBaseline
# ============================================================================


class FidelityBaseline:
    """Load, compare against and update a fidelity baseline file.

    Constructing the object eagerly reads the file so that an incompatible
    format version fails immediately.  A missing file is a valid empty
    baseline; nothing is written until :meth:`compare` is called with
    ``update=True``.

    Args:
        path: Path to the JSON baseline file.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        self._data: dict[str, Any] = self._read_latest()

    # -- reading ------------------------------------------------------------

    def _read_latest(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return _scaffold()
        return _load_and_validate(self.path)

    def reload(self) -> None:
        """Re-read the file from disk (another process may have updated it)."""
        self._data = self._read_latest()

    def keys(self) -> list[str]:
        """All stored baseline keys, in sorted order."""
        return sorted(self._data[_ENTRIES_TOP_KEY].keys())

    def entries_for(self, key: str) -> list[BaselineEntry]:
        """Return the stored entries for ``key`` (empty list if unknown)."""
        record = self._data[_ENTRIES_TOP_KEY].get(key)
        if record is None:
            return []
        return _entries_from_record(record, key=key, path=self.path)

    def __contains__(self, key: object) -> bool:
        return key in self._data[_ENTRIES_TOP_KEY]

    # -- comparison ---------------------------------------------------------

    def compare(
        self,
        key: str,
        diffs: Sequence[FidelityDiff | BaselineEntry],
        *,
        update: bool = False,
        source_provider: str | None = None,
        target_provider: str | None = None,
        direction: Literal["request", "response"] = "request",
        fingerprint: str | None = None,
    ) -> BaselineComparisonReport:
        """Compare current ``diffs`` with the baseline stored under ``key``.

        Args:
            key: Stable key from :func:`make_baseline_key`.
            diffs: Current fidelity differences (or pre-built entries).
            update: Read-only when ``False`` (the default): only produce a
                report.  When ``True``, rewrite the baseline record for
                ``key`` with the current entries under an exclusive file
                lock.
            source_provider: Optional provider-pair metadata stored with the
                record on update.
            target_provider: See ``source_provider``.
            direction: Request/response direction stored on update.
            fingerprint: Full fingerprint stored on update.

        Returns:
            A report computed against the *previous* baseline state, even
            when ``update=True``, so callers still see what changed while
            accepting the new snapshot.
        """
        current_entries = normalize_entries(diffs)
        if not update:
            # Re-read so long-lived processes see baselines updated elsewhere.
            self._data = self._read_latest()
            old_entries = self.entries_for(key)
            return _build_report(key, old_entries, current_entries, updated=False)
        return self._compare_and_update(
            key,
            current_entries,
            source_provider=source_provider,
            target_provider=target_provider,
            direction=direction,
            fingerprint=fingerprint,
        )

    def _compare_and_update(
        self,
        key: str,
        current_entries: list[BaselineEntry],
        *,
        source_provider: str | None,
        target_provider: str | None,
        direction: str,
        fingerprint: str | None,
    ) -> BaselineComparisonReport:
        with _baseline_lock(self.path):
            # Re-read inside the lock: another process may have committed
            # between our construction and this update.
            data = self._read_latest()
            record = data[_ENTRIES_TOP_KEY].get(key)
            old_entries = (
                _entries_from_record(record, key=key, path=self.path)
                if record is not None
                else []
            )
            report = _build_report(key, old_entries, current_entries, updated=True)
            data[_ENTRIES_TOP_KEY][key] = self._build_record(
                current_entries,
                source_provider=source_provider,
                target_provider=target_provider,
                direction=direction,
                fingerprint=fingerprint,
            )
            self._write_atomic(data)
            self._data = data
        return report

    @staticmethod
    def _build_record(
        entries: Sequence[BaselineEntry],
        *,
        source_provider: str | None,
        target_provider: str | None,
        direction: str,
        fingerprint: str | None,
    ) -> dict[str, Any]:
        return {
            "source_provider": source_provider or "",
            "target_provider": target_provider or "",
            "direction": direction,
            "fingerprint": fingerprint or "",
            # Empty list is a meaningful record: this case is now clean.
            _DIFFS_RECORD_KEY: [entry.to_dict() for entry in entries],
        }

    # -- writing ------------------------------------------------------------

    def _write_atomic(self, data: dict[str, Any]) -> None:
        """Atomically replace the baseline file with deterministic content.

        ``sort_keys=True`` makes the on-disk byte order independent of
        insertion history, so restarts and rewrites stay byte-stable, and
        ``os.replace`` guarantees readers never observe a half-written file.
        """
        directory = os.path.dirname(os.path.realpath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".fidelity-baseline-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    data,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


# ============================================================================
# High-level convenience
# ============================================================================


def check_fidelity_against_baseline(
    baseline: FidelityBaseline,
    *,
    source_provider: str,
    target_provider: str,
    original: dict[str, Any],
    roundtripped: dict[str, Any],
    direction: Literal["request", "response"] = "request",
    mode: Literal["critical", "full"] = "critical",
    format_name: str | None = None,
    features: Mapping[str, Any] | None = None,
    update: bool = False,
    checker: FidelityChecker | None = None,
) -> BaselineComparisonReport:
    """Run a fidelity check and compare its diffs with the stored baseline.

    Args:
        baseline: Open baseline store.
        source_provider: Format of the original body (provider pair side).
        target_provider: Format the body round-tripped through.
        original: Original request/response body.  Also feeds the stable
            fingerprint unless ``features`` is given.
        roundtripped: Body produced by the A -> IR -> A (or A -> IR -> B ->
            IR -> A) round-trip.
        direction: Whether ``original`` is a request or a response.
        mode: ``FidelityChecker`` mode ("critical"/"full").
        format_name: Format used for critical-path selection; defaults to
            ``source_provider``.
        features: Explicit fingerprint characteristics; defaults to the
            whole ``original`` body.
        update: Read-only by default; set ``True`` to accept the current
            diffs as the new baseline.
        checker: Reuse an existing :class:`FidelityChecker` instead of
            creating one.

    Returns:
        The comparison report.
    """
    active_checker = checker or FidelityChecker(
        mode=mode, format_name=format_name or source_provider
    )
    diffs = active_checker.compare(original, roundtripped, direction=direction)
    fingerprint = compute_fingerprint(
        source_provider,
        target_provider,
        direction=direction,
        body=None if features is not None else original,
        features=features,
    )
    key = make_baseline_key(
        source_provider, target_provider, fingerprint, direction=direction
    )
    return baseline.compare(
        key,
        diffs,
        update=update,
        source_provider=source_provider,
        target_provider=target_provider,
        direction=direction,
        fingerprint=fingerprint,
    )

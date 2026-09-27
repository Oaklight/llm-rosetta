"""Tests for persistent fidelity baselines (``fidelity_baseline``).

Covers:
- baseline write / read (including simulated process restart)
- stable fingerprint / baseline key generation
- stable ordering and byte-deterministic files
- new, disappeared and severity-change detection
- incompatible / missing / corrupt format version errors
- read-only mode never writing
- empty report output on clean comparisons
- cross-process file-lock serialization
- the high-level ``check_fidelity_against_baseline`` convenience
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

from llm_rosetta.fidelity import FidelityChecker, FidelityDiff
from llm_rosetta.fidelity_baseline import (
    BASELINE_FORMAT_VERSION,
    BaselineComparisonReport,
    BaselineEntry,
    BaselineFormatError,
    FidelityBaseline,
    canonical_json,
    check_fidelity_against_baseline,
    compute_fingerprint,
    make_baseline_key,
    normalize_entries,
)

KEY = "openai_chat->anthropic:request:0123456789abcdef"


def _entry(
    path: str, kind: str = "missing", severity: str = "critical"
) -> BaselineEntry:
    return BaselineEntry(path=path, kind=kind, severity=severity)


# ============================================================================
# Severity derivation
# ============================================================================


class TestSeverity:
    def test_kind_default_severity(self) -> None:
        assert FidelityDiff(path="a", kind="missing").effective_severity == "critical"
        assert (
            FidelityDiff(path="a", kind="type_changed").effective_severity == "critical"
        )
        assert FidelityDiff(path="a", kind="changed").effective_severity == "warning"
        assert FidelityDiff(path="a", kind="added").effective_severity == "info"

    def test_unknown_kind_defaults_to_warning(self) -> None:
        assert FidelityDiff(path="a", kind="weird").effective_severity == "warning"

    def test_explicit_severity_override(self) -> None:
        diff = FidelityDiff(path="a", kind="missing", severity="info")
        assert diff.effective_severity == "info"

    def test_existing_construction_still_works(self) -> None:
        # Backward compatibility: no severity argument.
        diff = FidelityDiff("model", "gpt-4o", "gpt-4o-mini", "changed")
        assert diff.severity is None
        assert diff.effective_severity == "warning"


# ============================================================================
# Fingerprint / stable identifier
# ============================================================================


class TestFingerprint:
    def test_same_body_different_key_order_same_fingerprint(self) -> None:
        body_a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        body_b = {
            "messages": [{"content": "hi", "role": "user"}],
            "model": "gpt-4o",
        }
        fp_a = compute_fingerprint("openai_chat", "anthropic", body=body_a)
        fp_b = compute_fingerprint("openai_chat", "anthropic", body=body_b)
        assert fp_a == fp_b

    def test_different_request_characteristics_change_fingerprint(self) -> None:
        fp_a = compute_fingerprint(
            "openai_chat", "anthropic", body={"messages": [{"content": "a"}]}
        )
        fp_b = compute_fingerprint(
            "openai_chat", "anthropic", body={"messages": [{"content": "b"}]}
        )
        assert fp_a != fp_b

    def test_provider_pair_and_direction_participate(self) -> None:
        body = {"messages": [{"content": "hi"}]}
        fp = compute_fingerprint("openai_chat", "anthropic", body=body)
        assert (
            compute_fingerprint("anthropic", "anthropic", body=body) != fp
        )  # other source
        assert (
            compute_fingerprint("openai_chat", "openai_chat", body=body) != fp
        )  # other target
        assert (
            compute_fingerprint(
                "openai_chat", "anthropic", direction="response", body=body
            )
            != fp
        )

    def test_repeated_run_is_stable(self) -> None:
        body = {"model": "x", "messages": [{"role": "user", "content": "n=42"}]}
        fp_1 = compute_fingerprint("openai_chat", "anthropic", body=body)
        fp_2 = compute_fingerprint("openai_chat", "anthropic", body=body)
        assert fp_1 == fp_2

    def test_features_override_body(self) -> None:
        fp = compute_fingerprint(
            "openai_chat",
            "anthropic",
            body={"volatile": "x", "stable": 1},
            features={"stable": 1},
        )
        same = compute_fingerprint(
            "openai_chat",
            "anthropic",
            body={"volatile": "y", "stable": 1},
            features={"stable": 1},
        )
        different = compute_fingerprint(
            "openai_chat", "anthropic", features={"stable": 2}
        )
        assert fp == same != different

    def test_requires_body_or_features(self) -> None:
        with pytest.raises(ValueError, match="body.*features"):
            compute_fingerprint("openai_chat", "anthropic")

    def test_key_uses_full_digest_and_pair(self) -> None:
        fp = compute_fingerprint("openai_chat", "anthropic", body={"a": 1})
        key = make_baseline_key("openai_chat", "anthropic", fp)
        assert key == f"openai_chat->anthropic:request:{fp}"

    def test_canonical_json_ignores_dict_order(self) -> None:
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


# ============================================================================
# Baseline write / read / restart
# ============================================================================


class TestBaselineWriteRead:
    def test_missing_file_is_empty_but_not_created(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        baseline = FidelityBaseline(path)
        assert baseline.keys() == []
        assert KEY not in baseline
        assert not path.exists()  # read-only construction writes nothing

    def test_update_writes_entries_with_path_kind_severity(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "baseline.json"
        baseline = FidelityBaseline(path)
        diffs = [
            FidelityDiff(path="messages.*.role", kind="missing"),
            FidelityDiff(path="model", kind="changed"),
            FidelityDiff(path="extra", kind="added"),
        ]
        report = baseline.compare(
            KEY,
            diffs,
            update=True,
            source_provider="openai_chat",
            target_provider="anthropic",
            fingerprint=KEY.rsplit(":", 1)[-1],
        )
        assert path.exists()
        assert report.updated is True

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["format_version"] == BASELINE_FORMAT_VERSION
        record = raw["entries"][KEY]
        assert record["source_provider"] == "openai_chat"
        assert record["target_provider"] == "anthropic"
        assert record["direction"] == "request"
        # Missing-field paths ("lost fields") and severity are persisted.
        stored = {(d["path"], d["kind"], d["severity"]) for d in record["diffs"]}
        assert ("messages.*.role", "missing", "critical") in stored
        assert ("model", "changed", "warning") in stored
        assert ("extra", "added", "info") in stored

    def test_restart_reload_sees_same_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        FidelityBaseline(path).compare(
            KEY,
            [FidelityDiff(path="a.b", kind="missing")],
            update=True,
        )

        # Simulate a brand-new process opening the same file.
        restarted = FidelityBaseline(path)
        assert KEY in restarted
        assert restarted.entries_for(KEY) == [_entry("a.b", "missing", "critical")]

    def test_repeated_rewrite_is_byte_deterministic(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        diffs = [
            FidelityDiff(path="z", kind="added"),
            FidelityDiff(path="a", kind="missing"),
            FidelityDiff(path="m", kind="changed"),
        ]
        FidelityBaseline(path).compare(
            KEY, diffs, update=True, source_provider="s", target_provider="t"
        )
        first_bytes = path.read_bytes()

        # Rewrite from a second "restarted" instance with shuffled input.
        FidelityBaseline(path).compare(
            KEY,
            [diffs[2], diffs[0], diffs[1]],
            update=True,
            source_provider="s",
            target_provider="t",
        )
        assert path.read_bytes() == first_bytes

    def test_entries_key_order_on_disk_is_sorted(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        baseline = FidelityBaseline(path)
        for name in ("key-c", "key-a", "key-b"):
            baseline.compare(
                name, [FidelityDiff(path="p", kind="missing")], update=True
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(raw["entries"].keys()) == ["key-a", "key-b", "key-c"]

    def test_update_clean_case_stores_empty_diff_list(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        FidelityBaseline(path).compare(KEY, [], update=True)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["entries"][KEY]["diffs"] == []
        assert FidelityBaseline(path).entries_for(KEY) == []

    def test_duplicate_identity_deduped(self, tmp_path: Path) -> None:
        entries = normalize_entries(
            [
                FidelityDiff(path="a", kind="missing"),
                FidelityDiff(path="a", kind="missing", severity="info"),
                BaselineEntry(path="b", kind="added", severity="info"),
            ]
        )
        assert [(e.path, e.kind) for e in entries] == [("a", "missing"), ("b", "added")]
        # First occurrence wins.
        assert entries[0].severity == "critical"


# ============================================================================
# New / disappeared / severity-change comparison
# ============================================================================


class TestComparison:
    def _seeded(self, path: Path, entries: list[BaselineEntry]) -> FidelityBaseline:
        baseline = FidelityBaseline(path)
        baseline.compare(KEY, entries, update=True)
        return baseline

    def test_all_new_when_no_baseline(self, tmp_path: Path) -> None:
        baseline = FidelityBaseline(tmp_path / "b.json")
        report = baseline.compare(KEY, [FidelityDiff(path="a", kind="missing")])
        assert report.updated is False
        assert len(report.new_entries) == 1
        assert report.new_entries[0].path == "a"
        assert report.disappeared_entries == []
        assert report.severity_changes == []
        assert report.baseline_count == 0
        assert report.current_count == 1

    def test_new_entries_detected(self, tmp_path: Path) -> None:
        baseline = self._seeded(tmp_path / "b.json", [_entry("a")])
        report = baseline.compare(KEY, [_entry("a"), _entry("b")])
        assert [e.path for e in report.new_entries] == ["b"]
        assert report.disappeared_entries == []
        assert report.severity_changes == []

    def test_disappeared_entries_detected(self, tmp_path: Path) -> None:
        baseline = self._seeded(tmp_path / "b.json", [_entry("a"), _entry("b")])
        report = baseline.compare(KEY, [_entry("a")])
        assert report.new_entries == []
        assert [e.path for e in report.disappeared_entries] == ["b"]
        assert report.severity_changes == []

    def test_severity_change_detected(self, tmp_path: Path) -> None:
        baseline = self._seeded(
            tmp_path / "b.json", [_entry("a", "missing", "critical")]
        )
        report = baseline.compare(KEY, [_entry("a", "missing", "warning")])
        assert report.new_entries == []
        assert report.disappeared_entries == []
        assert len(report.severity_changes) == 1
        change = report.severity_changes[0]
        assert change.path == "a"
        assert change.kind == "missing"
        assert change.old_severity == "critical"
        assert change.new_severity == "warning"

    def test_kind_change_is_disappeared_plus_new(self, tmp_path: Path) -> None:
        baseline = self._seeded(
            tmp_path / "b.json", [_entry("a", "missing", "critical")]
        )
        report = baseline.compare(KEY, [_entry("a", "changed", "warning")])
        assert [e.kind for e in report.disappeared_entries] == ["missing"]
        assert [e.kind for e in report.new_entries] == ["changed"]
        assert report.severity_changes == []

    def test_unchanged_entries_produce_empty_report(self, tmp_path: Path) -> None:
        baseline = self._seeded(
            tmp_path / "b.json",
            [_entry("a", "missing", "critical"), _entry("b", "added", "info")],
        )
        report = baseline.compare(
            KEY,
            [_entry("b", "added", "info"), _entry("a", "missing", "critical")],
        )
        assert report.is_empty
        assert report.baseline_count == 2
        assert report.current_count == 2

    def test_update_returns_report_against_old_then_persists(
        self, tmp_path: Path
    ) -> None:
        baseline = self._seeded(tmp_path / "b.json", [_entry("a"), _entry("b")])
        report = baseline.compare(KEY, [_entry("a")], update=True)
        # Report reflects the comparison against the previous baseline.
        assert [e.path for e in report.disappeared_entries] == ["b"]
        assert report.updated is True
        # A follow-up read-only run is clean against the rewritten baseline.
        follow_up = FidelityBaseline(tmp_path / "b.json").compare(KEY, [_entry("a")])
        assert follow_up.is_empty

    def test_report_lists_are_stably_sorted(self, tmp_path: Path) -> None:
        baseline = self._seeded(
            tmp_path / "b.json",
            [_entry("z"), _entry("a"), _entry("m", "changed", "warning")],
        )
        report = baseline.compare(
            KEY,
            [
                _entry("z", "missing", "info"),  # severity change
                _entry("a"),  # unchanged
                _entry("y"),  # new
            ],
        )
        assert [e.path for e in report.new_entries] == ["y"]
        assert [e.path for e in report.disappeared_entries] == ["m"]
        assert [c.path for c in report.severity_changes] == ["z"]

    def test_report_to_dict_and_str(self, tmp_path: Path) -> None:
        baseline = self._seeded(tmp_path / "b.json", [_entry("a")])
        changed = baseline.compare(
            KEY,
            [
                _entry("a", "missing", "warning"),
                _entry("b", "added", "info"),
            ],
        )
        payload = changed.to_dict()
        assert payload["has_changes"] is True
        assert payload["updated"] is False
        assert payload["new_entries"] == [
            {"path": "b", "kind": "added", "severity": "info"}
        ]
        assert payload["severity_changes"][0]["old_severity"] == "critical"
        text = str(changed)
        assert "+ new:" in text and "b" in text
        assert "severity" in text and "critical -> warning" in text


# ============================================================================
# Format version / corruption handling
# ============================================================================


class TestFormatVersion:
    def test_incompatible_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"format_version": 99, "entries": {}}), encoding="utf-8"
        )
        with pytest.raises(BaselineFormatError, match="version 99"):
            FidelityBaseline(path)

    def test_missing_version_raises_not_empty_baseline(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"entries": {}}), encoding="utf-8")
        with pytest.raises(BaselineFormatError, match="format_version"):
            FidelityBaseline(path)

    def test_legacy_version_zero_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"format_version": 0, "entries": {}}), encoding="utf-8"
        )
        with pytest.raises(BaselineFormatError, match="version 0"):
            FidelityBaseline(path)

    def test_non_integer_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"format_version": "1", "entries": {}}), encoding="utf-8"
        )
        with pytest.raises(BaselineFormatError, match="non-integer"):
            FidelityBaseline(path)

    def test_bool_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"format_version": True, "entries": {}}), encoding="utf-8"
        )
        with pytest.raises(BaselineFormatError):
            FidelityBaseline(path)

    def test_corrupt_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BaselineFormatError):
            FidelityBaseline(path)

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(BaselineFormatError, match="JSON object"):
            FidelityBaseline(path)

    def test_incompatible_version_never_treated_as_empty(self, tmp_path: Path) -> None:
        # Even a read-only comparison must fail loudly on a bad version.
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"format_version": 99, "entries": {}}), encoding="utf-8"
        )
        with pytest.raises(BaselineFormatError):
            FidelityBaseline(path).compare(
                KEY, [FidelityDiff(path="a", kind="missing")]
            )

    def test_malformed_entry_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps(
                {
                    "format_version": BASELINE_FORMAT_VERSION,
                    "entries": {KEY: {"diffs": [{"path": "a"}]}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(BaselineFormatError, match="'kind'"):
            FidelityBaseline(path).compare(KEY, [])


# ============================================================================
# Read-only mode
# ============================================================================


class TestReadOnly:
    def test_readonly_without_file_creates_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        report = FidelityBaseline(path).compare(
            KEY, [FidelityDiff(path="a", kind="missing")]
        )
        assert not path.exists()
        assert not Path(str(path) + ".lock").exists()
        assert len(report.new_entries) == 1

    def test_readonly_does_not_modify_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        FidelityBaseline(path).compare(KEY, [_entry("a")], update=True)
        before = path.read_bytes()
        mtime_before = path.stat().st_mtime_ns

        FidelityBaseline(path).compare(
            KEY,
            [_entry("a"), _entry("b"), _entry("c")],  # drift reported...
        )

        assert path.read_bytes() == before
        assert path.stat().st_mtime_ns == mtime_before
        # ...but the baseline is unchanged on disk.
        restarted = FidelityBaseline(path)
        assert [e.path for e in restarted.entries_for(KEY)] == ["a"]

    def test_explicit_update_does_modify_file(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        FidelityBaseline(path).compare(KEY, [_entry("a")], update=True)
        before = path.read_bytes()
        FidelityBaseline(path).compare(KEY, [_entry("a"), _entry("b")], update=True)
        assert path.read_bytes() != before
        assert [e.path for e in FidelityBaseline(path).entries_for(KEY)] == ["a", "b"]

    def test_lock_sidecar_created_only_on_update(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        FidelityBaseline(path).compare(KEY, [_entry("a")])
        assert not Path(str(path) + ".lock").exists()
        FidelityBaseline(path).compare(KEY, [_entry("a")], update=True)
        assert Path(str(path) + ".lock").exists()


# ============================================================================
# Empty report
# ============================================================================


class TestEmptyReport:
    def test_no_diffs_no_baseline_is_empty(self, tmp_path: Path) -> None:
        report = FidelityBaseline(tmp_path / "b.json").compare(KEY, [])
        self._assert_empty(report)
        assert "No fidelity drift" in str(report)

    def test_no_diffs_after_baselining_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        FidelityBaseline(path).compare(KEY, [], update=True)
        report = FidelityBaseline(path).compare(KEY, [])
        self._assert_empty(report)

    def test_stable_entries_with_diffs_is_empty_report(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        diffs = [FidelityDiff(path="a", kind="missing")]
        FidelityBaseline(path).compare(KEY, diffs, update=True)
        report = FidelityBaseline(path).compare(KEY, diffs)
        self._assert_empty(report)

    def test_disappeared_drift_is_not_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        FidelityBaseline(path).compare(
            KEY, [FidelityDiff(path="a", kind="missing")], update=True
        )
        report = FidelityBaseline(path).compare(KEY, [])
        assert report.has_changes
        assert [e.path for e in report.disappeared_entries] == ["a"]

    @staticmethod
    def _assert_empty(report: BaselineComparisonReport) -> None:
        assert report.is_empty
        assert not report.has_changes
        assert report.new_entries == []
        assert report.disappeared_entries == []
        assert report.severity_changes == []
        payload = report.to_dict()
        assert payload["has_changes"] is False
        assert payload["new_entries"] == []
        assert payload["disappeared_entries"] == []
        assert payload["severity_changes"] == []


# ============================================================================
# Concurrent writers (threads in one process and multiple worker processes)
# ============================================================================


class TestConcurrentWriters:
    def test_threads_sharing_one_file(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        barrier = threading.Barrier(8)

        def worker(worker_id: int) -> None:
            baseline = FidelityBaseline(path)
            barrier.wait()
            for i in range(10):
                key = f"w{worker_id:02d}-k{i:02d}"
                baseline.compare(
                    key,
                    [FidelityDiff(path=f"p{worker_id}.{i}", kind="missing")],
                    update=True,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        baseline = FidelityBaseline(path)
        assert len(baseline.keys()) == 80
        # File is well-formed and every record survived.
        for worker_id in range(8):
            for i in range(10):
                key = f"w{worker_id:02d}-k{i:02d}"
                entries = baseline.entries_for(key)
                assert entries == [_entry(f"p{worker_id}.{i}")]


def _multiprocess_worker(path: str, work: list[tuple[str, str]]) -> None:
    """Top-level worker (picklable under spawn) updating distinct keys."""
    baseline = FidelityBaseline(path)
    for key, field_path in work:
        baseline.compare(
            key,
            [FidelityDiff(path=field_path, kind="missing")],
            update=True,
            source_provider="openai_chat",
            target_provider="anthropic",
        )


def test_multiple_processes_serialized_by_file_lock(tmp_path: Path) -> None:
    path = str(tmp_path / "b.json")
    # spawn: works on every platform and avoids fork-in-multithreaded-process
    # issues; the worker is a picklable top-level function.
    ctx = multiprocessing.get_context("spawn")

    # Every process writes its own keys plus one shared key with identical
    # content, so the final state is deterministic.
    plans: list[list[tuple[str, str]]] = []
    for worker_id in range(6):
        plan = [
            (f"proc{worker_id:02d}-key{i:02d}", f"path.{worker_id}.{i}")
            for i in range(6)
        ]
        plan.append(("shared-key", "shared.path"))
        plans.append(plan)

    processes = [
        ctx.Process(target=_multiprocess_worker, args=(path, plan)) for plan in plans
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker failed with exitcode {proc.exitcode}"

    # Strict reload (format validation) must succeed after the storm.
    baseline = FidelityBaseline(path)
    assert len(baseline.keys()) == 37  # 6*6 distinct + 1 shared
    assert baseline.entries_for("shared-key") == [_entry("shared.path")]
    for worker_id in range(6):
        for i in range(6):
            entries = baseline.entries_for(f"proc{worker_id:02d}-key{i:02d}")
            assert entries == [_entry(f"path.{worker_id}.{i}")]

    # On-disk key order stays sorted after interleaved process writes.
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert list(raw["entries"].keys()) == sorted(raw["entries"].keys())
    assert os.path.exists(path + ".lock")


# ============================================================================
# End-to-end convenience helper with a real FidelityChecker
# ============================================================================


class TestCheckAgainstBaseline:
    def _openai_request(self, *, with_tools: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather?"}],
        }
        if with_tools:
            body["tools"] = [{"type": "function", "function": {"name": "get_weather"}}]
        return body

    def test_accept_then_regression_and_fix_cycle(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        baseline = FidelityBaseline(path)
        original = self._openai_request()

        # Round-trip drops the tools list — accept as the initial baseline.
        broken = self._openai_request(with_tools=False)
        first = check_fidelity_against_baseline(
            baseline,
            source_provider="openai_chat",
            target_provider="openai_chat",
            original=original,
            roundtripped=broken,
            mode="critical",
            update=True,
        )
        assert first.updated is True
        assert len(first.new_entries) >= 1
        assert any(e.kind == "missing" for e in first.new_entries)

        # Re-running the same conversion is stable: nothing drifts.
        stable = check_fidelity_against_baseline(
            baseline,
            source_provider="openai_chat",
            target_provider="openai_chat",
            original=original,
            roundtripped=broken,
            mode="critical",
        )
        assert stable.is_empty
        assert stable.updated is False

        # Converter fixed: tools survive → the old missing entries disappear.
        fixed = check_fidelity_against_baseline(
            baseline,
            source_provider="openai_chat",
            target_provider="openai_chat",
            original=original,
            roundtripped=dict(original),
            mode="critical",
        )
        assert fixed.new_entries == []
        assert len(fixed.disappeared_entries) >= 1
        assert all(e.kind == "missing" for e in fixed.disappeared_entries)

        # Read-only by default: the fixed state was not accepted.
        restarted = FidelityBaseline(path)
        assert len(restarted.entries_for(stable.key)) >= 1

    def test_same_request_same_key_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        original = self._openai_request()
        broken = self._openai_request(with_tools=False)

        report_1 = check_fidelity_against_baseline(
            FidelityBaseline(path),
            source_provider="openai_chat",
            target_provider="anthropic",
            original=original,
            roundtripped=broken,
            update=True,
        )
        report_2 = check_fidelity_against_baseline(
            FidelityBaseline(path),
            source_provider="openai_chat",
            target_provider="anthropic",
            original=dict(reversed(list(original.items()))),  # reordered keys
            roundtripped=broken,
        )
        assert report_1.key == report_2.key
        assert report_2.is_empty

    def test_full_mode_response_direction(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        original = {"id": "resp-1", "choices": [{"finish_reason": "stop"}]}
        roundtripped = {"id": "resp-1", "choices": [{"finish_reason": "length"}]}
        report = check_fidelity_against_baseline(
            FidelityBaseline(path),
            source_provider="openai_chat",
            target_provider="openai_chat",
            original=original,
            roundtripped=roundtripped,
            direction="response",
            mode="full",
        )
        assert report.has_changes
        assert any("finish_reason" in e.path for e in report.new_entries)
        assert "response" in report.key

    def test_reuses_injected_checker(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        checker = FidelityChecker(mode="full")
        body = {"a": 1}
        report = check_fidelity_against_baseline(
            FidelityBaseline(path),
            source_provider="anthropic",
            target_provider="anthropic",
            original=body,
            roundtripped=body,
            checker=checker,
        )
        assert report.is_empty

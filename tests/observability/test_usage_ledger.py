"""Tests for the in-memory usage ledger.

The ledger's job is to be the *same* ledger as the stored one, minus the store.
So two things are asserted here: that it behaves like a ledger (records, ages
records out of windows, sums per subject), and that for the same rows it gives
the same report a SQLite-backed gateway gives — which is the whole claim.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from llm_rosetta.observability import PersistenceManager
from llm_rosetta.observability.usage import (
    LEDGER_COLUMNS,
    USAGE_WINDOW_LABELS,
    UsageLedger,
    UsageReport,
    usage_columns,
)
from llm_rosetta.observability.usage.report import NO_SUBJECT_LABEL


def _stamp(**kwargs) -> tuple[str, float]:
    """One instant, as the stored ledger writes it and as the clock reads it."""
    moment = datetime.now(timezone.utc) - timedelta(**kwargs)
    return moment.isoformat(), moment.timestamp()


class _FixedClock:
    """A clock that only moves when a test moves it."""

    def __init__(self, started_at: float = 1_800_000_000.0) -> None:
        self.now = started_at

    def __call__(self) -> float:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs).total_seconds()


class TestRecording:
    def test_a_usage_dict_with_no_token_count_records_nothing(self):
        ledger = UsageLedger()
        assert ledger.record(None) is None
        assert ledger.record({}) is None
        assert ledger.record({"model": "gpt-5-nano"}) is None
        assert len(ledger) == 0

    def test_a_recorded_usage_becomes_the_ledgers_columns(self):
        ledger = UsageLedger()
        ledger.record({"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 50})
        counters = ledger.report().total("24h")
        assert counters["input_tokens"] == 40
        assert counters["output_tokens"] == 5
        assert counters["total_tokens"] == 50
        assert counters["request_count"] == 1

    def test_the_canonical_total_is_filled_when_the_provider_sends_none(self):
        """A Messages payload has no total, and its cache tokens are additive."""
        ledger = UsageLedger()
        ledger.record(
            {
                "prompt_tokens": 40,
                "completion_tokens": 5,
                "cache_read_tokens": 20,
                "cache_creation_tokens": 2,
            }
        )
        counters = ledger.report().total("24h")
        assert counters["cache_read_tokens"] == 20
        assert counters["cache_creation_tokens"] == 2
        assert counters["total_tokens"] == 67

    def test_a_counter_the_usage_did_not_name_stays_zero(self):
        ledger = UsageLedger()
        ledger.record({"prompt_tokens": 1, "completion_tokens": 1})
        counters = ledger.report().total("24h")
        assert counters["cache_read_tokens"] == 0
        assert counters["reasoning_tokens"] == 0

    def test_records_carry_the_model_and_its_type(self):
        ledger = UsageLedger()
        ledger.record(
            {"prompt_tokens": 1},
            model="text-embedding-3-small",
            model_type="embedding",
        )
        (record,) = ledger.records()
        assert record.model == "text-embedding-3-small"
        assert record.model_type == "embedding"

    def test_the_model_type_defaults_to_llm(self):
        ledger = UsageLedger()
        ledger.record({"prompt_tokens": 1}, model="gpt-5-nano")
        assert ledger.records()[0].model_type == "llm"

    def test_clearing_forgets_everything(self):
        ledger = UsageLedger()
        ledger.record({"prompt_tokens": 1})
        assert len(ledger) == 1
        ledger.clear()
        assert len(ledger) == 0
        assert ledger.report().total("24h")["total_tokens"] == 0


class TestWindows:
    def test_a_record_ages_out_of_the_narrow_windows(self):
        clock = _FixedClock()
        ledger = UsageLedger(clock=clock)
        ledger.record({"total_tokens": 10})
        clock.advance(hours=6)

        assert ledger.report().total("5h")["total_tokens"] == 0
        assert ledger.report().total("24h")["total_tokens"] == 10
        assert ledger.report().total("30d")["total_tokens"] == 10

    def test_a_record_can_be_dated_when_it_is_written(self):
        """Replaying a log is the reason the ledger takes a time at all."""
        clock = _FixedClock()
        ledger = UsageLedger(clock=clock)
        ledger.record(
            {"total_tokens": 10}, at=clock.now - timedelta(days=3).total_seconds()
        )
        ledger.record({"total_tokens": 1})

        assert ledger.report().total("24h")["total_tokens"] == 1
        assert ledger.report().total("7d")["total_tokens"] == 11

    def test_a_subject_older_than_the_widest_window_is_absent(self):
        """The stored reader leaves it out, so this one does too."""
        clock = _FixedClock()
        ledger = UsageLedger(clock=clock)
        ledger.record(
            {"total_tokens": 10},
            subject="key-1",
            at=clock.now - timedelta(days=40).total_seconds(),
        )
        assert ledger.groups() == []
        # The record is still held: it is the *window* that has nothing to say.
        assert len(ledger) == 1

    def test_every_window_is_measured_from_now(self):
        clock = _FixedClock()
        ledger = UsageLedger(clock=clock)
        for days, tokens in ((0, 1), (2, 2), (10, 4), (40, 8)):
            ledger.record(
                {"total_tokens": tokens},
                at=clock.now - timedelta(days=days).total_seconds(),
            )
        counters = ledger.report().total
        assert counters("5h")["total_tokens"] == 1
        assert counters("24h")["total_tokens"] == 1
        assert counters("7d")["total_tokens"] == 3
        assert counters("30d")["total_tokens"] == 7


class TestGroups:
    def test_records_are_summed_per_subject_and_label(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 3}, subject="key-1", label="prod")
        ledger.record({"total_tokens": 4}, subject="key-1", label="prod")
        ledger.record({"total_tokens": 5}, subject="key-2", label="dev")

        groups = {group["bucket"]: group for group in ledger.groups()}
        assert set(groups) == {"key-1", "key-2"}
        assert groups["key-1"]["windows"]["24h"]["total_tokens"] == 7
        assert groups["key-1"]["windows"]["24h"]["request_count"] == 2
        assert groups["key-1"]["label"] == "prod"
        assert groups["key-2"]["windows"]["24h"]["total_tokens"] == 5

    def test_rows_with_no_subject_are_grouped_under_no_bucket(self):
        """Not dropped: the dashboard counts them, so the report must."""
        ledger = UsageLedger()
        ledger.record({"total_tokens": 6})
        (group,) = ledger.groups()
        assert group["bucket"] is None
        assert group["label"] is None

    def test_a_group_carries_every_window(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 1}, subject="key-1")
        (group,) = ledger.groups()
        assert tuple(group["windows"]) == USAGE_WINDOW_LABELS

    def test_a_group_carries_every_ledger_column(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 1}, subject="key-1")
        (group,) = ledger.groups()
        for name in USAGE_WINDOW_LABELS:
            assert set(LEDGER_COLUMNS) <= set(group["windows"][name])


class TestReport:
    def test_a_ledger_with_no_subjects_still_reports_every_token(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 7})
        ledger.record({"total_tokens": 8})

        report = ledger.report()
        assert report.by_subject == {}
        assert report.total("24h")["total_tokens"] == 15

    def test_the_report_attributes_the_subjects_the_ledger_has_seen(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 7}, subject="key-1", label="prod")
        ledger.record({"total_tokens": 8})

        report = ledger.report()
        assert list(report.by_subject) == ["key-1"]
        assert report.attributed("key-1", "24h")["total_tokens"] == 7
        assert report.total("24h")["total_tokens"] == 15

    def test_the_caller_can_name_the_subjects_instead(self):
        """A ledger may have recorded traffic for a subject it does not own."""
        ledger = UsageLedger()
        ledger.record({"total_tokens": 7}, subject="key-1", label="prod")

        report = ledger.report(subjects=[("somebody-else", "theirs")])
        assert report.attributed("somebody-else", "24h")["total_tokens"] == 0
        assert report.total("24h")["total_tokens"] == 7
        assert report.unattributed is not None

    def test_internal_traffic_is_kept_apart(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 7}, subject="internal", label="internal")
        ledger.record({"total_tokens": 8}, subject="key-1", label="prod")

        report = ledger.report(
            subjects=[("key-1", "prod")],
            internal_subject="internal",
            internal_label="internal",
        )
        assert report.internal is not None
        assert report.internal["24h"]["total_tokens"] == 7
        assert report.attributed("key-1", "24h")["total_tokens"] == 8
        assert report.total("24h")["total_tokens"] == 15

    def test_the_total_is_every_class_at_once(self):
        """The parts add up to the whole, whatever class each row fell in."""
        ledger = UsageLedger()
        ledger.record({"total_tokens": 1}, subject="key-1", label="prod")
        ledger.record({"total_tokens": 2}, subject="internal", label="internal")
        ledger.record({"total_tokens": 4}, subject="gone", label="deleted")
        ledger.record({"total_tokens": 8})

        report = ledger.report(
            subjects=[("key-1", "prod")],
            internal_subject="internal",
            internal_label="internal",
        )
        assert (
            sum(part["total_tokens"] for part in report.window_parts("24h"))
            == report.total("24h")["total_tokens"]
            == 15
        )

    def test_the_payload_names_subjects_the_way_the_caller_does(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 5}, subject="bucket-1", label="prod")
        report = ledger.report(subjects=[("bucket-1", "prod")])
        payload = report.to_payload({"bucket-1": "key-id-1"})
        assert list(payload["keys"]) == ["key-id-1"]
        assert payload["keys"]["key-id-1"]["label"] == "prod"


class TestTheInMemoryLedgerIsTheStoredLedger:
    """Same rows, same report — the two ledgers are one subsystem."""

    @staticmethod
    def _rows(
        rng: random.Random,
    ) -> list[tuple[str, float, dict, str | None, str | None]]:
        """Rows: written-at (both spellings), usage, subject, label."""
        rows = []
        for i in range(rng.randint(1, 14)):
            kind = rng.choice(["key", "key", "internal", "orphan", "none"])
            subject = {
                "key": f"key-{i % 3}",
                "internal": "internal",
                "orphan": f"orphan-{i}",
                "none": None,
            }[kind]
            label = {
                "key": f"label-{i % 3}",
                "internal": "internal",
                "orphan": "gone",
                "none": None,
            }[kind]
            usage: dict = {
                "prompt_tokens": rng.randint(0, 200),
                "completion_tokens": rng.randint(0, 100),
            }
            if rng.random() < 0.5:
                usage["cache_read_tokens"] = rng.randint(0, 50)
                usage["reasoning_tokens"] = rng.randint(0, 20)
            if rng.random() < 0.3:
                # A provider total, which the canonical total must prefer.
                usage["total_tokens"] = sum(usage.values())
            when = _stamp(minutes=rng.choice([1, 30, 240, 4000, 20000]))
            rows.append((*when, usage, subject, label))
        return rows

    @pytest.mark.parametrize("seed", range(25))
    async def test_the_same_rows_give_the_same_report(self, tmp_path, seed):
        rng = random.Random(seed)
        rows = self._rows(rng)

        stored = await PersistenceManager.create(str(tmp_path))
        ledger = UsageLedger()
        try:
            await stored.insert_log_entries(
                [
                    {
                        "id": f"row-{i}",
                        "timestamp": stamp,
                        "model": "m",
                        "source_provider": "chat_completions",
                        "target_provider": "chat_completions",
                        "is_stream": False,
                        "status_code": 200,
                        "duration_ms": 1.0,
                        "api_key_bucket": subject,
                        "api_key_label": label,
                        **usage_columns(usage),
                    }
                    for i, (stamp, _at, usage, subject, label) in enumerate(rows)
                ]
            )
            for _stamp_text, at, usage, subject, label in rows:
                ledger.record(usage, subject=subject, label=label, at=at)

            subjects = [
                ("key-0", "label-0"),
                ("key-1", "label-1"),
                ("key-2", "label-2"),
            ]
            from_stored = UsageReport.build(
                groups=(await stored.query_usage_groups())["groups"],
                subjects=subjects,
                internal_subject="internal",
                internal_label="internal",
            )
        finally:
            await stored.close()

        from_memory = ledger.report(
            subjects=subjects,
            internal_subject="internal",
            internal_label="internal",
        )

        for window in USAGE_WINDOW_LABELS:
            assert from_memory.total(window) == from_stored.total(window), window
        assert from_memory.to_payload() == from_stored.to_payload()

    @pytest.mark.parametrize("seed", range(10))
    async def test_the_total_equals_the_rows_summed(self, tmp_path, seed):
        """The invariant, stated against the rows themselves rather than a peer."""
        rng = random.Random(seed)
        rows = self._rows(rng)
        ledger = UsageLedger()
        expected = {name: 0 for name in LEDGER_COLUMNS}
        for _stamp_text, at, usage, subject, label in rows:
            ledger.record(usage, subject=subject, label=label, at=at)
            for name, value in usage_columns(usage).items():
                expected[name] += value

        total = ledger.report().total("30d")
        for name, value in expected.items():
            assert total[name] == value, name


class TestTheRecordsThatAreNotDroppedSilently:
    """A usage dict that measures as nothing has two causes, and one signal.

    A provider that reported no usage and a caller who misspelled a key look
    identical from the outside, so the drop is observable instead: an explicit
    zeros reading is a *record*, and a non-empty dict that yields no columns
    says so through the sink.
    """

    def test_explicit_zeros_are_a_request_and_are_recorded(self):
        ledger = UsageLedger()
        record = ledger.record({"prompt_tokens": 0, "completion_tokens": 0})
        assert record is not None
        assert record.columns == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        total = ledger.total("24h")
        assert total["request_count"] == 1
        assert total["total_tokens"] == 0

    def test_a_usage_that_measures_as_nothing_says_so(self):
        ledger = UsageLedger()
        warnings: list[str] = []
        assert (
            ledger.record({"prompt_token": 5}, model="gpt-5-nano", warnings=warnings)
            is None
        )
        assert len(warnings) == 1
        assert "gpt-5-nano" in warnings[0]
        assert "prompt_token" in warnings[0]
        assert "not recorded" in warnings[0]

    def test_an_empty_usage_is_not_a_mistake_and_is_not_reported(self):
        """A response that carried no usage at all is not a caller error."""
        ledger = UsageLedger()
        warnings: list[str] = []
        assert ledger.record({}, warnings=warnings) is None
        assert ledger.record(None, warnings=warnings) is None
        assert warnings == []

    def test_a_count_that_arrived_as_a_string_is_the_count_it_spells(self):
        """A provider that JSON-strings its numbers still sent counts."""
        ledger = UsageLedger()
        warnings: list[str] = []
        assert ledger.record({"prompt_tokens": "10"}, warnings=warnings) is not None
        assert warnings == []
        assert ledger.total("24h")["input_tokens"] == 10

    def test_a_provider_total_that_arrived_as_a_string_still_wins(self):
        """The #1014 shape: adding cache on top of a totalled prompt.

        Read as "not a count" and recomputed, the cached tokens — which this
        provider had already counted into its total — would be added again.
        """
        ledger = UsageLedger()
        ledger.record(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "cache_read_tokens": 800,
                "total_tokens": "1050",
            }
        )
        assert ledger.total("24h")["total_tokens"] == 1050
        assert ledger.total("24h")["cache_read_tokens"] == 800

    def test_a_provider_total_on_its_own_is_a_total(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": "1050"})
        assert ledger.total("24h")["total_tokens"] == 1050

    def test_a_usage_that_names_an_uncountable_count_is_refused_whole(self):
        """Not measured in part: a half-sum is a quietly short total."""
        ledger = UsageLedger()
        warnings: list[str] = []
        assert (
            ledger.record(
                {"prompt_tokens": "ten", "completion_tokens": 5}, warnings=warnings
            )
            is None
        )
        assert len(warnings) == 1
        assert ledger.total("24h")["request_count"] == 0
        # The countable half of that same payload is not written either: a row
        # reporting output tokens with no total is a breakdown its total never
        # had, and the report would sum it as zero on one side and show it on
        # the other.
        assert ledger.total("24h")["output_tokens"] == 0

    def test_a_payload_with_no_additive_counter_at_all_is_not_a_refusal(self):
        """A response that reported only reasoning is not a malformed one."""
        ledger = UsageLedger()
        warnings: list[str] = []
        assert ledger.record({"reasoning_tokens": 5}, warnings=warnings) is not None
        assert warnings == []
        total = ledger.total("24h")
        assert total["reasoning_tokens"] == 5
        # Reasoning rides inside the output, which nobody reported, so it adds
        # to no total — the asymmetry the declaration already states.
        assert total["total_tokens"] == 0


class TestNamingTheSubjects:
    """``subjects=`` takes names, and says so when it is given something else."""

    def test_a_name_on_its_own_is_its_own_label(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 5}, subject="batch-7")
        report = ledger.report(subjects=["batch-7"])
        assert report.attributed("batch-7", "24h")["total_tokens"] == 5
        assert report.total("24h")["total_tokens"] == 5

    def test_a_pair_carries_the_display_name(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 5}, subject="a-bucket", label="prod")
        report = ledger.report(subjects=[("a-bucket", "prod")])
        assert report.by_subject["a-bucket"].label == "prod"

    def test_the_two_forms_may_be_mixed(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 1}, subject="one")
        ledger.record({"total_tokens": 2}, subject="two")
        report = ledger.report(subjects=["one", ("two", "Second")])
        assert set(report.by_subject) == {"one", "two"}
        assert report.total("24h")["total_tokens"] == 3

    def test_something_else_is_refused_with_the_shape_spelled_out(self):
        ledger = UsageLedger()
        # A caller who got the shape wrong is the case this covers, so the
        # wrong shape has to be passed on purpose.
        malformed: Any = [("one", "two", "three")]
        with pytest.raises(ValueError, match="a subject is a name or a"):
            ledger.report(subjects=malformed)

    def test_a_record_with_no_label_reads_the_same_way_twice(self):
        """The two readings agree on everything a consumer can compare.

        The label differs on purpose — the report needs something to display,
        and the groups keep what the row carried, which is what the stored
        ledger's reader reports for the same rows.
        """
        ledger = UsageLedger()
        ledger.record({"total_tokens": 5}, subject="batch-7")
        (group,) = ledger.groups()
        assert group["label"] is None
        assert ledger.subjects() == [("batch-7", "batch-7")]
        assert ledger.report().attributed("batch-7", "24h")["total_tokens"] == 5


class TestTheRemainderIsNamed:
    def test_rows_with_no_label_are_filed_under_a_readable_name(self):
        ledger = UsageLedger()
        ledger.record({"total_tokens": 5})
        report = ledger.report()
        assert NO_SUBJECT_LABEL in report.unattributed_by_label


class TestTheDocumentedRecipe:
    """The example in the API reference.  A recipe that does not run is a lie."""

    def test_measuring_usage_without_a_gateway(self):
        resp = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        }

        ledger = UsageLedger()
        ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")
        ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")

        total = ledger.total("24h")
        assert total["request_count"] == 2
        assert total["input_tokens"] == 40
        assert total["output_tokens"] == 10
        assert total["total_tokens"] == 50

    def test_the_documented_report_attributes_a_subject(self):
        ledger = UsageLedger()
        ledger.record({"prompt_tokens": 3, "completion_tokens": 1}, subject="k1")
        report = ledger.report(subjects=[("k1", "batch")])
        total = report.total("24h")
        assert total["input_tokens"] == 3
        assert report.attributed("k1", "24h")["total_tokens"] == 4

"""Tests for the usage metering core.

These are library-level tests: nothing here imports the gateway, and the last
class asserts that nothing in the package does.
"""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import llm_rosetta.observability as observability
from llm_rosetta.observability import PersistenceManager
from llm_rosetta.observability.usage import (
    ADDITIVE_CACHE_STANDARDS,
    GOOGLE_METADATA,
    MESSAGES_FIELDS,
    OPENAI_DETAILS,
    STANDARD_SHAPES,
    USAGE_SHAPES,
    USAGE_WINDOW_LABELS,
    UsageReport,
    cache_is_additive,
    token_total,
    usage_columns,
)
from llm_rosetta.observability.usage.normalize import (
    DEFAULT_MODEL_TYPE,
    MODEL_TYPES,
    MODEL_TYPE_FIELDS,
)
from llm_rosetta.observability.usage.report import NO_SUBJECT_LABEL

LEDGER = ("input_tokens", "output_tokens", "total_tokens")


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


def _windows(**counters) -> dict[str, dict[str, int]]:
    """One row's worth of windows, every window alike."""
    window = {
        "request_count": 1,
        **{name: counters.get(name, 0) for name in observability.LEDGER_COLUMNS},
    }
    return dict.fromkeys(USAGE_WINDOW_LABELS, dict(window))


def _group(bucket, label, model_type=DEFAULT_MODEL_TYPE, **counters) -> dict:
    return {
        "bucket": bucket,
        "label": label,
        "model_type": model_type,
        "windows": _windows(**counters),
    }


class TestTokenTotal:
    """The one rule that fills `total_tokens`, wherever usage comes from."""

    def test_provider_total_wins(self):
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": 5,
            "total_tokens": 50,
            "cache_read_tokens": 20,
        }
        assert token_total(usage) == 50

    def test_falls_back_to_the_additive_sum(self):
        """The Messages API sends no total, and its cache tokens are additive."""
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": 5,
            "cache_read_tokens": 20,
            "cache_creation_tokens": 2,
        }
        assert token_total(usage) == 67

    def test_reasoning_is_never_added(self):
        """It rides inside the output tokens."""
        assert (
            token_total(
                {"prompt_tokens": 10, "completion_tokens": 50, "reasoning_tokens": 40}
            )
            == 60
        )

    def test_no_usage_at_all(self):
        assert token_total(None) is None
        assert token_total({}) is None
        assert token_total({"model": "gpt-4o"}) is None

    def test_partial_usage_still_totals(self):
        assert token_total({"completion_tokens": 5}) == 5

    def test_a_count_that_arrived_as_a_string_is_the_count_it_spells(self):
        """A provider that JSON-strings its numbers still sent counts."""
        assert token_total({"prompt_tokens": "10", "completion_tokens": "5"}) == 15
        assert token_total({"prompt_tokens": "10", "completion_tokens": 5}) == 15

    def test_a_string_total_is_the_total_the_provider_sent(self):
        """The canonical rule with the cache counters on both sides of it.

        Read as "not a count" this total would be recomputed additively, and
        the cached tokens — which this provider had already counted into its
        total — would be added again: 1050 becomes 1850.  That is the shape
        #1014 was about, so the provider's own total has to win however it
        arrived.
        """
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cache_read_tokens": 800,
            "total_tokens": "1050",
        }
        assert token_total(usage) == 1050
        assert usage_columns(usage)["total_tokens"] == 1050
        assert token_total({"total_tokens": "1050"}) == 1050

    def test_a_counter_that_is_not_a_count_refuses_the_whole_usage(self):
        """Skipping it would report a total that is quietly short."""
        for usage in (
            {"prompt_tokens": "ten", "completion_tokens": 5},
            {"prompt_tokens": 10.5},
            {"prompt_tokens": True, "completion_tokens": 5},
            {"cache_read_tokens": {"nested": 1}, "prompt_tokens": 10},
        ):
            assert token_total(usage) is None, usage
            assert usage_columns(usage) == {}, usage
        # Nothing was declared, so there is nothing to refuse.
        assert token_total({"reasoning_tokens": 5}) is None
        assert usage_columns({"reasoning_tokens": 5}) == {"reasoning_tokens": 5}


class TestUsageColumns:
    def test_columns_are_named_as_the_ledger_expects(self):
        columns = usage_columns(
            {
                "prompt_tokens": 40,
                "completion_tokens": 5,
                "cache_read_tokens": 20,
                "reasoning_tokens": 3,
            }
        )
        assert columns == {
            "input_tokens": 40,
            "output_tokens": 5,
            "cache_read_tokens": 20,
            "reasoning_tokens": 3,
            "total_tokens": 65,
        }

    def test_a_column_nobody_reported_is_absent_not_zero(self):
        """ "No cache in play" and "cache reported as zero" are different rows."""
        columns = usage_columns({"prompt_tokens": 1, "completion_tokens": 2})
        assert "cache_read_tokens" not in columns

    def test_no_usage_is_no_columns(self):
        assert usage_columns(None) == {}
        assert usage_columns({}) == {}


class TestEachModelTypeNormalizesFromItsOwnIR:
    """One field map per family, read from the IR type that family declares.

    Chat asks the provider for ``prompt_tokens`` and ``completion_tokens``; the
    decision IR already reports ``input_tokens``; embedding and rerank have no
    output side at all.  A map that used chat's names for every family would
    read zero out of three of them, and do it silently.
    """

    def test_chat_is_unchanged(self):
        assert usage_columns(
            {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45}
        ) == {"input_tokens": 40, "output_tokens": 5, "total_tokens": 45}

    def test_embedding(self):
        """``EmbeddingUsageInfo``: ``total_tokens``, ``prompt_tokens``."""
        assert usage_columns({"prompt_tokens": 5, "total_tokens": 5}, "embedding") == {
            "input_tokens": 5,
            "total_tokens": 5,
        }
        # Voyage reports only the total; the input column stays absent.
        assert usage_columns({"total_tokens": 32}, "embedding") == {"total_tokens": 32}

    def test_rerank(self):
        """``RerankUsageInfo``: ``total_tokens``, ``prompt_tokens``, ``cached_tokens``."""
        assert usage_columns(
            {"prompt_tokens": 12, "total_tokens": 12, "cached_tokens": 4}, "rerank"
        ) == {"input_tokens": 12, "total_tokens": 12, "cache_read_tokens": 4}

    def test_decision(self):
        """``DecisionUsageInfo``: ``input_tokens`` and ``output_tokens`` only."""
        assert usage_columns({"input_tokens": 10, "output_tokens": 5}, "decision") == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_token_total_reads_the_familys_own_fields(self):
        assert token_total({"prompt_tokens": 10, "completion_tokens": 5}) == 15
        assert token_total({"prompt_tokens": 10}, "embedding") == 10
        assert token_total({"prompt_tokens": 7}, "rerank") == 7
        assert token_total({"input_tokens": 10, "output_tokens": 5}, "decision") == 15

    def test_the_canonical_rule_is_the_same_for_every_family(self):
        """The provider's own total wins wherever the family declares one."""
        assert token_total({"prompt_tokens": 4, "total_tokens": 9}, "embedding") == 9
        assert (
            token_total(
                {"prompt_tokens": 4, "cached_tokens": 2, "total_tokens": 9}, "rerank"
            )
            == 9
        )
        # The decision IR declares no total, so the additive rule always runs.
        assert token_total({"input_tokens": 4, "output_tokens": 1}, "decision") == 5

    def test_a_family_that_reports_nothing_totals_nothing(self):
        assert token_total({"model": "text-embedding-3"}, "embedding") is None
        assert usage_columns({"model": "text-embedding-3"}, "embedding") == {}

    def test_an_unknown_model_type_is_refused_not_guessed(self):
        """A writer's typo must not quietly record chat's field names."""
        with pytest.raises(KeyError):
            usage_columns({"prompt_tokens": 1}, "familly")
        with pytest.raises(KeyError):
            token_total({"prompt_tokens": 1}, "familly")


class TestTheModelTypesAreTheRegistrys:
    """The column's values are the gateway registry's canonical names.

    ``gateway/model_types.py`` owns the vocabulary and this package may not
    import it, so the names are restated in ``normalize`` and pinned here —
    both as literals, which say what the vocabulary is, and against the live
    registry, which fails on a rename on either side.
    """

    def test_the_canonical_names(self):
        assert MODEL_TYPES == ("llm", "embedding", "rerank", "decision")
        assert DEFAULT_MODEL_TYPE == "llm"

    def test_the_registry_registers_exactly_these(self):
        """A rename on either side must break a test, not a report."""
        from llm_rosetta.gateway.model_types import MODEL_TYPE_REGISTRY

        assert set(MODEL_TYPE_REGISTRY) == set(MODEL_TYPES)
        # The default is a registered name too, or every row without one would
        # be written with a type no descriptor answers to.
        assert DEFAULT_MODEL_TYPE in MODEL_TYPE_REGISTRY

    def test_every_type_declares_a_field_map(self):
        assert set(MODEL_TYPE_FIELDS) == set(MODEL_TYPES)
        for model_type, fields in MODEL_TYPE_FIELDS.items():
            # A family with no input side could not report usage at all.
            assert "input_tokens" in fields, model_type

    def test_a_maps_columns_are_ledger_columns(self):
        for fields in MODEL_TYPE_FIELDS.values():
            assert set(fields) <= set(observability.LEDGER_COLUMNS)
            assert all(field for field in fields.values())


class TestSemantics:
    """The declaration is the only encoding of either fact."""

    def test_every_standard_resolves_to_a_declared_shape(self):
        for standard, shape in STANDARD_SHAPES.items():
            assert shape in USAGE_SHAPES, f"{standard} -> {shape.name}"

    def test_additivity_is_derived_not_restated(self):
        derived = {s for s, shape in STANDARD_SHAPES.items() if shape.cache_additive}
        assert ADDITIVE_CACHE_STANDARDS == derived
        assert {"messages", "anthropic"} <= derived

    def test_an_unknown_standard_claims_nothing(self):
        assert cache_is_additive("google_interactions") is False
        assert cache_is_additive("no-such-standard") is False

    def test_only_the_messages_shape_is_additive(self):
        assert MESSAGES_FIELDS.cache_additive is True
        assert OPENAI_DETAILS.cache_additive is False
        assert GOOGLE_METADATA.cache_additive is False

    def test_a_shape_reads_only_the_counters_it_declares(self):
        assert OPENAI_DETAILS.extended(
            {
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens_details": {"reasoning_tokens": 3},
                "cache_read_input_tokens": 99,
            }
        ) == {"cache_read_tokens": 7, "reasoning_tokens": 3}

    def test_google_accepts_either_spelling(self):
        assert GOOGLE_METADATA.extended({"cachedContentTokenCount": 5}) == {
            "cache_read_tokens": 5
        }
        assert GOOGLE_METADATA.extended({"cached_content_token_count": 5}) == {
            "cache_read_tokens": 5
        }

    def test_a_negative_or_missing_counter_is_not_a_counter(self):
        assert (
            OPENAI_DETAILS.extended({"prompt_tokens_details": {"cached_tokens": -1}})
            == {}
        )
        assert OPENAI_DETAILS.extended({"prompt_tokens_details": None}) == {}

    def test_shapes_are_declared_for_every_family_a_writer_may_meet(self):
        """A shape with no field is a declaration that claims nothing."""
        for shape in USAGE_SHAPES:
            assert shape.cache_read or shape.reasoning or shape.cache_creation


class TestConvertersAgreeWithTheDeclaration:
    """The drift guard the two-encoding bug would have tripped.

    A converter normalizes its own provider's usage, and the ledger derives a
    total from what it produced.  If a converter starts reporting a cache
    counter that the declaration calls a subset as if it were additive — or the
    declaration flips without the converter — the totals silently change.  So:
    for every standard whose converter reports a cache counter, the shape the
    declaration names must actually read that counter.
    """

    @pytest.mark.parametrize(
        "standard, payload",
        [
            (
                "chat_completions",
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "total_tokens": 1050,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            ),
            (
                "messages",
                {
                    "input_tokens": 40,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 2,
                },
            ),
            (
                "google_generate",
                {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 110,
                    "cachedContentTokenCount": 60,
                },
            ),
        ],
    )
    def test_the_declared_shape_reads_the_provider_cache_counter(
        self, standard, payload
    ):
        from llm_rosetta.auto_detect import get_converter_for_provider

        converter = get_converter_for_provider(standard)
        ir_usage = converter._build_p_usage_to_ir(payload)
        declared = STANDARD_SHAPES[standard]

        from_declaration = declared.extended(payload)
        assert "cache_read_tokens" in from_declaration, (
            f"{standard}'s shape ({declared.name}) reads no cache counter, but "
            f"its converter reported {ir_usage.get('cache_read_tokens')}"
        )
        assert "cache_read_tokens" in ir_usage
        assert declared.cache_additive == cache_is_additive(standard)


class TestUsageReport:
    def test_no_subjects_means_everything_is_unattributed(self):
        """A library user with no credentials still gets a correct answer."""
        report = UsageReport.build(
            groups=[_group("b1", "someone", input_tokens=4, output_tokens=6)],
        )
        assert report.by_subject == {}
        assert report.unattributed is not None
        assert report.total("24h")["input_tokens"] == 4

    def test_a_live_subject_is_attributed_and_zero_filled(self):
        report = UsageReport.build(
            groups=[_group("b1", "k", total_tokens=10)],
            subjects=[("b1", "k"), ("b2", "idle")],
        )
        assert report.attributed("b1", "24h")["total_tokens"] == 10
        assert report.attributed("b2", "24h")["total_tokens"] == 0
        assert report.unattributed is None

    def test_the_remainder_keeps_its_label_split(self):
        report = UsageReport.build(
            groups=[
                _group("gone", "deleted", total_tokens=70),
                _group(None, None, total_tokens=30),
            ],
        )
        assert report.unattributed is not None
        assert report.unattributed["24h"]["total_tokens"] == 100
        assert set(report.unattributed_by_label) == {"deleted", NO_SUBJECT_LABEL}

    def test_internal_traffic_is_separate(self):
        report = UsageReport.build(
            groups=[
                _group("internal", "internal", total_tokens=25),
                _group(None, "internal", total_tokens=5),
                _group(None, None, total_tokens=1),
            ],
            internal_subject="internal",
            internal_label="internal",
        )
        assert report.internal is not None
        assert report.internal["24h"]["total_tokens"] == 30
        assert report.unattributed is not None
        assert report.unattributed["24h"]["total_tokens"] == 1

    def test_a_live_subject_owns_the_internal_label(self):
        """Rows labelled `internal` predate the sentinel: they belong to it."""
        report = UsageReport.build(
            groups=[_group(None, "internal", total_tokens=9)],
            subjects=[("b1", "internal")],
            internal_subject="internal",
            internal_label="internal",
        )
        assert report.internal is None  # never adopted the legacy rows
        assert report.attributed("b1", "24h")["total_tokens"] == 0
        assert report.unattributed is not None
        assert report.unattributed["24h"]["total_tokens"] == 9

    def test_a_shared_label_is_flagged(self):
        report = UsageReport.build(
            groups=[],
            subjects=[("b1", "dup"), ("b2", "dup"), ("b3", "unique")],
        )
        assert report.by_subject["b1"].shared_label is True
        assert report.by_subject["b2"].shared_label is True
        assert report.by_subject["b3"].shared_label is False

    def test_payload_files_subjects_under_their_own_name(self):
        report = UsageReport.build(
            groups=[_group("b1", "k", total_tokens=10)],
            subjects=[("b1", "k")],
        )
        payload = report.to_payload()
        assert payload["windows"] == list(USAGE_WINDOW_LABELS)
        assert payload["keys"]["b1"]["label"] == "k"
        assert "unattributed" not in payload

    def test_payload_files_subjects_under_the_callers_names(self):
        report = UsageReport.build(
            groups=[_group("b1", "k", total_tokens=10)],
            subjects=[("b1", "k")],
        )
        payload = report.to_payload({"b1": "key-id"})
        assert list(payload["keys"]) == ["key-id"]

    def test_an_empty_window_is_counted_as_zero_not_dropped(self):
        report = UsageReport.build(
            groups=[_group("b1", "k", total_tokens=10)], subjects=[("b1", "k")]
        )
        assert report.total("5h")["total_tokens"] == 10
        assert report.total("30d")["total_tokens"] == 10


class TestTheReportBreaksDownByModelType:
    """The second cut over the same rows: what kind of call spent the tokens."""

    def test_each_type_is_reported_on_its_own(self):
        report = UsageReport.build(
            groups=[
                _group("b1", "k", model_type="llm", total_tokens=100),
                _group("b1", "k", model_type="embedding", total_tokens=10),
                _group("b1", "k", model_type="rerank", total_tokens=5),
                _group("b1", "k", model_type="decision", total_tokens=1),
            ],
            subjects=[("b1", "k")],
        )
        assert set(report.by_model_type) == set(MODEL_TYPES)
        assert report.for_model_type("embedding", "24h")["total_tokens"] == 10
        # The cut is orthogonal to the classes: the subject's own total is every
        # type summed, and so is the report's.
        assert report.attributed("b1", "24h")["total_tokens"] == 116
        assert report.total("24h")["total_tokens"] == 116
        assert report.total("24h")["request_count"] == 4

    def test_a_type_with_no_traffic_is_absent_not_a_measured_zero(self):
        report = UsageReport.build(groups=[_group(None, None, total_tokens=5)])
        assert set(report.by_model_type) == {"llm"}
        assert report.for_model_type("embedding", "24h")["total_tokens"] == 0

    def test_a_group_that_names_no_type_counts_as_chat(self):
        """A library user's groups predate the column, or never had one."""
        report = UsageReport.build(
            groups=[
                {"bucket": None, "label": None, "windows": _windows(total_tokens=3)}
            ]
        )
        assert set(report.by_model_type) == {DEFAULT_MODEL_TYPE}
        assert report.total("24h")["total_tokens"] == 3


class TestTheReportReconcilesWithTheLedger:
    """The invariant, over generated ledgers rather than picked fixtures.

    The dashboard sums the ledger's own column; the panel sums the report's
    classes.  They agree only if the report accounts for *every* row, which is
    what the remainder class is for — and that has to hold with every model
    type in the ledger, not just chat.
    """

    @staticmethod
    def _random_ledger(rng: random.Random) -> tuple[list[dict], list[tuple[str, str]]]:
        """A ledger and the subjects that may be attributed in it."""
        subjects = [(f"live{i}", f"key{i}") for i in range(rng.randint(0, 3))]
        if subjects and rng.random() < 0.3:
            # Two live keys sharing a label: neither owns its traffic.
            subjects.append(("live-dup", subjects[0][1]))
        groups: list[dict] = []
        for i in range(rng.randint(0, 12)):
            kind = rng.choice(["live", "orphan", "none", "internal"])
            if kind == "live" and subjects:
                bucket, label = rng.choice(subjects)
            elif kind == "internal":
                bucket, label = "internal", rng.choice(["internal", None])
            elif kind == "orphan":
                bucket, label = f"orphan{i}", rng.choice([f"gone{i}", "key0", None])
            else:
                bucket, label = None, rng.choice([None, f"gone{i}", "internal"])
            total = rng.randint(0, 500)
            groups.append(
                _group(
                    bucket,
                    label,
                    model_type=rng.choice(MODEL_TYPES),
                    input_tokens=total // 2,
                    output_tokens=total - total // 2,
                    total_tokens=total,
                )
            )
        return groups, subjects

    def test_the_generator_produces_every_model_type(self):
        """All four families are present, so the property actually exercises them."""
        seen: set[str] = set()
        for seed in range(25):
            groups, _subjects = self._random_ledger(random.Random(seed))
            seen |= {group["model_type"] for group in groups}
        assert seen == set(MODEL_TYPES)

    @pytest.mark.parametrize("seed", range(25))
    async def test_report_total_equals_the_ledger_sum(self, tmp_path, seed):
        rng = random.Random(seed)
        groups, subjects = self._random_ledger(rng)

        pm = await PersistenceManager.create(str(tmp_path))
        try:
            await pm.insert_log_entries(
                [
                    {
                        "id": f"row-{i}",
                        "timestamp": _ago(minutes=1),
                        "model": "m",
                        "source_provider": "chat_completions",
                        "target_provider": "chat_completions",
                        "is_stream": False,
                        "status_code": 200,
                        "duration_ms": 1.0,
                        "api_key_bucket": group["bucket"],
                        "api_key_label": group["label"],
                        "model_type": group["model_type"],
                        "input_tokens": group["windows"]["24h"]["input_tokens"],
                        "output_tokens": group["windows"]["24h"]["output_tokens"],
                        "total_tokens": group["windows"]["24h"]["total_tokens"],
                    }
                    for i, group in enumerate(groups)
                ]
            )
            read_back = (await pm.query_usage_groups())["groups"]
            report = UsageReport.build(
                groups=read_back,
                subjects=subjects,
                internal_subject="internal",
                internal_label="internal",
            )
            ledger = await pm.query_rolling_24h_tokens()
        finally:
            await pm.close()

        assert report.total("24h")["total_tokens"] == ledger["total_tokens_24h"]
        assert report.total("24h")["input_tokens"] == ledger["input_tokens_24h"]
        assert report.total("24h")["output_tokens"] == ledger["output_tokens_24h"]
        # And the read-back rows kept their type, so the breakdown is the
        # ledger's own split rather than a merge that happens to sum right.
        assert set(report.by_model_type) == {g["model_type"] for g in read_back}

    @pytest.mark.parametrize("seed", range(25))
    def test_every_row_lands_in_exactly_one_class(self, seed):
        rng = random.Random(seed)
        groups, subjects = self._random_ledger(rng)
        report = UsageReport.build(
            groups=groups,
            subjects=subjects,
            internal_subject="internal",
            internal_label="internal",
        )
        # total() sums the classes; it must also equal the rows fed in, so
        # nothing is double-counted and nothing is dropped.
        expected = sum(g["windows"]["24h"]["total_tokens"] for g in groups)
        assert report.total("24h")["total_tokens"] == expected


def _gateway_imports(tree: ast.Module, *, everywhere: bool) -> list[str]:
    """Module names the tree imports from the gateway.

    With ``everywhere=False`` only module-level statements are read, which is
    what decides whether importing the module pulls the gateway in; with
    ``everywhere=True`` a lazily imported one is a finding too.
    """
    nodes = ast.walk(tree) if everywhere else tree.body
    found: list[str] = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("llm_rosetta.gateway"):
                found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("llm_rosetta.gateway")
            )
    return found


class TestThePackageStandsAlone:
    """Usage must be meterable without a gateway — the point of the package."""

    @staticmethod
    def _usage_dir() -> Path:
        return Path(observability.__file__).parent / "usage"

    @staticmethod
    def _observability_dir() -> Path:
        return Path(observability.__file__).parent

    def test_no_usage_module_imports_the_gateway(self):
        """A static guard, so new code cannot quietly add the dependency."""
        offenders = [
            f"{path.name}: {module}"
            for path in self._usage_dir().rglob("*.py")
            for module in _gateway_imports(
                ast.parse(path.read_text(), filename=str(path)), everywhere=True
            )
        ]
        assert not offenders, f"usage must not depend on the gateway: {offenders}"

    def test_importing_observability_pulls_in_no_gateway_module(self):
        """The whole package, at module level.

        One function-local import exists in ``error_dump`` (credential
        scrubbing, which lives with the gateway's sanitizer) and is not a
        module-level dependency; a new one at module level would be.
        """
        offenders = [
            f"{path.name}: {module}"
            for path in self._observability_dir().glob("*.py")
            for module in _gateway_imports(
                ast.parse(path.read_text(), filename=str(path)), everywhere=False
            )
        ]
        assert not offenders, f"module-level gateway import: {offenders}"

    def test_importing_observability_does_not_pull_in_the_gateway(self):
        """Transitively too: a fresh interpreter that only meters usage."""
        src = str(self._observability_dir().parent.parent)
        code = (
            "import sys; import llm_rosetta.observability.usage as u; "
            "u.token_total({'prompt_tokens': 1}); "
            "print(any(m.split('.')[:2] == ['llm_rosetta', 'gateway'] "
            "for m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": src},
        )
        assert result.stdout.strip() == "False", result.stdout + result.stderr

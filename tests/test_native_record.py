"""One record for what a provider said in its own words.

The record replaced two keys with two shapes (``_native_finish_reason`` and
``_native_prompt_feedback``).  What is pinned here is the property that made
the replacement safe: a value produced by an earlier release reads the same as
one produced today, and a write only ever produces the new shape — otherwise
the library would keep two records forever.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.ir.primitives import (
    LEGACY_FINISH_REASON_KEY,
    LEGACY_PROMPT_FEEDBACK_KEY,
    NATIVE_RESPONSE_KEY,
    native_record,
    record_native,
)

ORIGIN = "google_generate"


class TestTheRecord:
    def test_it_is_written_under_one_key(self):
        metadata: dict[str, Any] = {}

        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        assert metadata == {
            NATIVE_RESPONSE_KEY: {"origin": ORIGIN, "finish_reason": "OTHER"}
        }
        assert LEGACY_FINISH_REASON_KEY not in metadata
        assert LEGACY_PROMPT_FEEDBACK_KEY not in metadata

    def test_two_fields_share_one_record(self):
        metadata: dict[str, Any] = {}

        record_native(metadata, origin=ORIGIN, finish_reason="SAFETY")
        record_native(
            metadata,
            origin=ORIGIN,
            prompt_feedback={"blockReason": "SAFETY"},
        )

        assert metadata[NATIVE_RESPONSE_KEY] == {
            "origin": ORIGIN,
            "finish_reason": "SAFETY",
            "prompt_feedback": {"blockReason": "SAFETY"},
        }

    def test_a_field_left_out_keeps_what_was_there(self):
        metadata: dict[str, Any] = {}
        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        record_native(metadata, origin=ORIGIN, prompt_feedback={"blockReason": "X"})

        record = native_record(metadata, origin=ORIGIN)
        assert record["finish_reason"] == "OTHER"
        assert record["prompt_feedback"] == {"blockReason": "X"}

    def test_it_is_read_back(self):
        metadata = {
            NATIVE_RESPONSE_KEY: {"origin": ORIGIN, "finish_reason": "LANGUAGE"}
        }

        assert native_record(metadata, origin=ORIGIN)["finish_reason"] == "LANGUAGE"

    def test_no_record_is_an_empty_read(self):
        assert native_record({}, origin=ORIGIN) == {}
        assert native_record(None, origin=ORIGIN) == {}
        assert native_record("not a mapping", origin=ORIGIN) == {}

    def test_a_record_holding_only_an_origin_reads_as_absent(self):
        """``origin`` names whose fact it would be, not a fact.

        An empty record is a real state — a write with neither field — and a
        caller using the record as an "is there a fact" test must see none.
        """
        metadata = {NATIVE_RESPONSE_KEY: {"origin": ORIGIN}}

        assert native_record(metadata, origin=ORIGIN) == {}

    def test_a_field_with_no_value_is_not_a_word(self):
        """A present-but-``None`` field must not read as a recorded word."""
        metadata = {NATIVE_RESPONSE_KEY: {"origin": ORIGIN, "finish_reason": None}}

        assert "finish_reason" not in native_record(metadata, origin=ORIGIN)

    def test_a_legacy_record_without_a_word_reads_as_absent(self):
        """The legacy fold carries a word only when the old shape had one."""
        metadata = {LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": None}}

        assert native_record(metadata, origin=ORIGIN) == {}

    def test_a_write_with_no_fact_leaves_no_record(self):
        """A write that adds nothing stores nothing — no origin-only record."""
        metadata: dict[str, Any] = {}

        written = record_native(metadata, origin=ORIGIN)

        assert written == {}
        assert NATIVE_RESPONSE_KEY not in metadata


class TestTheLegacyShapes:
    """A value from an earlier release converts the same as today's."""

    def test_the_finish_reason_key_still_reads(self):
        metadata = {LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "SPII"}}

        assert native_record(metadata, origin=ORIGIN) == {
            "origin": ORIGIN,
            "finish_reason": "SPII",
        }

    def test_the_prompt_feedback_key_still_reads(self):
        metadata = {
            LEGACY_PROMPT_FEEDBACK_KEY: {
                "base": ORIGIN,
                "prompt_feedback": {"blockReason": "OTHER"},
            }
        }

        assert native_record(metadata, origin=ORIGIN) == {
            "origin": ORIGIN,
            "prompt_feedback": {"blockReason": "OTHER"},
        }

    def test_both_legacy_keys_fold_into_one_record(self):
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
            LEGACY_PROMPT_FEEDBACK_KEY: {
                "base": ORIGIN,
                "prompt_feedback": {"blockReason": "OTHER"},
            },
        }

        record = native_record(metadata, origin=ORIGIN)

        assert record["finish_reason"] == "OTHER"
        assert record["prompt_feedback"] == {"blockReason": "OTHER"}

    def test_a_conversion_of_an_old_record_writes_the_new_shape(self):
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
        }

        record_native(metadata, origin=ORIGIN, prompt_feedback={"blockReason": "X"})

        assert NATIVE_RESPONSE_KEY in metadata
        assert metadata[NATIVE_RESPONSE_KEY]["finish_reason"] == "OTHER"
        assert metadata[NATIVE_RESPONSE_KEY]["prompt_feedback"] == {"blockReason": "X"}


class TestTheBoundaryWithPartCarriers:
    """``_native`` already holds a provider's native *part*; the record is not it.

    Three dialects keep a part payload under ``_native`` (``{"base": ...,
    "part"|"block"|"item": ...}``).  One key cannot mean two shapes, so the
    record has its own; a part carrier must survive a record write untouched.
    """

    def test_a_part_carrier_is_not_read_as_the_record(self):
        metadata = {
            "_native": {"base": ORIGIN, "part": {"executableCode": {"code": "x"}}}
        }

        assert native_record(metadata, origin=ORIGIN) == {}

    def test_a_record_write_leaves_a_part_carrier_alone(self):
        part_record = {"base": ORIGIN, "part": {"executableCode": {"code": "x"}}}
        metadata = {"_native": part_record}

        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        assert metadata["_native"] == part_record
        assert native_record(metadata, origin=ORIGIN)["finish_reason"] == "OTHER"


class TestTheMigration:
    """A write leaves one spelling behind, not two."""

    def test_the_legacy_keys_are_consumed_by_a_write(self):
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
            LEGACY_PROMPT_FEEDBACK_KEY: {
                "base": ORIGIN,
                "prompt_feedback": {"blockReason": "OTHER"},
            },
        }

        record_native(metadata, origin=ORIGIN, finish_reason="LANGUAGE")

        assert LEGACY_FINISH_REASON_KEY not in metadata
        assert LEGACY_PROMPT_FEEDBACK_KEY not in metadata
        assert metadata[NATIVE_RESPONSE_KEY]["finish_reason"] == "LANGUAGE"
        assert metadata[NATIVE_RESPONSE_KEY]["prompt_feedback"] == {
            "blockReason": "OTHER"
        }

    def test_normalising_a_factless_legacy_key_leaves_nothing_behind(self):
        """The old key goes even when it held no word; no empty record appears."""
        metadata = {LEGACY_FINISH_REASON_KEY: {"base": ORIGIN}}

        written = record_native(metadata, origin=ORIGIN)

        assert written == {}
        assert metadata == {}

    def test_a_partial_current_record_does_not_hide_a_legacy_field(self):
        metadata = {
            NATIVE_RESPONSE_KEY: {
                "origin": ORIGIN,
                "prompt_feedback": {"blockReason": "X"},
            },
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
        }

        record = native_record(metadata, origin=ORIGIN)

        assert record["finish_reason"] == "OTHER"
        assert record["prompt_feedback"] == {"blockReason": "X"}

    def test_the_current_record_wins_a_field_it_holds(self):
        metadata = {
            NATIVE_RESPONSE_KEY: {"origin": ORIGIN, "finish_reason": "LANGUAGE"},
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
        }

        assert native_record(metadata, origin=ORIGIN)["finish_reason"] == "LANGUAGE"


class TestTheOrigin:
    """A record belongs to the dialect that wrote it."""

    def test_another_origins_record_is_not_read(self):
        metadata = {
            NATIVE_RESPONSE_KEY: {"origin": "chat_completions", "finish_reason": "X"}
        }

        assert native_record(metadata, origin=ORIGIN) == {}

    def test_another_origins_legacy_record_is_not_read(self):
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": "chat_completions", "reason": "X"}
        }

        assert native_record(metadata, origin=ORIGIN) == {}

    def test_a_legacy_record_for_this_origin_still_reads(self):
        metadata = {LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "X"}}

        assert native_record(metadata, origin=ORIGIN)["finish_reason"] == "X"

    def test_a_write_by_another_dialect_replaces_the_record(self):
        """The documented asymmetry: reads are origin-filtered, writes are not.

        Unreachable today — a record is written by the reader that produced
        the turn, so a dict carries one dialect's — and it is the shape to
        revisit in P6 if a dict must hold two.
        """
        metadata = {
            NATIVE_RESPONSE_KEY: {"origin": "chat_completions", "finish_reason": "X"}
        }

        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        assert native_record(metadata, origin=ORIGIN)["finish_reason"] == "OTHER"
        assert native_record(metadata, origin="chat_completions") == {}

    def test_a_factless_write_leaves_a_foreign_record_alone(self):
        """A write with nothing to say must not delete what another said."""
        metadata = {
            NATIVE_RESPONSE_KEY: {"origin": "chat_completions", "finish_reason": "X"}
        }

        written = record_native(metadata, origin=ORIGIN)

        assert written == {}
        assert metadata == {
            NATIVE_RESPONSE_KEY: {"origin": "chat_completions", "finish_reason": "X"}
        }

    def test_a_write_leaves_a_foreign_legacy_key_alone(self):
        """Only the legacy key this write folded is consumed."""
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": "chat_completions", "reason": "X"}
        }

        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        assert metadata[LEGACY_FINISH_REASON_KEY] == {
            "base": "chat_completions",
            "reason": "X",
        }
        assert metadata[NATIVE_RESPONSE_KEY]["finish_reason"] == "OTHER"

    def test_a_write_consumes_its_own_key_beside_a_foreign_one(self):
        """The foreign key survives while the write's own key is consumed."""
        metadata = {
            LEGACY_FINISH_REASON_KEY: {"base": ORIGIN, "reason": "OTHER"},
            LEGACY_PROMPT_FEEDBACK_KEY: {
                "base": "chat_completions",
                "prompt_feedback": {"blockReason": "X"},
            },
        }

        record_native(metadata, origin=ORIGIN, finish_reason="LANGUAGE")

        assert LEGACY_FINISH_REASON_KEY not in metadata
        assert metadata[LEGACY_PROMPT_FEEDBACK_KEY]["base"] == "chat_completions"

    def test_a_current_record_with_a_stale_base_is_still_ours(self):
        """Ownership of the current key is asked by ``origin``, as the read is.

        A value carrying both spellings must not be classified by ``base``:
        ``native_record`` reads this key by ``origin``, so a write that judged
        it by ``base`` would disagree with the read it was told to consume.
        """
        metadata = {NATIVE_RESPONSE_KEY: {"origin": ORIGIN, "base": "chat_completions"}}

        record_native(metadata, origin=ORIGIN)

        assert metadata == {}

    def test_a_legacy_key_with_a_stray_origin_is_judged_by_base(self):
        """A legacy key is read by ``base``, so only ``base`` may claim it."""
        metadata = {
            LEGACY_FINISH_REASON_KEY: {
                "base": "chat_completions",
                "origin": ORIGIN,
                "reason": "X",
            }
        }

        record_native(metadata, origin=ORIGIN, finish_reason="OTHER")

        assert metadata[LEGACY_FINISH_REASON_KEY]["reason"] == "X"

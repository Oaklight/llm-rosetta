"""Tests for truncate_with_digest helper."""

import pytest

from llm_rosetta.converters.base.helpers.truncate import (
    HASH_SUFFIX_LEN,
    truncate_with_digest,
)


class TestTruncateWithDigest:
    def test_short_text_unchanged(self):
        assert truncate_with_digest("hello", 64) == "hello"

    def test_exact_length_unchanged(self):
        text = "a" * 64
        assert truncate_with_digest(text, 64) == text

    def test_long_text_truncated_with_digest(self):
        text = "a" * 100
        result = truncate_with_digest(text, 64)
        assert len(result) == 64
        assert result.endswith("_" + result.split("_")[-1])
        assert len(result.split("_")[-1]) == HASH_SUFFIX_LEN

    def test_deterministic(self):
        text = "a" * 100
        assert truncate_with_digest(text, 64) == truncate_with_digest(text, 64)

    def test_different_inputs_different_outputs(self):
        a = truncate_with_digest("a" * 100, 64)
        b = truncate_with_digest("b" * 100, 64)
        assert a != b

    def test_seed_overrides_hash_source(self):
        text = "a" * 100
        result_no_seed = truncate_with_digest(text, 64)
        result_with_seed = truncate_with_digest(text, 64, seed="custom_seed")
        assert result_no_seed != result_with_seed

    def test_force_appends_digest_to_short_text(self):
        result = truncate_with_digest("hello", 64, force=True)
        assert result != "hello"
        assert "_" in result
        assert len(result) <= 64

    def test_force_with_seed(self):
        r1 = truncate_with_digest("hello", 64, seed="ns1\0hello", force=True)
        r2 = truncate_with_digest("hello", 64, seed="ns2\0hello", force=True)
        assert r1 != r2

    def test_max_length_minimum_valid(self):
        min_len = HASH_SUFFIX_LEN + 1
        result = truncate_with_digest("a" * 100, min_len)
        assert len(result) == min_len

    def test_max_length_too_small_raises(self):
        with pytest.raises(ValueError, match="max_length must be >="):
            truncate_with_digest("hello", HASH_SUFFIX_LEN)

    def test_max_length_zero_raises(self):
        with pytest.raises(ValueError, match="max_length must be >="):
            truncate_with_digest("hello", 0)

    def test_result_fits_budget(self):
        for length in [10, 20, 30, 64]:
            result = truncate_with_digest("x" * 200, length)
            assert len(result) <= length

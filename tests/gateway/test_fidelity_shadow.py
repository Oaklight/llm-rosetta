"""Tests for always-on fidelity shadow diff in gateway proxy."""

from __future__ import annotations

import logging

import pytest

from llm_rosetta.gateway.pipelines._fidelity import check_same_format_fidelity
from llm_rosetta.gateway.proxy import _check_fidelity


class TestCheckFidelity:
    """Unit tests for _check_fidelity helper."""

    def test_cross_format_skips_check(self):
        profile: dict = {}
        result = _check_fidelity(
            "openai_chat", "messages", {"a": 1}, {"b": 2}, "request", profile
        )
        assert result is False
        assert "fidelity" not in profile

    def test_same_format_no_diff(self):
        profile: dict = {}
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = _check_fidelity(
            "openai_chat", "openai_chat", body, dict(body), "request", profile
        )
        assert result is False
        assert "fidelity" not in profile

    def test_detects_critical_field_change(self):
        profile: dict = {}
        original = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        converted = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        _check_fidelity(
            "openai_chat", "openai_chat", original, converted, "request", profile
        )
        assert "fidelity" in profile
        fidelity = profile["fidelity"]["request"]
        assert fidelity["diff_count"] > 0
        assert "max_severity" in fidelity
        assert isinstance(fidelity["diffs"], list)
        assert any("max_tokens" in d["path"] for d in fidelity["diffs"])

    def test_detects_role_change(self):
        profile: dict = {}
        original = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        converted = {
            "model": "m",
            "messages": [{"role": "assistant", "content": "hi"}],
        }
        _check_fidelity(
            "openai_chat", "openai_chat", original, converted, "request", profile
        )
        assert "fidelity" in profile
        assert profile["fidelity"]["request"]["diff_count"] > 0

    def test_response_direction(self):
        profile: dict = {}
        original = {
            "id": "resp-1",
            "model": "m",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        converted = {
            "id": "resp-1",
            "model": "m",
        }
        _check_fidelity(
            "openai_chat", "openai_chat", original, converted, "response", profile
        )
        assert "fidelity" in profile
        assert "response" in profile["fidelity"]

    def test_both_directions_coexist(self):
        profile: dict = {}
        _check_fidelity(
            "openai_chat",
            "openai_chat",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            "request",
            profile,
        )
        _check_fidelity(
            "openai_chat",
            "openai_chat",
            {"id": "1", "model": "m", "usage": {"prompt_tokens": 1}},
            {"id": "1", "model": "m"},
            "response",
            profile,
        )
        assert "request" in profile["fidelity"]
        assert "response" in profile["fidelity"]

    def test_diffs_capped_at_20(self):
        profile: dict = {}
        original = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": f"type_{i}"} for i in range(25)],
                }
            ],
        }
        converted = {
            "model": "m",
            "messages": [{"role": "user", "content": []}],
        }
        _check_fidelity(
            "openai_chat", "openai_chat", original, converted, "request", profile
        )
        if "fidelity" in profile:
            assert len(profile["fidelity"]["request"]["diffs"]) <= 20

    def test_returns_true_on_critical_severity(self):
        profile: dict = {}
        original = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        converted = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = _check_fidelity(
            "openai_chat", "openai_chat", original, converted, "request", profile
        )
        assert result is True


class TestNonLlmFidelityShadow:
    """The non-LLM handlers warn on same-format round-trip loss."""

    @pytest.fixture(autouse=True)
    def _propagate(self, monkeypatch):
        # The gateway logger sets propagate=False; caplog needs it to reach
        # the root handler.
        from llm_rosetta.gateway.logging import _logger

        monkeypatch.setattr(_logger, "propagate", True)

    def test_same_format_diff_logs_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            check_same_format_fidelity(
                "embedding",
                "openai",
                "openai",
                {"model": "m", "input": ["hi"], "dimensions": 8},
                {"model": "m", "input": ["hi"]},
                "request",
                model="m",
            )
        assert any("[FIDELITY]" in record.message for record in caplog.records)

    def test_cross_format_is_skipped(self, caplog):
        with caplog.at_level(logging.WARNING):
            check_same_format_fidelity(
                "embedding",
                "openai",
                "cohere",
                {"texts": ["hi"]},
                {"model": "m"},
                "request",
            )
        assert not any("[FIDELITY]" in record.message for record in caplog.records)

    def test_identical_bodies_do_not_warn(self, caplog):
        body = {"model": "m", "input": ["hi"]}
        with caplog.at_level(logging.WARNING):
            check_same_format_fidelity(
                "embedding", "openai", "openai", body, dict(body), "request"
            )
        assert not any("[FIDELITY]" in record.message for record in caplog.records)

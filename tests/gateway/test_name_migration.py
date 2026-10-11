"""Tests for the gateway one-time provider/shim name correction."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.name_migration import (
    DOCS_URL,
    correct_legacy_names,
    legacy_name_error,
)


@pytest.mark.parametrize(
    ("slot", "value", "expected"),
    [
        # shim: slot — unambiguous shim identity.
        ("shim", "openai", "openai--chat_completions"),
        ("shim", "anthropic", "anthropic--messages"),
        ("shim", "google", "google--google_generate"),
        ("shim", "deepseek--openai_chat", "deepseek--chat_completions"),
        ("shim", "openai--chat_completions", "openai--chat_completions"),
        # type: slot — names a format.
        ("type", "openai_chat", "chat_completions"),
        ("type", "google", "google_generate"),
        ("type", "anthropic", "messages"),
        ("type", "openai_responses", "openai--open_responses"),
        ("type", "openai", "openai--chat_completions"),
        # A canonical base format must NOT be rewritten to a shim.
        ("type", "google_generate", "google_generate"),
        ("type", "google_interactions", "google_interactions"),
        ("type", "messages", "messages"),
        ("type", "chat_completions", "chat_completions"),
    ],
)
def test_correct_legacy_names(slot: str, value: str, expected: str) -> None:
    raw = {"providers": {"p": {slot: value}}}

    assert correct_legacy_names(raw) is (value != expected)
    assert raw["providers"]["p"][slot] == expected


def test_correct_legacy_names_is_idempotent() -> None:
    raw = {"providers": {"p": {"shim": "openai", "type": None}}}

    assert correct_legacy_names(raw) is True
    # Second pass changes nothing.
    assert correct_legacy_names(raw) is False


def test_correct_legacy_names_handles_missing_providers() -> None:
    assert correct_legacy_names({}) is False
    assert correct_legacy_names({"providers": "nope"}) is False


@pytest.mark.parametrize(
    "cfg",
    [
        {"shim": "openai"},
        {"shim": "anthropic"},
        {"type": "openai_chat"},
        {"type": "google"},
        {"type": "openai"},
    ],
)
def test_legacy_name_error_flags_legacy(cfg: dict) -> None:
    message = legacy_name_error({"p": cfg})

    assert message is not None
    assert DOCS_URL in message


@pytest.mark.parametrize(
    "cfg",
    [
        {"shim": "openai--chat_completions"},
        {"shim": "anthropic--messages"},
        {"type": "google_generate"},
        {"type": "messages"},
        {"type": "chat_completions"},
        {"base_url": "https://example.invalid"},
    ],
)
def test_legacy_name_error_passes_canonical(cfg: dict) -> None:
    assert legacy_name_error({"p": cfg}) is None

"""Tests for the gateway guard against retired provider and shim names.

The one-time rewrite (``correct_legacy_names``) is gone (#919): a retired name
is rejected with the retired-names diagnostic instead of being rewritten.
"""

from __future__ import annotations

import pytest

from llm_rosetta import retired_names
from llm_rosetta_gateway.maintenance import names
from llm_rosetta_gateway.maintenance.names import legacy_name_error


def test_the_rewrite_is_gone() -> None:
    assert not hasattr(names, "correct_legacy_names")
    import llm_rosetta_gateway.maintenance as maintenance

    assert "correct_legacy_names" not in maintenance.__all__
    assert not hasattr(maintenance, "correct_legacy_names")


@pytest.mark.parametrize(
    ("cfg", "replacement"),
    [
        (dict(shim="openai"), "openai--chat_completions"),
        (dict(shim="anthropic"), "anthropic--messages"),
        (dict(shim="google"), "google--google_generate"),
        (dict(shim="volcengine--openai_chat"), "volcengine--chat_completions"),
        (dict(type="openai_chat"), "chat_completions"),
        (dict(type="google"), "google_generate"),
        (dict(type="anthropic"), "messages"),
        (dict(type="openai_responses"), "openai--open_responses"),
        (dict(type="openai"), "openai--chat_completions"),
    ],
)
def test_legacy_name_error_names_the_replacement(cfg: dict, replacement: str) -> None:
    message = legacy_name_error(dict(p=cfg))

    assert message is not None
    assert repr(replacement) in message
    assert message.endswith(retired_names.MIGRATE_HINT)
    assert "'p'" in message


def test_a_standalone_package_shim_names_its_preset() -> None:
    message = legacy_name_error(dict(p=dict(shim="alcf--sophia")))

    assert message is not None
    assert "'alcf_sophia--chat_completions'" in message
    assert "shims.presets: ['alcf']" in message


@pytest.mark.parametrize(
    "cfg",
    [
        dict(shim="openai--chat_completions"),
        dict(shim="anthropic--messages"),
        dict(type="google_generate"),
        dict(type="google_interactions"),
        dict(type="messages"),
        dict(type="chat_completions"),
        dict(base_url="https://example.invalid"),
    ],
)
def test_legacy_name_error_passes_canonical(cfg: dict) -> None:
    assert legacy_name_error(dict(p=cfg)) is None


def test_the_guard_ignores_non_mapping_entries() -> None:
    assert legacy_name_error(dict(p="nope")) is None


def test_the_guard_reads_the_table_not_a_copy() -> None:
    for row in retired_names.RETIRED_NAMES:
        if row.kind == retired_names.MODEL_TYPE:
            continue
        slot = "shim" if row.kind == retired_names.SHIM else "type"
        message = legacy_name_error(dict(p={slot: row.name}))
        assert message is not None, row
        assert repr(row.replacement) in message or retired_names.lookup(
            row.name, retired_names.PROVIDER_TYPE
        ), row

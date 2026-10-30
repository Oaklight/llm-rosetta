"""What each dialect loses, by subject — the guard for the P5b migration.

Epic #1238 migrates the warning sites dialect by dialect.  Rewording a
message is allowed; changing *which* losses a conversion reports, or which
leg and code they carry, is not.  This module converts a fixed matrix of IR
values through every dialect and records the ``(leg, code, subject)`` triples
they report, so a migration that drops one, invents one, or moves it to
another leg fails here rather than in a user's log.

The baseline is ``warning_subjects.json``; regenerate it with
``python scripts/dev/regen_warning_subjects.py`` when a *deliberate*
behaviour change lands.

What it does **not** cover, so the migration does not over-trust it: the
matrix converts request and response bodies only — no ``stream`` leg — and a
dialect that reports nothing for these values contributes nothing.  A leg
move at a site the matrix never reaches stays invisible; P5b's per-dialect
tests are where those land.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import (
    GoogleInteractionsConverter,
)
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

DIALECTS: dict[str, Callable[[], Any]] = {
    "chat_completions": ChatCompletionsConverter,
    "open_responses": OpenResponsesConverter,
    "messages": MessagesConverter,
    "google_generate": GoogleGenerateConverter,
    "google_interactions": GoogleInteractionsConverter,
}


def _request() -> dict[str, Any]:
    """A request carrying the fields dialects are known to refuse."""
    return {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "generation": {
            "top_k": 5,
            "logprobs": True,
            "top_logprobs": 2,
            "seed": 1,
            "stop_sequences": ["END"],
        },
        "tools": [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch.",
                "format": {"type": "grammar", "grammar": {"syntax": "lark"}},
            },
        ],
    }


def _response() -> dict[str, Any]:
    """A response with the parts a writer may have to let go of."""
    return {
        "id": "resp_1",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "reasoning": "think"},
                        {"type": "text", "text": "hi"},
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "tool_name": "apply_patch",
                            "tool_type": "custom",
                            "tool_input": {"input": "patch"},
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "tool_name": "web_search",
                            "tool_type": "intrinsic",
                            "intrinsic_kind": "web_search",
                            "tool_input": {"query": "x"},
                        },
                    ],
                },
                "finish_reason": {"reason": "tool_calls"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def build_warning_subjects() -> dict[str, list[list[str]]]:
    """Convert the matrix through every dialect and record what it lost.

    Returns:
        ``{dialect: [[leg, code, subject], ...]}``, sorted, for comparison
        against the committed baseline.
    """
    subjects: dict[str, list[list[str]]] = {}
    for dialect, factory in DIALECTS.items():
        converter = factory()
        seen: set[tuple[str, str, str]] = set()

        ctx = ConversionContext()
        converter.request_to_provider(_request(), context=ctx)
        seen |= {(w.leg, w.code, w.subject) for w in ctx.warnings}

        ctx = ConversionContext()
        converter.response_to_provider(_response(), context=ctx)
        seen |= {(w.leg, w.code, w.subject) for w in ctx.warnings}

        subjects[dialect] = [list(item) for item in sorted(seen)]
    return subjects


def _baseline_path() -> Path:
    return Path(__file__).with_name("warning_subjects.json")


def test_the_losses_each_dialect_reports_are_unchanged() -> None:
    """The migration may reword a message; it may not move a loss."""
    baseline = json.loads(_baseline_path().read_text())
    assert build_warning_subjects() == baseline, (
        "a warning subject moved: if the change is intended, regenerate with "
        "python scripts/dev/regen_warning_subjects.py"
    )


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_every_dialect_reports_something(dialect: str) -> None:
    """A dialect that reports nothing would make the guard vacuous for it."""
    assert build_warning_subjects()[dialect], dialect

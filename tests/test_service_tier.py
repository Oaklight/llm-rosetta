"""``service_tier`` crosses the formats that have it, and is reported where not.

The request side of it was not modelled at all: the field was dropped on every
route, same-format included, while the response side has had
``IRResponse.service_tier`` all along.  The two OpenAI-family formats share one
value space (``auto``, ``default``, ``flex``, ``priority``, ...), so the tier
travels as the source wrote it.  Anthropic's is a different axis -- which
capacity may serve the request, not which tier to schedule it on -- so only
``auto`` maps, in either direction.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta import ConversionPipeline

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
RESPONSES = {"model": "m", "input": "hi"}
MESSAGES = {
    "model": "m",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "hi"}],
}


def _convert(
    source: str, request: dict[str, Any], target: str
) -> tuple[dict[str, Any], list[str]]:
    pipe = ConversionPipeline(source, target)
    return pipe.convert_request(request), list(pipe.warnings)


@pytest.mark.parametrize(
    ("source", "base", "target"),
    [
        ("open_responses", RESPONSES, "open_responses"),
        ("chat_completions", CHAT, "chat_completions"),
        ("open_responses", RESPONSES, "chat_completions"),
        ("chat_completions", CHAT, "open_responses"),
    ],
)
def test_a_tier_travels_between_the_formats_that_share_it(
    source: str, base: dict[str, Any], target: str
) -> None:
    body, warnings = _convert(source, {**base, "service_tier": "priority"}, target)
    assert body["service_tier"] == "priority"
    assert warnings == []


@pytest.mark.parametrize(
    ("source", "base"), [("open_responses", RESPONSES), ("chat_completions", CHAT)]
)
def test_messages_keeps_auto_and_reports_a_real_tier(
    source: str, base: dict[str, Any]
) -> None:
    body, warnings = _convert(source, {**base, "service_tier": "auto"}, "messages")
    assert body["service_tier"] == "auto"
    assert warnings == []

    body, warnings = _convert(source, {**base, "service_tier": "priority"}, "messages")
    assert "service_tier" not in body
    assert warnings == [
        "service_tier='priority' cannot be expressed by the Messages API; "
        "the request is sent without a tier"
    ]


def test_messages_auto_carries_out() -> None:
    body, warnings = _convert(
        "messages", {**MESSAGES, "service_tier": "auto"}, "open_responses"
    )
    assert body["service_tier"] == "auto"
    assert warnings == []


def test_messages_standard_only_is_reported_rather_than_reset() -> None:
    """It constrains capacity, which the IR tier vocabulary cannot say."""
    body, warnings = _convert(
        "messages", {**MESSAGES, "service_tier": "standard_only"}, "open_responses"
    )
    assert "service_tier" not in body
    assert warnings == [
        "service_tier='standard_only' (Messages) has no equivalent in the IR's "
        "tier vocabulary; it was not carried"
    ]


def test_a_request_without_a_tier_stays_quiet() -> None:
    for source, base, target in (
        ("open_responses", RESPONSES, "messages"),
        ("messages", MESSAGES, "open_responses"),
    ):
        body, warnings = _convert(source, base, target)
        assert "service_tier" not in body
        assert warnings == []

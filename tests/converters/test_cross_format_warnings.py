"""Every cross-format loss this batch names must be attributable by its path.

The property in ``test_cross_format_roundtrip`` attributes a loss through
``explained()``: a warning explains a path when its ``subject`` is that path,
the path's leaf, or a prefix of it.  A warning whose subject is the *dialect's*
field name, a comma-joined list, or the container it emptied explains nothing
-- that is how the losses in #1328 stayed silent while being reported.

Each test here writes one IR request to one target and asserts two things:
the warning's subject is the IR path, and ``explained()`` accepts it for that
path and rejects an unrelated one.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.conversion_warnings import WarningSink, recording_to
from tests.converters.test_roundtrip_property import FORMATS
from tools.conformance.suites.wire import explained

#: A path no warning in this module should ever claim to explain.
UNRELATED = "generation.max_tokens: lost"


def _write(
    target: str, ir: dict[str, Any], *, output_format: str | None = None
) -> tuple[dict[str, Any], list[Any]]:
    """Write *ir* as *target*, collecting what the writer reports."""
    fmt = FORMATS[target]
    ctx = ConversionContext()
    sink = WarningSink(leg="request")
    ctx.warnings = sink
    kwargs: dict[str, Any] = {}
    if output_format is not None:
        kwargs["output_format"] = output_format
    with recording_to(sink):
        body, _ = fmt.converter().request_to_provider(ir, context=ctx, **kwargs)
    return body, list(sink)


def _base(**extra: Any) -> dict[str, Any]:
    ir: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    ir.update(extra)
    return ir


def _subjects(warnings: list[Any]) -> set[str]:
    return {getattr(warning, "subject", "") for warning in warnings}


def _assert_explains(warnings: list[Any], path: str) -> None:
    """The one assertion every warning in this module must satisfy.

    The subject set must be *exactly* the path: a superset would let a warning
    that names something else ride along, and a membership check alone would
    accept a leaf-only subject (``budget_tokens``), which is the form this
    batch removes.
    """
    assert _subjects(warnings) == {path}, (path, warnings)
    assert explained(f"{path}: lost", warnings)
    assert not explained(UNRELATED, warnings), warnings


@pytest.mark.parametrize(
    "target", ["messages", "open_responses", "google_generate", "google_interactions"]
)
def test_logit_bias_is_named_by_its_ir_path(target: str) -> None:
    body, warnings = _write(target, _base(generation={"logit_bias": {"1": 1}}))

    assert "logit_bias" not in str(body.get("logit_bias", ""))
    _assert_explains(warnings, "generation.logit_bias")


@pytest.mark.parametrize(
    "target", ["open_responses", "google_generate", "google_interactions"]
)
def test_include_usage_is_named_by_its_ir_path(target: str) -> None:
    _, warnings = _write(target, _base(stream={"enabled": True, "include_usage": True}))

    _assert_explains(warnings, "stream.include_usage")


@pytest.mark.parametrize("target", ["chat_completions", "messages", "google_generate"])
def test_truncation_is_named_by_its_ir_path(target: str) -> None:
    _, warnings = _write(target, _base(generation={"truncation": "auto"}))

    _assert_explains(warnings, "generation.truncation")


def test_the_google_generate_cache_retention_is_named_by_its_ir_path() -> None:
    body, warnings = _write("google_generate", _base(cache={"retention": 60}))

    assert "retention" not in body.get("config", {})
    _assert_explains(warnings, "cache.retention")


def test_the_anthropic_budget_fallback_is_named_by_its_ir_path() -> None:
    """``enabled`` without a budget falls back to a mode; say why."""
    _, warnings = _write("messages", _base(reasoning={"mode": "enabled"}))

    _assert_explains(warnings, "reasoning.budget_tokens")


@pytest.mark.parametrize(
    "cache, path",
    [({"key": "k1"}, "cache.key"), ({"retention": 60}, "cache.retention")],
)
def test_the_interactions_cache_loss_is_named_by_its_ir_path(
    cache: dict[str, Any], path: str
) -> None:
    body, warnings = _write("google_interactions", _base(cache=cache))

    assert "cached_content" not in body
    _assert_explains(warnings, path)


@pytest.mark.parametrize(
    "target, reasoning, path",
    [
        ("chat_completions", {"mode": "enabled"}, "reasoning.mode"),
        ("chat_completions", {"budget_tokens": 1024}, "reasoning.budget_tokens"),
        ("open_responses", {"mode": "enabled"}, "reasoning.mode"),
        ("open_responses", {"budget_tokens": 1024}, "reasoning.budget_tokens"),
        ("google_interactions", {"mode": "enabled"}, "reasoning.mode"),
        ("google_interactions", {"budget_tokens": 1024}, "reasoning.budget_tokens"),
    ],
)
def test_a_reasoning_axis_the_target_lacks_is_named_by_its_ir_path(
    target: str, reasoning: dict[str, Any], path: str
) -> None:
    _, warnings = _write(target, _base(reasoning=reasoning))

    _assert_explains(warnings, path)


@pytest.mark.parametrize("target", ["google_generate", "google_interactions"])
def test_a_service_tier_with_no_google_equivalent_is_reported(target: str) -> None:
    body, warnings = _write(target, _base(service_tier="auto"))

    assert "service_tier" not in body
    _assert_explains(warnings, "service_tier")


def test_a_shared_service_tier_travels_to_both_google_dialects() -> None:
    """``flex``/``priority``/``standard`` are the same word on both routes."""
    for target in ("google_generate", "google_interactions"):
        body, warnings = _write(target, _base(service_tier="flex"))

        assert warnings == []
        if target == "google_generate":
            assert body["config"]["service_tier"] == "flex"
        else:
            assert body["service_tier"] == "flex"

    rest, warnings = _write(
        "google_generate", _base(service_tier="priority"), output_format="rest"
    )
    assert warnings == []
    assert rest["service_tier"] == "priority"


def test_an_explicit_stream_off_survives_the_interactions_round_trip() -> None:
    """The API takes the flag in the body, so ``enabled: false`` is not absence.

    Measured live 2026-11-08: ``stream: false`` answers 200 JSON and
    ``stream: true`` answers 200 SSE.
    """
    body, warnings = _write("google_interactions", _base(stream={"enabled": False}))

    assert body["stream"] is False
    assert warnings == []
    restored = FORMATS["google_interactions"].converter().request_from_provider(body)
    assert restored["stream"] == {"enabled": False}

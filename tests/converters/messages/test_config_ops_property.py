"""Laws of ``MessagesConfigOps`` over generated configs (Hypothesis).

``test_config_ops.py`` pins each mapping with one example.  The checks here
state the law behind them once and run it over generated input:

* the generation writer maps exactly the five supported fields, drops every
  unsupported one and reports each drop once, and the reader gives the
  supported fields back;
* the generation and reasoning readers copy only what they know and answer
  ``{}`` to anything that is not a dict;
* the stream writer maps ``enabled`` to ``stream`` and reports ``include_usage``;
  the stream reader gives ``enabled`` back;
* the reasoning reader maps ``thinking.type``, ``budget_tokens``, ``display``
  and ``output_config.effort`` to the IR fields the docstring names;
* a provider reasoning body that survives ``reader -> writer`` keeps its
  thinking type, budget, display and effort.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.conversion_warnings import WarningSink, recording_to  # noqa: E402
from llm_rosetta.converters.llm.messages.config_ops import (  # noqa: E402
    MessagesConfigOps as Ops,
)
from llm_rosetta.ir.model import GenerationConfig  # noqa: E402

_SUPPORTED = ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences")
_UNSUPPORTED = (
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "logprobs",
    "top_logprobs",
    "n",
)

_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)

_FIELD_VALUES = {
    "max_tokens": st.integers(min_value=1, max_value=100_000),
    "temperature": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "top_p": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "top_k": st.integers(min_value=0, max_value=1000),
    "stop_sequences": st.lists(st.text(max_size=8), max_size=4),
}
_ANY = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=6))


@st.composite
def _supported(draw) -> dict[str, Any]:
    names = draw(st.lists(st.sampled_from(_SUPPORTED), unique=True))
    return {name: draw(_FIELD_VALUES[name]) for name in names}


@st.composite
def _unsupported(draw) -> dict[str, Any]:
    names = draw(st.lists(st.sampled_from(_UNSUPPORTED), unique=True))
    return {name: draw(_ANY) for name in names}


@_SETTINGS
@given(supported=_supported(), unsupported=_unsupported())
def test_the_generation_writer_maps_supported_and_reports_the_rest(
    supported: dict[str, Any], unsupported: dict[str, Any]
) -> None:
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_generation_config_to_p(
            cast(GenerationConfig, {**supported, **unsupported})
        )

    assert written == {
        name: (float(v) if name == "temperature" else v)
        for name, v in supported.items()
    }
    warnings = list(sink)
    assert len(warnings) == len(unsupported)
    for name in unsupported:
        needle = f"does not support {name};"
        assert sum(needle in warning for warning in warnings) == 1, name


@_SETTINGS
@given(supported=_supported())
def test_the_generation_reader_returns_what_the_writer_wrote(
    supported: dict[str, Any],
) -> None:
    written = Ops.ir_generation_config_to_p(cast(GenerationConfig, supported))
    assert Ops.p_generation_config_to_ir(written) == written


@_SETTINGS
@given(
    supported=_supported(),
    extras=st.dictionaries(st.text(min_size=1, max_size=8), _ANY, max_size=4),
)
def test_the_generation_reader_copies_only_known_fields(
    supported: dict[str, Any], extras: dict[str, Any]
) -> None:
    body = {**extras, **supported}
    assert Ops.p_generation_config_to_ir(body) == {
        name: body[name] for name in _SUPPORTED if name in body
    }


_NOT_A_DICT = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(),
    st.lists(_ANY, max_size=3),
)


@_SETTINGS
@given(value=_NOT_A_DICT)
def test_the_readers_answer_empty_to_anything_that_is_not_a_dict(value: Any) -> None:
    assert Ops.p_generation_config_to_ir(value) == {}
    assert Ops.p_reasoning_config_to_ir(value) == {}
    assert Ops.p_stream_config_to_ir(value) == {}


_THINKING = st.one_of(
    st.just({"type": "adaptive"}),
    st.builds(
        lambda budget: {"type": "enabled", "budget_tokens": budget},
        st.integers(min_value=1024, max_value=32_000),
    ),
).flatmap(
    lambda thinking: st.one_of(
        st.just(thinking),
        st.sampled_from(["summarized", "omitted"]).map(
            lambda display: {**thinking, "display": display}
        ),
    )
)
_EFFORT = st.sampled_from(["low", "medium", "high"])


@st.composite
def _reasoning_bodies(draw) -> dict[str, Any]:
    body: dict[str, Any] = {"thinking": draw(_THINKING)}
    if draw(st.booleans()):
        body["output_config"] = {"effort": draw(_EFFORT)}
    return body


@_SETTINGS
@given(body=_reasoning_bodies())
def test_a_reasoning_body_survives_reader_then_writer(body: dict[str, Any]) -> None:
    again = Ops.ir_reasoning_config_to_p(Ops.p_reasoning_config_to_ir(body))
    assert again["thinking"] == body["thinking"]
    assert again.get("output_config") == body.get("output_config")


@_SETTINGS
@given(
    enabled=st.one_of(st.none(), st.booleans()),
    include_usage=st.one_of(st.none(), st.booleans()),
)
def test_the_stream_writer_maps_enabled_and_reports_include_usage(
    enabled: bool | None, include_usage: bool | None
) -> None:
    ir: dict[str, Any] = {}
    if enabled is not None:
        ir["enabled"] = enabled
    if include_usage is not None:
        ir["include_usage"] = include_usage
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_stream_config_to_p(cast(Any, ir))

    assert written == ({} if enabled is None else {"stream": enabled})
    reported = [w for w in sink if "include_usage" in w]
    assert len(reported) == (1 if include_usage else 0)
    assert len(list(sink)) == len(reported)


@_SETTINGS
@given(stream=st.one_of(st.none(), st.booleans()), extra=_ANY)
def test_the_stream_reader_returns_enabled_only_when_stream_is_set(
    stream: bool | None, extra: Any
) -> None:
    body: dict[str, Any] = {"unrelated": extra}
    if stream is not None:
        body["stream"] = stream
    expected = {} if stream is None else {"enabled": stream}
    assert Ops.p_stream_config_to_ir(body) == expected
    assert Ops.ir_stream_config_to_p(cast(Any, expected)) == (
        {} if stream is None else {"stream": stream}
    )


@st.composite
def _provider_reasoning(draw) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if draw(st.booleans()):
        thinking: dict[str, Any] = {}
        if draw(st.booleans()):
            thinking["type"] = draw(
                st.sampled_from(["adaptive", "enabled", "disabled", "other"])
            )
        if draw(st.booleans()):
            thinking["budget_tokens"] = draw(st.integers(min_value=0, max_value=50_000))
        if draw(st.booleans()):
            thinking["display"] = draw(st.sampled_from(["summarized", "omitted", "x"]))
        body["thinking"] = thinking
    if draw(st.booleans()):
        output_config: dict[str, Any] = {}
        if draw(st.booleans()):
            output_config["effort"] = draw(st.sampled_from(["low", "high", "max"]))
        body["output_config"] = output_config
    return body


@_SETTINGS
@given(body=_provider_reasoning())
def test_the_reasoning_reader_follows_its_documented_mapping(
    body: dict[str, Any],
) -> None:
    """The mapping is written out here independently of the reader."""
    expected: dict[str, Any] = {}
    thinking = body.get("thinking", {})
    expected_mode = {"adaptive": "auto", "enabled": "enabled", "disabled": "disabled"}
    if thinking.get("type") in expected_mode:
        expected["mode"] = expected_mode[thinking["type"]]
    if "budget_tokens" in thinking:
        expected["budget_tokens"] = thinking["budget_tokens"]
    if thinking.get("display") == "omitted":
        expected.update(include_thoughts=False, summary="none")
    elif thinking.get("display") == "summarized":
        expected.update(include_thoughts=True, summary="auto")
    effort = body.get("output_config", {}).get("effort")
    if effort is not None:
        expected["effort"] = effort
        expected.setdefault("mode", "auto")

    assert Ops.p_reasoning_config_to_ir(body) == expected

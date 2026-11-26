"""Laws of ``GoogleGenerateConfigOps`` over generated configs (Hypothesis).

``test_config_ops.py`` pins each mapping with one example.  The checks here
state the law behind them once and run it over generated input:

* the generation writer maps the seven direct fields, renames ``max_tokens``
  and ``n``, drops ``logit_bias`` / ``logprobs`` / ``top_logprobs`` and reports
  each drop once, by its IR path;
* the generation reader accepts the snake_case and the camelCase spelling, the
  snake_case one winning, and copies nothing it does not know;
* what the writer wrote, the reader gives back;
* the response-format writer and reader follow the envelope/raw-schema rule
  (#1327), and an envelope key with no raw spelling is reported once;
* the reasoning reader follows the mapping its docstring states, and the
  writer's own output reads back to the IR that produced it;
* the stream and cache ops map their one field each, and every reader answers
  ``{}`` to anything that is not a dict.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.conversion_warnings import WarningSink, recording_to  # noqa: E402
from llm_rosetta.converters.llm.google_generate import (  # noqa: E402
    GoogleGenerateConverter,
)
from llm_rosetta.converters.llm.google_generate.config_ops import (  # noqa: E402
    GoogleGenerateConfigOps as Ops,
)

_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)

_DIRECT = (
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)
_UNSUPPORTED = ("logit_bias", "logprobs", "top_logprobs")

_VALUES: dict[str, st.SearchStrategy[Any]] = {
    "temperature": st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    "top_p": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    "top_k": st.integers(min_value=0, max_value=1000),
    "stop_sequences": st.lists(st.text(max_size=8), max_size=4),
    "frequency_penalty": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    "presence_penalty": st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    "seed": st.integers(min_value=0, max_value=2**31),
    "max_tokens": st.integers(min_value=1, max_value=100_000),
    "n": st.integers(min_value=1, max_value=8),
}
_ANY = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=6))
_NOT_A_DICT = st.one_of(
    st.none(), st.booleans(), st.integers(), st.text(), st.lists(_ANY, max_size=3)
)

_RENAMED = {"max_tokens": "max_output_tokens", "n": "candidate_count"}


def test_the_target_is_the_product_s_own() -> None:
    assert GoogleGenerateConverter.config_ops_class is Ops


@st.composite
def _supported(draw) -> dict[str, Any]:
    names = draw(st.lists(st.sampled_from([*_DIRECT, *_RENAMED]), unique=True))
    return {name: draw(_VALUES[name]) for name in names}


@st.composite
def _unsupported(draw) -> dict[str, Any]:
    names = draw(st.lists(st.sampled_from(_UNSUPPORTED), unique=True))
    return {name: draw(_ANY) for name in names}


@_SETTINGS
@given(supported=_supported(), unsupported=_unsupported())
def test_the_generation_writer_maps_renames_and_reports_the_rest(
    supported: dict[str, Any], unsupported: dict[str, Any]
) -> None:
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_generation_config_to_p(cast(Any, {**supported, **unsupported}))

    assert written == {_RENAMED.get(name, name): v for name, v in supported.items()}
    warnings = list(sink)
    assert sorted(w.subject for w in warnings) == sorted(
        f"generation.{name}" for name in unsupported
    )
    assert {w.leg for w in warnings} <= {"request"}
    for name in unsupported:
        assert sum(f"support {name};" in str(w) for w in warnings) == 1, name


@_SETTINGS
@given(supported=_supported())
def test_the_generation_reader_returns_what_the_writer_wrote(
    supported: dict[str, Any],
) -> None:
    written = Ops.ir_generation_config_to_p(cast(Any, supported))
    assert Ops.p_generation_config_to_ir(written) == supported


_CAMEL = {
    "stop_sequences": "stopSequences",
    "frequency_penalty": "frequencyPenalty",
    "presence_penalty": "presencePenalty",
    "max_output_tokens": "maxOutputTokens",
    "candidate_count": "candidateCount",
}


@_SETTINGS
@given(
    snake=_supported(),
    camel_names=st.lists(st.sampled_from(sorted(_CAMEL)), unique=True),
    extras=st.dictionaries(st.text(min_size=1, max_size=8), _ANY, max_size=4),
    data=st.data(),
)
def test_the_generation_reader_accepts_both_spellings_and_snake_wins(
    snake: dict[str, Any],
    camel_names: list[str],
    extras: dict[str, Any],
    data: st.DataObject,
) -> None:
    """A field in both spellings reads as the snake_case one; extras are ignored."""
    wire_snake = {_RENAMED.get(k, k): v for k, v in snake.items()}
    # Alternate spellings carry a value of the same kind but a different one.
    wire_camel: dict[str, Any] = {}
    for snake_name in camel_names:
        ir_name = {v: k for k, v in _RENAMED.items()}.get(snake_name, snake_name)
        wire_camel[_CAMEL[snake_name]] = data.draw(_VALUES[ir_name])
    body = {**extras, **wire_camel, **wire_snake}

    expected: dict[str, Any] = {}
    for key in (*_DIRECT, "max_output_tokens", "candidate_count"):
        ir_name = {v: k for k, v in _RENAMED.items()}.get(key, key)
        if key in wire_snake:
            expected[ir_name] = wire_snake[key]
        elif key in _CAMEL and _CAMEL[key] in wire_camel:
            expected[ir_name] = wire_camel[_CAMEL[key]]
        elif key in ("temperature", "top_p", "top_k", "seed") and key in extras:
            expected[ir_name] = extras[key]
    assert Ops.p_generation_config_to_ir(body) == expected


@_SETTINGS
@given(value=_NOT_A_DICT)
def test_every_reader_answers_empty_to_anything_that_is_not_a_dict(value: Any) -> None:
    assert Ops.p_generation_config_to_ir(value) == {}
    assert Ops.p_response_format_to_ir(value) == {}
    assert Ops.p_stream_config_to_ir(value) == {}
    assert Ops.p_reasoning_config_to_ir(value) == {}
    assert Ops.p_cache_config_to_ir(value) == {}


# --- response format ---------------------------------------------------------

_RAW_SCHEMA = st.fixed_dictionaries(
    {"type": st.just("object")},
    optional={
        "description": st.text(max_size=8),
        "name": st.text(min_size=1, max_size=8),
        "properties": st.just({"a": {"type": "string"}}),
    },
)


@st.composite
def _envelope(draw) -> dict[str, Any]:
    env: dict[str, Any] = {"schema": draw(_RAW_SCHEMA)}
    if draw(st.booleans()):
        env["name"] = draw(st.text(min_size=1, max_size=8))
    if draw(st.booleans()):
        env["description"] = draw(st.text(min_size=1, max_size=8))
    if draw(st.booleans()):
        env["strict"] = draw(st.booleans())
    return env


@_SETTINGS
@given(env=_envelope())
def test_an_envelope_is_unwrapped_and_its_lost_keys_reported_once(
    env: dict[str, Any],
) -> None:
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_response_format_to_p(
            cast(Any, {"type": "json_schema", "json_schema": env})
        )
    assert written == {
        "response_mime_type": "application/json",
        "response_schema": env["schema"],
    }
    lost = [k for k in ("name", "description") if k in env]
    if env.get("strict") is True:
        lost.append("strict")
    warnings = list(sink)
    assert [w.subject for w in warnings] == (
        ["response_format.json_schema"] if lost else []
    )
    for key in lost:
        assert key in str(warnings[0])
    if lost:
        assert str(warnings[0]).endswith(
            " are dropped" if len(lost) > 1 else " is dropped"
        )


@_SETTINGS
@given(raw=_RAW_SCHEMA)
def test_a_raw_schema_is_written_whole_and_reports_nothing(
    raw: dict[str, Any],
) -> None:
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_response_format_to_p(
            cast(Any, {"type": "json_schema", "json_schema": raw})
        )
    assert written == {
        "response_mime_type": "application/json",
        "response_schema": raw,
    }
    assert list(sink) == []


@pytest.mark.parametrize("fmt", [{}, {"type": "text"}])
def test_text_or_no_response_format_writes_nothing(fmt: dict[str, Any]) -> None:
    assert Ops.ir_response_format_to_p(cast(Any, fmt)) == {}


def test_json_object_writes_only_the_mime_type() -> None:
    assert Ops.ir_response_format_to_p(cast(Any, {"type": "json_object"})) == {
        "response_mime_type": "application/json"
    }


@_SETTINGS
@given(
    spelling=st.sampled_from(["snake", "camel"]),
    mime=st.sampled_from(["application/json", "text/plain", "text/x.other", ""]),
    schema=st.one_of(st.none(), _RAW_SCHEMA),
)
def test_the_response_format_reader_follows_mime_type_and_schema(
    spelling: str, mime: str, schema: dict[str, Any] | None
) -> None:
    mime_key, schema_key = (
        ("response_mime_type", "response_schema")
        if spelling == "snake"
        else ("responseMimeType", "responseSchema")
    )
    body: dict[str, Any] = {mime_key: mime}
    if schema is not None:
        body[schema_key] = schema
    if mime != "application/json":
        expected: dict[str, Any] = {}
    elif schema:
        expected = {"type": "json_schema", "json_schema": {"schema": schema}}
    else:
        expected = {"type": "json_object"}
    assert Ops.p_response_format_to_ir(body) == expected


@_SETTINGS
@given(raw=_RAW_SCHEMA)
def test_a_response_schema_survives_reader_then_writer(raw: dict[str, Any]) -> None:
    body = {"response_mime_type": "application/json", "response_schema": raw}
    assert Ops.ir_response_format_to_p(Ops.p_response_format_to_ir(body)) == body


# --- stream and cache --------------------------------------------------------


@_SETTINGS
@given(stream=st.one_of(st.none(), st.booleans()), extra=_ANY)
def test_the_stream_ops_map_enabled_and_stream_both_ways(
    stream: bool | None, extra: Any
) -> None:
    body: dict[str, Any] = {"unrelated": extra}
    ir: dict[str, Any] = {}
    if stream is not None:
        body["stream"] = stream
        ir["enabled"] = stream
    assert Ops.p_stream_config_to_ir(body) == ir
    assert Ops.ir_stream_config_to_p(cast(Any, ir)) == (
        {} if stream is None else {"stream": stream}
    )


@_SETTINGS
@given(
    key=st.one_of(st.none(), st.text(min_size=1, max_size=12)),
    retention=st.one_of(st.none(), st.sampled_from(["short", "long"])),
)
def test_the_cache_writer_maps_key_and_reports_retention(
    key: str | None, retention: str | None
) -> None:
    ir: dict[str, Any] = {}
    if key is not None:
        ir["key"] = key
    if retention is not None:
        ir["retention"] = retention
    sink = WarningSink()
    with recording_to(sink):
        written = Ops.ir_cache_config_to_p(cast(Any, ir))
    assert written == ({} if key is None else {"cached_content": key})
    assert [w.subject for w in sink] == (
        [] if retention is None else ["cache.retention"]
    )


@_SETTINGS
@given(
    snake=st.one_of(st.none(), st.text(max_size=8)),
    camel=st.one_of(st.none(), st.text(max_size=8)),
)
def test_the_cache_reader_takes_the_first_non_empty_spelling(
    snake: str | None, camel: str | None
) -> None:
    body: dict[str, Any] = {}
    if snake is not None:
        body["cached_content"] = snake
    if camel is not None:
        body["cachedContent"] = camel
    key = snake or camel
    assert Ops.p_cache_config_to_ir(body) == ({"key": key} if key else {})


# --- reasoning ---------------------------------------------------------------


@st.composite
def _thinking_configs(draw) -> dict[str, Any]:
    spelling = draw(st.sampled_from(["snake", "camel"]))
    names = (
        ("thinking_budget", "thinking_level", "include_thoughts")
        if spelling == "snake"
        else ("thinkingBudget", "thinkingLevel", "includeThoughts")
    )
    config: dict[str, Any] = {}
    if draw(st.booleans()):
        config[names[0]] = draw(
            st.one_of(st.just(0), st.just(-1), st.integers(1, 32_000))
        )
    if draw(st.booleans()):
        config[names[1]] = draw(st.sampled_from(["low", "medium", "high"]))
    if draw(st.booleans()):
        config[names[2]] = draw(st.booleans())
    return config


@_SETTINGS
@given(
    config=_thinking_configs(),
    key=st.sampled_from(["thinking_config", "thinkingConfig"]),
)
def test_the_reasoning_reader_follows_its_documented_mapping(
    config: dict[str, Any], key: str
) -> None:
    """The mapping is written out here independently of the reader."""
    expected: dict[str, Any] = {}
    budget = config.get("thinking_budget", config.get("thinkingBudget"))
    if budget is not None:
        if budget == 0:
            expected["mode"] = "disabled"
        elif budget == -1:
            expected["mode"] = "auto"
        else:
            expected.update(mode="enabled", budget_tokens=budget)
    level = config.get("thinking_level", config.get("thinkingLevel"))
    if level is not None:
        expected["effort"] = level
        expected.setdefault("mode", "auto")
    include = config.get("include_thoughts", config.get("includeThoughts"))
    if include is not None:
        expected["include_thoughts"] = include
        expected["summary"] = "auto" if include else "none"

    body = {key: config}
    assert Ops.p_reasoning_config_to_ir(body) == (expected if config else {})


@_SETTINGS
@given(config=_thinking_configs())
def test_the_writer_s_reasoning_output_reads_back_to_its_ir(
    config: dict[str, Any],
) -> None:
    ir = Ops.p_reasoning_config_to_ir({"thinking_config": config})
    written = Ops.ir_reasoning_config_to_p(cast(Any, ir))
    # A disabled turn does not think, so the writer keeps no effort for it.
    kept = {
        k: v for k, v in ir.items() if not (k == "effort" and ir["mode"] == "disabled")
    }
    assert Ops.p_reasoning_config_to_ir(written) == kept


@_SETTINGS
@given(body=st.dictionaries(st.text(min_size=1, max_size=8), _ANY, max_size=4))
def test_a_body_with_no_thinking_config_reads_as_no_reasoning(
    body: dict[str, Any],
) -> None:
    body.pop("thinking_config", None)
    body.pop("thinkingConfig", None)
    assert Ops.p_reasoning_config_to_ir(body) == {}

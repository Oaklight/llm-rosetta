"""Same-format round trips on generated bodies — epic #1223, property 2.

``A → IR → A`` must be the identity on the subset of A the IR can represent.
The wire suite checks that on one canonical body per format; this module
generates one body per declared top-level request field instead, so a field
that only one dialect carries is still exercised.

Most values are sampled from the SDK's own annotation — the first literal, the
first union member, a one-field object for a nested model.  A sampled value can
be meaningless in its field (``text: {}`` names no format), and a meaningless
body proves nothing, so :data:`_VALUES` pins a plausible value for the fields
where the generic sample is not one.  A field whose value cannot be sampled at
all is skipped rather than guessed.

A case is only a test if its value reaches the IR.  ``read(field) == read()``
is still "the identity" for a reader that never looked, so every case also
asserts that the first read differs from the read of the bare fixture.  Which
fields may reach the IR is not a judgement call here: it comes from the
``request-fields`` ledger, which records what the reader does with each declared
field.  Only ``mapped`` fields generate a case — a ``dropped`` field cannot
change the IR by definition, and asserting the identity on one would be
vacuous.

:data:`_EXCLUSIONS` names the cases the property does not cover, each with the
reason.  Every other difference fails the case and names the field.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import (
    GoogleInteractionsConverter,
)
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from tools.conformance.suites.fields import _hints, _members  # noqa: E402

#: Container fields whose value the sampler cannot invent: a message list, a
#: tool list or a system prompt has to be built for the dialect.  The fixture
#: carries them, and the sampler leaves them alone.
_BUILT_BY_HAND = frozenset(
    {
        "messages",
        "input",
        "contents",
        "system",
        "system_instruction",
        "instructions",
        "tools",
        "functions",
    }
)

#: The ``request-fields`` ledger: one class per declared request field saying
#: what the reader does with it.  It is the same statement the conformance
#: suite gates on, so this module cannot disagree with it about what is mapped.
_LEDGER = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "conformance"
        / "baselines"
        / "request-fields.json"
    ).read_text()
)["formats"]

#: Formats whose SDK fields live in a nested object on the wire, not at the
#: top level of the body.  The Google SDK takes ``GenerateContentConfig`` as a
#: ``config`` argument, so ``temperature`` is ``config.temperature`` on the
#: wire and the reader looks for it there.
_NESTED: dict[str, str] = {"google_generate": "config"}

#: A field the reader only reads when a companion field is present.  The
#: companion is merged into the same object the case writes.
#:
#: ``response_schema`` is a Google wire rule, not a property of the schema: the
#: reader builds a response format only when ``response_mime_type`` is present,
#: and ``p_response_format_to_ir`` reads the schema only when that mime type is
#: ``application/json``.  If a bare schema ever lands, the entry is dead and
#: ``test_every_companion_is_still_needed`` says so.
_COMPANIONS: dict[tuple[str, str], dict[str, Any]] = {
    ("google_generate", "response_schema"): {"response_mime_type": "application/json"},
}

#: A value the generic sample gets wrong: an empty object, a literal with no
#: meaning in its field, or a container the reader needs a companion field for.
_VALUES: dict[tuple[str, str], Any] = {
    ("chat_completions", "reasoning_effort"): "high",
    ("chat_completions", "response_format"): {"type": "json_object"},
    ("chat_completions", "stop"): ["END"],
    ("chat_completions", "stream_options"): {"include_usage": True},
    ("open_responses", "text"): {"format": {"type": "json_object"}},
    ("open_responses", "reasoning"): {"effort": "high"},
    ("open_responses", "tool_choice"): "auto",
    ("open_responses", "stream_options"): {"include_usage": True},
    ("messages", "thinking"): {"type": "enabled", "budget_tokens": 1024},
    ("messages", "output_config"): {"effort": "high"},
    ("messages", "tool_choice"): {"type": "auto"},
    ("messages", "system"): [{"type": "text", "text": "be brief"}],
    ("google_generate", "response_mime_type"): "application/json",
    ("google_generate", "thinking_config"): {"thinking_budget": 1024},
    ("google_generate", "tool_config"): {"function_calling_config": {"mode": "AUTO"}},
    ("google_generate", "response_schema"): {"type": "object"},
    ("google_generate", "top_k"): 3,
}

#: Cases the property does not cover, each for a stated reason.
_EXCLUSIONS: dict[tuple[str, str], str] = {
    (
        "open_responses",
        "prompt_cache_options",
    ): "profile-only: recorded, written only when the target declares it",
    (
        "open_responses",
        "moderation",
    ): "profile-only: recorded, written only when the target declares it",
    (
        "open_responses",
        "access_programs",
    ): "profile-only: recorded, written only when the target declares it",
    (
        "open_responses",
        "stream_options",
    ): "not round-trippable by design: the writer never emits it (Responses "
    "always reports usage), see test_stream_config_round_trip",
}


def _sample_first_member(annotation: Any, depth: int) -> Any:
    """Return a sample of the first union member that has one.

    Args:
        annotation: A union annotation.
        depth: How deep the sampling already is.

    Returns:
        A value of the first sampleable member's shape.

    Raises:
        ValueError: When no member has a sampleable shape.
    """
    for arg in typing.get_args(annotation):
        if arg is type(None):
            continue
        try:
            return _sample(arg, depth)
        except ValueError:
            continue
    raise ValueError("union with no sampleable member")


def _sample_model(annotation: Any, depth: int) -> Any:
    """Return a one-field object for a nested SDK model.

    Args:
        annotation: The annotation naming the model.
        depth: How deep the sampling already is.

    Returns:
        ``{first sampleable field: its value}``.

    Raises:
        ValueError: When the model has no sampleable field, or the depth is spent.
    """
    classes = _members(annotation)
    if classes and depth < 3:
        for name, child in _hints(classes[0]).items():
            try:
                return {name: _sample(child, depth + 1)}
            except ValueError:
                continue
    raise ValueError(f"no sampleable shape for {annotation!r}")


def _sample(annotation: Any, depth: int = 0) -> Any:
    """Return a deterministic value the reader will read, or raise ValueError.

    Args:
        annotation: The field's SDK annotation.
        depth: How deep the sampling already is; nested models stop at 3.

    Returns:
        A value of roughly the annotated shape.

    Raises:
        ValueError: When the annotation has no sampleable shape.
    """
    origin = typing.get_origin(annotation)
    wrapper = str(origin).rsplit(".", 1)[-1] if origin is not None else ""
    # ``Required``/``NotRequired`` survive ``get_type_hints`` on Python 3.10,
    # where the SDK's ``typing_extensions`` wrapper is not stripped the way
    # ``typing`` strips it from 3.11 on.
    if wrapper in ("Annotated", "Required", "NotRequired"):
        return _sample(typing.get_args(annotation)[0], depth)
    if origin is typing.Union or str(origin) == "types.UnionType":
        return _sample_first_member(annotation, depth)
    if origin is typing.Literal:
        return typing.get_args(annotation)[0]
    if origin in (list, tuple, set, Sequence):
        args = typing.get_args(annotation)
        return [_sample(args[0], depth + 1)] if args else []
    if origin in (dict, Mapping):
        args = typing.get_args(annotation)
        return {"k": _sample(args[1], depth + 1)} if len(args) == 2 else {"k": "v"}
    if annotation is str:
        return "x"
    if annotation is bool:
        return True
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    return _sample_model(annotation, depth)


class Format:
    """One chat format's round-trip case: its converter, root and fixture."""

    def __init__(
        self,
        name: str,
        converter: Callable[[], Any],
        root: Callable[[], Any],
        fixture: dict[str, Any],
        values: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.converter = converter
        self.root = root
        self.fixture = fixture
        self.values = values

    def annotations(self) -> dict[str, Any]:
        """Return the format's declared top-level request fields."""
        if self.values is not None:
            return self.values
        return _hints(self.root())


def _chat_root() -> Any:
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
    )

    return CompletionCreateParamsNonStreaming


def _responses_root() -> Any:
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
    )

    return ResponseCreateParamsNonStreaming


def _messages_root() -> Any:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    return MessageCreateParamsNonStreaming


def _google_root() -> Any:
    from google.genai.types import GenerateContentConfig

    return GenerateContentConfig


#: The Interactions API ships no Python type, so its top-level surface is
#: listed with the value each field is exercised with.
_INTERACTIONS_VALUES: dict[str, Any] = {
    "model": "m2",
    "input": "hi",
    "system_instruction": "be brief",
    "stream": True,
    "generation_config": {"temperature": 0.5},
    "response_format": {"type": "text", "mime_type": "application/json"},
    "store": True,
    "background": False,
    "previous_interaction_id": "i_1",
    "safety_settings": [],
    "service_tier": "default",
    "labels": {"k": "v"},
}

FORMATS: dict[str, Format] = {
    "chat_completions": Format(
        "chat_completions",
        ChatCompletionsConverter,
        _chat_root,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ),
    "open_responses": Format(
        "open_responses",
        OpenResponsesConverter,
        _responses_root,
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
            "stream": True,
        },
    ),
    "messages": Format(
        "messages",
        MessagesConverter,
        _messages_root,
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ),
    "google_generate": Format(
        "google_generate",
        GoogleGenerateConverter,
        _google_root,
        {"model": "m", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    ),
    "google_interactions": Format(
        "google_interactions",
        GoogleInteractionsConverter,
        dict,
        {"model": "m", "input": "hi"},
        values=_INTERACTIONS_VALUES,
    ),
}


def _class_of(format_name: str, field: str) -> str:
    """Return what the ``request-fields`` ledger says the reader does with it.

    Args:
        format_name: The format's name in :data:`FORMATS`.
        field: The declared request field.

    Returns:
        ``mapped``, ``dropped``, ``not_on_wire`` or ``unclassified``.
    """
    entry = _LEDGER[format_name]["fields"].get(field)
    return entry["class"] if entry else "unclassified"


def _value_for(format_name: str, field: str, annotation: Any) -> Any:
    """Return the value a case uses, or raise ValueError to skip it."""
    pinned = _VALUES.get((format_name, field))
    if pinned is not None:
        return pinned
    if FORMATS[format_name].values is not None:
        return annotation
    return _sample(annotation)


def _body(
    format: Format, field: str, value: Any, *, companions: bool = True
) -> dict[str, Any]:
    """Return the fixture with the case's field set where the reader looks.

    Args:
        format: The format whose fixture is the base.
        field: The field under test.
        value: Its value.
        companions: Whether to merge the field's companions in.

    Returns:
        The request body the reader is called with.
    """
    nest = _NESTED.get(format.name)
    if nest is None:
        return {**format.fixture, field: value}
    inner = {**format.fixture.get(nest, {})}
    if companions:
        inner.update(_COMPANIONS.get((format.name, field), {}))
    inner[field] = value
    return {**format.fixture, nest: inner}


def _cases() -> list[tuple[str, str, Any]]:
    """Return ``(format, field, value)`` for every exercisable top-level field."""
    cases: list[tuple[str, str, Any]] = []
    for name, format in FORMATS.items():
        for field, annotation in format.annotations().items():
            if field in _BUILT_BY_HAND:
                continue
            if (name, field) in _EXCLUSIONS:
                continue
            if _class_of(name, field) != "mapped":
                continue
            try:
                value = _value_for(name, field, annotation)
            except ValueError:
                continue
            cases.append((name, field, value))
    return cases


CASES = _cases()


@pytest.mark.parametrize(
    ("format_name", "field", "value"),
    CASES,
    ids=[f"{name}-{field}" for name, field, _value in CASES],
)
def test_a_same_format_round_trip_is_the_identity(
    format_name: str, field: str, value: Any
) -> None:
    """Reading what this dialect wrote yields the IR the first read produced."""
    format = FORMATS[format_name]
    converter = format.converter()

    base = converter.request_from_provider(dict(format.fixture))
    first = converter.request_from_provider(_body(format, field, value))
    assert first != base, (
        f"{format_name}.{field}: the reader ignores the field, so this case "
        "would pass with the reader deleted — the ledger calls it mapped"
    )

    written, _warnings = converter.request_to_provider(first)
    second = converter.request_from_provider(written)

    assert second == first, (
        f"{format_name}.{field} does not survive A -> IR -> A: the second read "
        "differs from the first"
    )


def test_every_format_contributes_cases() -> None:
    """A format whose fields all failed to sample would make its cases vacuous."""
    for name in FORMATS:
        assert any(case[0] == name for case in CASES), name


def test_every_mapped_field_becomes_a_case() -> None:
    """A field the ledger calls mapped has to be tested, or the gap is silent."""
    covered = {(case[0], case[1]) for case in CASES}
    for name, format in FORMATS.items():
        for field in format.annotations():
            if _class_of(name, field) != "mapped":
                continue
            if field in _BUILT_BY_HAND or (name, field) in _EXCLUSIONS:
                continue
            assert (name, field) in covered, (
                f"{name}.{field} is mapped in the request ledger but no case covers it"
            )


def test_every_case_names_a_declared_field() -> None:
    """A case for a field the format no longer declares would never apply."""
    for name, field, _value in CASES:
        assert field in FORMATS[name].annotations(), (name, field)


def test_every_companion_is_still_needed() -> None:
    """A companion the reader no longer needs would pin a pair for nothing."""
    for name, field in _COMPANIONS:
        format = FORMATS[name]
        converter = format.converter()
        value = _value_for(name, field, format.annotations()[field])
        base = converter.request_from_provider(dict(format.fixture))
        alone = converter.request_from_provider(
            _body(format, field, value, companions=False)
        )
        assert alone == base, (
            f"{name}.{field}: the companion {sorted(_COMPANIONS[(name, field)])} "
            "is no longer needed — the reader now reads the field on its own, "
            "so the case would prove less than it looks like it does"
        )


def test_every_exclusion_says_why() -> None:
    """An exclusion with no reason hides a case nobody decided to hide."""
    for key, reason in _EXCLUSIONS.items():
        assert reason, key


def test_every_pinned_value_belongs_to_a_declared_field() -> None:
    """A pin for a field the format no longer declares would never apply."""
    for name, field in _VALUES:
        assert field in FORMATS[name].annotations(), (name, field)

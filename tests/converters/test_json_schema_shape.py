"""The IR's ``json_schema`` is one shape, whichever dialect wrote it (#1327).

The field used to carry two: Chat Completions and Responses read and wrote the
OpenAI envelope — ``{name, description, schema, strict}`` — while the two
Google dialects read and wrote the raw JSON Schema.  A Google request detoured
through Responses therefore reached a writer that found no ``schema`` key,
wrote a format with none, and came back as ``json_object``: a schema the caller
sent became a request for "some JSON".

The IR now spells the envelope, every reader wraps a raw schema as its
``schema``, and every writer unwraps it.  These cases use the body shapes the
issue measured, not a simplified stand-in.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta.conversion_warnings import WarningSink, recording_to
from llm_rosetta.converters import (
    ChatCompletionsConverter,
    GoogleGenerateConverter,
    GoogleInteractionsConverter,
    OpenResponsesConverter,
)

#: The schema the issue's repro carries: a two-key object with one property.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name"],
}

#: The issue's own minimal shape, kept because it is what the report recorded.
_MINIMAL_SCHEMA: dict[str, Any] = {"type": "object"}

#: A Responses ``text.format`` envelope, as the Responses API defines it.
_ENVELOPE: dict[str, Any] = {
    "type": "json_schema",
    "name": "person",
    "description": "a person",
    "strict": True,
    "schema": _SCHEMA,
}


@pytest.fixture
def google() -> GoogleGenerateConverter:
    return GoogleGenerateConverter()


@pytest.fixture
def responses() -> OpenResponsesConverter:
    return OpenResponsesConverter()


def _google_body(schema: dict[str, Any]) -> dict[str, Any]:
    """A ``google_generate`` request carrying *schema*, as the issue's repro is."""
    return {
        "model": "gemini-2.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "Give me a person."}]}],
        "config": {
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    }


def _responses_body(format_: dict[str, Any]) -> dict[str, Any]:
    """An ``open_responses`` request whose ``text.format`` is *format_*."""
    return {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "Give me a person."}],
        "text": {"format": format_},
    }


def _read(converter: Any, body: dict[str, Any]) -> tuple[Any, list[Any]]:
    sink = WarningSink(leg="request")
    with recording_to(sink):
        ir = converter.request_from_provider(body)
    return ir, list(sink)


def _write(converter: Any, ir: Any) -> tuple[dict[str, Any], list[Any]]:
    sink = WarningSink(leg="request")
    with recording_to(sink):
        body, _ = converter.request_to_provider(ir)
    return body, list(sink)


def _google_schema(body: dict[str, Any]) -> Any:
    return (body.get("config") or {}).get("response_schema")


class TestGoogleThroughResponses:
    """``google_generate`` → ``open_responses`` → ``google_generate``."""

    def test_the_issue_repro_keeps_its_schema(self, google, responses):
        """The exact case from the issue: ``{type: object}`` survives the detour."""
        ir, _ = _read(google, _google_body(_MINIMAL_SCHEMA))
        assert ir["response_format"]["json_schema"] == {"schema": _MINIMAL_SCHEMA}

        written, _ = _write(responses, ir)
        assert written["text"]["format"] == {
            "type": "json_schema",
            "schema": _MINIMAL_SCHEMA,
        }

        back_ir, _ = _read(responses, written)
        assert back_ir["response_format"]["type"] == "json_schema"

        back, _ = _write(google, back_ir)
        assert _google_schema(back) == _MINIMAL_SCHEMA

    def test_every_field_of_the_schema_survives(self, google, responses):
        """The comparison is of the whole schema, not of a summary of it."""
        ir, _ = _read(google, _google_body(_SCHEMA))
        written, _ = _write(responses, ir)
        back_ir, _ = _read(responses, written)
        back, _ = _write(google, back_ir)

        assert _google_schema(back) == _SCHEMA

    def test_the_type_is_not_downgraded_to_json_object(self, google, responses):
        """The failure the issue recorded: the format came back as ``json_object``."""
        ir, _ = _read(google, _google_body(_SCHEMA))
        written, _ = _write(responses, ir)
        back_ir, _ = _read(responses, written)

        assert back_ir["response_format"]["type"] == "json_schema"


class TestResponsesThroughGoogle:
    """``open_responses`` → ``google_generate`` → ``open_responses``."""

    def test_google_is_written_the_raw_schema_not_the_envelope(self, responses, google):
        """Google takes a JSON Schema; the envelope's keys are not schema keywords."""
        ir, _ = _read(responses, _responses_body(_ENVELOPE))
        assert ir["response_format"]["json_schema"] == {
            key: _ENVELOPE[key] for key in ("name", "description", "strict", "schema")
        }

        written, _ = _write(google, ir)
        assert _google_schema(written) == _SCHEMA

    def test_the_schema_comes_back_unchanged(self, responses, google):
        ir, _ = _read(responses, _responses_body(_ENVELOPE))
        written, _ = _write(google, ir)
        back_ir, _ = _read(google, written)
        back, _ = _write(responses, back_ir)

        assert back["text"]["format"]["schema"] == _SCHEMA

    def test_the_keys_google_cannot_carry_are_reported(self, responses, google):
        """``name`` / ``description`` / ``strict`` have no Google spelling."""
        ir, _ = _read(responses, _responses_body(_ENVELOPE))
        _, warnings = _write(google, ir)

        subjects = {warning.subject for warning in warnings}
        assert "response_format.json_schema" in subjects
        assert any("name" in str(warning) for warning in warnings)

    def test_a_format_without_those_keys_is_not_reported(self, responses, google):
        """A bare schema loses nothing on the Google leg, so nothing is warned."""
        ir, _ = _read(
            responses, _responses_body({"type": "json_schema", "schema": _SCHEMA})
        )
        _, warnings = _write(google, ir)

        assert [w for w in warnings if w.subject.startswith("response_format")] == []


class TestInteractionsSharesTheGoogleShape:
    """The other Google dialect takes the same raw schema (#1327)."""

    def test_the_envelope_is_unwrapped_for_interactions(self, responses):
        interactions = GoogleInteractionsConverter()
        ir, _ = _read(responses, _responses_body(_ENVELOPE))

        written, _ = _write(interactions, ir)
        assert written["response_format"]["response_schema"] == _SCHEMA

    def test_a_raw_schema_is_wrapped_on_the_way_back(self):
        interactions = GoogleInteractionsConverter()
        ir, _ = _read(
            interactions,
            {
                "model": "gemini-3.6-flash",
                "input": "hi",
                "response_format": {
                    "mime_type": "application/json",
                    "response_schema": _SCHEMA,
                },
            },
        )

        assert ir["response_format"]["json_schema"] == {"schema": _SCHEMA}


class TestTheWritersStillAcceptABareSchema:
    """An IR producer built before the envelope was the contract is tolerated."""

    def test_responses_writes_a_bare_schema_rather_than_dropping_it(self, responses):
        written, _ = _write(
            responses,
            {
                "model": "gpt-5",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "response_format": {"type": "json_schema", "json_schema": _SCHEMA},
            },
        )

        assert written["text"]["format"] == {"type": "json_schema", "schema": _SCHEMA}

    def test_chat_writes_a_bare_schema_rather_than_dropping_it(self):
        chat = ChatCompletionsConverter()
        written, _ = _write(
            chat,
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "response_format": {"type": "json_schema", "json_schema": _SCHEMA},
            },
        )

        assert written["response_format"] == {
            "type": "json_schema",
            "json_schema": {"schema": _SCHEMA},
        }


class TestARawSchemaIsNotReportedAsALoss:
    """A raw schema loses nothing on the Google leg — including its keywords.

    ``description`` and ``name`` are ordinary JSON Schema keywords; the first
    version of the loss report counted them by key presence, so it warned
    "description is dropped" on the same call that wrote it to the wire.
    """

    #: A raw schema that uses the envelope's key *names* as JSON Schema keywords.
    _RAW_WITH_KEYWORDS: dict[str, Any] = {
        "type": "object",
        "description": "a person",
        "properties": {"name": {"type": "string", "description": "their name"}},
    }

    @staticmethod
    def _ir(json_schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }

    def test_google_generate_writes_the_keywords_and_reports_nothing(self, google):
        written, warnings = _write(google, self._ir(self._RAW_WITH_KEYWORDS))

        assert _google_schema(written) == self._RAW_WITH_KEYWORDS
        assert [w for w in warnings if w.subject.startswith("response_format")] == []

    def test_google_interactions_writes_the_keywords_and_reports_nothing(self):
        interactions = GoogleInteractionsConverter()
        written, warnings = _write(interactions, self._ir(self._RAW_WITH_KEYWORDS))

        assert written["response_format"]["response_schema"] == self._RAW_WITH_KEYWORDS
        assert [w for w in warnings if w.subject.startswith("response_format")] == []

    def test_google_interactions_reports_the_envelope_it_drops(self):
        interactions = GoogleInteractionsConverter()
        _, warnings = _write(
            interactions,
            self._ir({"name": "person", "schema": _SCHEMA, "strict": True}),
        )

        assert any(
            w.subject == "response_format.json_schema" and "name" in str(w)
            for w in warnings
        )

    @pytest.mark.parametrize(
        ("strict", "reported"),
        [(False, False), (True, True)],
    )
    def test_strict_is_only_a_loss_when_it_asks_for_more(
        self, google, strict: bool, reported: bool
    ):
        """``strict: false`` asks for less than Google already enforces."""
        _, warnings = _write(google, self._ir({"schema": _SCHEMA, "strict": strict}))

        subjects = [
            w.subject for w in warnings if w.subject == "response_format.json_schema"
        ]
        assert bool(subjects) is reported

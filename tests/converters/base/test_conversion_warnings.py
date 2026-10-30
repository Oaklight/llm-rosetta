"""One sink for a conversion's losses, reachable from every leg.

The three channels this replaces are described in
:mod:`llm_rosetta.conversion_warnings`: free strings on the
request context, ``logger.warning`` calls inside converters, and stdlib
``warnings``.  What is pinned here is the property the module exists for — a
caller that reads ``pipeline.warnings`` has heard about every loss, on every
leg, exactly once — plus the compatibility that made the change a lifting
rather than a rewrite: a warning is still the string it always was.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

from llm_rosetta import ConversionPipeline
from llm_rosetta.conversion_warnings import (
    DOWNGRADED,
    DROPPED,
    FAILED,
    ConversionWarning,
    WarningSink,
    recording_to,
    report_drop,
    report_loss,
    use_sink,
)

REQUEST: dict = {"model": "test-model", "input": "hello"}


class TestTheWarning:
    """A loss is its message, plus provenance that rides alongside."""

    def test_it_is_the_message_it_always_was(self):
        warning = ConversionWarning("a part was dropped", code=DROPPED)

        assert warning == "a part was dropped"
        assert str(warning) == "a part was dropped"
        assert f"{warning}" == "a part was dropped"
        assert ["a part was dropped"] == [warning]

    def test_it_carries_its_provenance(self):
        warning = ConversionWarning(
            "dropped",
            code=DOWNGRADED,
            leg="stream",
            converter="a->b",
            subject="web_search",
        )

        assert warning.as_dict() == {
            "code": DOWNGRADED,
            "leg": "stream",
            "converter": "a->b",
            "subject": "web_search",
            "message": "dropped",
        }

    def test_provenance_survives_a_copy(self):
        warning = ConversionWarning("dropped", code=DOWNGRADED, subject="file")

        cloned = copy.deepcopy(warning)

        assert isinstance(cloned, ConversionWarning)
        assert cloned.as_dict() == warning.as_dict()


class TestTheSink:
    """The sink is still a list; recording is where the logging happens."""

    def test_a_plain_string_arrives_with_the_sinks_provenance(self):
        sink = WarningSink()
        sink.leg = "response"
        sink.converter = "open_responses->messages"

        sink.append("a block was dropped")

        [warning] = sink
        assert isinstance(warning, ConversionWarning)
        assert warning.code == DROPPED
        assert warning.leg == "response"
        assert warning.converter == "open_responses->messages"

    def test_an_annotated_warning_keeps_its_own_provenance(self):
        sink = WarningSink()
        sink.leg = "request"

        sink.append(ConversionWarning("x", code=FAILED, leg="stream", subject="t"))

        [warning] = sink
        assert warning.leg == "stream"
        assert warning.code == FAILED
        assert warning.subject == "t"

    def test_recording_writes_one_log_line(self, caplog):
        sink = WarningSink()
        sink.leg = "stream"

        with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
            sink.append("a part was dropped")

        [record] = [r for r in caplog.records if r.name == "llm_rosetta"]
        assert record.getMessage() == (
            "Conversion warning (stream, dropped): a part was dropped"
        )

    def test_a_silent_sink_records_without_logging(self, caplog):
        sink = WarningSink()
        sink.log = False

        with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
            sink.append("a shadow round trip lost something")

        assert [r for r in caplog.records if r.name == "llm_rosetta"] == []
        assert list(sink) == ["a shadow round trip lost something"]

    def test_warn_records_the_code_and_subject(self):
        sink = WarningSink()

        warning = sink.warn("down", code=DOWNGRADED, subject="refusal")

        assert sink == [warning]
        assert (warning.code, warning.subject) == (DOWNGRADED, "refusal")

    def test_extending_records_every_message(self, caplog):
        sink = WarningSink()
        sink.log = False

        sink.extend(["one", "two"])

        assert [w.leg for w in sink] == ["request", "request"]

    def test_iadd_records_through_the_sink(self, caplog):
        sink = WarningSink()

        with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
            sink += ["added with +="]

        [warning] = sink
        assert isinstance(warning, ConversionWarning)
        assert any(r.name == "llm_rosetta" for r in caplog.records)


class TestReportDrop:
    """A converter reports once, wherever it is and whoever is listening."""

    def test_an_explicit_sink_wins_over_the_ambient_one(self, caplog):
        explicit, ambient = WarningSink(), WarningSink()
        token = use_sink(ambient)
        try:
            with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
                report_drop("dropped", warnings=explicit)
        finally:
            from llm_rosetta.conversion_warnings import reset_sink

            reset_sink(token)

        assert explicit == ["dropped"]
        assert ambient == []

    def test_the_conversion_in_progress_is_used_when_no_sink_is_named(self):
        ambient = WarningSink()
        token = use_sink(ambient)
        try:
            report_drop("dropped from a helper with no context")
        finally:
            from llm_rosetta.conversion_warnings import reset_sink

            reset_sink(token)

        assert ambient == ["dropped from a helper with no context"]

    def test_with_no_sink_it_logs(self, caplog):
        with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
            warning = report_drop("nobody is collecting")

        assert warning == "nobody is collecting"
        assert any(
            r.getMessage()
            == "Conversion warning (unknown, dropped): nobody is collecting"
            for r in caplog.records
        )

    def test_a_plain_list_is_the_callers_and_is_not_logged(self, caplog):
        collected: list = []

        with caplog.at_level(logging.WARNING, logger="llm_rosetta"):
            report_drop("for a caller that collects its own", warnings=collected)

        assert collected == ["for a caller that collects its own"]
        assert [r for r in caplog.records if r.name == "llm_rosetta"] == []


class TestReportLoss:
    """One API for all three kinds of loss, and a leg it can name."""

    def test_it_records_the_code_and_subject(self):
        sink = WarningSink(leg="request", converter="a->b")
        warning = report_loss(
            "a field was dropped", code=DOWNGRADED, subject="field", warnings=sink
        )
        assert warning.code == DOWNGRADED
        assert warning.subject == "field"
        assert sink == [warning]

    def test_the_leg_override_leaves_the_cursor_alone(self):
        """One warning on another leg must not mislabel the next one."""
        sink = WarningSink(leg="request")
        on_stream = report_loss("stream loss", leg="stream", warnings=sink)
        after = report_loss("request loss", warnings=sink)

        assert on_stream.leg == "stream"
        assert after.leg == "request"
        assert sink.leg == "request"

    def test_without_a_leg_it_takes_the_sinks(self):
        sink = WarningSink(leg="response")
        warning = report_loss("a loss", warnings=sink)
        assert warning.leg == "response"

    def test_the_alias_records_through_the_same_route(self, monkeypatch):
        import llm_rosetta.conversion_warnings as warnings_module

        calls: list[tuple[str, dict]] = []

        def _spy(message, **kwargs):
            calls.append((message, kwargs))
            return ConversionWarning(message)

        monkeypatch.setattr(warnings_module, "report_loss", _spy)
        report_drop("old spelling", code=DROPPED, subject="s")
        assert calls == [
            ("old spelling", {"code": DROPPED, "subject": "s", "warnings": None})
        ]


class TestACallersOwnSinkWins:
    """A sink the caller installed keeps receiving; the entry point backs off."""

    def test_an_installed_sink_is_not_stolen(self):
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.chat_completions import (
            ChatCompletionsConverter,
        )

        caller = WarningSink(leg="caller")
        ctx = ConversionContext()
        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "generation": {"top_k": 5},
        }

        with recording_to(caller):
            ChatCompletionsConverter().request_to_provider(ir, context=ctx)

        assert [w.subject for w in caller] == ["top_k"]
        assert caller.leg == "caller", "the caller's cursor must not move"
        assert ctx.warnings == [], "the entry point must not install its own sink"

    def test_a_tool_config_loss_also_reaches_the_caller(self):
        """The site that was missed: an append straight onto the context.

        A loss that writes to ``ctx.warnings`` itself bypasses the sink the
        caller installed, so one request could deliver two losses to two
        different lists.
        """
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.chat_completions import (
            ChatCompletionsConverter,
        )

        caller = WarningSink(leg="caller")
        ctx = ConversionContext()
        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            # A tool definition keeps the request-level repair from stripping
            # the tool_config as orphaned before the writer sees it.
            "tools": [{"type": "function", "name": "fn", "parameters": {}}],
            "tool_config": {"max_calls": 3},
        }

        with recording_to(caller):
            ChatCompletionsConverter().request_to_provider(ir, context=ctx)

        assert [w.subject for w in caller] == ["max_tool_calls"]
        assert ctx.warnings == []

    def test_a_tool_choice_loss_reaches_the_caller_too(self):
        """The Responses writer stopped handing its context sink to the helper.

        The undeclared-name report lives in a tool-choice helper that has no
        context.  It used to be reachable only through an explicit
        ``warnings=ctx.warnings`` argument, so a caller that installed its own
        sink heard nothing; the entry point's sink is now the one it lands in.
        """
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = WarningSink(leg="caller")
        ctx = ConversionContext()
        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [{"type": "function", "name": "declared", "parameters": {}}],
            "tool_choice": {
                "mode": "auto",
                "allowed_tools": [{"name": "declared"}, {"name": "missing"}],
            },
        }

        with recording_to(caller):
            OpenResponsesConverter().request_to_provider(ir, context=ctx)

        assert [w.subject for w in caller] == ["allowed_tools"]
        assert caller[0].code == DOWNGRADED
        assert ctx.warnings == []


class TestTheSinkBeatsALegacyList:
    """A list a helper builds no longer outranks the conversion in progress.

    Helpers that predate the sink appended to a ``list[str]`` and their caller
    extended the conversion's sink with it.  That silently delivered nothing to
    a caller that installed its own sink: the list was filled, the caller's
    sink was not, and the conversion threw the list away.  Each probe below is
    a site where that used to happen.
    """

    def _caller(self):
        return WarningSink(leg="caller")

    @staticmethod
    def _request(messages: list) -> Any:
        return {"model": "m", "messages": messages}

    def test_an_unknown_part_reaches_the_caller(self):
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = self._caller()
        ir = self._request([{"role": "user", "content": [{"type": "mystery"}]}])

        with recording_to(caller):
            OpenResponsesConverter().request_to_provider(ir)

        assert [(w.code, w.subject) for w in caller] == [("dropped", "mystery")]

    def test_an_unknown_part_in_an_assistant_turn_reaches_the_caller(self):
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = self._caller()
        ir = self._request(
            [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "mystery"}]},
            ]
        )

        with recording_to(caller):
            OpenResponsesConverter().request_to_provider(ir)

        assert [(w.code, w.subject) for w in caller] == [("dropped", "mystery")]

    def test_a_harvested_additional_tools_loss_reaches_the_caller(self):
        """The harvest helper took the context's sink positionally."""
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = self._caller()
        provider_request = {
            "model": "m",
            "input": [{"type": "additional_tools", "tools": ["not a tool"]}],
        }

        with recording_to(caller):
            OpenResponsesConverter().request_from_provider(provider_request)

        assert [w.subject for w in caller] == ["additional_tools"]

    def test_an_unrenderable_server_result_reaches_the_caller(self):
        """The response leg handed the context's sink to the tool writer."""
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = self._caller()
        ir: Any = {
            "id": "r",
            "object": "response",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": "c1",
                                "tool_result": "x",
                                "tool_type": "intrinsic",
                                "intrinsic_kind": "mystery_search",
                            }
                        ],
                    },
                    "finish_reason": {"reason": "stop"},
                }
            ],
        }

        with recording_to(caller):
            OpenResponsesConverter().response_to_provider(ir)

        assert [w.subject for w in caller] == ["mystery_search"]
        # The caller's sink keeps its own leg cursor, so a site that does not
        # name its leg reports a response loss as a request one.
        assert caller[0].leg == "response"

    def test_a_read_side_allow_list_loss_reaches_the_caller(self):
        """The shared helper in ``ir/tools`` had no ambient route at all."""
        from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

        caller = self._caller()
        provider_request = {
            "model": "m",
            "input": "hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function"}],
            },
        }

        with recording_to(caller):
            OpenResponsesConverter().request_from_provider(provider_request)

        assert [(w.code, w.subject) for w in caller] == [
            ("downgraded", "tool_choice.allowed_tools")
        ]

    def test_a_list_does_not_outrank_the_conversion_in_progress(self):
        """Directly: the rule the five probes above depend on."""
        sink = WarningSink(leg="response")
        legacy: list[str] = []

        with recording_to(sink):
            report_loss("a part was dropped", subject="part", warnings=legacy)

        assert [w.subject for w in sink] == ["part"]
        assert legacy == [], "the list the caller handed in stays empty"

    def test_google_interactions_losses_reach_the_caller_with_their_leg(self):
        """Both directions: the definition on the way out, the mode on the way in."""
        from llm_rosetta.converters.llm.google_interactions import (
            GoogleInteractionsConverter,
        )

        converter = GoogleInteractionsConverter()

        outgoing = self._caller()
        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "type": "intrinsic",
                    "name": "mystery",
                    "intrinsic_kind": "mystery_kind",
                }
            ],
        }
        with recording_to(outgoing):
            converter.request_to_provider(ir)

        incoming = self._caller()
        provider_request = {
            "model": "m",
            "input": "hi",
            "generation_config": {
                "tool_choice": {
                    "allowed_tools": {
                        "mode": "whenever",
                        "tools": [{"type": "function", "name": "x"}],
                    }
                }
            },
        }
        with recording_to(incoming):
            converter.request_from_provider(provider_request)

        assert [(w.code, w.subject, w.leg) for w in outgoing] == [
            ("dropped", "mystery_kind", "request")
        ]
        # Reading a provider request is the request leg — the leg is the
        # payload being converted, not the direction the data flows.
        assert [(w.code, w.subject, w.leg) for w in incoming] == [
            ("downgraded", "tool_choice.allowed_tools", "request")
        ]

    def test_google_generate_losses_reach_the_caller_with_their_leg(self):
        """The leg is named by the site, on all three legs and both read paths."""
        from llm_rosetta.converters.llm.google_generate import (
            GoogleGenerateConverter,
        )

        converter = GoogleGenerateConverter()

        def read(call):
            caller = self._caller()
            with recording_to(caller):
                call()
            return [(w.code, w.subject, w.leg) for w in caller]

        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            # A server tool Google has no inline form for, and a
            # restriction it can only state with mode ANY.
            "tools": [
                {
                    "type": "intrinsic",
                    "name": "mystery",
                    "intrinsic_kind": "mystery_kind",
                }
            ],
            "tool_choice": {
                "mode": "auto",
                "allowed_tools": [{"name": "mystery"}],
            },
        }
        outgoing = read(lambda: converter.request_to_provider(ir))
        assert outgoing == [
            ("dropped", "mystery_kind", "request"),
            ("downgraded", "allowed_tools", "request"),
        ]

        # The reader serves both legs, so the leg is a parameter, not a constant.
        request_read = read(
            lambda: converter.request_from_provider(
                {
                    "model": "m",
                    "contents": [{"role": "user", "parts": [{"unknownPart": {}}]}],
                }
            )
        )
        assert request_read == [("dropped", "unknownPart", "request")]

        response_read = read(
            lambda: converter.response_from_provider(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"unknownPart": {}}],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            )
        )
        assert response_read == [("dropped", "unknownPart", "response")]

        # A replayed server call is history, not the turn being returned: the
        # writer has to say so, or the template takes its response default —
        # which drops the call from the request body and labels the loss with
        # the wrong leg.
        history = read(
            lambda: converter.request_to_provider(
                cast(
                    Any,
                    {
                        "model": "m",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_call",
                                        "tool_call_id": "c1",
                                        "tool_name": "mystery_search",
                                        "tool_input": {},
                                        "tool_type": "intrinsic",
                                        "intrinsic_kind": "mystery_search",
                                    }
                                ],
                            },
                        ],
                    },
                )
            )
        )
        assert history == [("dropped", "mystery_search", "request")]

        # No context: the sink is the only place the loss can land, so the
        # site's own word is the only thing that says which leg it was.
        stream_read = read(
            lambda: converter.stream_response_to_provider(
                cast(
                    Any,
                    {
                        "type": "tool_result",
                        "tool_call_id": "c1",
                        "result": "R",
                        "tool_type": "intrinsic",
                        "provider_metadata": {"intrinsic_kind": "web_search"},
                    },
                )
            )
        )
        assert stream_read == [("dropped", "web_search", "stream")]

    def test_an_explicit_sink_still_wins(self):
        """A sink is the conversion's own answer to where losses go."""
        conversion = WarningSink(leg="request")
        other = WarningSink(leg="caller")

        with recording_to(other):
            report_loss("a part was dropped", subject="part", warnings=conversion)

        assert [w.subject for w in conversion] == ["part"]
        assert other == []


class TestTheBaseStreamDefaults:
    """The dispatch table's own handler is the rule for a dialect without a form."""

    def test_a_dialect_with_no_refusal_form_reports_the_words(self):
        """Base ``refusal_delta`` is unreachable today — and must still report.

        Every dialect in the tree overrides it, so the fallback is what the
        *next* one inherits: without it, a wire with no refusal slot would
        drop the words silently, which is the loss class the sink exists for.
        Called directly, because "every dialect overrides it" is exactly why
        no conversion can reach it.
        """
        from llm_rosetta.converters.base import BaseConverter
        from llm_rosetta.converters.llm.messages import MessagesConverter

        sink = WarningSink()
        event: Any = {"type": "refusal_delta", "refusal": "I can't help with that."}

        with recording_to(sink):
            result = BaseConverter._handle_ir_refusal_delta_to_p(
                MessagesConverter(), event, None
            )

        assert result == {}
        assert [warning.subject for warning in sink] == ["refusal"]
        assert sink[0].code == DROPPED
        assert "no form in this target" in sink[0]

    def test_a_messages_allowed_tools_loss_reaches_the_caller(self):
        """The Messages writer stopped handing its context sink to the helper.

        Anthropic has no allow-list tool choice, so the choice is widened to
        "any declared tool".  That loss used to travel through an explicit
        ``warnings=ctx.warnings`` argument; a caller with its own sink heard
        nothing.
        """
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.messages import MessagesConverter

        caller = WarningSink(leg="caller")
        ctx = ConversionContext()
        ir: Any = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [{"type": "function", "name": "declared", "parameters": {}}],
            "tool_choice": {"mode": "auto", "allowed_tools": [{"name": "declared"}]},
        }

        with recording_to(caller):
            MessagesConverter().request_to_provider(ir, context=ctx)

        assert [w.subject for w in caller] == ["allowed_tools"]
        assert caller[0].code == DOWNGRADED
        assert ctx.warnings == []


class TestTheEntryPointsNameTheirLeg:
    """The act knows its leg; it does not wait for the driver to say."""

    def test_a_response_read_on_a_fresh_context_is_a_response(self, monkeypatch):
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.messages import MessagesConverter

        def _read(self, provider_response, *, context, **kwargs):
            report_loss("lost while reading", subject="x", warnings=context.warnings)
            return {
                "id": "msg_1",
                "object": "response",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": []},
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            }

        monkeypatch.setattr(MessagesConverter, "_do_response_from_provider", _read)
        ctx = ConversionContext()
        assert ctx.warnings.leg == "request"

        MessagesConverter().response_from_provider({"id": "msg_1"}, context=ctx)

        [warning] = ctx.warnings
        assert warning.leg == "response"

    def test_a_request_write_on_a_response_leg_sink_is_a_request(self, monkeypatch):
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.llm.messages import MessagesConverter

        def _write(self, ir_request, *, context, **kwargs):
            report_loss("lost while writing", subject="y", warnings=context.warnings)
            return {"model": "m", "messages": []}

        monkeypatch.setattr(MessagesConverter, "_do_request_to_provider", _write)
        ctx = ConversionContext()
        ctx.warnings.leg = "response"

        MessagesConverter().request_to_provider(
            {"model": "m", "messages": []}, context=ctx
        )

        [warning] = ctx.warnings
        assert warning.leg == "request"


class TestThePipelineSink:
    """The pipeline is where the legs meet."""

    def test_a_request_leg_loss_carries_its_label(self):
        pipe = ConversionPipeline("open_responses", "messages")

        pipe.convert_request(
            {
                "model": "test-model",
                "input": "hello",
                "previous_response_id": "resp_abc",
            }
        )

        warnings = [w for w in pipe.warnings if "previous_response_id" in w]
        assert warnings, pipe.warnings
        assert warnings[0].leg == "request"
        assert warnings[0].converter == "open_responses->messages"

    def test_the_response_leg_reports_into_the_same_sink(self):
        pipe = ConversionPipeline("chat_completions", "chat_completions")
        pipe.convert_request(dict(REQUEST))

        pipe.convert_response(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        assert pipe.warnings is pipe.context.warnings
        assert all(w.leg == "response" for w in pipe.warnings)

    def test_a_stream_processor_reports_into_the_pipeline_sink(self):
        pipe = ConversionPipeline("chat_completions", "messages")
        pipe.convert_request(dict(REQUEST))

        processor = pipe.create_stream_processor()
        processor.source_context.warnings.append("dropped while streaming")

        [warning] = [w for w in pipe.warnings if "while streaming" in w]
        assert warning.leg == "stream"
        assert warning.converter == "chat_completions->messages"

    def test_the_converter_runs_under_the_pipeline_sink(self):
        """The ambient path is what a helper with no context uses."""
        pipe = ConversionPipeline("messages", "chat_completions")

        pipe.convert_request(
            {
                "model": "test-model",
                "max_tokens": 16,
                "top_k": 40,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )

        [warning] = [w for w in pipe.warnings if w.subject == "top_k"]
        assert warning.leg == "request"
        assert warning.code == DROPPED
        assert warning.converter == "messages->chat_completions"

    def test_a_report_from_inside_a_chunk_reaches_the_sink(self, monkeypatch):
        """The stream leg, reported the way a converter inside it would."""
        from llm_rosetta.converters.llm.messages import MessagesConverter

        original = MessagesConverter.stream_response_from_provider

        def reported(self, chunk, context=None):
            report_drop("a chunk lost something", subject="chunk-1")
            return original(self, chunk, context=context)

        monkeypatch.setattr(
            MessagesConverter, "stream_response_from_provider", reported
        )
        pipe = ConversionPipeline("chat_completions", "messages")
        pipe.convert_request(dict(REQUEST))
        processor = pipe.create_stream_processor()

        processor.process_chunk(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "model": "test-model",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            }
        )

        [warning] = [w for w in pipe.warnings if w.subject == "chunk-1"]
        assert warning.leg == "stream"
        assert warning.converter == "chat_completions->messages"

    def test_the_idle_sink_is_one_object(self):
        pipe = ConversionPipeline("chat_completions", "messages")

        assert pipe.warnings is pipe.warnings
        assert pipe.warnings == []

    def test_the_ambient_sink_ends_with_the_conversion(self):
        from llm_rosetta.conversion_warnings import current_sink

        pipe = ConversionPipeline("chat_completions", "messages")
        pipe.convert_request(dict(REQUEST))

        assert current_sink() is None

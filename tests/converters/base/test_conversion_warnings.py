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

from llm_rosetta import ConversionPipeline
from llm_rosetta.conversion_warnings import (
    DOWNGRADED,
    DROPPED,
    FAILED,
    ConversionWarning,
    WarningSink,
    report_drop,
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

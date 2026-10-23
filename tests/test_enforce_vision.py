"""The vision pass that runs on every request.

``enforce_vision`` is the library-level policy: an IR request bound for a model
that did not declare ``vision`` carries text placeholders instead of images, so
the target never answers with a content-type error.  It used to be shadowed by
a shim-facing factory that duplicated the decision and had no production
caller; these tests cover the pass itself.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.conversion_warnings import (
    DROPPED,
    ConversionWarning,
    WarningSink,
    recording_to,
)
from llm_rosetta.ir.passes import enforce_vision
from llm_rosetta.pipeline import ConversionPipeline

IMAGE: dict[str, Any] = {
    "type": "image",
    "source": {"type": "base64", "data": "AAAA", "media_type": "image/png"},
}
PLACEHOLDER = {"type": "text", "text": "[image not available]"}


def _request(*parts: Any) -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": list(parts)}],
    }


def _images(request: dict[str, Any]) -> list[Any]:
    return [
        part
        for message in request["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    ]


class TestTheDecision:
    def test_unknown_capabilities_leave_the_request_alone(self):
        request = _request({"type": "text", "text": "hi"}, IMAGE)

        assert enforce_vision(request, model_capabilities=None) is request

    def test_a_vision_model_keeps_its_images(self):
        request = _request(IMAGE)

        result = enforce_vision(request, model_capabilities=["text", "vision"])

        assert result is request
        assert len(_images(result)) == 1

    def test_an_undeclared_vision_capability_strips(self):
        request = _request(IMAGE)

        result = enforce_vision(request, model_capabilities=["text"])

        assert _images(result) == []
        assert result["messages"][0]["content"] == [PLACEHOLDER]

    def test_a_declined_capability_list_strips(self):
        """``[]`` is "declined"; only ``None`` means unknown."""
        result = enforce_vision(_request(IMAGE), model_capabilities=[])

        assert _images(result) == []


class TestWhatItTouches:
    def test_text_beside_an_image_survives_in_place(self):
        request = _request(
            {"type": "text", "text": "describe"}, IMAGE, {"type": "text", "text": "?"}
        )

        result = enforce_vision(request, model_capabilities=["text"])

        assert result["messages"][0]["content"] == [
            {"type": "text", "text": "describe"},
            PLACEHOLDER,
            {"type": "text", "text": "?"},
        ]

    def test_an_image_inside_a_tool_result_is_stripped_too(self):
        tool_result = {
            "type": "tool_result",
            "tool_call_id": "call_1",
            "result": [{"type": "text", "text": "screenshot:"}, IMAGE],
        }
        request = _request(tool_result)

        result = enforce_vision(request, model_capabilities=["text"])

        assert result["messages"][0]["content"][0]["result"] == [
            {"type": "text", "text": "screenshot:"},
            PLACEHOLDER,
        ]

    def test_a_request_without_images_is_returned_unchanged(self):
        request = _request({"type": "text", "text": "hi"})

        assert enforce_vision(request, model_capabilities=["text"]) is request

    def test_the_callers_request_is_not_mutated(self):
        request = _request(IMAGE)

        enforce_vision(request, model_capabilities=["text"])

        assert len(_images(request)) == 1


class TestTheReport:
    def test_the_strip_names_the_subject(self):
        sink = WarningSink()
        with recording_to(sink):
            enforce_vision(_request(IMAGE), model_capabilities=["text"], model="m")

        assert [w.subject for w in sink] == ["image"]
        assert [w.code for w in sink] == [DROPPED]

    def test_the_default_request_id_is_not_a_prefix(self):
        """``-`` is the sentinel default, not an id worth showing a reader."""
        sink = WarningSink()
        with recording_to(sink):
            enforce_vision(_request(IMAGE), model_capabilities=["text"], model="m")

        assert str(sink[0]) == (
            "stripped 1 images: model m does not have vision capability"
        )

    def test_a_real_request_id_is_kept(self):
        sink = WarningSink()
        with recording_to(sink):
            enforce_vision(
                _request(IMAGE),
                model_capabilities=["text"],
                model="m",
                request_id="r1",
            )

        assert str(sink[0]).startswith("[r1] stripped 1 images")

    def test_no_report_when_nothing_is_stripped(self):
        sink = WarningSink()
        with recording_to(sink):
            enforce_vision(_request(IMAGE), model_capabilities=["text", "vision"])

        assert sink == []


class TestThroughThePipeline:
    """The pass is wired into the request leg, not only importable."""

    REQUEST: dict[str, Any] = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
    }

    def test_a_non_vision_target_never_receives_the_image(self):
        pipe = ConversionPipeline(
            "chat_completions",
            "chat_completions",
            model_capabilities=["text"],
        )

        body = pipe.convert_request(dict(self.REQUEST))

        parts = body["messages"][0]["content"]
        assert all(part.get("type") != "image_url" for part in parts)
        assert any(part.get("text") == "[image not available]" for part in parts)
        assert [w.subject for w in pipe.warnings] == ["image"]

    def test_a_vision_target_keeps_it(self):
        pipe = ConversionPipeline(
            "chat_completions",
            "chat_completions",
            model_capabilities=["text", "vision"],
        )

        body = pipe.convert_request(dict(self.REQUEST))

        parts = body["messages"][0]["content"]
        assert any(part.get("type") == "image_url" for part in parts)
        assert not any(
            isinstance(w, ConversionWarning) and w.subject == "image"
            for w in pipe.warnings
        )

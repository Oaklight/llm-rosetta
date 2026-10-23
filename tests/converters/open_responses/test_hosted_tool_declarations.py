"""The OpenAI profile's hosted (intrinsic) tool declarations.

The profile declares the server-tool kinds a *foreign* request may be
translated into.  A kind qualifies only when its wire payload can be built
without configuration the client would have had to supply — measured against
the live API:

===========================  ==========================================
``image_generation``         200 with ``{"type": "image_generation"}``
``code_interpreter``         needs ``container``; synthesised as ``auto``
``file_search``              needs ``vector_store_ids``
``mcp``                      needs ``server_label``
``computer_use_preview``     needs ``environment``
``shell`` / ``tool_search``  model-gated (400 "not supported with <model>")
===========================  ==========================================

The second group stays source-only: it is recognised on the way in and
restored same-format from the preserved native payload, but a cross-format
request never invents one.  When a target *cannot* take a kind the caller
declared, the definition is dropped and reported — not silently removed.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta import ConversionPipeline, get_converter_for_provider
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.llm.open_responses.tool_ops import (
    _RESPONSES_INTRINSIC_TOOLS,
)
from llm_rosetta.ir.tools.intrinsic import intrinsic_kind_of
from llm_rosetta.shims.provider_shim import resolve_shim

REQUEST: dict[str, Any] = {
    "model": "m",
    "input": "draw a cat",
    "tools": [{"type": "image_generation"}],
}


def _ir_request(kind: str) -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"type": "intrinsic", "intrinsic_kind": kind}],
    }


class TestTheDeclaration:
    def test_the_profile_declares_the_synthesizable_kinds(self):
        shim = resolve_shim("openai--open_responses")

        assert shim is not None
        declared = set(shim.tools.intrinsic_tools)
        assert {"web_search", "code_interpreter", "image_generation"} <= declared
        # Every declared kind has to be one this converter can actually write,
        # or the shim would promise a translation that fails at the wire.
        assert declared <= set(_RESPONSES_INTRINSIC_TOOLS)

    def test_the_neutral_base_declares_none(self):
        """A bare ``open_responses`` provider is not the OpenAI route."""
        shim = resolve_shim("open_responses")

        assert shim is None or shim.tools.intrinsic_tools == ()


class TestTheWrite:
    def test_image_generation_is_written_bare(self):
        converter = get_converter_for_provider("open_responses")

        body, _ = converter.request_to_provider(
            _ir_request("image_generation"), context=ConversionContext()
        )

        assert body["tools"] == [{"type": "image_generation"}]

    def test_a_same_format_read_keeps_the_kind(self):
        converter = get_converter_for_provider("open_responses")

        back = converter.request_from_provider(
            dict(REQUEST), context=ConversionContext()
        )

        [tool] = back["tools"]
        assert tool["type"] == "intrinsic"
        assert intrinsic_kind_of(tool) == "image_generation"
        # The native payload is preserved for the same-format round trip.
        assert tool["metadata"]["_native"]["tool"] == {"type": "image_generation"}


class TestTheDrop:
    def test_a_target_without_the_kind_drops_and_reports_it(self):
        pipe = ConversionPipeline("open_responses", "chat_completions")

        body = pipe.convert_request(dict(REQUEST))

        assert "tools" not in body or body["tools"] == []
        [warning] = [
            w for w in pipe.warnings if getattr(w, "subject", "") == "image_generation"
        ]
        assert warning.code == "dropped"
        assert "does not support intrinsic kind" in warning

    def test_the_same_format_route_keeps_it_and_reports_nothing(self):
        pipe = ConversionPipeline("open_responses", "open_responses")

        pipe.convert_request(dict(REQUEST))

        assert pipe.warnings == []

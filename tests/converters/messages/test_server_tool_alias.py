"""Anthropic server-tool version aliases on the read path.

Anthropic names a server tool's runtime call/result after the tool *version*.
The newer code-execution versions (``code_execution_20250825`` and later) run
bash under the hood and report it under a different runtime name:

    call   -> ``server_tool_use`` named ``bash_code_execution``
    result -> ``bash_code_execution_tool_result`` block

The shim declares the canonical kind ``code_execution``.  The read path must
fold the alias onto that kind, or a newer-version response carries a kind no
shim declares.
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind

_OLD_CALL = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "code_execution",
    "input": {"code": "print(1)"},
}
_NEW_CALL = {
    "type": "server_tool_use",
    "id": "srvtoolu_2",
    "name": "bash_code_execution",
    "input": {"command": "print(1)"},
}
_OLD_RESULT = {
    "type": "code_execution_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": {
        "type": "code_execution_result",
        "stdout": "1",
        "stderr": "",
        "return_code": 0,
        "content": [],
        "abort_reason": None,
    },
}
_NEW_RESULT = {
    "type": "bash_code_execution_tool_result",
    "tool_use_id": "srvtoolu_2",
    "content": {
        "type": "bash_code_execution_result",
        "stdout": "1",
        "stderr": "",
        "return_code": 0,
        "content": [],
    },
}


class TestCallReadNormalization:
    @pytest.mark.parametrize("block", [_OLD_CALL, _NEW_CALL], ids=["old", "new"])
    def test_call_lands_on_canonical_kind(self, block):
        part = MessagesToolOps.p_tool_call_to_ir(block)
        assert part["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(part) == "code_execution"

    def test_call_keeps_tool_name_in_step_with_kind(self):
        part = MessagesToolOps.p_tool_call_to_ir(_NEW_CALL)
        assert part["tool_name"] == "code_execution"

    def test_non_alias_kind_untouched(self):
        part = MessagesToolOps.p_tool_call_to_ir(
            {
                "type": "server_tool_use",
                "id": "srvtoolu_3",
                "name": "web_search",
                "input": {"query": "x"},
            }
        )
        assert get_intrinsic_kind(part) == "web_search"
        assert part["tool_name"] == "web_search"


class TestResultReadNormalization:
    @pytest.mark.parametrize("block", [_OLD_RESULT, _NEW_RESULT], ids=["old", "new"])
    def test_result_lands_on_canonical_kind(self, block):
        part = MessagesToolOps.p_server_tool_result_to_ir(block)
        assert part["tool_type"] == "intrinsic"
        assert get_intrinsic_kind(part) == "code_execution"

    def test_old_and_new_share_one_kind(self):
        old = MessagesToolOps.p_server_tool_result_to_ir(_OLD_RESULT)
        new = MessagesToolOps.p_server_tool_result_to_ir(_NEW_RESULT)
        assert get_intrinsic_kind(old) == get_intrinsic_kind(new)


class TestRoundTrip:
    def test_new_call_round_trips_to_canonical_kind(self):
        """IR -> provider -> IR keeps the canonical kind for the new shape."""
        part = MessagesToolOps.p_tool_call_to_ir(_NEW_CALL)
        provider = MessagesToolOps.ir_tool_call_to_p(part)
        assert provider["type"] == "server_tool_use"
        assert provider["name"] == "code_execution"
        restored = MessagesToolOps.p_tool_call_to_ir(provider)
        assert get_intrinsic_kind(restored) == "code_execution"

    def test_new_result_write_keeps_the_canonical_kind(self):
        """Phase 4: the writer re-emits the server-tool result block.

        The read path folds the runtime name onto the canonical kind; the
        history-leg write now keeps that kind in the block type instead of
        flattening it to a generic ``tool_result``.
        """
        part = MessagesToolOps.p_server_tool_result_to_ir(_NEW_RESULT)
        provider = MessagesToolOps.ir_tool_result_to_p(part)
        # The block read from this dialect is restored verbatim (runtime name
        # and all); the point is that it is a server-tool result block again.
        assert provider == _NEW_RESULT
        assert provider["type"].endswith("_tool_result")

    def test_new_definition_preserves_native_version(self):
        """A newer code-execution definition round-trips verbatim via ``_native``.

        The read path already canonicalizes the *definition* to kind
        ``code_execution`` while storing the native payload, so the write path
        restores the requested version instead of the canonical
        ``code_execution_20250522`` fallback.
        """
        native = {"type": "code_execution_20250825", "name": "code_execution"}
        ir = MessagesToolOps.p_tool_definition_to_ir(dict(native))
        assert ir["intrinsic_kind"] == "code_execution"
        assert MessagesToolOps.ir_tool_definition_to_p(ir) == native

    def test_history_round_trip_end_to_end(self):
        """Messages A -> IR -> A: the bash call/result keep kind ``code_execution``."""
        converter = MessagesConverter()
        provider_request = {
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [
                {"role": "assistant", "content": [dict(_NEW_CALL)]},
                {"role": "user", "content": [dict(_NEW_RESULT)]},
            ],
        }
        ir = converter.request_from_provider(provider_request)
        kinds = [
            get_intrinsic_kind(part)
            for msg in cast(list[dict[str, Any]], ir["messages"])
            for part in cast(list[dict[str, Any]], msg["content"])
            if part.get("tool_type") == "intrinsic"
        ]
        assert kinds == ["code_execution", "code_execution"]

        back, _warnings = converter.request_to_provider(ir)
        call_block = cast(dict[str, Any], back["messages"][0]["content"][0])
        assert call_block["type"] == "server_tool_use"
        assert call_block["name"] == "code_execution"


class TestStreamingFold:
    """The streaming reader folds the same alias as the block reader.

    Folding only the non-streaming path left the two disagreeing about the
    same response: a streamed ``bash_code_execution`` call came out with the
    raw runtime name while its non-streamed twin came out canonical.
    """

    def _start(self, name: str):
        converter = MessagesConverter()
        return cast(
            list[dict[str, Any]],
            converter.stream_response_from_provider(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_2",
                        "name": name,
                        "input": {},
                    },
                }
            ),
        )

    def test_streaming_call_start_folds_the_alias(self):
        events = self._start("bash_code_execution")

        assert [e["tool_name"] for e in events] == ["code_execution"]

    def test_streaming_call_start_leaves_a_non_alias_alone(self):
        events = self._start("web_search")

        assert [e["tool_name"] for e in events] == ["web_search"]

"""Read-path tests for provider-hosted intrinsic server items.

A real OpenAI Responses ``web_search_call`` / ``code_interpreter_call``
output item carries its payload under type-specific keys (``action`` /
``code`` …), not the ``arguments`` string a ``function_call`` uses.  Reading
it back to IR must keep that payload and label the call with the canonical
intrinsic kind, matching the definition side.
"""

from typing import Any, cast

from llm_rosetta.converters.open_responses.content_ops import OpenResponsesContentOps
from llm_rosetta.converters.open_responses.message_ops import OpenResponsesMessageOps
from llm_rosetta.converters.open_responses.tool_ops import OpenResponsesToolOps

# The two item shapes as returned by the live OpenAI endpoint.
WEB_SEARCH_ITEM = {
    "id": "ws_abc",
    "type": "web_search_call",
    "status": "completed",
    "action": {
        "type": "search",
        "query": "finance: NVDA",
        "queries": ["finance: NVDA"],
    },
}

CODE_INTERPRETER_ITEM = {
    "id": "ci_abc",
    "type": "code_interpreter_call",
    "status": "completed",
    "code": "print(1+1)",
    "container_id": "cntr_1",
    "outputs": [{"type": "logs", "logs": "2"}],
}

# A multi-query search: the wire carries the full list, the writer used to
# collapse it to `[query]`.
WEB_SEARCH_MULTI_QUERY_ITEM = {
    "id": "ws_multi",
    "type": "web_search_call",
    "status": "completed",
    "action": {
        "type": "search",
        "query": "finance: NVDA",
        "queries": ["finance: NVDA", "NVDA stock"],
    },
}

FILE_SEARCH_ITEM = {
    "id": "fs_abc",
    "type": "file_search_call",
    "status": "completed",
    "queries": ["q1", "q2"],
    "results": [{"file_id": "f1", "text": "hit"}],
}


def _message_ops() -> OpenResponsesMessageOps:
    return OpenResponsesMessageOps(OpenResponsesContentOps(), OpenResponsesToolOps())


class TestIntrinsicServerItemReadPath:
    """Server output items → IR ToolCallPart via ``p_tool_call_to_ir``."""

    def test_web_search_call_keeps_query_and_canonical_name(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(WEB_SEARCH_ITEM)

        assert ir["type"] == "tool_call"
        assert ir["tool_type"] == "intrinsic"
        assert ir["tool_call_id"] == "ws_abc"
        # Canonical kind, not the raw wire type.
        assert ir["tool_name"] == "web_search"
        assert ir["provider_metadata"]["intrinsic_kind"] == "web_search"
        # The query survives; it used to read as an empty ``arguments``.
        assert ir["tool_input"]["query"] == "finance: NVDA"
        assert ir["tool_input"]["queries"] == ["finance: NVDA"]

    def test_code_interpreter_call_keeps_payload_and_canonical_name(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(CODE_INTERPRETER_ITEM)

        assert ir["type"] == "tool_call"
        assert ir["tool_type"] == "intrinsic"
        assert ir["tool_call_id"] == "ci_abc"
        assert ir["tool_name"] == "code_interpreter"
        assert ir["provider_metadata"]["intrinsic_kind"] == "code_interpreter"
        assert ir["tool_input"]["code"] == "print(1+1)"
        assert ir["tool_input"]["container_id"] == "cntr_1"
        assert ir["tool_input"]["outputs"] == [{"type": "logs", "logs": "2"}]

    def test_other_server_item_uses_its_own_payload_fields(self):
        """A recognised kind without a key map lifts its own payload fields."""
        item = {
            "id": "fs_abc",
            "type": "file_search_call",
            "status": "completed",
            "queries": ["annual report"],
            "results": [{"file_id": "file_1"}],
        }
        ir = OpenResponsesToolOps.p_tool_call_to_ir(item)

        assert ir["tool_name"] == "file_search"
        assert ir["tool_input"] == {
            "queries": ["annual report"],
            "results": [{"file_id": "file_1"}],
        }


class TestIntrinsicServerItemMessagePath:
    """The flat message read path funnels through the same function."""

    def test_p_messages_to_ir_keeps_query_and_code(self):
        ir_messages = _message_ops().p_messages_to_ir(
            [WEB_SEARCH_ITEM, CODE_INTERPRETER_ITEM]
        )

        first = cast(dict[str, Any], ir_messages[0])
        parts = cast(list[dict[str, Any]], first["content"])
        by_name = {part["tool_name"]: part for part in parts}
        assert by_name["web_search"]["tool_input"]["query"] == "finance: NVDA"
        assert by_name["code_interpreter"]["tool_input"]["code"] == "print(1+1)"
        assert all(part["tool_type"] == "intrinsic" for part in parts)


class TestIntrinsicServerItemRoundTrip:
    """The write path reads ``tool_input["query"]`` / ``["code"]``."""

    def test_web_search_query_survives_round_trip(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(WEB_SEARCH_ITEM)
        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

        assert restored["type"] == "web_search_call"
        assert restored["action"]["query"] == "finance: NVDA"

    def test_code_interpreter_code_survives_round_trip(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(CODE_INTERPRETER_ITEM)
        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

        assert restored["type"] == "code_interpreter_call"
        assert restored["code"] == "print(1+1)"

    def test_every_web_search_action_shape_survives(self):
        """`action` is discriminated, and all three shapes must round-trip.

        `search` carries query/queries, `open_page` a url, `find` a pattern
        (the `Action*` types on the OpenAI side).  Lifting only the search
        keys left the other two reading back as `{}` and replaying as a bare
        `web_search_call` with no action at all.
        """
        for action in (
            {"type": "search", "query": "a", "queries": ["a", "b"]},
            {"type": "open_page", "url": "https://example.com/x"},
            {"type": "find", "pattern": "revenue"},
        ):
            item = {"id": "ws_1", "type": "web_search_call", "action": action}
            ir = OpenResponsesToolOps.p_tool_call_to_ir(item)
            assert ir["tool_input"], f"{action['type']}: payload not lifted"
            restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)
            assert restored["action"] == action, action["type"]

    def test_web_search_keeps_every_query_on_replay(self):
        """The writer used to hardcode `queries: [query]`, dropping the rest."""
        ir = OpenResponsesToolOps.p_tool_call_to_ir(WEB_SEARCH_MULTI_QUERY_ITEM)
        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

        assert restored["action"]["queries"] == ["finance: NVDA", "NVDA stock"]

    def test_file_search_query_reaches_the_writer(self):
        """The read side lifts `queries`; the writer used to read `query`.

        With the fallback producing `queries`/`results`, the writer emitted an
        empty `query` and discarded what the read side had just preserved.
        """
        ir = OpenResponsesToolOps.p_tool_call_to_ir(FILE_SEARCH_ITEM)

        assert ir["tool_input"]["queries"] == ["q1", "q2"]
        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)
        assert restored["query"] == "q1"


class TestIntrinsicServerItemWriteShape:
    """The write leg emits the fields the API requires (#981).

    A native server item identifies itself with ``id`` and carries its payload
    under type-specific keys; it must not carry the ``call_id`` / ``arguments``
    pair a client ``function_call`` uses.  Replaying a written
    ``code_interpreter_call`` without ``id`` answers
    ``400 Missing required parameter: 'input[2].id'`` on the live API.
    """

    def test_web_search_call_shape(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(WEB_SEARCH_ITEM)

        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

        assert restored == {
            "type": "web_search_call",
            "id": "ws_abc",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "finance: NVDA",
                "queries": ["finance: NVDA"],
            },
        }

    def test_code_interpreter_call_shape(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(CODE_INTERPRETER_ITEM)

        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

        assert restored == {
            "type": "code_interpreter_call",
            "id": "ci_abc",
            "code": "print(1+1)",
            "container_id": "cntr_1",
            "outputs": [{"type": "logs", "logs": "2"}],
        }

    def test_server_item_has_no_client_fields(self):
        for item in (WEB_SEARCH_ITEM, CODE_INTERPRETER_ITEM):
            ir = OpenResponsesToolOps.p_tool_call_to_ir(item)
            restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)

            assert "call_id" not in restored
            assert "arguments" not in restored
            assert restored["id"] == item["id"]

    def test_status_is_restored_from_the_read_side(self):
        item = dict(WEB_SEARCH_ITEM, status="in_progress")

        restored = OpenResponsesToolOps.ir_tool_call_to_p(
            OpenResponsesToolOps.p_tool_call_to_ir(item)
        )

        assert restored["status"] == "in_progress"

    def test_status_defaults_to_completed_without_a_read_status(self):
        restored = OpenResponsesToolOps.ir_tool_call_to_p(
            {
                "type": "tool_call",
                "tool_call_id": "ws_1",
                "tool_name": "web_search",
                "tool_input": {"query": "q"},
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "web_search"},
            }
        )

        assert restored["status"] == "completed"

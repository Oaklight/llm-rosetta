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
        """The read side lifts `queries`/`results`; the writer used to read
        `query`.

        The singular ``query`` the writer emitted is rejected live
        (``400 Missing required parameter: 'input[1].queries'``), so the native
        ``queries``/``results`` now reach the wire instead.
        """
        ir = OpenResponsesToolOps.p_tool_call_to_ir(FILE_SEARCH_ITEM)

        assert ir["tool_input"]["queries"] == ["q1", "q2"]
        restored = OpenResponsesToolOps.ir_tool_call_to_p(ir)
        assert restored["queries"] == ["q1", "q2"]
        assert restored["results"] == [{"file_id": "f1", "text": "hit"}]
        assert "query" not in restored


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

    def test_shell_and_computer_items_carry_call_id(self):
        # The published schema requires ``call_id`` on both, and the read side
        # keeps the wire ``call_id`` as the IR's ``tool_call_id``, so the write
        # leg can supply it.
        for item in (
            {
                "type": "shell_call",
                "id": "sc1",
                "call_id": "c1",
                "action": {"commands": ["ls"]},
                "status": "completed",
            },
            {
                "type": "computer_call",
                "id": "cc1",
                "call_id": "c2",
                "action": {"type": "screenshot"},
                "pending_safety_checks": [],
                "status": "completed",
            },
        ):
            restored = OpenResponsesToolOps.ir_tool_call_to_p(
                OpenResponsesToolOps.p_tool_call_to_ir(item)
            )

            assert restored["type"] == item["type"]
            assert restored["call_id"] == item["call_id"]


class TestServerItemHistoryLeg:
    """A replayed turn keeps the native server item when the IR has its payload.

    ``for_history=True`` is the request leg (an assistant turn fed back as
    input).  It used to degrade every non-web-search server item to a client
    ``function_call``, dropping ``code`` / ``container_id`` / ``outputs`` — the
    path issue #985 measured.  It now re-emits the native item whenever the IR
    carries the fields the API requires, and degrades only when it does not.
    """

    def test_code_interpreter_replays_natively(self):
        ir = OpenResponsesToolOps.p_tool_call_to_ir(CODE_INTERPRETER_ITEM)

        written = OpenResponsesToolOps.ir_tool_call_to_p(ir, for_history=True)

        assert written["type"] == "code_interpreter_call"
        assert written["id"] == "ci_abc"
        assert written["container_id"] == "cntr_1"
        assert written["outputs"] == [{"type": "logs", "logs": "2"}]
        assert "call_id" not in written
        assert "arguments" not in written

    def test_code_interpreter_without_container_degrades(self):
        # A foreign request that never had a container_id cannot be replayed
        # natively (live: 400 missing 'input[1].container_id').
        item = {k: v for k, v in CODE_INTERPRETER_ITEM.items() if k != "container_id"}
        ir = OpenResponsesToolOps.p_tool_call_to_ir(item)

        written = OpenResponsesToolOps.ir_tool_call_to_p(ir, for_history=True)

        assert written["type"] == "function_call"
        assert written["call_id"] == "ci_abc"

    def test_file_search_without_complete_results_degrades(self):
        # The live API rejects a file-search result without `filename`.
        ir = OpenResponsesToolOps.p_tool_call_to_ir(FILE_SEARCH_ITEM)

        written = OpenResponsesToolOps.ir_tool_call_to_p(ir, for_history=True)

        assert written["type"] == "function_call"

    def test_shell_call_degrades_because_the_ir_cannot_supply_its_fields(self):
        # `shell_call` / `computer_call` need `call_id` and `status` (published
        # schema), which the read side does not lift, so replaying them natively
        # would write an item the API rejects.  Keep the function_call
        # degradation the history leg used before the gate was added.
        ir = OpenResponsesToolOps.p_tool_call_to_ir(
            {
                "type": "shell_call",
                "id": "sc1",
                "call_id": "c1",
                "action": {"commands": ["ls"]},
                "status": "completed",
            }
        )

        written = OpenResponsesToolOps.ir_tool_call_to_p(ir, for_history=True)

        assert written["type"] == "function_call"


class TestIntrinsicResultInputLeg:
    """A server result is dropped from the input leg, never client-shaped.

    The payload rides on the call item, the input schema has no
    ``*_call_output`` item for a code interpreter / file search / web search,
    and a ``function_call_output`` with a server call's id pairs with no client
    call (measured live 2026-10-13).
    """

    def test_code_interpreter_result_is_dropped(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "code_interpreter_call_output",
                "call_id": "ci_abc",
                "output": [{"type": "logs", "logs": "2"}],
            }
        )

        assert ir["tool_type"] == "intrinsic"
        assert OpenResponsesToolOps.ir_tool_result_to_p(ir) == {}

    def test_dropped_result_reports_an_error_flag(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "code_interpreter_call_output",
                "call_id": "ci_abc",
                "output": [{"type": "logs", "logs": "2"}],
            }
        )
        ir["is_error"] = True
        warnings: list[str] = []

        assert OpenResponsesToolOps.ir_tool_result_to_p(ir, warnings=warnings) == {}
        assert warnings

    def test_shell_result_pairs_with_its_degraded_call(self):
        # The shell call degrades to a client ``function_call`` on the history
        # leg, so its output is written as a ``function_call_output`` that pairs
        # with it — the blanket drop would leave the call unpaired.
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "shell_call_output",
                "call_id": "c1",
                "output": [{"stdout": "a"}],
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "c1"

    def test_computer_result_pairs_with_its_degraded_call(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "computer_call_output",
                "call_id": "c2",
                "output": [{"type": "computer_screenshot", "image_url": "x"}],
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "c2"

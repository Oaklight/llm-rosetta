"""Read-path tests for provider-hosted intrinsic server items.

A real OpenAI Responses ``web_search_call`` / ``code_interpreter_call``
output item carries its payload under type-specific keys (``action`` /
``code`` …), not the ``arguments`` string a ``function_call`` uses.  Reading
it back to IR must keep that payload and label the call with the canonical
intrinsic kind, matching the definition side.
"""

from typing import Any, cast

from llm_rosetta.converters.base.tools import (
    make_intrinsic_tool_call,
    make_intrinsic_tool_result,
)
from llm_rosetta.converters.open_responses.content_ops import OpenResponsesContentOps
from llm_rosetta.converters.open_responses.message_ops import OpenResponsesMessageOps
from llm_rosetta.converters.open_responses.tool_ops import OpenResponsesToolOps
from llm_rosetta.types.ir import IRInputItem, ToolCallPart, ToolResultPart

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


def _intrinsic_call(
    kind: str, call_id: str, tool_input: dict[str, Any]
) -> ToolCallPart:
    """IR tool call part for an intrinsic kind, including one with no item type."""
    return cast(
        ToolCallPart,
        make_intrinsic_tool_call(
            tool_call_id=call_id,
            tool_name=kind,
            tool_input=tool_input,
            intrinsic_kind=kind,
        ),
    )


def _intrinsic_result(
    kind: str,
    call_id: str,
    result: Any,
    *,
    is_error: bool = False,
) -> ToolResultPart:
    """IR tool result part for an intrinsic kind, including one with no item type."""
    return cast(
        ToolResultPart,
        make_intrinsic_tool_result(
            tool_call_id=call_id,
            result=result,
            intrinsic_kind=kind,
            is_error=is_error,
        ),
    )


def _conversation(call: Any, result: Any) -> list[IRInputItem]:
    """A minimal assistant tool-call + tool-result message pair."""
    return cast(
        list[IRInputItem],
        [
            {"role": "assistant", "content": [call]},
            {"role": "tool", "content": [result]},
        ],
    )


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


class TestIntrinsicResultHistoryLeg:
    """On the history leg a result pairs with the call the writer emits.

    The call writer degrades a non-web_search intrinsic call to a client
    ``function_call`` only when ``_server_item_renderable`` says the native item
    cannot be replayed.  The result writer is handed that same decision
    (``degraded_intrinsic_calls``): a degraded call takes a matching
    ``function_call_output``, a native server item takes no output item at all
    (its ``*_call_output`` type is rejected, and a ``function_call_output``
    carrying a server call's id pairs with no client call).  ``web_search`` is
    always ``{}`` — its call stays native and carries the output inline.
    """

    def test_web_search_result_is_dropped(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "web_search_call_output",
                "call_id": "ws_abc",
                "output": "search results",
            }
        )

        assert ir["tool_type"] == "intrinsic"
        assert OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True) == {}

    def test_web_search_error_flag_is_warned_and_dropped(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "web_search_call_output",
                "call_id": "ws_abc",
                "output": "error",
            }
        )
        ir["is_error"] = True
        warnings: list[str] = []

        result = OpenResponsesToolOps.ir_tool_result_to_p(
            ir, warnings=warnings, for_history=True
        )

        assert result == {}
        assert warnings

    def test_shell_result_pairs_with_its_degraded_call(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "shell_call_output",
                "call_id": "c1",
                "output": [{"stdout": "a"}],
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True)

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

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "c2"

    def test_code_interpreter_result_pairs_with_its_degraded_call(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "code_interpreter_call_output",
                "call_id": "ci_abc",
                "output": [{"type": "logs", "logs": "2"}],
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "ci_abc"

    def test_file_search_result_pairs_with_its_degraded_call(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "file_search_call_output",
                "call_id": "fs_abc",
                "output": "file search results",
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "fs_abc"

    def test_default_for_history_is_true(self):
        """Callers that omit ``for_history`` get history-leg behaviour."""
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "shell_call_output",
                "call_id": "c1",
                "output": "ok",
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir)

        assert written["type"] == "function_call_output"

    def test_unknown_kind_result_pairs_as_a_client_output(self):
        # A kind with no native item type is written as a client
        # ``function_call`` on both legs, so its result pairs with it too.
        ir = _intrinsic_result("code_execution", "ce1", "ok")

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=True)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "ce1"

    def test_native_call_result_is_dropped_on_the_history_leg(self):
        # ``code_interpreter`` replays natively when the IR carries its
        # ``container_id``, so no input item can pair with it: a
        # ``function_call_output`` here would carry an id no client call
        # answers.
        ir = _intrinsic_result(
            "code_interpreter", "ci_abc", [{"type": "logs", "logs": "2"}]
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(
            ir, for_history=True, degraded_intrinsic_calls={"ci_abc": False}
        )

        assert written == {}

    def test_dropped_native_result_reports_an_error_flag(self):
        ir = _intrinsic_result(
            "code_interpreter",
            "ci_abc",
            [{"type": "logs", "logs": "2"}],
            is_error=True,
        )
        warnings: list[str] = []

        written = OpenResponsesToolOps.ir_tool_result_to_p(
            ir,
            warnings=warnings,
            for_history=True,
            degraded_intrinsic_calls={"ci_abc": False},
        )

        assert written == {}
        assert warnings

    def test_call_id_absent_from_the_map_is_treated_as_degraded(self):
        # A caller that converts a result without its call keeps the paired
        # client output rather than dropping it.
        ir = _intrinsic_result(
            "code_interpreter", "ci_abc", [{"type": "logs", "logs": "2"}]
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(
            ir, for_history=True, degraded_intrinsic_calls={}
        )

        assert written["type"] == "function_call_output"


class TestIntrinsicResultResponseLeg:
    """On the response leg the call stays native, so the result must pair
    with the native output type.
    """

    def test_web_search_result_is_dropped(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "web_search_call_output",
                "call_id": "ws_abc",
                "output": "search results",
            }
        )

        assert OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False) == {}

    def test_shell_result_emits_native_output_type(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "shell_call_output",
                "call_id": "c1",
                "output": "ok",
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False)

        assert written["type"] == "shell_call_output"
        assert written["call_id"] == "c1"

    def test_computer_result_emits_native_output_type(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "computer_call_output",
                "call_id": "c2",
                "output": "screenshot",
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False)

        assert written["type"] == "computer_call_output"
        assert written["call_id"] == "c2"

    def test_code_interpreter_result_emits_native_output_type(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "code_interpreter_call_output",
                "call_id": "ci_abc",
                "output": [{"type": "logs", "logs": "2"}],
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False)

        assert written["type"] == "code_interpreter_call_output"
        assert written["call_id"] == "ci_abc"

    def test_file_search_result_emits_native_output_type(self):
        ir = OpenResponsesToolOps.p_tool_result_to_ir(
            {
                "type": "file_search_call_output",
                "call_id": "fs_abc",
                "output": "results",
            }
        )

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False)

        assert written["type"] == "file_search_call_output"
        assert written["call_id"] == "fs_abc"

    def test_unknown_kind_result_stays_a_client_output(self):
        # The response leg keeps a native call native, but a kind with no
        # native item type is still written as a client ``function_call``
        # there, so its result stays a ``function_call_output``.
        ir = _intrinsic_result("code_execution", "ce1", "ok")

        written = OpenResponsesToolOps.ir_tool_result_to_p(ir, for_history=False)

        assert written["type"] == "function_call_output"
        assert written["call_id"] == "ce1"


class TestIntrinsicPairingThroughMessages:
    """Every intrinsic call the message writer emits is answered by its result.

    Issue #996: the call writer decides the shape from ``(kind, fields, leg)``
    while the result writer branched on the kind alone, so a call that degraded
    was left without an output, and a native call could be handed a client one.
    Driving each kind through the message writer — the shape the review
    measured — asserts the pairing contract directly.
    """

    def test_each_kind_writes_the_paired_items(self):
        # (label, call item, result item, item types the writer must emit)
        cases: list[tuple[str, dict[str, Any], dict[str, Any], list[str]]] = [
            (
                "web_search",
                WEB_SEARCH_ITEM,
                {
                    "type": "web_search_call_output",
                    "call_id": "ws_abc",
                    "output": "results",
                },
                ["web_search_call"],
            ),
            (
                "code_interpreter (replayable)",
                CODE_INTERPRETER_ITEM,
                {
                    "type": "code_interpreter_call_output",
                    "call_id": "ci_abc",
                    "output": [{"type": "logs", "logs": "2"}],
                },
                ["code_interpreter_call"],
            ),
            (
                "code_interpreter (no container_id)",
                {
                    key: value
                    for key, value in CODE_INTERPRETER_ITEM.items()
                    if key != "container_id"
                },
                {
                    "type": "code_interpreter_call_output",
                    "call_id": "ci_abc",
                    "output": [{"type": "logs", "logs": "2"}],
                },
                ["function_call", "function_call_output"],
            ),
            (
                "file_search (no complete results)",
                FILE_SEARCH_ITEM,
                {
                    "type": "file_search_call_output",
                    "call_id": "fs_abc",
                    "output": "results",
                },
                ["function_call", "function_call_output"],
            ),
            (
                "shell",
                {
                    "type": "shell_call",
                    "id": "sc1",
                    "call_id": "c1",
                    "action": {"commands": ["ls"]},
                    "status": "completed",
                },
                {
                    "type": "shell_call_output",
                    "call_id": "c1",
                    "output": [
                        {
                            "stdout": "a",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                },
                ["function_call", "function_call_output"],
            ),
        ]

        for label, call_item, result_item, expected in cases:
            call = OpenResponsesToolOps.p_tool_call_to_ir(call_item)
            result = OpenResponsesToolOps.p_tool_result_to_ir(result_item)
            items, _ = _message_ops().ir_messages_to_p(_conversation(call, result))

            assert [item["type"] for item in items] == expected, label

    def test_no_client_output_is_left_without_its_call(self):
        # The #996 failure shape: a ``function_call_output`` whose id no
        # ``function_call`` answers.  Covers kinds with no native item type
        # (``code_execution``, ``image_generation``, ``mcp``), which the kind
        # set the old writer branched on did not list.
        cases = [
            ("code_execution", "ce1", {"code": "1+1"}, "ok"),
            ("image_generation", "ig1", {"prompt": "a cat"}, "img"),
            ("mcp", "mcp1", {"name": "tool"}, "{}"),
        ]

        for kind, call_id, tool_input, result in cases:
            call = _intrinsic_call(kind, call_id, tool_input)
            result_part = _intrinsic_result(kind, call_id, result)
            items, _ = _message_ops().ir_messages_to_p(_conversation(call, result_part))

            calls = {i.get("call_id") for i in items if i["type"].endswith("_call")}
            outputs = {i["call_id"] for i in items if i["type"].endswith("_output")}

            assert outputs == {call_id}, kind
            assert outputs <= calls, kind

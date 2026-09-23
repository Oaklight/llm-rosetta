"""Tests for namespace round-tripping on namespaced tool calls.

Tools arriving inside a ``namespace`` container are flattened to bare
names on the request leg, because no target format can carry a namespace.
Clients that dispatch on ``(name, namespace)`` — Codex's ``multi_agent_v2``
tools, for instance — need the namespace back on the returned
``function_call`` item, or the call cannot be routed to its handler.
"""

from typing import Any

from llm_rosetta.capabilities import (
    ToolNameMap,
    build_tool_name_map,
    restore_client_tool_names,
)
from llm_rosetta.pipeline import ConversionPipeline


def _namespace_container(namespace: str, *names: str) -> dict[str, Any]:
    return {
        "type": "namespace",
        "name": namespace,
        "tools": [
            {
                "type": "function",
                "name": name,
                "description": f"The {name} tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }
            for name in names
        ],
    }


def _custom_namespace_container(namespace: str, *names: str) -> dict[str, Any]:
    """Same, for ``custom`` tools — free-text input instead of parameters."""
    return {
        "type": "namespace",
        "name": namespace,
        "tools": [
            {
                "type": "custom",
                "name": name,
                "description": f"The {name} tool.",
                "format": {"type": "text"},
            }
            for name in names
        ],
    }


def _request(*containers: dict[str, Any], stream: bool = False) -> dict[str, Any]:
    """A Responses request carrying tools the Codex way: in ``input``."""
    return {
        "model": "test-model",
        "stream": stream,
        "tool_choice": "auto",
        "input": [
            {
                "type": "additional_tools",
                "id": "at_test",
                "role": "developer",
                "tools": list(containers),
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "spawn a subagent"}],
            },
        ],
    }


def _chat_completion(tool_name: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": '{"message":"hi"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat_chunks(tool_name: str) -> list[dict[str, Any]]:
    def chunk(choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": choices,
        }

    return [
        chunk(
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        ),
        chunk(
            [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"message":"hi"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        ),
        chunk([{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        # Empty-choices chunk closes the stream, producing response.completed.
        chunk([]),
    ]


def _name_map(*entries: tuple[str, ...]) -> ToolNameMap:
    """Build a map from ``(upstream, namespace[, client name])`` triples."""
    ir_tools = []
    for entry in entries:
        upstream, namespace = entry[0], entry[1]
        metadata: dict[str, Any] = {"namespace": namespace}
        if len(entry) > 2:
            metadata["_original_name"] = entry[2]
        ir_tools.append({"name": upstream, "metadata": metadata})
    return build_tool_name_map({"tools": ir_tools})


def _tool_call_items(
    events: list[dict[str, Any]], item_type: str
) -> list[tuple[str, dict[str, Any]]]:
    """Every item of *item_type* in a Responses event stream, with its event type."""
    found: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == item_type:
            found.append((event["type"], item))
        response = event.get("response")
        if isinstance(response, dict):
            for out in response.get("output") or []:
                if isinstance(out, dict) and out.get("type") == item_type:
                    found.append((event["type"], out))
    return found


def _function_calls(events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return _tool_call_items(events, "function_call")


def _custom_tool_calls(
    events: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    return _tool_call_items(events, "custom_tool_call")


# ---------------------------------------------------------------------------
# build_tool_name_map
# ---------------------------------------------------------------------------


class TestBuildToolNameMap:
    def test_maps_both_directions(self):
        ir = {"tools": [{"name": "spawn_agent", "metadata": {"namespace": "agents"}}]}
        name_map = build_tool_name_map(ir)
        assert name_map.to_upstream("spawn_agent", "agents") == "spawn_agent"
        assert name_map.to_client("spawn_agent") == ("spawn_agent", "agents")

    def test_uses_original_name_when_renamed(self):
        ir = {
            "tools": [
                {
                    "name": "agents_wait",
                    "metadata": {"namespace": "agents", "_original_name": "wait"},
                }
            ]
        }
        name_map = build_tool_name_map(ir)
        assert name_map.to_upstream("wait", "agents") == "agents_wait"
        assert name_map.to_client("agents_wait") == ("wait", "agents")

    def test_disambiguates_same_name_across_namespaces(self):
        """The bare name alone is ambiguous; the namespace resolves it."""
        ir = {
            "tools": [
                {
                    "name": "functions_wait",
                    "metadata": {"namespace": "functions", "_original_name": "wait"},
                },
                {
                    "name": "agents_wait",
                    "metadata": {"namespace": "agents", "_original_name": "wait"},
                },
            ]
        }
        name_map = build_tool_name_map(ir)
        assert name_map.to_upstream("wait", "functions") == "functions_wait"
        assert name_map.to_upstream("wait", "agents") == "agents_wait"

    def test_skips_tools_that_were_not_renamed(self):
        ir = {"tools": [{"name": "plain", "metadata": {}}, {"name": "bare"}]}
        assert not build_tool_name_map(ir)

    def test_upstream_name_two_tools_share_is_not_attributed(self):
        """Qualification can fail, leaving two tools under one upstream name.

        Guessing an owner is worse than not translating: the provider may
        well have meant the *other* one, and a wrong namespace dispatches
        the call to a handler that never asked for it.
        """
        ir = {
            "tools": [
                {"name": "exec", "metadata": {}},  # top-level, kept its name
                {"name": "exec", "metadata": {"namespace": "ns"}},  # unqualifiable
            ]
        }
        name_map = build_tool_name_map(ir)
        assert name_map.to_client("exec") == ("exec", None)

    def test_empty_when_no_tools(self):
        assert not build_tool_name_map({})

    def test_unknown_names_pass_through_both_ways(self):
        name_map = ToolNameMap()
        assert name_map.to_upstream("mystery") == "mystery"
        assert name_map.to_client("mystery") == ("mystery", None)


# ---------------------------------------------------------------------------
# apply_upstream_tool_names — the selector fields
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# restore_client_tool_names
# ---------------------------------------------------------------------------


class TestRestoreClientToolNames:
    @staticmethod
    def _response(tool_name: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_call_id": "call_1",
                                "tool_name": tool_name,
                                "tool_input": {},
                            }
                        ],
                    }
                }
            ]
        }

    def test_sets_namespace_in_provider_metadata(self):
        ir = self._response("spawn_agent")
        restore_client_tool_names(ir, name_map=_name_map(("spawn_agent", "agents")))
        part = ir["choices"][0]["message"]["content"][0]
        assert part["provider_metadata"]["namespace"] == "agents"
        assert part["tool_name"] == "spawn_agent"

    def test_restores_pre_rename_name(self):
        ir = self._response("agents_wait")
        restore_client_tool_names(
            ir, name_map=_name_map(("agents_wait", "agents", "wait"))
        )
        part = ir["choices"][0]["message"]["content"][0]
        assert part["tool_name"] == "wait"
        assert part["provider_metadata"]["namespace"] == "agents"

    def test_preserves_existing_provider_metadata(self):
        ir = self._response("spawn_agent")
        ir["choices"][0]["message"]["content"][0]["provider_metadata"] = {
            "responses_item_id": "fc_keep"
        }
        restore_client_tool_names(ir, name_map=_name_map(("spawn_agent", "agents")))
        pm = ir["choices"][0]["message"]["content"][0]["provider_metadata"]
        assert pm["responses_item_id"] == "fc_keep"
        assert pm["namespace"] == "agents"

    def test_noop_when_empty_map(self):
        ir = self._response("spawn_agent")
        restore_client_tool_names(ir, name_map=ToolNameMap())
        assert "provider_metadata" not in ir["choices"][0]["message"]["content"][0]

    def test_only_restores_matching_names(self):
        ir = self._response("other_tool")
        restore_client_tool_names(ir, name_map=_name_map(("spawn_agent", "agents")))
        part = ir["choices"][0]["message"]["content"][0]
        assert part["tool_name"] == "other_tool"
        assert "provider_metadata" not in part


# ---------------------------------------------------------------------------
# End-to-end through the pipeline
# ---------------------------------------------------------------------------


class TestNamespaceRoundTrip:
    def test_tools_reach_upstream_flattened(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        upstream = pipe.convert_request(
            _request(_namespace_container("agents", "spawn_agent"))
        )
        names = [t.get("function", t).get("name") for t in upstream["tools"]]
        assert names == ["spawn_agent"]

    def test_non_streaming_restores_namespace(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(_request(_namespace_container("agents", "spawn_agent")))
        out = pipe.convert_response(_chat_completion("spawn_agent"))

        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "spawn_agent"
        assert calls[0]["namespace"] == "agents"

    def test_streaming_restores_namespace_at_every_item(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(
            _request(_namespace_container("agents", "spawn_agent"), stream=True)
        )
        proc = pipe.create_stream_processor()

        events: list[dict[str, Any]] = []
        for chunk in _chat_chunks("spawn_agent"):
            events.extend(proc.process_chunk(chunk))

        calls = _function_calls(events)
        emitted = {event_type for event_type, _ in calls}
        assert "response.output_item.added" in emitted
        assert "response.output_item.done" in emitted
        assert "response.completed" in emitted

        for event_type, item in calls:
            assert item["name"] == "spawn_agent", event_type
            assert item.get("namespace") == "agents", event_type

    def test_collision_rename_is_reversed(self):
        """A name colliding across namespaces is qualified upstream, bare back."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        upstream = pipe.convert_request(
            _request(
                _namespace_container("functions", "wait"),
                _namespace_container("agents", "wait"),
            )
        )
        names = [t.get("function", t).get("name") for t in upstream["tools"]]
        assert names == ["functions_wait", "agents_wait"]

        out = pipe.convert_response(_chat_completion("agents_wait"))
        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        # The client dispatches on the bare name plus the namespace, so the
        # qualification has to be undone rather than handed back as-is.
        assert calls[0]["name"] == "wait"
        assert calls[0]["namespace"] == "agents"

    def test_second_turn_requalifies_history_tool_call(self):
        """A restored call sent back must be re-qualified on the way upstream.

        Turn 1 hands the client the bare ``wait`` plus ``namespace: agents``.
        The client echoes that back verbatim in turn 2's history.  The tool
        *definitions* are still qualified to ``agents_wait``, so unless the
        history item is qualified to match, the assistant message names a
        function that is not in the request's tool list.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container("functions", "wait"),
            _namespace_container("agents", "wait"),
        )
        request["input"].extend(
            [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "wait",
                    "namespace": "agents",
                    "arguments": '{"message":"hi"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ]
        )
        upstream = pipe.convert_request(request)

        tool_names = [t.get("function", t).get("name") for t in upstream["tools"]]
        assert tool_names == ["functions_wait", "agents_wait"]

        called = [
            tc["function"]["name"]
            for message in upstream["messages"]
            for tc in message.get("tool_calls") or []
        ]
        assert called == ["agents_wait"], (
            "history tool call must use the same upstream name as the tool "
            f"definition; got {called}, tools are {tool_names}"
        )

    def test_custom_tool_call_restores_namespace(self):
        """A namespaced custom tool round-trips like a namespaced function.

        The restore puts the namespace in provider_metadata whatever the
        tool type; only the ``function_call`` serializer used to read it
        back out, so a ``custom_tool_call`` reached the client bare and
        unattributable.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(
            _request(
                _custom_namespace_container("a", "edit"),
                _custom_namespace_container("b", "edit"),
            )
        )
        out = pipe.convert_response(_chat_completion("b_edit"))

        calls = [i for i in out["output"] if i.get("type") == "custom_tool_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "edit"
        assert calls[0]["namespace"] == "b"

    def test_custom_tool_call_namespace_survives_a_second_turn(self):
        """The namespace the client echoes back must re-qualify the history."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _custom_namespace_container("a", "edit"),
            _custom_namespace_container("b", "edit"),
        )
        request["input"].extend(
            [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "edit",
                    "namespace": "b",
                    "input": "hello",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "done",
                },
            ]
        )
        upstream = pipe.convert_request(request)

        # A Chat custom tool call nests under `custom`, not `function`.
        called = [
            (tc.get("function") or tc["custom"])["name"]
            for message in upstream["messages"]
            for tc in message.get("tool_calls") or []
        ]
        assert called == ["b_edit"]

    def test_streaming_custom_tool_call_restores_namespace(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(
            _request(
                _custom_namespace_container("a", "edit"),
                _custom_namespace_container("b", "edit"),
                stream=True,
            )
        )
        proc = pipe.create_stream_processor()

        events: list[dict[str, Any]] = []
        for chunk in _chat_chunks("b_edit"):
            events.extend(proc.process_chunk(chunk))

        calls = _custom_tool_calls(events)
        emitted = {event_type for event_type, _ in calls}
        assert "response.output_item.added" in emitted
        assert "response.output_item.done" in emitted
        assert "response.completed" in emitted

        for event_type, item in calls:
            assert item["name"] == "edit", event_type
            assert item.get("namespace") == "b", event_type

    def test_name_map_is_built_once_and_reused(self):
        """Response and streaming legs must use the request leg's own map."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(_request(_namespace_container("agents", "wait")))
        built = pipe._name_map

        pipe.convert_response(_chat_completion("wait"))
        pipe.create_stream_processor()
        assert pipe._name_map is built

    def test_non_namespaced_tools_get_no_namespace(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(
            {
                "model": "test-model",
                "tool_choice": "auto",
                "tools": [
                    {
                        "type": "function",
                        "name": "plain_tool",
                        "description": "A top-level tool.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "go"}],
                    }
                ],
            }
        )
        out = pipe.convert_response(_chat_completion("plain_tool"))
        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        assert "namespace" not in calls[0]

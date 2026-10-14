"""Tests for namespace round-tripping on namespaced tool calls.

Tools arriving inside a ``namespace`` container are flattened to bare
names on the request leg, because no target format can carry a namespace.
Clients that dispatch on ``(name, namespace)`` — Codex's ``multi_agent_v2``
tools, for instance — need the namespace back on the returned
``function_call`` item, or the call cannot be routed to its handler.
"""

from typing import Any, cast

import pytest

from llm_rosetta.capabilities import (
    ToolNameMap,
    apply_upstream_tool_names,
    build_tool_name_map,
    restore_client_tool_names,
)
from llm_rosetta.converters.base.helpers.tool_group import generated_group_id
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.types.ir.request import IRRequest
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ToolsConfig,
    register_shim,
    unregister_shim,
)


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


def _nameless_container(name: str, description: str | None = None) -> dict[str, Any]:
    """A namespace container with no name of its own.

    Codex always names its containers, and every upstream that implements
    ``namespace`` rejects one without a name (OpenAI, DeepSeek and Volcengine
    all 400 or 422), so this is malformed input.  Its children are flattened
    with ``tool_group == ""``, which qualifies nothing: two tools
    sharing a name here cannot be told apart at all, not even by the client
    that declared them.

    *description* overrides the child's own, so two of these can declare a
    genuinely conflicting pair — byte-identical children are collapsed by
    ``dedupe_tool_definitions`` (#913) and never reach the ambiguous case.
    """
    container = _namespace_container("", name)
    if description is not None:
        container["tools"][0]["description"] = description
    return container


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
    """Build a map from ``(upstream, group[, client name])`` triples."""
    ir_tools = []
    for entry in entries:
        upstream, group = entry[0], entry[1]
        tool: dict[str, Any] = {"name": upstream, "tool_group": group}
        if len(entry) > 2:
            tool["metadata"] = {"_original_name": entry[2]}
        ir_tools.append(tool)
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
        ir = {"tools": [{"name": "spawn_agent", "tool_group": "agents"}]}
        name_map = build_tool_name_map(ir)
        assert name_map.to_upstream("spawn_agent", "agents") == "spawn_agent"
        assert name_map.to_client("spawn_agent") == ("spawn_agent", "agents")

    def test_uses_original_name_when_renamed(self):
        ir = {
            "tools": [
                {
                    "name": "agents_wait",
                    "tool_group": "agents",
                    "metadata": {"_original_name": "wait"},
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
                    "tool_group": "functions",
                    "metadata": {"_original_name": "wait"},
                },
                {
                    "name": "agents_wait",
                    "tool_group": "agents",
                    "metadata": {"_original_name": "wait"},
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
                {"name": "exec", "tool_group": "ns"},  # unqualifiable
            ]
        }
        name_map = build_tool_name_map(ir)
        assert name_map.to_client("exec") == ("exec", None)

    def test_resolves_bare_name_while_one_tool_claims_it(self):
        """``tool_choice`` has no namespace field, so it resolves by name."""
        ir = {
            "tools": [
                {
                    "name": "agents_wait",
                    "tool_group": "agents",
                    "metadata": {"_original_name": "wait"},
                }
            ]
        }
        assert build_tool_name_map(ir).to_upstream("wait") == "agents_wait"

    def test_bare_name_unresolved_when_two_tools_claim_it(self):
        ir = {
            "tools": [
                {
                    "name": "functions_wait",
                    "tool_group": "functions",
                    "metadata": {"_original_name": "wait"},
                },
                {
                    "name": "agents_wait",
                    "tool_group": "agents",
                    "metadata": {"_original_name": "wait"},
                },
            ]
        }
        assert build_tool_name_map(ir).to_upstream("wait") == "wait"

    def test_bare_name_unresolved_when_an_untouched_tool_claims_it(self):
        """The untouched top-level tool is a claimant too, and blocks the guess."""
        ir = {
            "tools": [
                {"name": "wait", "metadata": {}},
                {
                    "name": "agents_wait",
                    "tool_group": "agents",
                    "metadata": {"_original_name": "wait"},
                },
            ]
        }
        assert build_tool_name_map(ir).to_upstream("wait") == "wait"

    def test_empty_when_no_tools(self):
        assert not build_tool_name_map({})

    def test_unknown_names_pass_through_both_ways(self):
        name_map = ToolNameMap()
        assert name_map.to_upstream("mystery") == "mystery"
        assert name_map.to_client("mystery") == ("mystery", None)


# ---------------------------------------------------------------------------
# apply_upstream_tool_names — the selector fields
# ---------------------------------------------------------------------------


class TestApplyUpstreamSelectors:
    """``tool_choice`` and ``allowed_tools`` name tools, so they translate too."""

    IR_TOOLS = [
        {
            "name": "agents_wait",
            "tool_group": "agents",
            "metadata": {"_original_name": "wait"},
        }
    ]

    # Two tools answer to the bare name, so a selector naming it resolves
    # to neither.
    AMBIGUOUS_TOOLS = IR_TOOLS + [
        {
            "name": "functions_wait",
            "tool_group": "functions",
            "metadata": {"_original_name": "wait"},
        }
    ]

    def _apply(self, ir: dict[str, Any]) -> list[str]:
        ir.setdefault("tools", self.IR_TOOLS)
        warnings: list[str] = []
        apply_upstream_tool_names(
            ir, name_map=build_tool_name_map(ir), warnings=warnings
        )
        return warnings

    def test_tool_choice_is_respelled(self):
        ir: dict[str, Any] = {"tool_choice": {"mode": "tool", "tool_name": "wait"}}
        assert self._apply(ir) == []
        assert ir["tool_choice"]["tool_name"] == "agents_wait"

    def test_allowed_tools_entries_are_respelled(self):
        ir: dict[str, Any] = {
            "provider_extensions": {
                "allowed_tools": [{"type": "function", "name": "wait"}, "wait"]
            }
        }
        assert self._apply(ir) == []
        assert ir["provider_extensions"]["allowed_tools"] == [
            {"type": "function", "name": "agents_wait"},
            "agents_wait",
        ]

    def test_ambiguous_allowed_tools_entry_warns(self):
        """An entry no single tool claims goes upstream naming nothing."""
        ir: dict[str, Any] = {
            "tools": self.AMBIGUOUS_TOOLS,
            "provider_extensions": {"allowed_tools": ["wait"]},
        }
        warnings = self._apply(ir)

        assert ir["provider_extensions"]["allowed_tools"] == ["wait"]
        assert any(
            "shared by tools in more than one namespace" in w for w in warnings
        ), f"expected an ambiguous-allowed_tools warning; got {warnings}"

    def test_selector_naming_no_tool_at_all_says_so(self):
        """A name nothing claims is a typo, not an ambiguity — say which."""
        ir: dict[str, Any] = {
            "tools": self.AMBIGUOUS_TOOLS,
            "tool_choice": {"mode": "tool", "tool_name": "totally_made_up"},
        }
        warnings = self._apply(ir)

        assert len(warnings) == 1, warnings
        assert "no tool of that name was declared" in warnings[0]
        assert "more than one namespace" not in warnings[0]

    def test_selector_is_checked_even_when_nothing_was_renamed(self):
        """Whether a selector names a real tool does not depend on namespaces.

        A request with no namespace containers builds an empty name map, but
        the typo it may carry is exactly as broken as it would be alongside
        one — so the check runs before the map is consulted.
        """
        ir: dict[str, Any] = {
            "tools": [{"name": "exec"}, {"name": "read"}],
            "tool_choice": {"mode": "tool", "tool_name": "totally_made_up"},
        }
        warnings = self._apply(ir)

        assert len(warnings) == 1, warnings
        assert "no tool of that name was declared" in warnings[0]

    def test_valid_selector_is_quiet_when_nothing_was_renamed(self):
        ir: dict[str, Any] = {
            "tools": [{"name": "exec"}, {"name": "read"}],
            "tool_choice": {"mode": "tool", "tool_name": "exec"},
        }
        assert self._apply(ir) == []
        assert ir["tool_choice"]["tool_name"] == "exec"

    def test_allowed_tools_accepts_the_wrapped_shape(self):
        ir: dict[str, Any] = {
            "provider_extensions": {
                "allowed_tools": {"mode": "auto", "tools": [{"name": "wait"}]}
            }
        }
        self._apply(ir)
        assert ir["provider_extensions"]["allowed_tools"]["tools"] == [
            {"name": "agents_wait"}
        ]

    def test_unrecognised_allowed_tools_shape_is_left_alone(self):
        ir: dict[str, Any] = {"provider_extensions": {"allowed_tools": "everything"}}
        assert self._apply(ir) == []
        assert ir["provider_extensions"]["allowed_tools"] == "everything"


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


class TestRestoreAgainstAGroupedTarget:
    """A container-carrying target hands a name back beside its namespace.

    There a child keeps the client's own name, so a top-level tool and a
    container child may share one, and the namespace the wire echoes is the
    only thing that says which the model meant.  The map has to be read as a
    ``(name, namespace)`` pair for those, not by name alone.
    """

    @staticmethod
    def _call(name: str, namespace: str | None = None) -> dict[str, Any]:
        part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": name,
            "tool_input": {},
        }
        if namespace is not None:
            part["provider_metadata"] = {"namespace": namespace}
        return {"choices": [{"message": {"role": "assistant", "content": [part]}}]}

    @staticmethod
    def _shared_name_map() -> ToolNameMap:
        """A top-level ``wait`` beside one inside ``tasks``, both bare."""
        return build_tool_name_map(
            {
                "tools": [
                    {"name": "wait", "tool_group": None},
                    {"name": "wait", "tool_group": "tasks"},
                ]
            }
        )

    def test_the_echoed_namespace_routes_to_the_container_child(self):
        ir = self._call("wait", "tasks")
        restore_client_tool_names(ir, name_map=self._shared_name_map())
        part = ir["choices"][0]["message"]["content"][0]
        assert part["tool_name"] == "wait"
        assert part["provider_metadata"]["namespace"] == "tasks"

    def test_a_bare_call_does_not_acquire_a_namespace(self):
        """The top-level tool of the shared name comes back with none."""
        ir = self._call("wait")
        restore_client_tool_names(ir, name_map=self._shared_name_map())
        part = ir["choices"][0]["message"]["content"][0]
        assert part["tool_name"] == "wait"
        assert "provider_metadata" not in part

    def test_a_generated_group_is_dropped_from_the_call(self):
        """A group we named is not the client's to receive.

        The container had no name, so its identifier is ours; the client has
        no namespace to route on and must not be handed one — the child comes
        back bare, like the top-level tool beside it.
        """
        generated = generated_group_id("", ["wait"])
        name_map = build_tool_name_map(
            {
                "tools": [
                    {"name": "wait", "tool_group": None},
                    {
                        "name": "wait",
                        "tool_group": generated,
                        "metadata": {"_generated_group": True},
                    },
                ]
            }
        )

        ir = self._call("wait", generated)
        restore_client_tool_names(ir, name_map=name_map)
        part = ir["choices"][0]["message"]["content"][0]
        assert part["tool_name"] == "wait"
        assert "namespace" not in part.get("provider_metadata", {})


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

    def test_ambiguous_tool_choice_warns_instead_of_guessing(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container("functions", "wait"),
            _namespace_container("agents", "wait"),
        )
        request["tool_choice"] = {"type": "function", "name": "wait"}
        upstream = pipe.convert_request(request)

        # Neither namespace can be assumed, so the name is left as sent —
        # but the caller is told it now matches nothing.
        assert upstream["tool_choice"]["function"]["name"] == "wait"
        assert any("tool_choice names 'wait'" in w for w in pipe.warnings), (
            f"expected an ambiguous-tool_choice warning; got {pipe.warnings}"
        )

    def test_undeclared_tool_choice_warns_with_no_namespaces_in_play(self):
        """The selector check must not hang off whether a rename happened.

        A request with no namespace containers builds an empty name map.
        The pipeline still has to run the check, so this goes through
        :class:`ConversionPipeline` rather than calling the helper directly —
        a unit test cannot see a caller that skips the call.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request: dict[str, Any] = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                }
            ],
            "tools": [{"type": "function", "name": "exec", "parameters": {}}],
            "tool_choice": {"type": "function", "name": "totally_made_up"},
        }
        pipe.convert_request(request)

        assert any("no tool of that name was declared" in w for w in pipe.warnings), (
            f"expected an undeclared-tool_choice warning; got {pipe.warnings}"
        )

    def test_valid_tool_choice_stays_quiet_with_no_namespaces_in_play(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request: dict[str, Any] = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                }
            ],
            "tools": [{"type": "function", "name": "exec", "parameters": {}}],
            "tool_choice": {"type": "function", "name": "exec"},
        }
        pipe.convert_request(request)

        assert pipe.warnings == [], pipe.warnings

    def test_ambiguous_allowed_tools_warns_through_the_pipeline(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container("functions", "wait"),
            _namespace_container("agents", "wait"),
        )
        request["allowed_tools"] = ["wait"]
        upstream = pipe.convert_request(request)

        names = [t.get("function", t).get("name") for t in upstream["tools"]]
        assert names == ["functions_wait", "agents_wait"]
        # The selector rides through as an extension, so it reaches the wire
        # naming a tool the same request no longer declares.
        assert upstream["allowed_tools"] == ["wait"]
        assert any("allowed_tools names 'wait'" in w for w in pipe.warnings), (
            f"expected an ambiguous-allowed_tools warning; got {pipe.warnings}"
        )

    @staticmethod
    def _with_history(*calls: tuple[str, str | None]) -> dict[str, Any]:
        """A two-namespace request replaying *calls* as history."""
        request = _request(
            _namespace_container("functions", "wait"),
            _namespace_container("agents", "wait"),
        )
        for i, (name, namespace) in enumerate(calls):
            call: dict[str, Any] = {
                "type": "function_call",
                "call_id": f"c{i}",
                "name": name,
                "arguments": "{}",
            }
            if namespace:
                call["namespace"] = namespace
            request["input"].append(call)
            request["input"].append(
                {"type": "function_call_output", "call_id": f"c{i}", "output": "ok"}
            )
        return request

    def test_history_call_dropping_its_namespace_warns(self):
        """A client that ignores the namespace field we added still gets told."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(self._with_history(("wait", None)))

        assert any("A history tool call names 'wait'" in w for w in pipe.warnings), (
            f"expected an unresolved-history warning; got {pipe.warnings}"
        )

    def test_repeated_history_calls_warn_once(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(self._with_history(*[("wait", None)] * 3))

        assert sum("A history tool call" in w for w in pipe.warnings) == 1

    def test_history_call_keeping_its_namespace_is_silent(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(self._with_history(("wait", "agents")))

        assert not [w for w in pipe.warnings if "A history tool call" in w]

    def test_history_call_for_a_top_level_tool_is_silent(self):
        """Omitting the namespace is how a client names the top-level tool.

        The bare name is ambiguous in the map — a namespaced tool declares it
        too — but the call is not, and the fallback already resolves it to the
        top-level tool's own upstream name.  Warning here would fire on the
        commonest way the two kinds of tool coexist.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(_namespace_container("agents", "wait"))
        request["tools"] = [{"type": "function", "name": "wait", "parameters": {}}]
        request["input"] += [
            {
                "type": "function_call",
                "call_id": "c0",
                "name": "wait",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c0", "output": "ok"},
        ]

        upstream = pipe.convert_request(request)

        # The collision did happen — otherwise the map is empty and the check
        # below passes without the warning path ever running.
        assert {t["function"]["name"] for t in upstream["tools"]} == {
            "wait",
            "agents_wait",
        }
        replayed = [
            call for msg in upstream["messages"] for call in msg.get("tool_calls") or []
        ]
        assert [c["function"]["name"] for c in replayed] == ["wait"]
        assert not [w for w in pipe.warnings if "A history tool call" in w], (
            f"expected silence for a top-level call; got {pipe.warnings}"
        )

    def test_history_call_warns_when_the_bare_name_belongs_to_a_namespace(self):
        """Being a name we send is not the same as being the tool meant.

        The nameless container's ``a`` has nothing to qualify it with, so it
        keeps the bare spelling, while ``other/a`` qualifies away to
        ``other_a``.  The replayed bare call therefore lands on a tool that is
        genuinely declared upstream, just not one a namespace-less call was
        ever meant to reach — it came from a container, and a client that
        meant it would have named that container.  Only asking whether a
        *top-level* tool claims the name tells that apart.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container("", "a"),
            _namespace_container("other", "a"),
        )
        request["input"] += [
            {"type": "function_call", "call_id": "c0", "name": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c0", "output": "ok"},
        ]

        upstream = pipe.convert_request(request)

        declared = {t["function"]["name"] for t in upstream["tools"]}
        # No top-level tool anywhere, yet the bare name is answered for by a
        # container's tool — the shape the old "is it a name we send" test
        # could not see past.
        assert declared == {f"{generated_group_id('', ['a'])}_a", "other_a"}
        replayed = [
            call for msg in upstream["messages"] for call in msg.get("tool_calls") or []
        ]
        assert [c["function"]["name"] for c in replayed] == ["a"]
        history = [w for w in pipe.warnings if "A history tool call" in w]
        assert len(history) == 1 and "matching none of them" in history[0], (
            f"expected an unresolved-history warning; got {pipe.warnings}"
        )

    def test_nameless_container_does_not_count_as_top_level(self):
        """An empty namespace is still a namespace, and must not suppress.

        A container with no name of its own gets one generated from its
        content, which is not a name the client ever wrote — the response leg
        hands back a bare name for it.  But on the way *up* the tool did come
        from a container, so a bare call cannot be meant for it, and reading
        its group as absent would say otherwise.  Here that tool keeps a name
        of its own while ``x/a`` qualifies away, so the bare call is ambiguous
        and only the second history arm can report it: exactly the arm
        ``has_top_level`` guards.
        """
        for label, container in (
            ("empty name", {"type": "namespace", "name": "", "tools": []}),
            ("no name key", {"type": "namespace", "tools": []}),
        ):
            container = {**container, "tools": _namespace_container("z", "a")["tools"]}
            pipe = ConversionPipeline("openai_responses", "openai_chat")
            request = _request(container, _namespace_container("x", "a"))
            request["input"] += [
                {
                    "type": "function_call",
                    "call_id": "c0",
                    "name": "a",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c0", "output": "ok"},
            ]

            upstream = pipe.convert_request(request)

            name_map = pipe._name_map
            assert not name_map.has_top_level("a"), (
                f"{label}: a container's tool is not a top-level declaration"
            )
            assert name_map.is_ambiguous("a"), label
            assert {t["function"]["name"] for t in upstream["tools"]} == {
                f"{generated_group_id('', ['a'])}_a",
                "x_a",
            }
            history = [w for w in pipe.warnings if "A history tool call" in w]
            assert len(history) == 1 and "matching none of them" in history[0], (
                f"{label}: expected the unresolved warning; got {pipe.warnings}"
            )

    def test_history_call_into_a_generated_group_is_not_asked_to_echo(self):
        """The advice has to be something the client can carry out.

        Two tools in one generated group share a bare name and cannot be told
        apart upstream.  The other ambiguity arm would tell the client to echo
        the namespace back — but a generated group never reaches the client,
        so there is nothing for it to echo and the call simply cannot be
        attributed.  Saying so is the honest answer.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        other = _nameless_container("a")
        other["tools"][0]["description"] = "The other a tool."
        request = _request(_nameless_container("a"), other)
        request["input"] += [
            {"type": "function_call", "call_id": "c0", "name": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c0", "output": "ok"},
        ]

        pipe.convert_request(request)

        history = [w for w in pipe.warnings if "A history tool call" in w]
        assert len(history) == 1, pipe.warnings
        assert "cannot tell which one" in history[0], history[0]
        assert "echo back" not in history[0], history[0]

    def test_history_call_for_an_undeclared_tool_is_silent(self):
        """The client's own stale history is not ours to complain about."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(self._with_history(("some_retired_tool", None)))

        assert not [w for w in pipe.warnings if "A history tool call" in w]

    def test_nameless_namespace_container_is_reported_on_both_legs(self):
        """A container with no name cannot do the one job a namespace has.

        The container is malformed — every upstream that implements containers
        rejects a missing or empty name — and the client is the only one who
        can fix it, so both legs have to say so.  Giving the container a
        generated name is what makes the two tools addressable rather than
        shadowing each other, and the response leg no longer has anything
        ambiguous to warn about.
        """
        for label, container in (
            ("empty name", {"type": "namespace", "name": "", "tools": None}),
            ("no name key", {"type": "namespace", "tools": None}),
        ):
            container = dict(container)
            container["tools"] = _namespace_container("x", "wait")["tools"]
            pipe = ConversionPipeline("openai_responses", "openai_chat")
            request = _request(container)
            request["tools"] = [{"type": "function", "name": "wait", "parameters": {}}]

            upstream = pipe.convert_request(request)

            # The generated group is what buys the second tool a name of its
            # own; without it the pair shadows each other upstream.
            group = generated_group_id("", ["wait"])
            assert [t["function"]["name"] for t in upstream["tools"]] == [
                "wait",
                f"{group}_wait",
            ], label
            assert any("arrived with no name of its own" in w for w in pipe.warnings), (
                f"{label}: expected a request-leg warning; got {pipe.warnings}"
            )

            before = len(pipe.warnings)
            out = pipe.convert_response(_chat_completion("wait"))

            calls = [i for i in out["output"] if i.get("type") == "function_call"]
            assert len(calls) == 1 and "namespace" not in calls[0], label
            assert not [
                w for w in pipe.warnings[before:] if "more than one declared" in w
            ], f"{label}: nothing is ambiguous now; got {pipe.warnings[before:]}"

    def test_byte_identical_nameless_children_lose_a_copy(self):
        """The duplicate a nameless container produced must not reach the wire.

        Two containers with no name of their own declare the same tool
        identically.  Nothing distinguishes the two — not the upstream, which
        would answer 400 to the repeated name, and not the client, which
        addresses its tools by a namespace it never gave.  So they are one
        duplicate, and only one goes upstream (#913).

        The malformed container is still reported, and the response leg has
        nothing left to be contested about: one tool answers to ``wait``.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(_nameless_container("wait"), _nameless_container("wait"))

        upstream = pipe.convert_request(request)

        assert [t["function"]["name"] for t in upstream["tools"]] == ["wait"]
        malformed = [w for w in pipe.warnings if "no name of its own" in w]
        assert len(malformed) == 1, (
            "one container, one cause, so one report however many children it "
            f"declares; got {pipe.warnings}"
        )

        before = len(pipe.warnings)
        out = pipe.convert_response(_chat_completion("wait"))

        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1 and "namespace" not in calls[0]
        assert not [
            w for w in pipe.warnings[before:] if "more than one declared" in w
        ], f"one tool is left, so nothing is contested; got {pipe.warnings[before:]}"

    def test_blank_container_name_is_no_name(self):
        """A blank name qualifies nothing, and the name it builds is illegal.

        ``_qualify_tool_name`` joins the two with an underscore, and providers
        match tool names against ``^[a-zA-Z0-9_-]+$`` — a space has no room in
        there, so a container named ``" "`` shipped ``" _wait"`` and a 400.
        Blank is read as no name at all, the same as a missing one, so these
        two spellings name the same thing and their identical children are one
        duplicate.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container(" ", "wait"), _namespace_container("", "wait")
        )

        upstream = pipe.convert_request(request)

        names = [t["function"]["name"] for t in upstream["tools"]]
        assert names == ["wait"], names
        assert [w for w in pipe.warnings if "no name of its own" in w]

    def test_unrenamed_call_is_quiet_on_the_response_leg(self):
        """A plain tool takes the same fallback path, and must stay silent.

        ``to_client`` hands back a bare name both for a contested tool and for
        one that was never renamed — the overwhelmingly common case.  Warning
        on the fallback itself would fire on every ordinary call, which is why
        the contested names are recorded at build time instead.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        # A namespaced tool as well, so the map is non-empty and the restore
        # actually runs — with only plain tools it short-circuits and the
        # check below would pass without proving anything.
        request = _request(_namespace_container("agents", "spawn_agent"))
        request["tools"] = [
            {"type": "function", "name": "plain_tool", "parameters": {}}
        ]
        pipe.convert_request(request)
        before = len(pipe.warnings)

        pipe.convert_response(_chat_completion("plain_tool"))

        assert pipe.warnings[before:] == []

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


# ---------------------------------------------------------------------------
# A target that can carry a group
# ---------------------------------------------------------------------------


class TestGroupsToACapableTarget:
    """A shim declaring ``tools.tool_groups`` gets the client's grouping back.

    Every other target keeps the flat list, which is what the rest of this
    module tests.  The flag is a per-provider declaration and not derivable
    from ``base``: OpenRouter's Chat Completions lane accepts the container
    while OpenAI's does not.
    """

    NAME = "test-tool-groups"

    @pytest.fixture(autouse=True)
    def _shim(self):
        register_shim(
            ProviderShim(
                name=self.NAME,
                base="open_responses",
                tools=ToolsConfig(tool_groups=True),
            )
        )
        yield
        unregister_shim(self.NAME)

    @staticmethod
    def _request(tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": "test-model",
            "tool_choice": "auto",
            "tools": tools,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        }

    def test_the_container_comes_back_at_its_first_member(self):
        pipe = ConversionPipeline("open_responses", self.NAME)

        body = pipe.convert_request(
            self._request(
                [
                    {"type": "function", "name": "plain", "parameters": {}},
                    _namespace_container("tasks", "wait", "edit"),
                ]
            )
        )

        tools = body["tools"]
        assert [t["name"] for t in tools if t["type"] != "namespace"] == ["plain"]
        container = tools[1]
        assert container["type"] == "namespace"
        assert container["name"] == "tasks"
        assert [c["name"] for c in container["tools"]] == ["wait", "edit"]
        # ``_namespace_container`` declares no description of its own, but a
        # container has to carry one — OpenAI and OpenRouter answer 400
        # without it — so an empty string stands in rather than text the
        # client never wrote.
        assert container["description"] == ""

    def test_a_flat_target_still_gets_the_flat_list(self):
        pipe = ConversionPipeline("open_responses", "chat_completions")

        body = pipe.convert_request(
            self._request(
                [
                    {"type": "function", "name": "plain", "parameters": {}},
                    _namespace_container("tasks", "wait"),
                ]
            )
        )

        names = [t["function"]["name"] for t in body["tools"]]
        assert names == ["plain", "wait"], "no container, and no group in the name"

    def test_a_child_sharing_a_top_level_name_keeps_its_bare_name(self):
        """A container-carrying target disambiguates by position, not by name.

        A top-level ``wait`` and one inside ``tasks`` share the bare name.  A
        flat target qualifies the grouped one to ``tasks_wait``; here the
        group travels beside it on the wire, so the naming pass — told this
        target carries groups — leaves the child alone and the model reads the
        name the client wrote.
        """
        pipe = ConversionPipeline("open_responses", self.NAME)

        body = pipe.convert_request(
            self._request(
                [
                    {"type": "function", "name": "wait", "parameters": {}},
                    _namespace_container("tasks", "wait"),
                ]
            )
        )

        assert [t["name"] for t in body["tools"] if t["type"] != "namespace"] == [
            "wait"
        ]
        container = next(t for t in body["tools"] if t["type"] == "namespace")
        assert [c["name"] for c in container["tools"]] == ["wait"]

    def test_the_flag_is_off_unless_a_provider_says_otherwise(self):
        """The default is half of this feature's contract.

        A provider that says nothing keeps the flat list it has always had, so
        the field is load-bearing in its default and worth naming here rather
        than leaving to whatever else notices.
        """
        assert ToolsConfig().tool_groups is False

    def test_a_group_all_of_whose_members_decline_is_dropped(self):
        """The writer's guard, on the one path that reaches it.

        Nothing the pipeline produces gets here — a tool the target cannot
        express is dropped from the IR and its group never forms — but the IR
        is a public ``TypedDict`` and the converter is public API, so a caller
        can hand ``request_to_provider`` an IR whose group holds only a tool
        this target declines.  Without the guard a childless container goes
        out, which every upstream that accepts one rejects.
        """
        from llm_rosetta.converters.base.context import ConversionContext
        from llm_rosetta.converters.open_responses.converter import (
            OpenResponsesConverter,
        )

        ctx = ConversionContext()
        ctx.options["tool_groups"] = True

        def _ir(kind: str) -> IRRequest:
            return cast(
                IRRequest,
                {
                    "model": "test-model",
                    "tools": [
                        {
                            "type": "function",
                            "name": "plain",
                            "description": "",
                            "parameters": {},
                        },
                        {
                            "type": "intrinsic",
                            "name": "web_search",
                            "description": "",
                            "parameters": {},
                            "tool_group": "g1",
                            "metadata": {"intrinsic_kind": kind},
                        },
                    ],
                },
            )

        declined, _ = OpenResponsesConverter().request_to_provider(
            _ir("no_such_kind"), context=ctx
        )
        assert [t.get("name") for t in declined["tools"]] == ["plain"], declined[
            "tools"
        ]

        # And the guard is targeted, not a blanket skip: an expressible member
        # leaves the container standing.
        kept, _ = OpenResponsesConverter().request_to_provider(
            _ir("web_search"), context=ctx
        )
        assert [t["type"] for t in kept["tools"]] == ["function", "namespace"]
        assert kept["tools"][1]["tools"], "the expressible member keeps the group"

    def test_a_group_the_target_cannot_express_is_dropped(self):
        """A container with nothing renderable in it never reaches the wire.

        The only member is an intrinsic tool this shim declares no server tool
        for, so it survives the reader and is then dropped from the IR, and the
        group never forms — which is *why* this asserts on the body rather than
        on the writer's empty-group guard: that guard sits behind the drop and
        cannot be reached from here.

        The child is **named** on purpose.  Left unnamed
        (``{"type": "web_search"}``) it never reaches the IR at all — the
        reader skips an unnamed child in a container, a pre-existing rule and a
        different reason — so that shape would pin something other than the
        one this docstring gives.
        """
        pipe = ConversionPipeline("open_responses", self.NAME)

        body = pipe.convert_request(
            self._request(
                [
                    {"type": "function", "name": "plain", "parameters": {}},
                    {
                        "type": "namespace",
                        "name": "g1",
                        "tools": [{"type": "web_search", "name": "web_search"}],
                    },
                ]
            )
        )

        assert [t.get("name") for t in body["tools"]] == ["plain"], body["tools"]

    def test_the_generated_group_is_carried_too(self):
        """A group we named still goes out; only the client leg drops it."""
        pipe = ConversionPipeline("open_responses", self.NAME)

        body = pipe.convert_request(self._request([_namespace_container("", "wait")]))

        container = body["tools"][0]
        assert container["type"] == "namespace"
        assert container["name"] == generated_group_id("", ["wait"])

    def test_children_of_different_containers_keep_the_shared_name(self):
        """Two containers may each hold a ``wait``; both keep it.

        The names never meet — each sits in its own container — so a naming
        pass counting collisions per group has nothing to separate, while a
        flat target would qualify them apart (see
        ``test_collision_rename_is_reversed``).
        """
        pipe = ConversionPipeline("open_responses", self.NAME)

        body = pipe.convert_request(
            self._request(
                [
                    _namespace_container("functions", "wait"),
                    _namespace_container("agents", "wait"),
                ]
            )
        )

        containers = [t for t in body["tools"] if t["type"] == "namespace"]
        assert [[c["name"] for c in t["tools"]] for t in containers] == [
            ["wait"],
            ["wait"],
        ]

    def test_a_container_carries_the_description_the_client_gave(self):
        """The description is the client's, passed through untouched."""
        pipe = ConversionPipeline("open_responses", self.NAME)

        declared = _namespace_container("tasks", "wait")
        declared["description"] = "Task tools."
        body = pipe.convert_request(
            self._request(
                [
                    {"type": "function", "name": "plain", "parameters": {}},
                    declared,
                ]
            )
        )

        container = next(t for t in body["tools"] if t["type"] == "namespace")
        assert container["description"] == "Task tools."

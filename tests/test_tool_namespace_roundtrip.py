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
    apply_upstream_tool_names,
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

    def test_resolves_bare_name_while_one_tool_claims_it(self):
        """``tool_choice`` has no namespace field, so it resolves by name."""
        ir = {
            "tools": [
                {
                    "name": "agents_wait",
                    "metadata": {"namespace": "agents", "_original_name": "wait"},
                }
            ]
        }
        assert build_tool_name_map(ir).to_upstream("wait") == "agents_wait"

    def test_bare_name_unresolved_when_two_tools_claim_it(self):
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
        assert build_tool_name_map(ir).to_upstream("wait") == "wait"

    def test_bare_name_unresolved_when_an_untouched_tool_claims_it(self):
        """The untouched top-level tool is a claimant too, and blocks the guess."""
        ir = {
            "tools": [
                {"name": "wait", "metadata": {}},
                {
                    "name": "agents_wait",
                    "metadata": {"namespace": "agents", "_original_name": "wait"},
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
            "metadata": {"namespace": "agents", "_original_name": "wait"},
        }
    ]

    # Two tools answer to the bare name, so a selector naming it resolves
    # to neither.
    AMBIGUOUS_TOOLS = IR_TOOLS + [
        {
            "name": "functions_wait",
            "metadata": {"namespace": "functions", "_original_name": "wait"},
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

    def test_history_call_for_a_contested_name_warns(self):
        """Reaching the provider is not the same as being attributable.

        Both tools kept their bare spelling, so the replayed call resolves to
        a name we really are sending — which is exactly why the ``declared``
        check cannot catch this one, and why ``is_contested`` has to be its
        own trigger rather than a refinement of the ambiguity test.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request, name = self._contested_request()
        request["input"] += [
            {"type": "function_call", "call_id": "c0", "name": name, "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c0", "output": "ok"},
        ]

        upstream = pipe.convert_request(request)

        replayed = [
            call for msg in upstream["messages"] for call in msg.get("tool_calls") or []
        ]
        assert [c["function"]["name"] for c in replayed] == [name]
        # The half that defeats the `declared` test: the name is one we send.
        assert name in {t["function"]["name"] for t in upstream["tools"]}
        history = [w for w in pipe.warnings if "A history tool call" in w]
        assert len(history) == 1 and "cannot tell which one" in history[0], (
            f"expected an unattributable-history warning; got {pipe.warnings}"
        )

    def test_contested_history_warning_does_not_invent_a_cause(self):
        """Qualification fails two ways, and the message must not pick one.

        Here ``x/a`` and ``y/a`` fail because top-level ``x_a`` and ``y_a``
        already hold their qualified spellings — nothing is near the 64-char
        budget.  A message blaming the length limit would send the reader off
        to shorten three-character names.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request = _request(
            _namespace_container("x", "a"),
            _namespace_container("y", "a"),
        )
        request["tools"] = [
            {"type": "function", "name": n, "parameters": {}} for n in ("x_a", "y_a")
        ]
        request["input"] += [
            {"type": "function_call", "call_id": "c0", "name": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c0", "output": "ok"},
        ]

        upstream = pipe.convert_request(request)

        longest = max(len(t["function"]["name"]) for t in upstream["tools"])
        assert longest < 10, "no name here is anywhere near the length budget"
        history = [w for w in pipe.warnings if "A history tool call" in w]
        assert len(history) == 1 and "length" not in history[0], (
            f"the cause is a taken spelling, not the length cap; got {history}"
        )
        # The accurate cause is carried by the qualification warnings, which
        # the message defers to, so they have to actually be there.
        assert [w for w in pipe.warnings if "still collides" in w]

    def test_history_call_for_an_undeclared_tool_is_silent(self):
        """The client's own stale history is not ours to complain about."""
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        pipe.convert_request(self._with_history(("some_retired_tool", None)))

        assert not [w for w in pipe.warnings if "A history tool call" in w]

    @staticmethod
    def _contested_request(*, stream: bool = False) -> tuple[dict[str, Any], str]:
        """A request where qualification cannot fit, and the shared name.

        Two tools in different namespaces share a name that already fills the
        64-char budget, so neither can be qualified and both go upstream
        spelled the same way.
        """
        name = "w" * 64
        return _request(
            _namespace_container("a", name),
            _namespace_container("b", name),
            stream=stream,
        ), name

    def test_contested_call_warns_on_the_response_leg(self):
        """The namespace is dropped on the way back, and the caller hears why.

        Goes through the pipeline rather than ``restore_client_tool_names``:
        the warnings list has to actually be threaded from the response leg,
        which a direct call to the helper cannot show.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request, name = self._contested_request()
        pipe.convert_request(request)
        before = len(pipe.warnings)

        out = pipe.convert_response(_chat_completion(name))

        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        # Refusing to guess is the right answer; it is also a broken call, so
        # it cannot pass silently.
        assert "namespace" not in calls[0]
        added = pipe.warnings[before:]
        assert any("more than one declared tool" in w for w in added), (
            f"expected a contested-call warning from the response leg; got {added}"
        )

    def test_contested_call_warns_on_the_streaming_leg(self):
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request, name = self._contested_request(stream=True)
        pipe.convert_request(request)
        before = len(pipe.warnings)
        proc = pipe.create_stream_processor()

        events: list[dict[str, Any]] = []
        for chunk in _chat_chunks(name):
            events.extend(proc.process_chunk(chunk))

        for event_type, item in _function_calls(events):
            assert "namespace" not in item, event_type
        added = pipe.warnings[before:]
        assert any("more than one declared tool" in w for w in added), (
            f"expected a contested-call warning from the stream; got {added}"
        )

    def test_contested_stream_warns_once_however_it_is_chunked(self):
        """Two calls, two chunks, still one warning — as non-streaming gives.

        The restore runs per chunk, so a set scoped to the chunk would report
        the same name once per frame.  Whether a provider packs its tool calls
        into one chunk or sends each in its own — the anthropic wire format
        does the latter — is not something the reader should hear about.
        """
        pipe = ConversionPipeline("openai_responses", "openai_chat")
        request, name = self._contested_request(stream=True)
        pipe.convert_request(request)
        before = len(pipe.warnings)
        proc = pipe.create_stream_processor()

        def start(index: int) -> dict[str, Any]:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": f"call_{index}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }

        for chunk in (start(0), start(1)):
            proc.process_chunk(chunk)

        added = [w for w in pipe.warnings[before:] if "more than one declared" in w]
        assert len(added) == 1, f"expected one warning for the stream; got {added}"

    def test_nameless_namespace_container_is_reported_on_both_legs(self):
        """A container with no name cannot do the one job a namespace has.

        Both safety nets used to be off at once.  ``_dedup_ir_tool_names``
        skipped qualification without ever reaching ``_qualify_tool_name``,
        which is where every *other* qualification failure is reported, and
        the contested set tested the namespace for truth rather than presence,
        so an empty one looked like a top-level tool and the response leg said
        nothing either.  Two tools shadowed each other in complete silence.
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

            # The shadowing itself is unchanged — this is about reporting it.
            assert [t["function"]["name"] for t in upstream["tools"]] == [
                "wait",
                "wait",
            ], label
            assert any("has no name of its own" in w for w in pipe.warnings), (
                f"{label}: expected a request-leg warning; got {pipe.warnings}"
            )

            before = len(pipe.warnings)
            out = pipe.convert_response(_chat_completion("wait"))

            calls = [i for i in out["output"] if i.get("type") == "function_call"]
            assert len(calls) == 1 and "namespace" not in calls[0], label
            assert any(
                "more than one declared tool" in w for w in pipe.warnings[before:]
            ), f"{label}: expected a response-leg warning; got {pipe.warnings[before:]}"

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

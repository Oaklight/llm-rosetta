"""The ChatGPT plan profile: what it declares, and what it must not.

`openai_chatgpt--open_responses` is a profile of the vendor-neutral
`open_responses` base, so a request converts exactly as any other Responses
request does.  What differs is the credential (a signed-in account, not an API
key), the model list (the route answers its own shape), and the capability
surface (the preview route refuses most hosted tools).
"""

from __future__ import annotations

import json

import pytest

from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.shims import get_shim
from llm_rosetta.shims.providers import get_model_list_transform, load_providers
from llm_rosetta_gateway.providers import build_provider_info

SHIM = "openai_chatgpt--open_responses"

#: The Responses events the plan route answers a one-word reply with, trimmed
#: to what a fold needs.  The terminal event carries an empty ``output`` on
#: purpose: the route does that, and the fold must reconstruct the turn from
#: the item events rather than from the terminal payload (#1061).
_PLAN_STREAM: list[dict] = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1763000000,
            "model": "gpt-5.1",
            "status": "in_progress",
            "output": [],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    },
    {
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {
        "type": "response.output_text.delta",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello",
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        },
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1763000000,
            "model": "gpt-5.1",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        },
    },
]


def _pipeline() -> ConversionPipeline:
    """A Messages → Responses pipeline with its request already converted.

    The fold writes through the response leg, which needs the request leg's
    context (custom-tool marks, the tool-name map, the source traits), so the
    test builds the pipeline the gateway would have built.
    """
    pipeline = ConversionPipeline("messages", "open_responses", target_shim=SHIM)
    pipeline.convert_request(
        {
            "model": "gpt-5.1",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    return pipeline


#: The shape the plan route answers, trimmed to what matters.
MODELS_BODY = {
    "models": [
        {
            "slug": "gpt-5.1-codex",
            "display_name": "GPT-5.1 Codex",
            "visibility": "list",
        },
        {"slug": "hidden-preview", "display_name": "Hidden", "visibility": "hidden"},
        {"slug": "gpt-5.1", "display_name": "GPT-5.1", "visibility": "list"},
        {"display_name": "No slug at all", "visibility": "list"},
    ]
}


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_providers()


def shim():
    """The loaded shim; every fact below is read off it."""
    loaded = get_shim(SHIM)
    assert loaded is not None
    return loaded


class TestTheProfile:
    def test_it_is_a_profile_of_the_neutral_base(self):
        profile = shim()

        assert profile.format == "open_responses"
        assert profile.model_type == "llm"

    def test_the_credential_comes_from_the_exchange(self):
        """No ``api_key_env``: an API key is not what this lane authenticates with.

        Declaring one would also hand the loader a static key beside the
        shim's own exchange, which it refuses.
        """
        profile = shim()

        assert profile.connection.api_key_env is None
        assert profile.connection.base_url == "https://api.openai.com/v1"
        assert profile.auth is not None
        assert profile.auth.kind == "oauth"
        # The token lives 3600 s; a 3600 s loop could renew after expiry.
        assert profile.connection.token_refresh_interval == 2880

    def test_the_route_is_stateless(self):
        profile = shim()

        assert profile.store_default is False
        assert profile.response_id_prefix == "resp_"
        assert profile.hoist_system_messages is True

    def test_the_tool_surface_is_the_preview_route_s(self):
        profile = shim()

        assert profile.tools.custom_tools is True
        assert profile.tools.tool_groups is True
        assert profile.tools.intrinsic_tools == ("web_search",)

    @pytest.mark.parametrize(
        "refused",
        [
            "code_interpreter",
            "file_search",
            "image_generation",
            "computer_use",
            "mcp",
            "tool_search",
        ],
    )
    def test_hosted_tools_the_route_refuses_are_not_declared(self, refused):
        """Advertising one would promise a capability the upstream rejects."""
        assert refused not in shim().tools.intrinsic_tools

    def test_the_picker_titles_the_group(self):
        """The fallback title is the mangled key; this one is read by operators."""
        profile = shim()

        assert profile.group_label == "OpenAI (ChatGPT plan)"
        assert profile.group is None  # the name split already files it


def transform():
    """The hook the loader registered for this shim."""
    hook = get_model_list_transform(SHIM)
    assert hook is not None
    return hook


class TestTheRequestSurface:
    """What the route refuses, and what the shim does about it."""

    def _apply(self, body: dict) -> dict:
        for transform in shim().post_ir_transforms:
            body = transform(body)
        return body

    @pytest.mark.parametrize(
        "field",
        [
            "background",
            "conversation",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "moderation",
            "multi_agent",
            "prompt",
            "prompt_cache_retention",
            "safety_identifier",
            "temperature",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        ],
    )
    def test_a_refused_field_is_stripped(self, field):
        """The client is entitled to send it; the route refuses the request."""
        out = self._apply({"model": "m", "input": [], field: "anything"})

        assert field not in out

    def test_the_declared_list_is_the_documented_one(self):
        """A guard against the strip list drifting from the docs."""
        from llm_rosetta.shims.providers.openai_chatgpt.llm.transforms import (
            UNSUPPORTED_FIELDS,
        )

        assert set(UNSUPPORTED_FIELDS) == {
            "background",
            "conversation",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "moderation",
            "multi_agent",
            "prompt",
            "prompt_cache_retention",
            "safety_identifier",
            "temperature",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
            "previous_response_id",
        }

    def test_store_is_forced_false(self):
        """`true` is a valid Responses value and an omitted one is refused too."""
        assert self._apply({"model": "m", "store": True})["store"] is False
        assert self._apply({"model": "m"})["store"] is False

    def test_the_history_pointer_is_dropped_but_the_history_is_not(self):
        """The route keeps no server-side state, so `input` must carry it."""
        out = self._apply(
            {
                "model": "m",
                "previous_response_id": "resp_abc",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        )

        assert "previous_response_id" not in out
        assert out["input"] == [{"type": "message", "role": "user", "content": "hi"}]

    def test_nothing_else_is_touched(self):
        body = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "f"}],
            "instructions": "be brief",
            "reasoning": {"effort": "low"},
        }

        assert self._apply(dict(body)) == {**body, "store": False}


class TestStreamingIsRequired:
    def test_the_shim_declares_it(self):
        """The route answers SSE only."""
        assert shim().connection.stream_required is True
        assert (
            build_provider_info("open_responses", {}, shim_name=SHIM).stream_required
            is True
        )

    def test_a_non_streaming_client_is_served_by_the_fold(self):
        """A Messages client gets one body back, not a refusal.

        The route answers SSE only, so the gateway streams upstream and folds
        the events back into the response they describe.  This used to be a
        documented ``400`` because reassembly existed for Chat Completions
        clients alone (#1061); the IR fold serves every source format, so a
        Messages or Responses client behind the plan lane now works.
        """
        import asyncio

        from llm_rosetta.routing import ResolvedRoute
        from llm_rosetta_gateway.proxy.engine import _non_streaming_from_stream

        route = ResolvedRoute(
            source_provider="messages",
            target_provider="open_responses",
            provider_name="chatgpt",
            shim_name=SHIM,
            upstream_model="gpt-5.1",
        )

        class _Stream:
            status_code = 200
            is_error = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def close(self):
                return None

            async def read_error(self):
                return ""

            def __aiter__(self):
                async def gen():
                    for chunk in _PLAN_STREAM:
                        yield chunk

                return gen()

        class _Transport:
            def __init__(self):
                self.streamed: list[bool] = []

            async def send(self, *args, **kwargs):
                raise AssertionError("a stream-only upstream is never asked for JSON")

            async def send_streaming(self, provider_info, url, body, **kwargs):
                provider_info.auth_headers()
                self.streamed.append(body.get("stream"))
                return _Stream()

            async def close(self):
                return None

        transport = _Transport()
        response, profile = asyncio.run(
            _non_streaming_from_stream(
                route=route,
                provider_info=build_provider_info("open_responses", {}, shim_name=SHIM),
                pipeline=_pipeline(),
                target_body={"model": "gpt-5.1", "input": "hi"},
                body={
                    "model": "gpt-5.1",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                model="gpt-5.1",
                transport=transport,
                metadata_store=None,
                extra_headers=None,
                persistence=None,
                capture_state=None,
                entry_id=None,
                profile={},
            )
        )

        assert transport.streamed == [True]
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["type"] == "message"
        assert body["content"] == [{"type": "text", "text": "hello", "citations": None}]
        assert body["stop_reason"] == "end_turn"
        assert profile["usage"]["prompt_tokens"] == 5


#: A plan-lane turn whose only item is a function call, ending on a plain
#: ``stop`` — the shape the route actually sends (milo measured it: the
#: ``response.completed`` carries ``status: completed`` and no tool terminal
#: reason, because the route has no such word).
_PLAN_TOOL_STREAM: list[dict] = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_2",
            "object": "response",
            "created_at": 1763000000,
            "model": "gpt-5.1",
            "status": "in_progress",
            "output": [],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_a",
            "name": "get_weather",
            "arguments": "",
            "status": "in_progress",
        },
    },
    {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": '{"city": ',
    },
    {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": '"Paris"}',
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_a",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        },
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_2",
            "object": "response",
            "created_at": 1763000000,
            "model": "gpt-5.1",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
        },
    },
]


class TestTheFoldedTurn:
    def test_a_plan_tool_turn_is_a_tool_turn_for_the_client(self):
        """The route says ``stop``; the folded parts are what make it a call.

        ``_finalize_completed_turn`` is the one line that turns it, and no
        other test reaches it: a Messages client would otherwise be told
        ``end_turn`` for a turn whose only content is a tool call, which is the
        difference between running the tool and believing the model finished.
        """
        pipeline = _pipeline()
        events: list[dict] = []
        processor = pipeline.create_stream_processor(on_ir_event=events.append)
        for chunk in _PLAN_TOOL_STREAM:
            processor.process_chunk(chunk)
        processor.process_chunk(None)

        body = pipeline.convert_stream_events(events)

        assert body["stop_reason"] == "tool_use"
        assert body["content"] == [
            {
                "type": "tool_use",
                "id": "call_a",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ]


class TestTheModelList:
    def test_only_visible_models_are_offered(self):
        ids, upstream = transform()(MODELS_BODY["models"])

        assert ids == ["gpt-5.1-codex", "gpt-5.1"]
        # The slug is what the route expects back, so nothing is renamed.
        assert upstream == {}

    def test_the_server_order_is_kept(self):
        ids, _ = transform()(
            [
                {"slug": "z", "visibility": "list"},
                {"slug": "a", "visibility": "list"},
            ]
        )

        assert ids == ["z", "a"]

    def test_the_admin_fetch_uses_the_hook_and_keeps_its_order(self):
        """The route answers ``{models: [...]}``, which the generic path ignores.

        The hook's order is the shim's opinion (newest capability first), so
        the caller keeps it rather than sorting an upstream that already had
        one.
        """
        from llm_rosetta_gateway.admin.routes.config.upstream import _extract_model_ids

        ids, upstream = _extract_model_ids(MODELS_BODY, "open_responses", SHIM, None)

        assert ids == ["gpt-5.1-codex", "gpt-5.1"]
        assert upstream == {}

    def test_a_generic_body_is_still_sorted(self):
        """Without a hook the caller sorts, so an unordered ``data`` is stable."""
        from llm_rosetta_gateway.admin.routes.config.upstream import _extract_model_ids

        ids, _ = _extract_model_ids(
            {"data": [{"id": "z"}, {"id": "a"}]}, "open_responses", None, None
        )

        assert ids == ["a", "z"]

    def test_a_nested_envelope_reaches_the_hook(self):
        """SiliconFlow wraps its catalogue as ``{data: {models: [...]}}``.

        The envelope is unwrapped before the hook sees it.  A hook handed the
        dict instead would iterate its keys and produce nothing.
        """
        from llm_rosetta_gateway.admin.routes.config.upstream import _extract_model_ids

        body = {"data": {"models": MODELS_BODY["models"]}}
        ids, upstream = _extract_model_ids(body, "open_responses", SHIM, None)

        assert ids == ["gpt-5.1-codex", "gpt-5.1"]
        assert upstream == {}

    def test_without_the_hook_the_body_yields_nothing(self):
        """What the profile would look like if the transform were dropped."""
        from llm_rosetta_gateway.admin.routes.config.upstream import _extract_model_ids

        ids, _ = _extract_model_ids(
            MODELS_BODY, "open_responses", "openai--open_responses", None
        )

        assert ids == []


class TestNoSystemItemReachesTheRoute:
    """The route refuses a system item anywhere in ``input``, first included.

    It answers ``400 {"detail":"System messages are not allowed"}``, so this is
    not the late-message case the hoist transform was written for — a Responses
    client that sends its prompt as an ``input`` item instead of ``instructions``
    trips it on its first request, and a Chat Completions source that keeps a
    mid-conversation system message keeps an item too.
    """

    def _convert(self, source: str, body: dict) -> dict:
        from llm_rosetta.pipeline.chat import ConversionPipeline

        return ConversionPipeline(
            source,
            "open_responses",
            target_shim=SHIM,
            upstream_model="gpt-5.1",
        ).convert_request(body)

    @staticmethod
    def _roles(body: dict) -> list[str]:
        return [
            item.get("role", "")
            for item in body.get("input", [])
            if isinstance(item, dict)
        ]

    def test_the_leading_item_is_hoisted_into_instructions(self):
        """The very shape the route refused: a prompt carried as an item."""
        out = self._convert(
            "open_responses",
            {
                "model": "gpt-5.1",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "You are Codex."}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    },
                ],
                "stream": True,
            },
        )

        assert self._roles(out) == ["user"]
        assert out["instructions"] == "You are Codex."

    def test_a_late_system_item_becomes_a_user_turn(self):
        out = self._convert(
            "open_responses",
            {
                "model": "gpt-5.1",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    },
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "note"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "again"}],
                    },
                ],
                "stream": True,
            },
        )

        assert self._roles(out) == ["user", "user", "user"]
        assert any(
            part.get("text") == "<system>\nnote\n</system>"
            for item in out["input"]
            for part in item.get("content", [])
        )

    def test_a_chat_completions_source_keeps_no_system_item(self):
        out = self._convert(
            "chat_completions",
            {
                "model": "gpt-5.1",
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                    {"role": "system", "content": "S2"},
                    {"role": "user", "content": "again"},
                ],
                "stream": True,
            },
        )

        assert "system" not in self._roles(out)
        assert "developer" not in self._roles(out)
        assert out["instructions"] == "S"

    def test_the_shim_declares_the_transform(self):
        """A capability the route requires, not a lane preference."""
        assert [repr(t) for t in shim().ir_transforms] == [
            "hoist_late_system_messages()"
        ]


class TestDroppedFieldsAreReported:
    """A drop the operator cannot see looks like it worked.

    Refusing the request instead would make the lane unusable — most clients
    send ``temperature`` unconditionally — so the drop is logged, and only
    when something was actually dropped.
    """

    def _apply(self, body: dict) -> dict:
        for transform in shim().post_ir_transforms:
            body = transform(body)
        return body

    def test_a_dropped_field_is_named_in_the_log(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="llm_rosetta.shims.providers"):
            self._apply({"model": "m", "max_output_tokens": 100, "temperature": 0.2})

        assert "max_output_tokens" in caplog.text
        assert "temperature" in caplog.text

    def test_a_clean_request_logs_nothing(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="llm_rosetta.shims.providers"):
            self._apply({"model": "m", "input": []})

        assert "dropped field" not in caplog.text

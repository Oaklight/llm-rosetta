"""The ChatGPT plan profile: what it declares, and what it must not.

`openai_chatgpt--open_responses` is a profile of the vendor-neutral
`open_responses` base, so a request converts exactly as any other Responses
request does.  What differs is the credential (a signed-in account, not an API
key), the model list (the route answers its own shape), and the capability
surface (the preview route refuses most hosted tools).
"""

from __future__ import annotations

import pytest

from llm_rosetta.shims import get_shim
from llm_rosetta.shims.providers import get_model_list_transform, load_providers
from llm_rosetta_gateway.providers import build_provider_info

SHIM = "openai_chatgpt--open_responses"

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
        from llm_rosetta.shims.providers.openai_chatgpt.llm.open_responses.transforms import (
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

    def test_a_non_streaming_responses_client_is_refused_clearly(self):
        """Not silently broken or empty: the gateway says what to do.

        Reassembly exists for Chat Completions sources only, so a Responses
        client that did not ask to stream is told to ask again rather than
        handed a synthesised body the gateway would have to guess at.
        """
        import asyncio

        from llm_rosetta.routing import ResolvedRoute
        from llm_rosetta_gateway.proxy.engine import _non_streaming_from_stream

        route = ResolvedRoute(
            source_provider="open_responses",
            target_provider="open_responses",
            provider_name="chatgpt",
            shim_name=SHIM,
            upstream_model="gpt-5.1",
        )
        response, _ = asyncio.run(
            _non_streaming_from_stream(
                route=route,
                provider_info=build_provider_info("open_responses", {}, shim_name=SHIM),
                # The refusal happens before either is touched, which is the
                # point: nothing is sent upstream for a client that cannot be
                # served.
                pipeline=None,  # ty: ignore[invalid-argument-type]
                target_body={},
                body={},
                model="m",
                transport=None,  # ty: ignore[invalid-argument-type]
                metadata_store=None,
                extra_headers=None,
                persistence=None,
                capture_state=None,
                entry_id=None,
                profile={},
            )
        )

        assert response.status_code == 400
        assert b"stream=true" in response.body


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

    def test_without_the_hook_the_body_yields_nothing(self):
        """What the profile would look like if the transform were dropped."""
        from llm_rosetta_gateway.admin.routes.config.upstream import _extract_model_ids

        ids, _ = _extract_model_ids(
            MODELS_BODY, "open_responses", "openai--open_responses", None
        )

        assert ids == []


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

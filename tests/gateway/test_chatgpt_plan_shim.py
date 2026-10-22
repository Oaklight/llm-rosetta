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

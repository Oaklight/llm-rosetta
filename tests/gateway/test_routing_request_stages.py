"""The stages that decide the upstream key and the upstream session identity."""

from __future__ import annotations

from llm_rosetta_gateway.routing import (
    ConversationIdentity,
    ProviderKeyAffinity,
    SelectionContext,
    SessionHeaderAffinity,
)
from llm_rosetta_gateway.routing.conversation import conversation_id
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth


def _pi(**kwargs) -> ProviderInfo:
    return ProviderInfo(
        name="p",
        api_key=kwargs.pop("api_key", "k"),
        base_url="https://x.example/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        **kwargs,
    )


def _ctx(pi: ProviderInfo, ir: dict | None, client: str | None = "clienthash"):
    ctx = SelectionContext("m", "chat_completions")
    ctx.with_client(client)
    ctx.with_provider_info(pi)
    ctx.with_ir_request(ir if ir is not None else {"messages": []})
    return ctx


def _ir(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class TestProviderKeyAffinity:
    def test_declines_without_a_provider(self):
        ctx = SelectionContext("m", "chat_completions")
        ctx.with_client("h")
        ProviderKeyAffinity().run(ctx)
        assert ctx.provider_info is None

    def test_declines_with_a_single_key(self):
        pi = _pi(api_key="only")
        ctx = _ctx(pi, _ir("hello"))
        ProviderKeyAffinity().run(ctx)
        assert ctx.provider_info is pi

    def test_declines_without_a_client(self):
        pi = _pi(api_key="k1,k2")
        ctx = _ctx(pi, _ir("hello"), client=None)
        ProviderKeyAffinity().run(ctx)
        assert ctx.provider_info is pi

    def test_declines_when_the_provider_turned_it_off(self):
        pi = _pi(api_key="k1,k2")
        ctx = _ctx(pi, _ir("hello"))
        ctx.with_key_affinity(False)
        ProviderKeyAffinity().run(ctx)
        assert ctx.provider_info is pi

    def test_pins_one_key_per_conversation(self):
        """Same client and prefix get the same key; round-robin would not."""
        pi = _pi(api_key="k1,k2")
        first = _ctx(pi, _ir("hello"))
        second = _ctx(pi, _ir("hello"))
        ProviderKeyAffinity().run(first)
        ProviderKeyAffinity().run(second)

        assert first.provider_info.auth_headers() == second.provider_info.auth_headers()

    def test_declines_on_an_empty_request(self):
        """No prefix to derive from, so no identity to pin on."""
        pi = _pi(api_key="k1,k2")
        ctx = _ctx(pi, {})
        ProviderKeyAffinity().run(ctx)
        assert ctx.provider_info is pi


class TestConversationIdentity:
    """The id the request log keeps, whatever lane the request takes."""

    def test_derives_a_digest_from_the_client_and_first_message(self):
        ctx = _ctx(_pi(), _ir("hello"))
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation == conversation_id("clienthash", _ir("hello"))
        assert len(ctx.identity.conversation) == 32

    def test_runs_for_a_provider_without_a_session_header(self):
        """The log needs the id even when the upstream is told nothing."""
        ctx = _ctx(_pi(), _ir("hello"))
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation is not None

    def test_a_conversation_keeps_one_id_across_turns(self):
        first = _ctx(_pi(), _ir("hello"))
        later = _ctx(
            _pi(),
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "more"},
                ]
            },
        )
        ConversationIdentity().run(first)
        ConversationIdentity().run(later)
        assert first.identity.conversation == later.identity.conversation

    def test_two_conversations_from_one_client_get_two_ids(self):
        a, b = _ctx(_pi(), _ir("hello")), _ctx(_pi(), _ir("different"))
        ConversationIdentity().run(a)
        ConversationIdentity().run(b)
        assert a.identity.conversation != b.identity.conversation

    def test_declines_without_a_client(self):
        ctx = _ctx(_pi(), _ir("hello"), client=None)
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation is None

    def test_declines_without_a_message_prefix(self):
        ctx = _ctx(_pi(), {})
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation is None


class TestSessionHeaderAffinity:
    def test_declines_without_a_declared_header(self):
        pi = _pi()
        ctx = _ctx(pi, _ir("hello"))
        SessionHeaderAffinity().run(ctx)
        assert ctx.provider_info is pi
        assert ctx.upstream_session_id is None

    def test_records_the_id_it_tells_the_upstream(self):
        ctx = _ctx(_pi(session_header="x-s"), _ir("hello"))
        SessionHeaderAffinity().run(ctx)
        assert ctx.upstream_session_id is not None
        assert ctx.provider_info.session_headers() == {"x-s": ctx.upstream_session_id}

    def test_a_conversation_keeps_one_id_across_turns(self):
        pi = _pi(session_header="x-s")
        first = _ctx(pi, _ir("hello"))
        later = _ctx(
            pi,
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "more"},
                ]
            },
        )
        SessionHeaderAffinity().run(first)
        SessionHeaderAffinity().run(later)
        assert first.upstream_session_id == later.upstream_session_id

    def test_two_conversations_get_two_ids(self):
        pi = _pi(session_header="x-s")
        a, b = _ctx(pi, _ir("hello")), _ctx(pi, _ir("different"))
        SessionHeaderAffinity().run(a)
        SessionHeaderAffinity().run(b)
        assert a.upstream_session_id != b.upstream_session_id

    def test_the_header_is_present_even_with_nothing_to_key_on(self):
        """Some upstreams answer 400 for a missing session, so never omit it."""
        ctx = _ctx(_pi(session_header="x-s"), {}, client=None)
        SessionHeaderAffinity().run(ctx)
        assert ctx.upstream_session_id
        assert ctx.provider_info.session_headers()["x-s"]

    def test_the_header_is_the_conversation_the_log_records(self):
        """One derived value serves both, so the panel names the session sent."""
        ctx = _ctx(_pi(session_header="x-s"), _ir("hello"))
        ConversationIdentity().run(ctx)
        SessionHeaderAffinity().run(ctx)
        assert ctx.upstream_session_id == ctx.identity.conversation

    def test_the_header_still_resolves_without_a_conversation(self):
        """No client means no conversation, but the header must still be set."""
        ctx = _ctx(_pi(session_header="x-s"), _ir("hello"), client=None)
        ConversationIdentity().run(ctx)
        SessionHeaderAffinity().run(ctx)
        assert ctx.identity.conversation is None
        assert ctx.provider_info.session_headers()["x-s"]

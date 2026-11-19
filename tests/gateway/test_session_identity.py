"""What identifies a conversation, and whether the answer may key state.

The point of the module under test is the *source*: a client that names its
conversation is believed, and a request that does not is only guessed at.  These
tests hold both halves — the carriers, in order, and the guess, with the
conditions that make it a guess.
"""

from __future__ import annotations

import hashlib

import pytest

from llm_rosetta_gateway.routing import (
    SESSION_HEADERS,
    ConversationIdentity,
    SelectionContext,
    SessionIdentity,
    identify,
)
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _user(text: str, **extra: object) -> dict:
    message = {"role": "user", "content": [{"type": "text", "text": text}]}
    message.update(extra)
    return message


class TestCarrierHeaders:
    def test_each_declared_header_is_read(self):
        for name in SESSION_HEADERS:
            identity = identify({name: "abc"}, None)
            assert identity.value == _digest("abc"), name
            assert identity.source == f"header:{name}"

    def test_the_order_decides_which_header_wins(self):
        headers = {"x-claude-code-session-id": "b", "x-opencode-session": "a"}
        assert identify(headers, None).value == _digest("a")

    def test_lookup_ignores_case_and_surrounding_space(self):
        identity = identify({"X-Claude-Code-Session-Id": "  abc  "}, None)
        assert identity.value == _digest("abc")
        assert identity.source == "header:x-claude-code-session-id"

    def test_a_blank_carrier_is_not_an_answer(self):
        identity = identify({"x-opencode-session": "   "}, {"messages": [_user("hi")]})
        assert identity.source == "derived:opening"

    def test_a_generic_session_header_is_not_read_by_default(self):
        """``x-session-id`` is set by middleware for its own purposes."""
        identity = identify({"x-session-id": "abc"}, None)
        assert identity.value is None
        assert identity.source == "none"

    def test_a_long_value_is_truncated_before_hashing(self):
        """An unbounded header is an unbounded log line, and a prefix is enough."""
        long = "x" * 200
        identity = identify({"x-opencode-session": long}, None)
        assert identity.value == _digest("x" * 128)


class TestTheVendorPattern:
    """``x-<vendor>-session-id`` is the convention everyone else follows.

    A named list cannot name a client that ships tomorrow, so the convention
    is read too — but only a value that looks like an id, and only after every
    named carrier has declined.  The shape test is the whole guard: without it
    a middleware's own header would be believed.
    """

    def test_a_vendor_header_is_read(self):
        identity = identify({"x-cursor-session-id": "9f3c1a77-abc"}, None)
        assert identity.value == _digest("9f3c1a77-abc")
        assert identity.source == "header:x-cursor-session-id"

    def test_a_multi_word_vendor_is_read(self):
        identity = identify({"x-github-copilot-session-id": "sess-12345"}, None)
        assert identity.source == "header:x-github-copilot-session-id"

    def test_a_listed_carrier_still_wins(self):
        headers = {
            "x-claude-code-session-id": "listed",
            "x-cursor-session-id": "patterned",
        }
        assert identify(headers, None).value == _digest("listed")

    def test_a_value_that_is_not_an_id_is_refused(self):
        """A sentence in a header is not a session the client declared."""
        for value in ("", "  ", "short", "two words here", "/var/run/session"):
            identity = identify({"x-cursor-session-id": value}, None)
            assert identity.source == "none", value

    def test_a_name_without_the_shape_is_not_read(self):
        for name in ("x-session-id", "x-cursor-session", "session-id"):
            assert identify({name: "9f3c1a77-abc"}, None).source == "none", name

    def test_the_first_by_name_wins_so_one_request_always_answers_alike(self):
        headers = {"x-zeta-session-id": "id-zeta-1", "x-alpha-session-id": "id-alpha-1"}
        first = identify(headers, None)
        assert first.source == "header:x-alpha-session-id"
        assert identify(dict(reversed(list(headers.items()))), None) == first

    def test_case_is_ignored(self):
        identity = identify({"X-Cursor-Session-Id": "9f3c1a77-abc"}, None)
        assert identity.source == "header:x-cursor-session-id"

    def test_a_patterned_header_beats_a_body_carrier(self):
        """The pattern is still a header, so it sits above the body's fields."""
        body = {"prompt_cache_key": "thread-1"}
        identity = identify({"x-cursor-session-id": "9f3c1a77-abc"}, body)
        assert identity.source == "header:x-cursor-session-id"


class TestCarrierBody:
    def test_user_id_as_an_object(self):
        body = {"metadata": {"user_id": {"session_id": "s-1"}}}
        identity = identify({}, body)
        assert identity.value == _digest("s-1")
        assert identity.source == "body:metadata.user_id.session_id"

    def test_user_id_as_json_text(self):
        """Claude Code sends the object as a JSON string."""
        body = {"metadata": {"user_id": '{"device_id":"d","session_id":"s-1"}'}}
        assert identify({}, body).source == "body:metadata.user_id.session_id"

    def test_unparseable_user_id_is_ignored(self):
        body = {"metadata": {"user_id": "not json"}, "messages": [_user("hi")]}
        assert identify({}, body).source == "derived:opening"

    def test_prompt_cache_key_names_the_thread(self):
        body = {"prompt_cache_key": "thread-1", "input": []}
        identity = identify({}, body)
        assert identity.value == _digest("thread-1")
        assert identity.source == "body:prompt_cache_key"

    def test_a_header_beats_a_body_carrier(self):
        body = {"metadata": {"user_id": {"session_id": "s-body"}}}
        identity = identify({"x-opencode-session": "s-header"}, body)
        assert identity.value == _digest("s-header")
        assert identity.source == "header:x-opencode-session"

    def test_metadata_beats_prompt_cache_key(self):
        body = {
            "metadata": {"user_id": {"session_id": "s-meta"}},
            "prompt_cache_key": "s-cache",
        }
        assert identify({}, body).source == "body:metadata.user_id.session_id"


class TestTheDerivedId:
    def test_the_preamble_is_skipped_for_the_users_own_words(self):
        body = {
            "messages": [
                _user("<system-reminder>project rules</system-reminder>"),
                _user("fix the parser"),
                _user("and then the tests"),
            ]
        }
        identity = identify({}, body)
        assert identity.source == "derived:opening"
        assert identity.declared is False

    def test_two_openings_differ(self):
        first = identify({}, {"messages": [_user("fix the parser")]})
        second = identify({}, {"messages": [_user("write the docs")]})
        assert first.value != second.value

    def test_a_moving_cache_mark_does_not_change_the_id(self):
        plain = {"messages": [_user("fix the parser")]}
        marked = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "fix the parser",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
        assert identify({}, plain).value == identify({}, marked).value

    def test_a_preamble_with_no_user_words_falls_back_to_the_preamble(self):
        """The known collision: two conversations in one project share an id.

        Documented rather than fixed: nothing in the request distinguishes
        them, which is exactly why this id may not key stored state.
        """
        body = {"messages": [_user("<system-reminder>rules</system-reminder>")]}
        first = identify({}, body)
        second = identify({}, body)
        assert first.value == second.value
        assert first.declared is False

    def test_the_client_separates_two_projects_that_open_alike(self):
        """Same preset preamble, two tenants: they are two conversations.

        The opening alone cannot tell them apart, and merging them would
        report one project's traffic under the other's key.
        """
        body = {"messages": [_user("<system-reminder>rules</system-reminder>")]}
        one = identify({}, body, client="client-a")
        two = identify({}, body, client="client-b")
        assert one.value != two.value

    def test_without_a_client_the_opening_alone_still_answers(self):
        body = {"messages": [_user("fix the parser")]}
        assert identify({}, body, client=None).value is not None

    def test_gemini_contents_are_read(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        assert identify({}, body).source == "derived:opening"

    def test_no_history_derives_nothing(self):
        assert identify({}, {"messages": []}).source == "none"

    def test_a_system_prompt_does_not_name_the_conversation(self):
        """The system prompt is the project's, identical for every conversation.

        Stopping the opening at it hashed the system prompt alone, so two
        conversations in one project shared a digest — the exact collapse this
        id must not have, since it is what the log groups by.
        """
        first = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                _user("fix the parser"),
            ]
        }
        second = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                _user("write the docs"),
            ]
        }
        one = identify({}, first)
        two = identify({}, second)
        assert one.source == "derived:opening"
        assert one.value != two.value

    def test_a_developer_prompt_does_not_name_it_either(self):
        first = {
            "messages": [
                {"role": "developer", "content": "Project rules."},
                _user("fix the parser"),
            ]
        }
        second = {
            "messages": [
                {"role": "developer", "content": "Project rules."},
                _user("write the docs"),
            ]
        }
        assert identify({}, first).value != identify({}, second).value


class TestFallbacks:
    def test_a_body_fallback_is_used_when_nothing_is_readable(self):
        fallback = SessionIdentity("f" * 32, "derived:opening")
        identity = identify({}, None, body_fallback=fallback)
        assert identity == fallback

    def test_a_declared_carrier_beats_the_fallback(self):
        fallback = SessionIdentity("f" * 32, "derived:opening")
        identity = identify(
            {"x-opencode-session": "real"}, None, body_fallback=fallback
        )
        assert identity.value == _digest("real")

    def test_declared_is_true_only_for_a_carrier(self):
        assert identify({"x-opencode-session": "s"}, None).declared is True
        assert identify({}, {"prompt_cache_key": "k"}).declared is True
        assert identify({}, {"messages": [_user("hi")]}).declared is False
        assert SessionIdentity().declared is False


class TestTheStage:
    def _ctx(self, headers=None, body=None) -> SelectionContext:
        ctx = SelectionContext("m", "chat_completions")
        ctx.with_client("clienthash")
        ctx.with_ir_request({"messages": [{"role": "user", "content": "hi"}]})
        ctx.with_source_request(headers or {}, body)
        return ctx

    def test_a_declared_header_is_recorded_with_its_source(self):
        ctx = self._ctx({"x-claude-code-session-id": "sess-1"})
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation == _digest("sess-1")
        assert ctx.identity.session.source == "header:x-claude-code-session-id"

    def test_without_a_carrier_the_ir_fallback_still_fills_the_log_column(self):
        ctx = self._ctx()
        ConversationIdentity().run(ctx)
        assert ctx.identity.conversation is not None
        assert len(ctx.identity.conversation) == 32
        assert ctx.identity.session.source == "derived:opening"

    def test_a_body_carrier_is_read_before_conversion(self):
        ctx = self._ctx({}, {"metadata": {"user_id": {"session_id": "sess-2"}}})
        ConversationIdentity().run(ctx)
        assert ctx.identity.session.source == "body:metadata.user_id.session_id"


class TestTheUpstreamSessionHeader:
    def test_a_declared_id_is_what_the_upstream_is_told(self):
        from llm_rosetta_gateway.routing import SessionHeaderAffinity

        provider = ProviderInfo(
            name="p",
            api_key="k",
            base_url="https://x.example/v1",
            auth_header_fn=openai_auth,
            url_template="{base_url}/chat/completions",
            session_header="x-opencode-session",
        )
        ctx = SelectionContext("m", "chat_completions")
        ctx.with_client("clienthash")
        ctx.with_provider_info(provider)
        ctx.with_ir_request({"messages": [{"role": "user", "content": "hi"}]})
        ctx.with_source_request({"x-claude-code-session-id": "sess-1"}, None)

        ConversationIdentity().run(ctx)
        SessionHeaderAffinity().run(ctx)

        assert ctx.upstream_session_id == _digest("sess-1")


class TestNoRawValueLeaks:
    @pytest.mark.parametrize(
        "headers,body",
        [
            ({"x-opencode-session": "secret-session-id"}, None),
            ({}, {"metadata": {"user_id": {"session_id": "secret-session-id"}}}),
            ({}, {"prompt_cache_key": "secret-session-id"}),
        ],
    )
    def test_the_digest_is_all_that_survives(self, headers, body):
        identity = identify(headers, body)
        assert identity.value is not None
        assert "secret-session-id" not in identity.value
        assert "secret-session-id" not in identity.source

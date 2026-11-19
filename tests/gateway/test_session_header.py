"""Per-conversation session header: shim declaration -> derived value -> transport."""

from __future__ import annotations

from llm_rosetta_gateway.proxy import _apply_request_stages
from llm_rosetta_gateway.routing import SelectionContext
from llm_rosetta_gateway.transport.http.transport import HttpTransport
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth


def _pi(**kwargs) -> ProviderInfo:
    return ProviderInfo(
        name="p",
        api_key="k",
        base_url="https://x.example/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        **kwargs,
    )


def _ir(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def _apply(
    pi: ProviderInfo,
    ir: dict,
    client_hash: str,
    *,
    key_affinity: bool = True,
    headers: dict | None = None,
    body: dict | None = None,
) -> ProviderInfo:
    """Run the stage-2 decisions the way a proxy handler does."""
    ctx = SelectionContext("m", "chat_completions")
    ctx.with_client(client_hash or None)
    ctx.with_key_affinity(key_affinity)
    ctx.with_provider_info(pi)
    ctx.with_ir_request(ir)
    ctx.with_source_request(headers or {}, body)
    return _apply_request_stages(ctx)


class TestSessionHeaders:
    def test_empty_without_header_or_id(self):
        assert _pi().session_headers() == {}
        # Declared but no id derived yet.
        assert _pi(session_header="x-s").session_headers() == {}

    def test_with_session_sets_id(self):
        pi = _pi(session_header="x-s").with_session("abc")
        assert pi.session_headers() == {"x-s": "abc"}

    def test_with_session_noop_without_header(self):
        pi = _pi()
        assert pi.with_session("abc") is pi

    def test_with_session_noop_on_empty_id(self):
        pi = _pi(session_header="x-s")
        assert pi.with_session("") is pi


class TestBuildHeaders:
    def test_session_header_merged_with_auth(self):
        transport = HttpTransport()
        pi = _pi(session_header="x-s").with_session("abc")

        headers = transport._build_headers(pi)

        assert headers["x-s"] == "abc"
        assert headers["Authorization"] == "Bearer k"

    def test_extra_headers_override_derived_session(self):
        transport = HttpTransport()
        pi = _pi(session_header="x-s").with_session("derived")

        headers = transport._build_headers(pi, {"x-s": "client"})

        assert headers["x-s"] == "client"


class TestSessionDerivation:
    def test_stable_within_conversation(self):
        pi = _pi(session_header="x-s")
        ir = _ir("hello")

        first = _apply(pi, ir, "clienthash")
        second = _apply(pi, ir, "clienthash")

        assert first.session_headers() == second.session_headers()
        assert first.session_headers()["x-s"]  # non-empty

    def test_stable_across_turns(self):
        # History is append-only. The id must not change between turn 1 and
        # turn 2 — a naive 2-message prefix would (turn 2 appends a1 + u2).
        pi = _pi(session_header="x-s")
        turn1 = _apply(pi, _ir("hello"), "clienthash")
        turn2 = _apply(
            pi,
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "more"},
                ]
            },
            "clienthash",
        )

        assert turn1.session_headers()["x-s"] == turn2.session_headers()["x-s"]

    def test_differs_across_conversations(self):
        pi = _pi(session_header="x-s")

        a = _apply(pi, _ir("hello"), "clienthash")
        b = _apply(pi, _ir("different"), "clienthash")

        assert a.session_headers()["x-s"] != b.session_headers()["x-s"]

    def test_header_present_without_identity(self):
        # No client key hash and an empty IR -> the fallback keeps the header
        # present so the upstream never sees a missing session.
        pi = _pi(session_header="x-s")

        out = _apply(pi, {}, "", key_affinity=False)

        assert out.session_headers()["x-s"]

    def test_noop_without_session_header(self):
        pi = _pi()

        out = _apply(pi, _ir("x"), "h")

        assert out is pi


class TestTheAgentLane:
    """An ACP lane is told the conversation, and only a declared one.

    The agent keeps a conversation inside a session it owns, so the id decides
    reuse — and a derived id names a group of log rows, not a conversation,
    which is why it must not reach the ledger.
    """

    def _acp(self) -> ProviderInfo:
        return _pi(transport="acp")

    def test_a_declared_header_becomes_the_conversation(self):
        out = _apply(
            self._acp(),
            _ir("x"),
            "h",
            headers={"x-claude-code-session-id": "sess-1"},
        )

        assert out.acp_conversation_id is not None
        assert len(out.acp_conversation_id) == 32

    def test_a_body_carrier_becomes_the_conversation(self):
        out = _apply(
            self._acp(),
            _ir("x"),
            "h",
            body={"metadata": {"user_id": {"session_id": "sess-2"}}},
        )

        assert out.acp_conversation_id is not None

    def test_a_derived_id_never_reaches_the_lane(self):
        """Nothing was declared, so the request gets a session of its own."""
        out = _apply(self._acp(), _ir("x"), "h")

        assert out.acp_conversation_id is None

    def test_an_http_lane_is_untouched(self):
        out = _apply(_pi(), _ir("x"), "h", headers={"x-opencode-session": "s"})

        assert out.acp_conversation_id is None

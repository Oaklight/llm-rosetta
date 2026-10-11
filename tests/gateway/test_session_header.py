"""Per-conversation session header: shim declaration -> derived value -> transport."""

from __future__ import annotations

from llm_rosetta.gateway.proxy import _maybe_apply_affinity
from llm_rosetta.gateway.transport.http.transport import HttpTransport
from llm_rosetta.gateway.transport.provider_info import ProviderInfo, openai_auth


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

        first = _maybe_apply_affinity(pi, ir, "clienthash", True)
        second = _maybe_apply_affinity(pi, ir, "clienthash", True)

        assert first.session_headers() == second.session_headers()
        assert first.session_headers()["x-s"]  # non-empty

    def test_stable_across_turns(self):
        # History is append-only. The id must not change between turn 1 and
        # turn 2 — a naive 2-message prefix would (turn 2 appends a1 + u2).
        pi = _pi(session_header="x-s")
        turn1 = _maybe_apply_affinity(pi, _ir("hello"), "clienthash", True)
        turn2 = _maybe_apply_affinity(
            pi,
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "more"},
                ]
            },
            "clienthash",
            True,
        )

        assert turn1.session_headers()["x-s"] == turn2.session_headers()["x-s"]

    def test_differs_across_conversations(self):
        pi = _pi(session_header="x-s")

        a = _maybe_apply_affinity(pi, _ir("hello"), "clienthash", True)
        b = _maybe_apply_affinity(pi, _ir("different"), "clienthash", True)

        assert a.session_headers()["x-s"] != b.session_headers()["x-s"]

    def test_header_present_without_identity(self):
        # No client key hash and an empty IR -> the fallback keeps the header
        # present so the upstream never sees a missing session.
        pi = _pi(session_header="x-s")

        out = _maybe_apply_affinity(pi, {}, "", False)

        assert out.session_headers()["x-s"]

    def test_noop_without_session_header(self):
        pi = _pi()

        out = _maybe_apply_affinity(pi, _ir("x"), "h", True)

        assert out is pi

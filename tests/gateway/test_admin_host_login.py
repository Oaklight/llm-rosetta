"""The admin routes for a sign-in that the vendor's own program performs.

The provider is built the way the gateway builds it and the exchange is the
real agy one, driven against the fake ``agy``.  What is asserted is what the
panel would receive: the start answer, the polled snapshot, what is scrubbed,
and the lifecycle events (re-seed, lane restart, readiness) that follow a
sign-in or sign-out.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes.credential import (
    begin_provider_login,
    cancel_provider_login,
    delete_provider_credential,
    get_provider_credential,
    get_provider_login_status,
    public_login_status,
)
from llm_rosetta_gateway.providers import build_provider_info

from .fake_agy import install_fake_agy, starts
from .test_agy_signin import MODULE


class _Transport:
    def __init__(self) -> None:
        self.closed: list[str] = []

    async def aclose_provider(self, name: str) -> None:
        self.closed.append(name)


class _Deferred:
    def __init__(self) -> None:
        self.ready: list[str] = []
        self.removed: list[str] = []

    async def provider_credential_ready(self, name: str, pinfo: Any) -> None:
        self.ready.append(name)

    async def provider_credential_removed(self, name: str, pinfo: Any) -> None:
        self.removed.append(name)


class _Request:
    def __init__(self, config: Any, transport: Any, deferred: Any) -> None:
        self.app = SimpleNamespace(
            gateway_config=config,
            transports={"acp": transport},
            deferred_startup=deferred,
        )

    def json(self) -> dict[str, Any]:
        raise ValueError("no body")


@pytest.fixture
def panel(tmp_path, monkeypatch):
    agy = install_fake_agy(tmp_path)
    host = tmp_path / "host"
    host.mkdir()
    monkeypatch.setenv("FAKE_AGY_HOST", str(host))
    monkeypatch.setenv("FAKE_AGY_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("FAKE_AGY_DELAY", "0.1")
    MODULE.reset_state()
    data = tmp_path / "data"
    data.mkdir()

    def entry(name: str, **extra: Any):
        return build_provider_info(
            "acp",
            {"acp_argv": [str(agy), "acp"], **extra},
            shim_name="antigravity--acp",
            data_dir=str(data),
            config_name=name,
            quiet=True,
        )

    pinfo = entry("agy", auth_fields={"agy_profile": "separate"})
    config = SimpleNamespace(providers={"agy": pinfo})
    transport, deferred = _Transport(), _Deferred()
    request = _Request(config, transport, deferred)
    yield SimpleNamespace(
        pinfo=pinfo,
        request=request,
        transport=transport,
        deferred=deferred,
        log=tmp_path / "log.jsonl",
        host=host,
        entry=entry,
        config=config,
    )
    MODULE.reset_state()


def call(handler: Any, request: Any, name: str = "agy") -> tuple[int, dict[str, Any]]:
    response = asyncio.run(handler(request, name))
    return response.status_code, json.loads(response.body)


def poll_until_done(panel: Any, seconds: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _, body = call(get_provider_login_status, panel.request)
        if body["login"]["state"] not in ("running", "verifying"):
            return body
        time.sleep(0.05)
    raise AssertionError("the sign-in never ended")


class TestTheCredentialPayload:
    def test_a_signed_out_entry_offers_a_host_sign_in_and_no_record(self, panel):
        status, body = call(get_provider_credential, panel.request)

        assert status == 200
        assert body["can_host_login"] is True
        assert body["can_install"] is False
        assert body["can_complete_login"] is False
        assert body["state"] == "needs_setup"
        assert body["login"]["state"] == "idle"
        assert body["kind"] == "agent"

    def test_the_shim_is_listed_as_a_host_login_shim(self, monkeypatch):
        from .test_admin_credential_payload import TestTheConfigPayload

        shims = TestTheConfigPayload()._payload(
            monkeypatch, SimpleNamespace(providers={})
        )["registered_shims"]
        by_name = {s["name"]: s for s in shims}

        agy = by_name["antigravity--acp"]
        assert agy["host_login"] is True
        assert agy["mints_credential"] is True
        assert agy["keyless"] is False
        assert [f["key"] for f in agy["fields"]] == ["agy_profile"]
        assert by_name["openai_chatgpt--open_responses"]["host_login"] is False
        assert by_name["openai--open_responses"]["host_login"] is False


class TestStartPollAndAdopt:
    def test_start_returns_the_first_snapshot_not_a_url_to_open(self, panel):
        status, body = call(begin_provider_login, panel.request)

        assert status == 200
        assert body["host"] is True and body["pending"] is True
        assert body["login"]["state"] in ("running", "verifying", "succeeded")
        assert "authorization_url" not in body
        poll_until_done(panel)

    def test_a_finished_sign_in_is_adopted_exactly_once(self, panel):
        assert panel.pinfo.credentials_pending is True
        call(begin_provider_login, panel.request)

        done = poll_until_done(panel)

        assert done["login"]["state"] == "succeeded"
        assert done["connected"] is True
        assert panel.pinfo.credentials_pending is False
        assert panel.transport.closed == ["agy"]
        assert panel.deferred.ready == ["agy"]
        # The next poll sees the same finished run and does nothing more.
        before = len(starts(panel.log))
        _, again = call(get_provider_login_status, panel.request)
        assert "connected" not in again
        assert panel.transport.closed == ["agy"]
        assert len(starts(panel.log)) == before

    def test_the_url_agy_printed_reaches_the_panel_and_the_token_does_not(self, panel):
        call(begin_provider_login, panel.request)
        done = poll_until_done(panel)

        assert done["login"]["url"].startswith("https://accounts.example/")
        assert "ya29" not in json.dumps(done)

    def test_after_the_sign_in_the_credential_endpoint_says_connected(self, panel):
        call(begin_provider_login, panel.request)
        poll_until_done(panel)

        _, body = call(get_provider_credential, panel.request)

        assert body["state"] == "connected"
        assert body["summary"]["profile"] == "separate"

    def test_cancel_stops_it_and_reports_a_cancelled_run(self, panel, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        call(begin_provider_login, panel.request)

        status, body = call(cancel_provider_login, panel.request)

        assert status == 200
        assert body["cancelled"] is True
        assert body["login"]["state"] == "cancelled"
        assert panel.transport.closed == []

    def test_a_start_that_cannot_run_agy_is_a_400_with_a_sentence(self, panel):
        pinfo = panel.entry(
            "agy",
            acp_argv=["/nonexistent/agy", "acp"],
            auth_fields={"agy_profile": "separate"},
        )
        panel.config.providers["agy"] = pinfo

        status, body = call(begin_provider_login, panel.request)

        assert status == 400
        assert "could not start" in body["error"]


class TestSignOut:
    def test_signing_out_a_separate_profile_restarts_the_lane(self, panel):
        call(begin_provider_login, panel.request)
        poll_until_done(panel)
        panel.transport.closed.clear()

        status, body = call(delete_provider_credential, panel.request)

        assert status == 200
        assert body["removed"] is True and body["revoked"] is False
        assert panel.transport.closed == ["agy"]
        assert panel.deferred.removed == ["agy"]

    def test_an_entry_on_the_host_login_cannot_be_signed_out_from_here(self, panel):
        host = panel.entry("agy")
        panel.config.providers["agy"] = host
        (panel.host / "signed-in").write_text("x")

        status, body = call(delete_provider_credential, panel.request)

        assert status == 200
        assert body["removed"] is False
        assert "gateway host" in body["error"]


class TestWhatReachesABrowser:
    def test_unknown_keys_tokens_and_unsafe_urls_are_dropped(self):
        out = public_login_status(
            {
                "state": "running",
                "url": "https://accounts.example/cb?code=4/0Abc",
                "lines": ["ok", "token ya29.A0ARrdaMsecretsecret", 7, None],
                "message": "Bearer abc.def",
                "access_token": "must-not-escape",
                "started_at": 1.5,
            }
        )

        assert out["url"] is None
        assert out["lines"][0] == "ok"
        assert "ya29" not in json.dumps(out)
        assert "Bearer abc" not in json.dumps(out)
        assert "access_token" not in out
        assert out["started_at"] == 1.5

    def test_lines_are_bounded_and_the_state_is_a_known_word(self):
        out = public_login_status(
            {"state": "<script>", "lines": ["x" * 5000] * 500, "url": "http://x"}
        )

        assert out["state"] == "idle"
        assert len(out["lines"]) == 40
        assert all(len(line) <= 400 for line in out["lines"])
        assert out["url"] is None

    def test_nothing_at_all_is_an_idle_snapshot(self):
        assert public_login_status(None)["state"] == "idle"


class TestARefusalForOtherExchanges:
    def test_an_exchange_with_no_host_sign_in_answers_501(self):
        class Plain:
            kind = "oauth"

            def needs_interactive_setup(self, ctx: Any) -> bool:
                return True

        pinfo = SimpleNamespace(
            auth_exchange=Plain(), auth_context=None, transport="http"
        )
        request = _Request(
            SimpleNamespace(providers={"x": pinfo}), _Transport(), _Deferred()
        )

        for handler in (get_provider_login_status, cancel_provider_login):
            status, _ = call(handler, request, "x")
            assert status == 501

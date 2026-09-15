"""Tests for reactive 401 token refresh and proxy retry logic."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from llm_rosetta.gateway.proxy import (
    ProviderMetadataStore,
    handle_non_streaming,
)
from llm_rosetta.gateway.transport._base import UpstreamResponse
from llm_rosetta.gateway.transport.provider_info import (
    ProviderInfo,
    openai_auth,
)
from llm_rosetta.gateway.transport.token_refresh import (
    force_refresh,
    reset_reactive_state,
)
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    *,
    name: str = "test-provider",
    token_command: list[str] | None = None,
) -> ProviderInfo:
    return ProviderInfo(
        name=name,
        api_key="initial-key",
        base_url="https://example.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        token_command=token_command,
    )


def _make_route(
    source: ProviderType = "openai_chat",
    target: ProviderType = "openai_chat",
) -> ResolvedRoute:
    return ResolvedRoute(
        source_provider=source,
        target_provider=target,
        provider_name="test-provider",
        shim_name=None,
        upstream_model="test-model",
    )


def _make_transport(
    *responses: UpstreamResponse,
) -> MagicMock:
    """Create a mock transport returning a sequence of responses."""
    transport = MagicMock()
    transport.send = AsyncMock(side_effect=list(responses))
    return transport


def _reset():
    reset_reactive_state()


# ---------------------------------------------------------------------------
# force_refresh
# ---------------------------------------------------------------------------


class TestForceRefresh:
    def test_no_token_command_returns_false(self):
        _reset()
        pinfo = _make_provider(token_command=None)
        assert asyncio.run(force_refresh(pinfo)) is False
        _reset()

    def test_success_refreshes_keyring(self):
        _reset()
        pinfo = _make_provider(token_command=["echo", "new-token"])
        with patch(
            "llm_rosetta.gateway.transport.token_refresh.run_token_command",
            new_callable=AsyncMock,
            return_value="new-token",
        ) as mock_cmd:
            result = asyncio.run(force_refresh(pinfo))

        assert result is True
        mock_cmd.assert_awaited_once()
        assert pinfo.key_ring.next() == "new-token"
        assert pinfo.token_status is not None
        assert pinfo.token_status["consecutive_failures"] == 0
        _reset()

    def test_failure_returns_false_keeps_old_key(self):
        _reset()
        pinfo = _make_provider(token_command=["echo", "x"])
        with patch(
            "llm_rosetta.gateway.transport.token_refresh.run_token_command",
            new_callable=AsyncMock,
            side_effect=RuntimeError("command failed"),
        ):
            result = asyncio.run(force_refresh(pinfo))

        assert result is False
        assert pinfo.key_ring.next() == "initial-key"
        # token_status should reflect the failure
        assert pinfo.token_status is not None
        assert pinfo.token_status["consecutive_failures"] == 1
        assert "command failed" in pinfo.token_status["last_error"]
        _reset()

    def test_debounce_skips_second_call(self):
        _reset()
        pinfo = _make_provider(token_command=["echo", "x"])

        async def _run():
            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                return_value="fresh-token",
            ) as mock_cmd:
                r1 = await force_refresh(pinfo)
                r2 = await force_refresh(pinfo)
            return r1, r2, mock_cmd.await_count

        r1, r2, count = asyncio.run(_run())
        assert r1 is True
        assert r2 is True
        assert count == 1
        _reset()

    def test_concurrent_calls_run_command_once(self):
        _reset()
        pinfo = _make_provider(token_command=["echo", "x"])
        call_count = 0

        async def slow_command(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "refreshed"

        async def _run():
            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                side_effect=slow_command,
            ):
                results = await asyncio.gather(
                    force_refresh(pinfo),
                    force_refresh(pinfo),
                    force_refresh(pinfo),
                )
            return results

        results = asyncio.run(_run())
        assert all(results)
        assert call_count == 1
        _reset()

    def test_different_providers_get_separate_locks(self):
        _reset()
        p1 = _make_provider(name="provider-a", token_command=["echo", "x"])
        p2 = _make_provider(name="provider-b", token_command=["echo", "y"])

        async def _run():
            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                return_value="token",
            ) as mock_cmd:
                await force_refresh(p1)
                await force_refresh(p2)
            return mock_cmd.await_count

        count = asyncio.run(_run())
        assert count == 2
        _reset()


# ---------------------------------------------------------------------------
# Proxy-level 401 retry integration
# ---------------------------------------------------------------------------

_OK_RESPONSE_BODY: dict[str, Any] = {
    "id": "resp-1",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0}],
}

_SIMPLE_REQUEST: dict[str, Any] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}],
}


class TestProxyRetryOn401:
    """End-to-end tests through handle_non_streaming for 401 retry wiring."""

    def test_401_then_200_retries_and_succeeds(self):
        """Normal 401 → reactive refresh → retry succeeds."""
        _reset()
        pinfo = _make_provider(token_command=["echo", "refreshed"])
        transport = _make_transport(
            UpstreamResponse(
                status_code=401,
                body=None,
                raw_content=b'{"error":"unauthorized"}',
            ),
            UpstreamResponse(
                status_code=200,
                body=_OK_RESPONSE_BODY,
                raw_content=json.dumps(_OK_RESPONSE_BODY).encode(),
            ),
        )

        async def _run():
            with patch(
                "llm_rosetta.gateway.proxy.force_refresh",
                new_callable=AsyncMock,
                return_value=True,
            ):
                resp, profile = await handle_non_streaming(
                    _make_route(),
                    pinfo,
                    _SIMPLE_REQUEST,
                    transport=transport,
                    metadata_store=ProviderMetadataStore(),
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        assert transport.send.await_count == 2
        _reset()

    def test_401_session_policy_returns_immediately(self):
        """ALCF session-policy 401 → no retry, rewritten error."""
        _reset()
        pinfo = _make_provider(token_command=["echo", "x"])
        policy_body = "rejected due to internal policies of the resource"
        transport = _make_transport(
            UpstreamResponse(
                status_code=401,
                body=None,
                raw_content=policy_body.encode(),
            ),
        )

        async def _run():
            resp, _ = await handle_non_streaming(
                _make_route(),
                pinfo,
                _SIMPLE_REQUEST,
                transport=transport,
                metadata_store=ProviderMetadataStore(),
            )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 401
        body = json.loads(resp.body)
        assert "alcf-token.py --login" in body["error"]["message"]
        # Should NOT have retried
        assert transport.send.await_count == 1
        _reset()

    def test_401_no_token_command_passes_through(self):
        """401 from a provider without token_command → no retry, raw passthrough."""
        _reset()
        pinfo = _make_provider(token_command=None)
        transport = _make_transport(
            UpstreamResponse(
                status_code=401,
                body=None,
                raw_content=b'{"error":"unauthorized"}',
            ),
        )

        async def _run():
            resp, _ = await handle_non_streaming(
                _make_route(),
                pinfo,
                _SIMPLE_REQUEST,
                transport=transport,
                metadata_store=ProviderMetadataStore(),
            )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 401
        assert transport.send.await_count == 1
        _reset()

    def test_401_refresh_fails_passes_through(self):
        """401 → reactive refresh fails → original error passed through."""
        _reset()
        pinfo = _make_provider(token_command=["echo", "x"])
        transport = _make_transport(
            UpstreamResponse(
                status_code=401,
                body=None,
                raw_content=b'{"error":"unauthorized"}',
            ),
        )

        async def _run():
            with patch(
                "llm_rosetta.gateway.proxy.force_refresh",
                new_callable=AsyncMock,
                return_value=False,
            ):
                resp, _ = await handle_non_streaming(
                    _make_route(),
                    pinfo,
                    _SIMPLE_REQUEST,
                    transport=transport,
                    metadata_store=ProviderMetadataStore(),
                )
            return resp

        resp = asyncio.run(_run())
        assert resp.status_code == 401
        assert transport.send.await_count == 1
        _reset()

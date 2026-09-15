"""Tests for reactive 401 token refresh and proxy retry logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from llm_rosetta.gateway.transport.provider_info import (
    ProviderInfo,
    openai_auth,
)
from llm_rosetta.gateway.transport.token_refresh import (
    force_refresh,
    reset_reactive_state,
)


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

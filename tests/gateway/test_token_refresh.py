"""Tests for the token_command dynamic key refresh feature."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_rosetta.gateway.transport.provider_info import (
    KeyRing,
    ProviderInfo,
    openai_auth,
)
from llm_rosetta.gateway.transport.token_refresh import (
    _refresh_loop,
    run_token_command,
    run_token_command_sync,
    start_token_refreshers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_info(
    *,
    api_key: str = "initial-key",
    base_url: str = "https://example.com/v1",
    token_command: list[str] | None = None,
    token_refresh_interval: int = 3600,
) -> ProviderInfo:
    return ProviderInfo(
        name="test-provider",
        api_key=api_key,
        base_url=base_url,
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        token_command=token_command,
        token_refresh_interval=token_refresh_interval,
    )


# ---------------------------------------------------------------------------
# KeyRing.refresh
# ---------------------------------------------------------------------------


class TestKeyRingRefresh:
    def test_same_count_returns_none(self):
        kr = KeyRing("a,b,c")
        assert kr.refresh("x,y,z") is None
        assert len(kr) == 3

    def test_different_count_returns_old(self):
        kr = KeyRing("a,b,c")
        assert kr.refresh("x,y") == 3
        assert len(kr) == 2

    def test_empty_is_noop(self):
        kr = KeyRing("a,b")
        assert kr.refresh("") is None
        assert len(kr) == 2
        assert kr.next() == "a"

    def test_keys_updated(self):
        kr = KeyRing("old1,old2")
        kr.refresh("new1,new2")
        keys = [kr.next(), kr.next()]
        assert set(keys) == {"new1", "new2"}

    def test_idx_clamped(self):
        kr = KeyRing("a,b,c,d,e")
        for _ in range(4):
            kr.next()
        assert kr._idx == 4
        kr.refresh("x,y")
        assert kr._idx == 0  # 4 % 2 = 0
        assert kr.next() == "x"

    def test_whitespace_stripped(self):
        kr = KeyRing("a")
        kr.refresh("  key1 , key2  ")
        assert kr.next() == "key1"
        assert kr.next() == "key2"


# ---------------------------------------------------------------------------
# run_token_command_sync
# ---------------------------------------------------------------------------


class TestRunTokenCommandSync:
    def test_success(self):
        token = run_token_command_sync(["echo", "my-token"])
        assert token == "my-token"

    def test_multi_key(self):
        token = run_token_command_sync(["echo", "key1,key2"])
        assert token == "key1,key2"

    def test_strips_whitespace(self):
        token = run_token_command_sync(["echo", "  spaced-token  "])
        assert token == "spaced-token"

    def test_nonzero_exit_raises(self):
        with pytest.raises(RuntimeError, match="exited with"):
            run_token_command_sync(["false"])

    def test_empty_output_raises(self):
        with pytest.raises(RuntimeError, match="empty output"):
            run_token_command_sync(["echo"])

    def test_timeout_raises(self):
        with pytest.raises(subprocess.TimeoutExpired):
            run_token_command_sync(["sleep", "10"], timeout=0.1)


# ---------------------------------------------------------------------------
# run_token_command (async)
# ---------------------------------------------------------------------------


class TestRunTokenCommandAsync:
    def test_success(self):
        token = asyncio.run(run_token_command(["echo", "async-key"]))
        assert token == "async-key"

    def test_nonzero_exit_raises(self):
        with pytest.raises(RuntimeError, match="exited with"):
            asyncio.run(run_token_command(["false"]))

    def test_empty_output_raises(self):
        with pytest.raises(RuntimeError, match="empty output"):
            asyncio.run(run_token_command(["echo"]))

    def test_timeout_raises(self):
        with pytest.raises(RuntimeError, match="timed out"):
            asyncio.run(run_token_command(["sleep", "10"], timeout=0.2))


# ---------------------------------------------------------------------------
# start_token_refreshers
# ---------------------------------------------------------------------------


class TestStartTokenRefreshers:
    def test_creates_tasks_for_token_command_providers(self):
        async def _run():
            p1 = _make_provider_info(token_command=["echo", "key1"])
            p2 = _make_provider_info()
            p3 = _make_provider_info(token_command=["echo", "key3"])
            tasks = await start_token_refreshers({"p1": p1, "p2": p2, "p3": p3})
            assert len(tasks) == 2
            for t in tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        asyncio.run(_run())

    def test_no_tasks_without_token_command(self):
        async def _run():
            p = _make_provider_info()
            tasks = await start_token_refreshers({"p": p})
            assert len(tasks) == 0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# _refresh_loop
# ---------------------------------------------------------------------------


class TestRefreshLoop:
    def test_refreshes_key(self):
        async def _run():
            pinfo = _make_provider_info(
                token_command=["echo", "refreshed-key"],
                token_refresh_interval=1,
            )
            assert pinfo.key_ring.next() == "initial-key"

            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                return_value="refreshed-key",
            ):
                # Use a very short interval via direct attribute override
                pinfo.token_refresh_interval = 0
                task = asyncio.create_task(_refresh_loop(pinfo))
                # Give the loop one iteration
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            assert pinfo.key_ring.next() == "refreshed-key"

        asyncio.run(_run())

    def test_keeps_old_key_on_failure(self):
        async def _run():
            pinfo = _make_provider_info(
                token_command=["false"],
                token_refresh_interval=1,
            )

            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                side_effect=RuntimeError("command failed"),
            ):
                pinfo.token_refresh_interval = 0
                task = asyncio.create_task(_refresh_loop(pinfo))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            assert pinfo.key_ring.next() == "initial-key"
            assert pinfo.token_status is not None
            assert pinfo.token_status["consecutive_failures"] >= 1

        asyncio.run(_run())

    def test_token_status_updated_on_success(self):
        async def _run():
            pinfo = _make_provider_info(
                token_command=["echo", "new"],
                token_refresh_interval=1,
            )

            with patch(
                "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                new_callable=AsyncMock,
                return_value="new",
            ):
                pinfo.token_refresh_interval = 0
                task = asyncio.create_task(_refresh_loop(pinfo))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            assert pinfo.token_status is not None
            assert pinfo.token_status["consecutive_failures"] == 0
            assert pinfo.token_status["last_error"] is None
            assert pinfo.token_status["last_refresh"] is not None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# build_provider_info with token_command
# ---------------------------------------------------------------------------


class TestBuildProviderInfoTokenCommand:
    def test_deferred_token_returns_sentinel(self):
        from llm_rosetta.gateway.providers import build_provider_info
        from llm_rosetta.gateway.transport.provider_info import TOKEN_PENDING_SENTINEL

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["echo", "seeded-key"],
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.ready is False
        assert pinfo.key_ring.next() == TOKEN_PENDING_SENTINEL
        assert pinfo.token_command == ["echo", "seeded-key"]
        assert pinfo.token_refresh_interval == 3600

    def test_custom_refresh_interval(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["echo", "key"],
            "token_refresh_interval": 7200,
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.token_refresh_interval == 7200

    def test_mutual_exclusivity_with_api_key(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "api_key": "static-key",
            "token_command": ["echo", "key"],
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_provider_info("openai_chat", cfg)

    def test_string_token_command_rejected(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": "echo key",
        }
        with pytest.raises(ValueError, match="must be an array"):
            build_provider_info("openai_chat", cfg)

    def test_empty_token_command_rejected(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": [],
        }
        with pytest.raises(ValueError, match="non-empty array"):
            build_provider_info("openai_chat", cfg)

    def test_low_refresh_interval_rejected(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["echo", "key"],
            "token_refresh_interval": 10,
        }
        with pytest.raises(ValueError, match=">= 60"):
            build_provider_info("openai_chat", cfg)

    def test_missing_binary_warns_but_builds(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["/nonexistent/binary-xyz"],
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.ready is False
        assert pinfo.token_command == ["/nonexistent/binary-xyz"]

    def test_without_token_command_unchanged(self):
        from llm_rosetta.gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "api_key": "static-key",
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.auth_headers() == {"Authorization": "Bearer static-key"}
        assert pinfo.token_command is None

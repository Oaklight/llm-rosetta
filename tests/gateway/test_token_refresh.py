"""Tests for the token_command dynamic key refresh feature."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_rosetta_gateway.transport.provider_info import (
    KeyRing,
    ProviderInfo,
    openai_auth,
)
from llm_rosetta_gateway.transport.token_refresh import (
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
                "llm_rosetta_gateway.transport.token_refresh.run_token_command",
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
                "llm_rosetta_gateway.transport.token_refresh.run_token_command",
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
                "llm_rosetta_gateway.transport.token_refresh.run_token_command",
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
        from llm_rosetta_gateway.providers import build_provider_info
        from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL

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
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["echo", "key"],
            "token_refresh_interval": 7200,
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.token_refresh_interval == 7200

    def test_mutual_exclusivity_with_api_key(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "api_key": "static-key",
            "token_command": ["echo", "key"],
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_provider_info("openai_chat", cfg)

    def test_string_token_command_rejected(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": "echo key",
        }
        with pytest.raises(ValueError, match="must be an array"):
            build_provider_info("openai_chat", cfg)

    def test_empty_token_command_rejected(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": [],
        }
        with pytest.raises(ValueError, match="non-empty array"):
            build_provider_info("openai_chat", cfg)

    def test_low_refresh_interval_rejected(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["echo", "key"],
            "token_refresh_interval": 10,
        }
        with pytest.raises(ValueError, match=">= 60"):
            build_provider_info("openai_chat", cfg)

    def test_missing_binary_warns_but_builds(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "token_command": ["/nonexistent/binary-xyz"],
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.ready is False
        assert pinfo.token_command == ["/nonexistent/binary-xyz"]

    def test_without_token_command_unchanged(self):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg: dict[str, Any] = {
            "base_url": "https://example.com/v1",
            "api_key": "static-key",
        }
        pinfo = build_provider_info("openai_chat", cfg)
        assert pinfo.auth_headers() == {"Authorization": "Bearer static-key"}
        assert pinfo.token_command is None


class TestTheCredentialChannelEnvironment:
    """The operator's key reaches the command, and only the command."""

    def _pinfo(self, *, env: str | None, key: str):
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo
        from llm_rosetta_gateway.providers import openai_auth

        return ProviderInfo(
            name="vendor",
            api_key="__token_pending__",
            base_url="https://vendor.example/v1",
            auth_header_fn=openai_auth,
            url_template="{base_url}/chat/completions",
            token_command=["python3", "-c", "print('minted')"],
            token_command_env=env,
            token_input=key,
        )

    def test_the_variable_carries_the_key(self):
        from llm_rosetta_gateway.transport.token_refresh import command_env

        env = command_env(self._pinfo(env="VENDOR_ACCOUNT_KEY", key="sk-account"))

        assert env is not None
        assert env["VENDOR_ACCOUNT_KEY"] == "sk-account"
        # The rest of the environment is inherited, not replaced.
        assert "PATH" in env

    def test_no_channel_means_no_override(self):
        from llm_rosetta_gateway.transport.token_refresh import command_env

        assert command_env(self._pinfo(env=None, key="sk-account")) is None
        assert command_env(self._pinfo(env="VENDOR_ACCOUNT_KEY", key="")) is None

    def test_the_key_is_in_the_environment_not_in_argv(self):
        """argv is world-readable in the process table; the env is not."""
        import asyncio

        from llm_rosetta_gateway.transport.token_refresh import (
            command_env,
            run_token_command,
        )

        pinfo = self._pinfo(env="VENDOR_ACCOUNT_KEY", key="sk-account")
        pinfo.token_command = [
            "python3",
            "-c",
            "import os; print(os.environ['VENDOR_ACCOUNT_KEY'])",
        ]

        token = asyncio.run(
            run_token_command(pinfo.token_command, env=command_env(pinfo))
        )

        assert token == "sk-account"
        assert "sk-account" not in pinfo.token_command


class _FakeExchange:
    """A shim-shipped exchange: one call, and it may demand a sign-in."""

    kind = "oauth"
    consumes_key = False

    def __init__(self, *, token: str = "minted", interactive: bool = False):
        self.token = token
        self.interactive = interactive
        self.calls = 0
        self.contexts: list[Any] = []

    def needs_interactive_setup(self, ctx: Any) -> bool:
        return self.interactive

    async def credential(self, ctx: Any) -> str:
        from llm_rosetta.shims.auth_exchange import InteractiveSetupRequired

        self.calls += 1
        self.contexts.append(ctx)
        if self.interactive:
            raise InteractiveSetupRequired("sign in to the vendor first")
        return self.token

    def account_id(self, ctx: Any) -> str | None:
        return None


def _exchange_pinfo(exchange: _FakeExchange, **kw: Any) -> ProviderInfo:
    from llm_rosetta.shims.auth_exchange import AuthContext
    from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL

    return ProviderInfo(
        name="vendor",
        api_key=TOKEN_PENDING_SENTINEL,
        base_url="https://vendor.example/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        auth_exchange=exchange,
        auth_context=AuthContext(provider="vendor", state_dir=kw.get("state_dir")),
        keyless=False,
    )


class TestMintCredential:
    """The one seam startup, the refresher and the 401 retry all use."""

    def test_a_command_is_run(self):
        from llm_rosetta_gateway.transport.token_refresh import mint_credential

        pinfo = _make_provider_info(token_command=["echo", "cmd-token"])

        assert asyncio.run(mint_credential(pinfo)) == "cmd-token"

    def test_an_exchange_is_awaited_with_the_state_dir(self):
        from llm_rosetta_gateway.transport.token_refresh import mint_credential

        exchange = _FakeExchange(token="oauth-token")
        pinfo = _exchange_pinfo(exchange, state_dir="/tmp/vendor-state")

        assert asyncio.run(mint_credential(pinfo)) == "oauth-token"
        assert exchange.contexts[0].state_dir == "/tmp/vendor-state"
        assert exchange.contexts[0].provider == "vendor"

    def test_a_provider_with_no_source_is_an_error_not_an_empty_key(self):
        from llm_rosetta_gateway.transport.token_refresh import mint_credential

        with pytest.raises(RuntimeError, match="mints no credential"):
            asyncio.run(mint_credential(_make_provider_info()))


class TestForceRefreshAcrossForms:
    """The reactive 401 retry does not care which form declared the source."""

    def setup_method(self):
        from llm_rosetta_gateway.transport.token_refresh import reset_reactive_state

        reset_reactive_state()

    def test_an_exchange_is_refreshed(self):
        from llm_rosetta_gateway.transport.token_refresh import force_refresh

        exchange = _FakeExchange(token="fresh")
        pinfo = _exchange_pinfo(exchange)

        assert asyncio.run(force_refresh(pinfo)) is True
        assert pinfo.key_ring.next() == "fresh"
        assert (pinfo.token_status or {})["consecutive_failures"] == 0

    def test_a_concurrent_burst_mints_once(self):
        """One rotating refresh token must not be spent by parallel 401s."""
        from llm_rosetta_gateway.transport.token_refresh import force_refresh

        exchange = _FakeExchange(token="fresh")
        pinfo = _exchange_pinfo(exchange)

        async def _burst():
            return await asyncio.gather(*(force_refresh(pinfo) for _ in range(5)))

        assert all(asyncio.run(_burst()))
        assert exchange.calls == 1

    def test_interactive_setup_is_reported_not_retried(self):
        from llm_rosetta_gateway.transport.token_refresh import force_refresh

        exchange = _FakeExchange(interactive=True)
        pinfo = _exchange_pinfo(exchange)

        assert asyncio.run(force_refresh(pinfo)) is False
        status = pinfo.token_status or {}
        assert status["interactive_setup_required"] is True
        assert "sign in" in status["last_error"]

    def test_a_provider_without_a_source_is_not_refreshed(self):
        from llm_rosetta_gateway.transport.token_refresh import force_refresh

        assert asyncio.run(force_refresh(_make_provider_info())) is False


class TestRefresherSchedulingAcrossForms:
    def test_an_exchange_provider_gets_a_loop(self):
        async def _run():
            pinfo = _exchange_pinfo(_FakeExchange())
            tasks = await start_token_refreshers({"p": pinfo})
            assert len(tasks) == 1
            for t in tasks:
                t.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await t

        asyncio.run(_run())

    def test_the_loop_mints_through_the_exchange(self):
        async def _run():
            exchange = _FakeExchange(token="rotated")
            pinfo = _exchange_pinfo(exchange)
            pinfo.token_refresh_interval = 0

            task = asyncio.create_task(_refresh_loop(pinfo))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert exchange.calls >= 1
            assert pinfo.key_ring.next() == "rotated"

        asyncio.run(_run())

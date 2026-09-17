"""Tests for the deferred startup manager.

Covers provider readiness states, routing skip logic, counter rebuild
merge, health endpoint integration, and token_command binary validation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_rosetta.gateway.deferred_startup import (
    DeferredStartup,
    ProviderInitState,
    ProviderNotReady,
)
from llm_rosetta.gateway.transport.provider_info import (
    TOKEN_PENDING_SENTINEL,
    ProviderInfo,
    openai_auth,
)
from llm_rosetta.observability.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pinfo(
    name: str = "test",
    *,
    api_key: str = "sk-real",
    token_command: list[str] | None = None,
    token_refresh_interval: int = 3600,
) -> ProviderInfo:
    return ProviderInfo(
        name=name,
        api_key=api_key,
        base_url="https://example.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        token_command=token_command,
        token_refresh_interval=token_refresh_interval,
    )


def _make_pending_pinfo(name: str = "test") -> ProviderInfo:
    """Provider with pending token (as if token_command deferred)."""
    return _make_pinfo(
        name, api_key=TOKEN_PENDING_SENTINEL, token_command=["echo", "x"]
    )


class _FakeApp:
    """Minimal app stub for DeferredStartup."""

    def __init__(self) -> None:
        self.metrics: MetricsCollector | None = None
        self.persistence: Any = None
        self.keystore: Any = None
        self.gateway_config: Any = None
        self._counter_rebuild_needed: bool = False
        self._token_refresh_tasks: list = []


class _FakeConfig:
    """Minimal GatewayConfig stub."""

    def __init__(self, providers: dict[str, ProviderInfo]) -> None:
        self.providers = providers
        self.models: dict = {}
        self.model_upstream_names: dict = {}


# ---------------------------------------------------------------------------
# ProviderInfo.ready
# ---------------------------------------------------------------------------


class TestProviderInfoReady:
    def test_ready_with_real_key(self):
        pinfo = _make_pinfo(api_key="sk-real")
        assert pinfo.ready is True

    def test_not_ready_with_sentinel(self):
        pinfo = _make_pinfo(api_key=TOKEN_PENDING_SENTINEL)
        assert pinfo.ready is False

    def test_ready_after_refresh(self):
        pinfo = _make_pinfo(api_key=TOKEN_PENDING_SENTINEL)
        assert pinfo.ready is False
        pinfo.key_ring.refresh("sk-actual-key")
        assert pinfo.ready is True

    def test_ready_with_multiple_keys(self):
        pinfo = _make_pinfo(api_key="key1,key2")
        assert pinfo.ready is True

    def test_not_ready_sentinel_only_key(self):
        pinfo = _make_pinfo(api_key=TOKEN_PENDING_SENTINEL)
        assert len(pinfo.key_ring) == 1
        assert pinfo.ready is False


# ---------------------------------------------------------------------------
# ProviderNotReady exception
# ---------------------------------------------------------------------------


class TestProviderNotReady:
    def test_pending_message(self):
        exc = ProviderNotReady("gpt-4o")
        assert "gpt-4o" in str(exc)
        assert "initializing" in str(exc)
        assert exc.failed is False

    def test_failed_message(self):
        exc = ProviderNotReady("gpt-4o", failed=True)
        assert "failed" in str(exc)
        assert exc.failed is True


# ---------------------------------------------------------------------------
# DeferredStartup — provider states
# ---------------------------------------------------------------------------


class TestDeferredStartupStates:
    def test_no_token_command_providers_all_ready(self):
        config = _FakeConfig({"openai": _make_pinfo("openai")})
        app = _FakeApp()
        ds = DeferredStartup(config, app)
        assert ds.is_fully_ready() is True
        assert ds.provider_state("openai") == ProviderInitState.READY

    def test_pending_provider_tracked(self):
        config = _FakeConfig(
            {
                "alcf": _make_pending_pinfo("alcf"),
            }
        )
        app = _FakeApp()
        ds = DeferredStartup(config, app)
        assert ds.provider_state("alcf") == ProviderInitState.PENDING
        assert ds.is_fully_ready() is False

    def test_status_structure(self):
        config = _FakeConfig(
            {
                "alcf": _make_pending_pinfo("alcf"),
                "openai": _make_pinfo("openai"),
            }
        )
        app = _FakeApp()
        ds = DeferredStartup(config, app)
        status = ds.status()
        assert status["ready"] is False
        assert "alcf" in status["providers"]
        assert status["providers"]["alcf"] == "pending"
        assert "openai" not in status["providers"]


# ---------------------------------------------------------------------------
# DeferredStartup — token seeding
# ---------------------------------------------------------------------------


class TestTokenSeeding:
    def test_seed_success(self):
        pinfo = _make_pending_pinfo("alcf")
        config = _FakeConfig({"alcf": pinfo})
        app = _FakeApp()
        ds = DeferredStartup(config, app)

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    new_callable=AsyncMock,
                    return_value="real-token",
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ):
                await ds._seed_tokens()

        asyncio.run(_run())

        assert ds.provider_state("alcf") == ProviderInitState.READY
        assert pinfo.ready is True
        assert pinfo.key_ring.next() == "real-token"

    def test_seed_failure(self):
        pinfo = _make_pending_pinfo("alcf")
        config = _FakeConfig({"alcf": pinfo})
        app = _FakeApp()
        ds = DeferredStartup(config, app)

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("command failed"),
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ):
                await ds._seed_tokens()

        asyncio.run(_run())

        assert ds.provider_state("alcf") == ProviderInitState.FAILED
        assert pinfo.ready is False

    def test_seed_parallel_mixed(self):
        """One provider succeeds, one fails — both tracked correctly."""
        p_ok = _make_pinfo(
            "ok-provider",
            api_key=TOKEN_PENDING_SENTINEL,
            token_command=["echo", "good"],
        )
        p_fail = _make_pinfo(
            "fail-provider",
            api_key=TOKEN_PENDING_SENTINEL,
            token_command=["echo", "bad"],
        )
        config = _FakeConfig({"ok-provider": p_ok, "fail-provider": p_fail})
        app = _FakeApp()
        ds = DeferredStartup(config, app)

        async def _mock_run_token(argv):
            if argv == ["echo", "good"]:
                return "good-token"
            raise RuntimeError("bad command")

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    side_effect=_mock_run_token,
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ):
                await ds._seed_tokens()

        asyncio.run(_run())

        assert ds.provider_state("ok-provider") == ProviderInitState.READY
        assert ds.provider_state("fail-provider") == ProviderInitState.FAILED
        assert "token_seed" not in ds._pending_tasks


# ---------------------------------------------------------------------------
# MetricsCollector.merge_rebuild
# ---------------------------------------------------------------------------


class TestMergeRebuild:
    def test_basic_merge_no_delta(self):
        """Rebuild with no live traffic during the window."""
        m = MetricsCollector()
        pre = m.export_counters()

        baseline = {
            "total_requests": 100,
            "total_errors": 5,
            "total_streams": 20,
            "by_model": {"gpt-4o": 60, "claude": 40},
            "by_source_provider": {"openai_chat": 100},
            "by_target_provider": {"openai": 60, "anthropic": 40},
            "by_status_code": {"200": 95, "500": 5},
            "total_input_tokens": 1000,
            "total_output_tokens": 2000,
            "by_model_tokens": {
                "gpt-4o": {"input_tokens": 600, "output_tokens": 1200},
            },
            "by_provider_tokens": {
                "openai": {"input_tokens": 600, "output_tokens": 1200},
            },
        }

        m.merge_rebuild(baseline, pre)

        assert m.total_requests == 100
        assert m.total_errors == 5
        assert m.by_model["gpt-4o"] == 60
        assert m.by_status_code[200] == 95

    def test_merge_with_delta(self):
        """Live traffic arrives during rebuild — delta is preserved."""
        m = MetricsCollector()
        m.total_requests = 10
        m.total_errors = 1
        m.by_model = {"gpt-4o": 10}

        pre = m.export_counters()  # snapshot before rebuild

        # Simulate 5 requests arriving during rebuild
        m.total_requests = 15
        m.total_errors = 2
        m.by_model = {"gpt-4o": 13, "claude": 2}

        baseline = {
            "total_requests": 100,
            "total_errors": 5,
            "total_streams": 0,
            "by_model": {"gpt-4o": 60, "claude": 40},
            "by_source_provider": {},
            "by_target_provider": {},
            "by_status_code": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model_tokens": {},
            "by_provider_tokens": {},
        }

        m.merge_rebuild(baseline, pre)

        # baseline + delta: 100 + (15 - 10) = 105
        assert m.total_requests == 105
        # 5 + (2 - 1) = 6
        assert m.total_errors == 6
        # gpt-4o: 60 + (13 - 10) = 63
        assert m.by_model["gpt-4o"] == 63
        # claude: 40 + (2 - 0) = 42
        assert m.by_model["claude"] == 42

    def test_merge_preserves_active_streams(self):
        """active_streams is live state, not rebuilt."""
        m = MetricsCollector()
        m.active_streams = 3
        pre = m.export_counters()

        baseline = {"total_requests": 50, "total_errors": 0, "total_streams": 10}
        m.merge_rebuild(baseline, pre)

        assert m.active_streams == 3  # untouched


# ---------------------------------------------------------------------------
# Token command binary validation
# ---------------------------------------------------------------------------


class TestTokenCommandValidation:
    def test_missing_binary_raises(self):
        from llm_rosetta.gateway.providers import _resolve_token_command

        with pytest.raises(ValueError, match="not found on PATH"):
            _resolve_token_command(
                "test-provider",
                {
                    "token_command": ["/nonexistent/binary-xyz", "--arg"],
                    "base_url": "https://example.com",
                },
            )

    def test_valid_binary_returns_sentinel(self):
        from llm_rosetta.gateway.providers import _resolve_token_command

        api_key, cmd, interval = _resolve_token_command(
            "test-provider",
            {
                "token_command": ["echo", "test"],
                "base_url": "https://example.com",
            },
        )
        assert api_key == TOKEN_PENDING_SENTINEL
        assert cmd == ["echo", "test"]

    def test_static_api_key_unchanged(self):
        from llm_rosetta.gateway.providers import _resolve_token_command

        api_key, cmd, interval = _resolve_token_command(
            "test-provider",
            {
                "api_key": "sk-static",
                "base_url": "https://example.com",
            },
        )
        assert api_key == "sk-static"
        assert cmd is None


# ---------------------------------------------------------------------------
# DeferredStartup — counter rebuild
# ---------------------------------------------------------------------------


class TestDeferredCounterRebuild:
    def test_rebuild_merges_correctly(self):
        """Background rebuild + live delta merged into live metrics."""
        metrics = MetricsCollector()
        metrics.total_requests = 10
        metrics.by_model = {"gpt-4o": 10}

        persistence = MagicMock()
        persistence.iter_log_rows_for_rebuild.return_value = [
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "openai",
                "target_provider_name": "openai",
                "is_stream": False,
                "status_code": 200,
            }
        ] * 50
        persistence.save_metrics = MagicMock()

        config = _FakeConfig({})
        app = _FakeApp()
        app.metrics = metrics
        app.persistence = persistence
        app._counter_rebuild_needed = True

        ds = DeferredStartup(config, app)
        asyncio.run(ds._rebuild_counters())

        assert metrics.total_requests == 50
        assert "counter_rebuild" not in ds._pending_tasks
        persistence.save_metrics.assert_called_once()


# ---------------------------------------------------------------------------
# DeferredStartup — backfills
# ---------------------------------------------------------------------------


class TestDeferredBackfills:
    def test_backfills_run_sequentially(self):
        """All three backfills are called in sequence."""
        from llm_rosetta.gateway.routing_strategy import ModelRoute, ProviderEntry

        persistence = MagicMock()
        persistence.backfill_provider_names.return_value = 3
        persistence.backfill_error_dump_log_ids.return_value = 1
        persistence.db_path = "/tmp/test.db"

        keystore = MagicMock()
        keystore.backfill_last_used.return_value = 2

        config = _FakeConfig({})
        config.models = {
            "gpt-4o": ModelRoute(providers=[ProviderEntry(name="openai")]),
        }
        config.model_upstream_names = {}

        app = _FakeApp()
        app.persistence = persistence
        app.keystore = keystore

        ds = DeferredStartup(config, app)
        ds._pending_tasks.add("backfills")
        asyncio.run(ds._run_backfills())

        persistence.backfill_provider_names.assert_called_once()
        keystore.backfill_last_used.assert_called_once_with("/tmp/test.db")
        persistence.backfill_error_dump_log_ids.assert_called_once()
        assert "backfills" not in ds._pending_tasks


# ---------------------------------------------------------------------------
# DeferredStartup — lifecycle
# ---------------------------------------------------------------------------


class TestDeferredStartupLifecycle:
    def test_start_and_shutdown(self):
        """start() creates tasks, shutdown() cancels them."""
        config = _FakeConfig(
            {
                "alcf": _make_pending_pinfo("alcf"),
            }
        )
        app = _FakeApp()
        app.persistence = MagicMock()
        app._counter_rebuild_needed = True

        ds = DeferredStartup(config, app)

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    new_callable=AsyncMock,
                    return_value="tok",
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ):
                await ds.start()
                assert len(ds._tasks) >= 2

                await asyncio.sleep(0.1)

            await ds.shutdown()
            assert all(t.done() or t.cancelled() for t in ds._tasks)

        asyncio.run(_run())

"""Tests for the lifecycle and config-reload Ops subclasses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.ops import (
    OpsAdminSetup,
    OpsConfigReload,
    OpsContext,
    OpsGatewayShutdown,
    OpsGatewayStartup,
    OpsStartupValidation,
)


class _Log:
    def __init__(self, *, fail: bool = False) -> None:
        self.rows: list[tuple[Any, bool]] = []
        self._fail = fail

    async def add(self, entry: Any, *, _skip_prune: bool = False) -> None:
        if self._fail:
            raise OSError("disk full")
        self.rows.append((entry, _skip_prune))


def _ctx(log: Any = None) -> OpsContext:
    return OpsContext(ops_log=log if log is not None else _Log())


@pytest.mark.asyncio
async def test_admin_setup() -> None:
    ctx = _ctx()
    await OpsAdminSetup(ctx).execute()
    [(entry, skip)] = ctx.ops_log.rows
    assert (entry.event_type, entry.severity, entry.source) == (
        "admin_setup",
        "info",
        "admin",
    )
    assert entry.message == "Admin panel initialized"
    assert entry.details is None or entry.details == {}
    assert skip is False


@pytest.mark.asyncio
async def test_gateway_startup() -> None:
    ctx = _ctx()
    config = SimpleNamespace(providers={"a": 1}, models={"m": 1, "n": 2})
    await OpsGatewayStartup(ctx, "0.0.0.0", 9000, config).execute()
    [(entry, _)] = ctx.ops_log.rows
    assert (entry.event_type, entry.source) == ("startup", "gateway")
    assert entry.message == "Gateway starting on 0.0.0.0:9000"
    assert entry.details == {
        "host": "0.0.0.0",
        "port": 9000,
        "provider_count": 1,
        "model_count": 2,
    }


@pytest.mark.asyncio
async def test_gateway_startup_without_a_config() -> None:
    ctx = _ctx()
    await OpsGatewayStartup(ctx, "h", 1, None).execute()
    assert ctx.ops_log.rows[0][0].details["provider_count"] == 0


@pytest.mark.asyncio
async def test_gateway_shutdown_skips_pruning() -> None:
    ctx = _ctx()
    await OpsGatewayShutdown(ctx).execute()
    [(entry, skip)] = ctx.ops_log.rows
    assert (entry.event_type, entry.source) == ("shutdown", "gateway")
    assert entry.message == "Gateway shutting down"
    assert skip is True


@pytest.mark.asyncio
async def test_a_row_that_does_not_skip_pruning_passes_no_keyword() -> None:
    class Strict:
        def __init__(self) -> None:
            self.rows: list[Any] = []

        async def add(self, entry: Any) -> None:
            self.rows.append(entry)

    log = Strict()
    await OpsAdminSetup(_ctx(log)).execute()
    assert len(log.rows) == 1


@pytest.mark.asyncio
async def test_config_reload() -> None:
    ctx = _ctx()
    config = SimpleNamespace(providers={"a": 1, "b": 2, "c": 3}, models={"m": 1})
    await OpsConfigReload(ctx, config).execute()
    [(entry, _)] = ctx.ops_log.rows
    assert (entry.event_type, entry.severity, entry.source) == (
        "config_reload",
        "info",
        "config",
    )
    assert entry.message == "Config reloaded (3 providers, 1 models)"
    assert entry.details == {"provider_count": 3, "model_count": 1}


@pytest.mark.asyncio
async def test_startup_validation_writes_the_row_for_a_changed_load() -> None:
    ctx = _ctx()
    result = {
        "disabled": [{"provider": "argo", "reason": "r", "category": "c"}],
        "recovered": [],
    }
    await OpsStartupValidation(ctx, result).execute()
    [(entry, _)] = ctx.ops_log.rows
    assert (entry.event_type, entry.severity, entry.source) == (
        "provider_validation",
        "warning",
        "config",
    )
    assert "argo" in entry.message
    assert entry.details["disabled"] == result["disabled"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [None, {}, {"disabled": [], "recovered": []}], ids=repr
)
async def test_startup_validation_writes_nothing_for_a_quiet_load(result: Any) -> None:
    ctx = _ctx()
    await OpsStartupValidation(ctx, result).execute()
    assert ctx.ops_log.rows == []


@pytest.mark.asyncio
async def test_a_failing_log_does_not_stop_the_operation() -> None:
    ctx = _ctx(_Log(fail=True))
    await OpsGatewayShutdown(ctx).execute()


@pytest.mark.asyncio
async def test_no_log_is_not_an_error() -> None:
    await OpsGatewayShutdown(OpsContext()).execute()

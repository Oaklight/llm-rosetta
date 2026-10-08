"""Tests for the OpsBase framework and OpsContext."""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.ops.base import OpsBase, OpsContext
from llm_rosetta.observability.ops_log import (
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SOURCE_ADMIN,
    SOURCE_PERSISTENCE,
)


# -- Concrete test subclass ------------------------------------------------


class _StubOp(OpsBase):
    """Minimal concrete subclass for testing the base protocol."""

    event_type: ClassVar[str] = "test_event"
    severity: ClassVar[str] = SEVERITY_INFO
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_return_value",)

    def __init__(self, ctx: OpsContext, *, return_value: Any = None) -> None:
        super().__init__(ctx)
        self._return_value = return_value

    async def _run(self) -> Any:
        return self._return_value

    def _message(self, result: Any) -> str:
        return f"stub ran: {result}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"result": result}


class _CustomRecordOp(OpsBase):
    """Subclass that overrides _record to test the override path."""

    event_type: ClassVar[str] = "custom_event"

    __slots__ = ("recorded", "recorded_error")

    def __init__(self, ctx: OpsContext) -> None:
        super().__init__(ctx)
        self.recorded = False
        self.recorded_error = None

    async def _run(self) -> Any:
        return "custom_result"

    def _message(self, result: Any) -> str:
        return "custom"

    def _details(self, result: Any) -> dict[str, Any]:
        return {}

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        self.recorded = True
        self.recorded_error = error


class _WarnOp(OpsBase):
    """Subclass with non-default severity and source."""

    event_type: ClassVar[str] = "warn_event"
    severity: ClassVar[str] = SEVERITY_WARNING
    source: ClassVar[str] = SOURCE_PERSISTENCE

    async def _run(self) -> Any:
        return None

    def _message(self, result: Any) -> str:
        return "warning"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"level": "warn"}


class _FailOp(OpsBase):
    """Always-failing subclass for testing failure recording."""

    event_type: ClassVar[str] = "fail_event"

    async def _run(self) -> Any:
        raise ValueError("boom")

    def _message(self, result: Any) -> str:
        return "failed op"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"attempted": True}


# -- OpsContext tests ------------------------------------------------------


class TestOpsContext:
    def test_slots(self):
        ctx = OpsContext()
        assert not hasattr(ctx, "__dict__")

    def test_all_none_by_default(self):
        ctx = OpsContext()
        assert ctx.ops_log is None
        assert ctx.request_log is None
        assert ctx.metrics is None
        assert ctx.persistence is None

    def test_init_with_services(self):
        ops_log = MagicMock()
        request_log = MagicMock()
        metrics = MagicMock()
        persistence = MagicMock()
        ctx = OpsContext(
            ops_log=ops_log,
            request_log=request_log,
            metrics=metrics,
            persistence=persistence,
        )
        assert ctx.ops_log is ops_log
        assert ctx.request_log is request_log
        assert ctx.metrics is metrics
        assert ctx.persistence is persistence


# -- OpsBase protocol tests ------------------------------------------------


class TestOpsBase:
    @pytest.mark.asyncio
    async def test_execute_calls_run_and_record(self):
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log)

        op = _StubOp(ctx, return_value=42)
        result = await op.execute()

        assert result == 42
        ops_log.add.assert_called_once()
        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == "test_event"
        assert entry.severity == SEVERITY_INFO
        assert entry.source == SOURCE_ADMIN
        assert "stub ran: 42" in entry.message
        assert entry.details == {"result": 42}

    @pytest.mark.asyncio
    async def test_execute_no_ops_log_graceful(self):
        ctx = OpsContext()
        op = _StubOp(ctx, return_value="ok")
        result = await op.execute()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_record_override(self):
        ctx = OpsContext()
        op = _CustomRecordOp(ctx)
        result = await op.execute()
        assert result == "custom_result"
        assert op.recorded is True
        assert op.recorded_error is None

    @pytest.mark.asyncio
    async def test_severity_and_source_classvar(self):
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log)

        op = _WarnOp(ctx)
        await op.execute()

        entry = ops_log.add.call_args[0][0]
        assert entry.severity == SEVERITY_WARNING
        assert entry.source == SOURCE_PERSISTENCE
        assert entry.details == {"level": "warn"}

    @pytest.mark.asyncio
    async def test_run_failure_records_and_reraises(self):
        """Failed _run() should still record (with error details) then re-raise."""
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log)

        op = _FailOp(ctx)
        with pytest.raises(ValueError, match="boom"):
            await op.execute()

        # Audit entry was still written
        ops_log.add.assert_called_once()
        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == "fail_event"
        assert entry.severity == SEVERITY_WARNING  # escalated on failure
        assert "boom" in entry.details.get("error", "")
        assert entry.details["attempted"] is True

    @pytest.mark.asyncio
    async def test_run_failure_no_ops_log_still_raises(self):
        """Failed _run() re-raises even when ops_log is None."""
        ctx = OpsContext()
        op = _FailOp(ctx)
        with pytest.raises(ValueError, match="boom"):
            await op.execute()

    @pytest.mark.asyncio
    async def test_record_failure_is_swallowed(self, caplog):
        """If _record() itself raises, the operation result is still returned."""
        ops_log = MagicMock()
        ops_log.add = AsyncMock(side_effect=RuntimeError("db full"))
        ctx = OpsContext(ops_log=ops_log)

        op = _StubOp(ctx, return_value="success")
        with caplog.at_level(logging.WARNING):
            result = await op.execute()

        assert result == "success"
        assert "ops audit record failed" in caplog.text

    @pytest.mark.asyncio
    async def test_record_failure_does_not_mask_run_failure(self, caplog):
        """If both _run() and _record() fail, the _run() exception wins."""
        ops_log = MagicMock()
        ops_log.add = AsyncMock(side_effect=RuntimeError("db full"))
        ctx = OpsContext(ops_log=ops_log)

        op = _FailOp(ctx)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="boom"):
                await op.execute()

        assert "ops audit record failed" in caplog.text

    @pytest.mark.asyncio
    async def test_record_override_receives_error(self):
        """Custom _record override receives the error kwarg on failure."""

        class _FailCustom(_CustomRecordOp):
            async def _run(self) -> Any:
                raise TypeError("type fail")

        ctx = OpsContext()
        op = _FailCustom(ctx)
        with pytest.raises(TypeError, match="type fail"):
            await op.execute()

        assert op.recorded is True
        assert isinstance(op.recorded_error, TypeError)

    def test_ops_base_is_abstract(self):
        with pytest.raises(TypeError):
            OpsBase(OpsContext())  # type: ignore[abstract]

    def test_slots_no_dict(self):
        ctx = OpsContext()
        op = _StubOp(ctx, return_value=1)
        assert not hasattr(op, "__dict__")

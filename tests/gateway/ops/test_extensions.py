"""``OpsExtensionCliInstall``: the audit record of an install from the panel (#1820)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.api import CliInstallError
from llm_rosetta.contracts.extension import ExternalInstall, ExternalProgram
from llm_rosetta_gateway.observability.ops_log import (
    ALL_EVENT_TYPES,
    EVENT_EXTENSION_CLI_INSTALL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SOURCE_ADMIN,
)
from llm_rosetta_gateway.ops import OpsExtensionCliInstall, extensions
from llm_rosetta_gateway.ops.base import OpsContext

PROGRAM = ExternalProgram(
    name="claude",
    check=("claude", "--version"),
    install=ExternalInstall(version="2.1.299", npm="@anthropic-ai/claude-code"),
)


def _ctx() -> OpsContext:
    ops_log = MagicMock()
    ops_log.add = AsyncMock()
    return OpsContext(ops_log=ops_log)


def _installer(monkeypatch, outcome):
    calls: list[tuple] = []

    def fake(ext_id, program, **kwargs):
        calls.append((ext_id, program, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(extensions, "install_cli", fake)
    return calls


def test_the_event_is_registered_and_the_class_exported():
    assert EVENT_EXTENSION_CLI_INSTALL in ALL_EVENT_TYPES
    assert OpsExtensionCliInstall.event_type == EVENT_EXTENSION_CLI_INSTALL
    assert OpsExtensionCliInstall.source == SOURCE_ADMIN


@pytest.mark.asyncio
async def test_an_install_runs_the_declared_program_and_is_recorded(monkeypatch):
    done = SimpleNamespace(state="installed", found="2.1.299")
    calls = _installer(monkeypatch, done)
    ctx = _ctx()

    result = await OpsExtensionCliInstall(
        ctx, ext_id="claude", program=PROGRAM
    ).execute()

    assert calls == [("claude", PROGRAM, {})]
    assert result == {"state": "installed", "found": "2.1.299"}
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.event_type == EVENT_EXTENSION_CLI_INSTALL
    assert entry.severity == SEVERITY_INFO
    assert entry.details == {
        "extension": "claude",
        "program": "claude",
        "version": "2.1.299",
        "state": "installed",
        "found": "2.1.299",
    }
    assert "claude/claude 2.1.299" in entry.message


@pytest.mark.asyncio
async def test_a_failure_is_recorded_with_the_installers_reason_and_raised(
    monkeypatch,
):
    _installer(monkeypatch, CliInstallError("npm exited with 1: ERR 404"))
    ctx = _ctx()

    with pytest.raises(CliInstallError, match="ERR 404"):
        await OpsExtensionCliInstall(ctx, ext_id="claude", program=PROGRAM).execute()

    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.severity == SEVERITY_WARNING
    assert entry.details["error"] == "npm exited with 1: ERR 404"
    assert entry.details["version"] == "2.1.299"
    assert "failed" in entry.message


@pytest.mark.asyncio
async def test_an_unexpected_error_text_never_reaches_the_record(monkeypatch):
    _installer(monkeypatch, RuntimeError("TOKEN=hunter2 in the environment"))
    ctx = _ctx()

    with pytest.raises(CliInstallError) as seen:
        await OpsExtensionCliInstall(ctx, ext_id="claude", program=PROGRAM).execute()

    assert "hunter2" not in str(seen.value)
    entry = ctx.ops_log.add.call_args[0][0]
    assert "hunter2" not in repr(entry.details) + entry.message
    assert "RuntimeError" in entry.details["error"]

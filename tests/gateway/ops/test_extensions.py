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


# -- remove and update (#1829) ------------------------------------------------

from llm_rosetta.api import ExtensionInstallError, PathEntry, RemovalPlan  # noqa: E402
from llm_rosetta_gateway.observability.ops_log import (  # noqa: E402
    EVENT_EXTENSION_REMOVE,
    EVENT_EXTENSION_UPDATE,
)
from llm_rosetta_gateway.ops import OpsExtensionRemove, OpsExtensionUpdate  # noqa: E402

PLAN = RemovalPlan(
    "claude",
    "0.1.1",
    "directory",
    True,
    delete=(PathEntry("/config/extensions/llm_rosetta_ext/claude", 10),),
    keep=(PathEntry("/config/tools/node", 0, "still used by: codex"),),
)


def _remover(monkeypatch, outcome):
    seen: list = []

    def fake(ext_id, **kwargs):
        seen.append((ext_id, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(extensions, "remove_with_programs", fake)
    return seen


def test_the_remove_and_update_events_are_registered_and_exported():
    for event, cls in (
        (EVENT_EXTENSION_REMOVE, OpsExtensionRemove),
        (EVENT_EXTENSION_UPDATE, OpsExtensionUpdate),
    ):
        assert event in ALL_EVENT_TYPES
        assert cls.event_type == event and cls.source == SOURCE_ADMIN


@pytest.mark.asyncio
async def test_a_removal_records_the_paths_deleted_and_kept(monkeypatch):
    done = SimpleNamespace(deleted=("/a", "/b"), failed=())
    seen = _remover(monkeypatch, done)
    ctx = _ctx()

    result = await OpsExtensionRemove(
        ctx, ext_id="claude", plan=PLAN, replace=True
    ).execute()

    assert seen == [("claude", {"path": "/config/extensions/llm_rosetta_ext/claude"})]
    assert result == {"deleted": ["/a", "/b"], "failed": []}
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.event_type == EVENT_EXTENSION_REMOVE
    assert entry.severity == SEVERITY_INFO
    assert entry.details == {
        "extension": "claude",
        "version": "0.1.1",
        "replace": True,
        "kept": ["/config/tools/node"],
        "deleted": ["/a", "/b"],
        "failed": [],
    }


@pytest.mark.asyncio
async def test_a_path_that_could_not_be_deleted_is_in_the_record(monkeypatch):
    _remover(
        monkeypatch,
        SimpleNamespace(deleted=("/a",), failed=(("/b", "PermissionError: denied"),)),
    )
    ctx = _ctx()
    await OpsExtensionRemove(ctx, ext_id="claude", plan=PLAN).execute()
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.details["failed"] == [["/b", "PermissionError: denied"]]
    assert "could not be deleted" in entry.message


@pytest.mark.asyncio
async def test_a_refused_removal_is_recorded_with_the_reason_and_raised(monkeypatch):
    _remover(monkeypatch, ExtensionInstallError("provided by the image"))
    ctx = _ctx()
    with pytest.raises(ExtensionInstallError, match="provided by the image"):
        await OpsExtensionRemove(ctx, ext_id="claude", plan=PLAN).execute()
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.severity == SEVERITY_WARNING
    assert entry.details["error"] == "provided by the image"
    assert "failed" in entry.message


@pytest.mark.asyncio
async def test_an_unexpected_removal_error_text_never_reaches_the_record(monkeypatch):
    _remover(monkeypatch, RuntimeError("TOKEN=hunter2"))
    ctx = _ctx()
    with pytest.raises(ExtensionInstallError) as seen:
        await OpsExtensionRemove(ctx, ext_id="claude", plan=PLAN).execute()
    assert "hunter2" not in str(seen.value)
    entry = ctx.ops_log.add.call_args[0][0]
    assert "hunter2" not in repr(entry.details) + entry.message
    assert "RuntimeError" in entry.details["error"]


@pytest.mark.asyncio
async def test_an_update_records_the_versions_before_and_after():
    ctx = _ctx()
    result = await OpsExtensionUpdate(
        ctx, ext_id="claude", before="0.1.1", work=lambda: "0.2.0"
    ).execute()
    assert result == {"after": "0.2.0"}
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.event_type == EVENT_EXTENSION_UPDATE
    assert entry.details == {"extension": "claude", "before": "0.1.1", "after": "0.2.0"}
    assert "0.1.1 -> 0.2.0" in entry.message


@pytest.mark.asyncio
async def test_a_failed_update_is_recorded_and_raised():
    def work():
        raise ExtensionInstallError("sha256 mismatch")

    ctx = _ctx()
    with pytest.raises(ExtensionInstallError, match="sha256"):
        await OpsExtensionUpdate(
            ctx, ext_id="claude", before="0.1.1", work=work
        ).execute()
    entry = ctx.ops_log.add.call_args[0][0]
    assert entry.severity == SEVERITY_WARNING
    assert entry.details == {
        "extension": "claude",
        "before": "0.1.1",
        "error": "sha256 mismatch",
    }


@pytest.mark.asyncio
async def test_an_unexpected_update_error_text_never_reaches_the_record():
    def work():
        raise RuntimeError("TOKEN=hunter2")

    ctx = _ctx()
    with pytest.raises(ExtensionInstallError):
        await OpsExtensionUpdate(ctx, ext_id="claude", before=None, work=work).execute()
    entry = ctx.ops_log.add.call_args[0][0]
    assert "hunter2" not in repr(entry.details) + entry.message

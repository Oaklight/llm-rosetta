"""Tests for security and key management Ops subclasses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.ops.keys import (
    OpsKeyCreate,
    OpsKeyDelete,
    OpsKeyRotate,
    OpsKeyUpdate,
)
from llm_rosetta.gateway.ops.security import (
    OpsPasswordChange,
    OpsSessionLogoutAll,
    OpsTokenRotate,
)
from llm_rosetta.observability.ops_log import (
    EVENT_KEY_CREATE,
    EVENT_KEY_DELETE,
    EVENT_KEY_ROTATE,
    EVENT_KEY_UPDATE,
    EVENT_PASSWORD_CHANGED,
    EVENT_SESSION_LOGOUT_ALL,
    EVENT_TOKEN_ROTATED,
    SOURCE_AUTH,
    SOURCE_KEYS,
)


def _ctx() -> OpsContext:
    ops_log = MagicMock()
    ops_log.add = AsyncMock()
    return OpsContext(ops_log=ops_log)


class TestSecurityOps:
    @pytest.mark.asyncio
    async def test_password_change(self):
        ctx = _ctx()
        await OpsPasswordChange(ctx).execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_PASSWORD_CHANGED
        assert entry.source == SOURCE_AUTH

    @pytest.mark.asyncio
    async def test_token_rotate(self):
        ctx = _ctx()
        await OpsTokenRotate(ctx).execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_TOKEN_ROTATED
        assert entry.source == SOURCE_AUTH

    @pytest.mark.asyncio
    async def test_session_logout_all(self):
        ctx = _ctx()
        result = await OpsSessionLogoutAll(ctx, count=5).execute()
        assert result["sessions_cleared"] == 5
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_SESSION_LOGOUT_ALL
        assert entry.details["sessions_cleared"] == 5
        assert "5 cleared" in entry.message

    @pytest.mark.asyncio
    async def test_no_ops_log_graceful(self):
        ctx = OpsContext()
        await OpsPasswordChange(ctx).execute()
        await OpsTokenRotate(ctx).execute()
        await OpsSessionLogoutAll(ctx, count=0).execute()


class TestKeyOps:
    @pytest.mark.asyncio
    async def test_key_create(self):
        ctx = _ctx()
        await OpsKeyCreate(ctx, key_id="k1", label="test-key").execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_KEY_CREATE
        assert entry.source == SOURCE_KEYS
        assert entry.details["key_id"] == "k1"
        assert entry.details["label"] == "test-key"
        assert "test-key" in entry.message

    @pytest.mark.asyncio
    async def test_key_create_no_label(self):
        ctx = _ctx()
        await OpsKeyCreate(ctx, key_id="k2", label="").execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert "(no label)" in entry.message

    @pytest.mark.asyncio
    async def test_key_update(self):
        ctx = _ctx()
        await OpsKeyUpdate(ctx, key_id="k1", changed_fields=["label"]).execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_KEY_UPDATE
        assert entry.details["changed_fields"] == ["label"]

    @pytest.mark.asyncio
    async def test_key_delete(self):
        ctx = _ctx()
        await OpsKeyDelete(ctx, key_id="k1", label="my-key").execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_KEY_DELETE
        assert entry.details["key_id"] == "k1"
        assert "my-key" in entry.message

    @pytest.mark.asyncio
    async def test_key_rotate(self):
        ctx = _ctx()
        await OpsKeyRotate(ctx, key_id="k1", label=None).execute()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_KEY_ROTATE
        assert "k1" in entry.message

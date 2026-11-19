"""The agent-process pool, the idle sweep, and the knobs that bound them.

Reuse (``test_acp_transport``) is what makes a session worth keeping; this
module is about how many of them exist and when they go away.  One process per
provider serialises the lane, so the operator may run several — and every
process so kept has to be given back when nobody is using it, on a timer
rather than on the next request.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_rosetta_gateway.providers import _resolve_acp_limits
from llm_rosetta_gateway.transport.acp import AcpTransport
from llm_rosetta_gateway.transport.acp_ledger import SessionKey

from .test_acp_transport import _kinds, _provider, _turn

PROVIDER = "antigravity"


def _key(conversation: str) -> SessionKey:
    return SessionKey(PROVIDER, conversation)


class TestThePool:
    """A provider may run more than one agent process."""

    async def test_two_conversations_land_on_two_processes(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        base = _provider(monkeypatch, tmp_path, "ok")
        base.acp_processes = 2
        try:
            await transport.send(
                base.with_acp_conversation("conv-a"), "", _turn("[User]\na")
            )
            await transport.send(
                base.with_acp_conversation("conv-b"), "", _turn("[User]\nb")
            )
            assert len(transport._pool(PROVIDER)) == 2
            entries = [transport.ledger.get(_key(c)) for c in ("conv-a", "conv-b")]
            assert all(entry is not None for entry in entries)
            assert {entry.agent_slot for entry in entries if entry} == {0, 1}
        finally:
            await transport.close()

    async def test_a_conversation_keeps_its_process(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-a")
        provider.acp_processes = 2
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            entry = transport.ledger.get(_key("conv-a"))
            assert entry is not None
            first_slot = entry.agent_slot
            await transport.send(provider, "", _turn("[User]\none", "[User]\ntwo"))
            assert entry.agent_slot == first_slot
            # Only one process was ever started: the second was never needed.
            assert len(transport._pool(PROVIDER)) == 1
        finally:
            await transport.close()

    async def test_one_process_by_default(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-a")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            assert len(transport._pool(PROVIDER)) == 1
        finally:
            await transport.close()

    async def test_a_third_conversation_reuses_the_least_loaded_process(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        base = _provider(monkeypatch, tmp_path, "ok")
        base.acp_processes = 2
        try:
            for conversation in ("conv-a", "conv-b", "conv-c"):
                await transport.send(
                    base.with_acp_conversation(conversation),
                    "",
                    _turn(f"[User]\n{conversation}"),
                )
            assert len(transport._pool(PROVIDER)) == 2
        finally:
            await transport.close()


class TestTheSweep:
    """An idle gateway gives its agent sessions back on a timer."""

    async def test_a_session_idle_past_its_provider_ttl_is_closed(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-a")
        provider.acp_session_ttl = 0.0
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            assert transport.ledger.get(_key("conv-a")) is not None
            await asyncio.sleep(0)
            closed = await transport.sweep()
        finally:
            await transport.close()

        assert closed == 1
        assert "close" in _kinds(tmp_path)

    async def test_a_fresh_session_survives_the_sweep(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-a")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            assert await transport.sweep() == 0
            assert transport.ledger.get(_key("conv-a")) is not None
        finally:
            await transport.close()


class TestTheConfigKnobs:
    """``acp:`` is the operator's block, validated at config load."""

    def test_the_block_is_read(self):
        assert _resolve_acp_limits(
            {"acp": {"processes": 3, "max_sessions": 8, "session_ttl": 60}},
            "p",
        ) == {
            "acp_processes": 3,
            "acp_max_sessions": 8,
            "acp_session_ttl": 60.0,
        }

    def test_no_block_means_the_defaults(self):
        assert _resolve_acp_limits({}, "p") == {}

    def test_a_partial_block_leaves_the_rest_to_defaults(self):
        assert _resolve_acp_limits({"acp": {"processes": 2}}, "p") == {
            "acp_processes": 2
        }

    @pytest.mark.parametrize(
        "block",
        [
            {"processes": 0},
            {"processes": "2"},
            {"processes": True},
            {"max_sessions": "8"},
            {"session_ttl": "soon"},
        ],
    )
    def test_a_bad_value_names_the_provider_and_the_key(self, block):
        with pytest.raises(ValueError, match="Provider 'p': acp\\."):
            _resolve_acp_limits({"acp": block}, "p")

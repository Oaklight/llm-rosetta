"""The session ledger: what it remembers, and what it refuses to.

The ledger answers one question per request — may this turn ride on the
session the agent already holds?  Getting it wrong in either direction is
expensive: a wrong "yes" hands one conversation's transcript to another, and a
wrong "no" replays the whole history into a fresh session.  These tests pin
the classification, the record it is computed from, and the bounds that keep a
long-lived gateway from accumulating them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from llm_rosetta_gateway.transport.acp_ledger import (
    DIVERGES,
    EXTENDS,
    SAME,
    AcpSessionLedger,
    LedgerEntry,
    SessionKey,
    block_digests,
    relationship,
)


def _digest(block: object) -> str:
    blob = json.dumps(block, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _blocks(*texts: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text} for text in texts]


def _entry(
    conversation: str = "conv",
    *,
    provider: str = "p",
    session_id: str = "sess-1",
    seen: tuple[str, ...] = (),
    live: bool = True,
    last_used: float | None = None,
) -> LedgerEntry:
    entry = LedgerEntry(
        key=SessionKey(provider, conversation),
        session_id=session_id,
        cwd="/tmp/ws",
        seen=seen,
        live=live,
    )
    if last_used is not None:
        entry.last_used = last_used
    return entry


class TestTheClassification:
    def test_an_extension_is_an_extension(self):
        assert relationship(("a", "b"), ("a", "b", "c")) == EXTENDS

    def test_the_same_history_is_not_an_extension(self):
        """It is a retry, and what a failed turn recorded is unknown."""
        assert relationship(("a", "b"), ("a", "b")) == SAME

    def test_a_rewritten_prefix_diverges(self):
        assert relationship(("a", "b"), ("a", "X", "c")) == DIVERGES

    def test_a_shorter_history_diverges(self):
        assert relationship(("a", "b", "c"), ("a", "b")) == DIVERGES

    def test_an_empty_record_extends_nothing(self):
        """A session that has seen nothing cannot be continued."""
        assert relationship((), ("a",)) == DIVERGES


class TestTheRecord:
    def test_a_digest_per_block_in_order(self):
        digests = block_digests(_blocks("one", "two"))
        assert digests == (_digest(_blocks("one")[0]), _digest(_blocks("two")[0]))

    def test_order_matters(self):
        forward = block_digests(_blocks("one", "two"))
        backward = block_digests(_blocks("two", "one"))
        assert forward != backward

    def test_a_changed_field_changes_the_digest(self):
        first = block_digests([{"type": "text", "text": "hi"}])
        second = block_digests([{"type": "text", "text": "hi", "cache": True}])
        assert first != second

    def test_an_image_block_is_digested_too(self):
        digests = block_digests(
            [
                {"type": "text", "text": "look"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            ]
        )
        assert len(digests) == 2

    def test_the_record_is_bounded(self):
        entry = _entry()
        entry.remember(("a",) * 10, max_blocks=8)
        assert entry.seen == ()
        entry.remember(("a",) * 8, max_blocks=8)
        assert entry.seen == ("a",) * 8


class TestExpiry:
    def test_an_idle_entry_is_returned_and_forgotten(self):
        ledger = AcpSessionLedger(idle_ttl=10.0)
        ledger.put(_entry("old", last_used=0.0))
        expired = ledger.expire(now=100.0)
        assert [e.key.conversation for e in expired] == ["old"]
        assert ledger.get(SessionKey("p", "old")) is None

    def test_a_fresh_entry_survives_the_sweep(self):
        ledger = AcpSessionLedger(idle_ttl=10.0)
        ledger.put(_entry("fresh", last_used=95.0))
        assert ledger.expire(now=100.0) == []
        assert ledger.get(SessionKey("p", "fresh")) is not None


class TestTheCap:
    def test_the_oldest_session_is_displaced_past_the_cap(self):
        ledger = AcpSessionLedger(max_sessions=2)
        ledger.put(_entry("one", last_used=1.0))
        ledger.put(_entry("two", last_used=2.0))
        displaced = ledger.put(_entry("three", last_used=3.0))
        assert [e.key.conversation for e in displaced] == ["one"]
        assert ledger.get(SessionKey("p", "one")) is None

    def test_a_session_mid_turn_is_never_displaced(self):
        """Evicting a session whose turn is in flight would cut the turn."""
        ledger = AcpSessionLedger(max_sessions=1)
        busy = _entry("busy", last_used=1.0)
        ledger.put(busy)

        async def hold() -> None:
            async with busy.lock:
                displaced = ledger.put(_entry("new", last_used=2.0))
                assert displaced == []

        asyncio.run(hold())
        assert ledger.get(SessionKey("p", "busy")) is busy

    def test_the_cap_counts_one_provider_only(self):
        ledger = AcpSessionLedger(max_sessions=1)
        ledger.put(_entry("mine", provider="p", last_used=1.0))
        displaced = ledger.put(_entry("theirs", provider="q", last_used=2.0))
        assert displaced == []
        assert ledger.get(SessionKey("p", "mine")) is not None

    def test_room_is_reported_before_a_session_is_opened(self):
        ledger = AcpSessionLedger(max_sessions=1)
        assert ledger.room_for("p") is True
        ledger.put(_entry("one"))
        assert ledger.room_for("p") is False


class TestDropping:
    def test_one_key_can_be_forgotten(self):
        ledger = AcpSessionLedger()
        entry = _entry()
        ledger.put(entry)
        assert ledger.drop(entry.key) is entry
        assert ledger.get(entry.key) is None

    def test_a_provider_can_be_forgotten_whole(self):
        ledger = AcpSessionLedger()
        ledger.put(_entry("a", provider="p"))
        ledger.put(_entry("b", provider="p"))
        ledger.put(_entry("c", provider="q"))
        dropped = ledger.drop_provider("p")
        assert len(dropped) == 2
        assert ledger.entries_of("p") == []
        assert len(ledger.entries_of("q")) == 1


class TestTheTurnLock:
    def test_two_turns_of_one_conversation_share_a_lock(self):
        ledger = AcpSessionLedger()
        key = SessionKey("p", "conv")
        assert ledger.lock_for(key) is ledger.lock_for(key)

    def test_a_lock_outlives_an_entry(self):
        """Two concurrent first turns must not both open a session."""
        ledger = AcpSessionLedger()
        key = SessionKey("p", "conv")
        lock = ledger.lock_for(key)
        ledger.put(_entry())
        ledger.drop(key)
        assert ledger.lock_for(key) is lock

"""The ledger of agent sessions: which conversation is on which session.

An ACP agent keeps a conversation inside a *session it owns*: the client sends
one turn, the agent appends it to the session's transcript, and the next turn
only has to carry what is new.  That is the whole reason to reuse a session —
the alternative is replaying the entire history on every request, which the
agent then sees twice.

The ledger is what makes reuse safe.  It holds, per
``(provider config name, conversation id)``, the agent's session id, the
process that serves it, and a digest of every prompt block the agent has been
shown.  A request is then classified against that record:

``extends``
    Its history begins with exactly what the agent has seen, and adds to it.
    The session is reused and only the new tail is sent.
``diverges``
    Its history differs from the record — a rewritten turn, a compaction, a
    different conversation on the same id.  The session is closed and a fresh
    one is opened with the whole history.
``same``
    It is the very history the agent already has.  Treated as ``diverges``: a
    client only resends an identical request after an attempt failed, and what
    the agent recorded of a failed turn is unknown.  Replaying into a fresh
    session is the only answer that cannot be wrong.

Nothing here is shared between processes and nothing is persisted: the ledger
is a view of live agent sessions, so a restart starts empty and the first turn
of each conversation simply opens a new session.  Entries are bounded twice —
by an idle TTL and by a per-provider cap — because every entry keeps an agent
session alive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from collections.abc import Iterable, Sequence

#: How long a session may sit unused before the ledger gives it back.  The
#: agent holds the transcript in memory, so an abandoned session is a cost the
#: provider pays without a request to justify it.
DEFAULT_IDLE_TTL = 1800.0

#: How many sessions one provider may hold open.  A client that starts a new
#: conversation per request would otherwise pin one agent session each.
DEFAULT_MAX_SESSIONS = 32

#: How many prompt blocks the ledger remembers per session.  The record is
#: per-block digests, not the blocks: 4096 of them are ~130 KB, past which the
#: session is dropped rather than tracked.
DEFAULT_MAX_BLOCKS = 4096

#: What a request's history is, relative to what the agent has seen.
EXTENDS = "extends"
DIVERGES = "diverges"
SAME = "same"


@dataclass(frozen=True, slots=True)
class SessionKey:
    """Which conversation of which provider an entry names."""

    provider: str
    conversation: str


@dataclass
class LedgerEntry:
    """One live agent session, and what it has been shown.

    Attributes:
        key: The conversation this session belongs to.
        session_id: The agent's own id for the session.
        cwd: The working directory the session was opened on.
        model: The model the session is configured with, as the agent names
            it.  A request that names another model cannot reuse the session.
        seen: One digest per prompt block the agent has been shown, in order.
        agent_slot: Which process of the provider's pool serves it.
        live: Whether the process still holds it open.  False after a restart
            or an eviction; the id may still be resumable.
        last_used: Monotonic time of the last turn that touched it.
        lock: Serialises turns of this one session, so two HTTP requests for
            one conversation cannot interleave inside one agent session.
    """

    key: SessionKey
    session_id: str
    cwd: str
    model: str = ""
    seen: tuple[str, ...] = ()
    agent_slot: int = 0
    live: bool = True
    last_used: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def touch(self, now: float | None = None) -> None:
        """Record that a turn used this session."""
        self.last_used = time.monotonic() if now is None else now

    def remember(self, digests: Sequence[str], *, max_blocks: int) -> None:
        """Record what the agent has now been shown.

        A history longer than *max_blocks* is not tracked: the entry keeps the
        last turn's digests so the next comparison is still exact for that
        turn, and the session is left to the TTL rather than growing without
        bound.
        """
        if len(digests) > max_blocks:
            self.seen = ()
            return
        self.seen = tuple(digests)


def block_digests(prompt: Iterable[Any]) -> tuple[str, ...]:
    """Return one digest per prompt block, in order.

    The digest is over the block's canonical JSON, so a block that differs in
    any field — text, image bytes, mime type — is a different block, and two
    requests that carry the same history produce the same tuple.
    """
    digests: list[str] = []
    for block in prompt:
        blob = json.dumps(block, sort_keys=True, ensure_ascii=False, default=str)
        digests.append(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32])
    return tuple(digests)


def relationship(seen: Sequence[str], incoming: Sequence[str]) -> str:
    """Classify an incoming history against what the agent has seen.

    Args:
        seen: The digests the session has been shown, in order.
        incoming: The digests this request carries, in order.

    Returns:
        :data:`EXTENDS` when *incoming* starts with *seen* and adds to it,
        :data:`SAME` when the two are equal, :data:`DIVERGES` otherwise.
    """
    if not seen:
        return DIVERGES
    if len(incoming) == len(seen) and tuple(incoming) == tuple(seen):
        return SAME
    if len(incoming) > len(seen) and tuple(incoming[: len(seen)]) == tuple(seen):
        return EXTENDS
    return DIVERGES


class AcpSessionLedger:
    """Which agent session serves which conversation, and whether it may.

    Args:
        idle_ttl: Seconds an unused session is kept (see
            :data:`DEFAULT_IDLE_TTL`).
        max_sessions: Sessions one provider may hold (see
            :data:`DEFAULT_MAX_SESSIONS`).
        max_blocks: Prompt blocks remembered per session (see
            :data:`DEFAULT_MAX_BLOCKS`).
    """

    def __init__(
        self,
        *,
        idle_ttl: float = DEFAULT_IDLE_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_blocks: int = DEFAULT_MAX_BLOCKS,
    ) -> None:
        self._entries: dict[SessionKey, LedgerEntry] = {}
        self._locks: dict[SessionKey, asyncio.Lock] = {}
        self._ttl = idle_ttl
        self._max_sessions = max_sessions
        self._max_blocks = max_blocks
        # Per-provider overrides, registered by the transport from the
        # provider's own config.  Absent means the defaults above apply.
        self._ttls: dict[str, float] = {}
        self._caps: dict[str, int] = {}

    # -- per-provider limits ------------------------------------------------

    def set_limits(
        self,
        provider: str,
        *,
        max_sessions: int | None = None,
        idle_ttl: float | None = None,
    ) -> None:
        """Apply one provider's own session limits.

        The operator may give a busy provider more sessions, or a quiet one a
        shorter idle life.  Either value may be left out, in which case the
        ledger's default applies — the same question answered per provider
        rather than once for the process.

        Args:
            provider: The provider's config name.
            max_sessions: How many of its sessions may be held at once.
            idle_ttl: Seconds one of its unconsumed sessions is kept.
        """
        if max_sessions is not None:
            self._caps[provider] = max(1, int(max_sessions))
        if idle_ttl is not None:
            self._ttls[provider] = max(0.0, float(idle_ttl))

    def ttl_of(self, provider: str) -> float:
        """The idle TTL that applies to *provider*'s sessions."""
        return self._ttls.get(provider, self._ttl)

    def cap_of(self, provider: str) -> int:
        """How many sessions *provider* may hold at once."""
        return self._caps.get(provider, self._max_sessions)

    # -- reading ------------------------------------------------------------

    @property
    def max_blocks(self) -> int:
        """How many prompt blocks one session's record may hold."""
        return self._max_blocks

    @property
    def idle_ttl(self) -> float:
        """How long an unused session is kept, in seconds."""
        return self._ttl

    def get(self, key: SessionKey) -> LedgerEntry | None:
        """Return the entry for *key*, if one is held."""
        return self._entries.get(key)

    def lock_for(self, key: SessionKey) -> asyncio.Lock:
        """Return the turn lock for *key*, creating it if this is the first turn.

        The lock outlives an entry: two concurrent first turns of one
        conversation must not both open a session.
        """
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def entries_of(self, provider: str) -> list[LedgerEntry]:
        """Every entry this provider holds, oldest use first."""
        return sorted(
            (e for e in self._entries.values() if e.key.provider == provider),
            key=lambda e: e.last_used,
        )

    def relationship_for(self, entry: LedgerEntry, digests: Sequence[str]) -> str:
        """Classify *digests* against *entry*'s record."""
        return relationship(entry.seen, digests)

    # -- writing ------------------------------------------------------------

    def put(self, entry: LedgerEntry, *, now: float | None = None) -> list[LedgerEntry]:
        """Record *entry*, returning whatever it displaced.

        The caller closes the displaced sessions: the ledger does not own
        processes, so it returns the entries it dropped rather than closing
        them itself.

        Returns:
            Entries evicted to respect the per-provider cap, oldest first.
            The caller must close them.
        """
        self._entries[entry.key] = entry
        self.lock_for(entry.key)
        return self._trim(entry)

    def drop(self, key: SessionKey) -> LedgerEntry | None:
        """Forget *key*, returning the entry that was there."""
        return self._entries.pop(key, None)

    def drop_provider(self, provider: str) -> list[LedgerEntry]:
        """Forget every entry of *provider*, returning them."""
        dropped = [e for e in self._entries.values() if e.key.provider == provider]
        for entry in dropped:
            self._entries.pop(entry.key, None)
            self._locks.pop(entry.key, None)
        return dropped

    def expire(self, *, now: float | None = None) -> list[LedgerEntry]:
        """Return, and forget, entries idle longer than their provider's TTL."""
        moment = time.monotonic() if now is None else now
        stale = [
            e
            for e in self._entries.values()
            if moment - e.last_used > self.ttl_of(e.key.provider)
        ]
        for entry in stale:
            self._entries.pop(entry.key, None)
            self._locks.pop(entry.key, None)
        return stale

    def _trim(self, keep: LedgerEntry) -> list[LedgerEntry]:
        """Drop the oldest sessions of *keep*'s provider beyond the cap.

        Two entries are never dropped: *keep* itself — the caller is about to
        use it — and one whose turn is in flight, which is locked and serving
        a request right now.  A provider can therefore sit over its cap while
        its sessions are busy, and come back under it once they are not.
        """
        held = self.entries_of(keep.key.provider)
        overflow = len(held) - self.cap_of(keep.key.provider)
        dropped: list[LedgerEntry] = []
        for entry in held:
            if overflow <= 0:
                break
            if entry is keep or entry.lock.locked():
                continue
            self._entries.pop(entry.key, None)
            self._locks.pop(entry.key, None)
            dropped.append(entry)
            overflow -= 1
        return dropped

    # -- capacity -----------------------------------------------------------

    def room_for(self, provider: str) -> bool:
        """Whether *provider* may open one more session under the cap.

        The caller asks before opening: a fresh session on a provider already
        at its cap must displace one, and the displacement has to happen
        before the new session exists to be counted.
        """
        return len(self.entries_of(provider)) < self.cap_of(provider)


__all__ = [
    "DEFAULT_IDLE_TTL",
    "DEFAULT_MAX_BLOCKS",
    "DEFAULT_MAX_SESSIONS",
    "DIVERGES",
    "EXTENDS",
    "SAME",
    "AcpSessionLedger",
    "LedgerEntry",
    "SessionKey",
    "block_digests",
    "relationship",
]

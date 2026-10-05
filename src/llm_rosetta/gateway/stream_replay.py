"""Resumable streaming — session IDs, sequence cursors, and event replay.

When a client disconnects mid-stream, a conventional proxy loses the rest
of the upstream response: reconnecting means paying for a brand-new request
and risking two inconsistent answers.  This module gives the gateway a
hub for *resumable* streams:

1. The pump that consumes the upstream appends every formatted SSE message
   to a :class:`StreamReplayManager` session under a monotonic, per-session
   sequence number.  The pump is a detached task — when the original
   subscriber (the HTTP response) disconnects, upstream consumption keeps
   going.

2. The client reconnects to the same endpoint carrying the session id and
   the last sequence number it received (cursor).  The gateway validates
   the cursor and replays buffered messages from that point without ever
   touching the upstream again — even if the upstream connection has
   already finished.

3. Sessions are backed either by an in-memory store (single process) or by
   SQLite (``stream_replay.db``), which lets other worker processes serve
   replays and keeps the buffered tail available across restarts.

Wire protocol (all headers are case-insensitive HTTP headers):

* Initial streaming response headers:
  ``X-Stream-Replay-Id`` (session id), ``X-Stream-Replay-Cursor: 0``,
  ``X-Stream-Replay-TTL`` (replay window after completion, seconds).
* Every SSE message is prefixed with the standard SSE id field,
  ``id: <session_id>:<sequence>\\n``, so the cursor is self-describing.
* Resume requests carry either ``Last-Event-ID: <session_id>:<sequence>``
  or the pair ``X-Stream-Replay-Id`` + ``X-Stream-Replay-Cursor``.
* Unavailable replays (unknown/expired session, out-of-range cursor,
  truncated window) are answered with an explicit HTTP error before the
  SSE stream begins — the gateway never silently restarts from scratch.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .logging import get_logger
from .transport.sse_format import (
    SSE_FORMATTERS,
    build_stream_error_events,
    format_sse_done,
)

logger = get_logger()

# ---------------------------------------------------------------------------
# Wire protocol constants
# ---------------------------------------------------------------------------

HEADER_REPLAY_ID = "x-stream-replay-id"
HEADER_REPLAY_CURSOR = "x-stream-replay-cursor"
HEADER_LAST_EVENT_ID = "last-event-id"
HEADER_REPLAY = "x-stream-replay"
HEADER_REPLAY_TTL = "x-stream-replay-ttl"
HEADER_REPLAY_ERROR = "x-stream-replay-error"

SESSION_PREFIX = "rs_"

#: Session is still being pumped by a live process.
STATE_STREAMING = "streaming"
#: Stream finished normally; all messages (incl. terminal markers) are cached.
STATE_COMPLETE = "complete"
#: Upstream failed mid-flight; synthesized terminal error events were cached.
STATE_FAILED = "failed"
#: The pumping process died (stale heartbeat); only the buffered tail exists.
STATE_INTERRUPTED = "interrupted"

_TERMINAL_STATES = frozenset({STATE_COMPLETE, STATE_FAILED, STATE_INTERRUPTED})

_DB_FILENAME = "stream_replay.db"


# ---------------------------------------------------------------------------
# Errors / settings
# ---------------------------------------------------------------------------


class ReplayRejected(Exception):
    """Raised when a resume request cannot be served.

    The HTTP layer converts this into a provider-formatted error response
    *before* any SSE bytes are committed.

    Attributes:
        status_code: Suggested HTTP status (400 for malformed cursors,
            410 Gone for sessions that require a fresh request).
        code: Short machine-readable error code (sent as
            ``X-Stream-Replay-Error``).
        message: Human-readable explanation.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StreamReplaySettings:
    """Configuration for :class:`StreamReplayManager`.

    Args:
        enabled: Master switch.  When False, no manager is attached.
        ttl_seconds: How long a finished session remains replayable.
        max_events: Per-session ring buffer size.  Events beyond the
            window are discarded; resuming from a discarded cursor fails
            explicitly instead of returning a gapped stream.
        max_sessions: Upper bound on concurrently retained sessions.
        persist: When True (and a data directory is configured) the
            cache lives in SQLite and survives across processes/restarts.
        poll_interval: Cross-process live-tail polling interval.
        orphan_timeout: A streaming session whose heartbeat is older
            than this is considered owned by a dead process.
        heartbeat_interval: How often the maintenance task refreshes
            heartbeats of locally pumped sessions.
    """

    enabled: bool = True
    ttl_seconds: float = 300.0
    max_events: int = 20_000
    max_sessions: int = 2_000
    persist: bool = True
    poll_interval: float = 0.05
    orphan_timeout: float = 30.0
    heartbeat_interval: float = 5.0

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> StreamReplaySettings:
        """Build settings from the ``server.stream_replay`` config object."""
        if not raw:
            return cls()

        def _num(key: str, default: float, *, minimum: float) -> float:
            if key not in raw:
                return default
            try:
                return max(minimum, float(raw[key]))
            except (TypeError, ValueError):
                logger.warning(
                    "config: invalid stream_replay.%s=%r, using default %s",
                    key,
                    raw.get(key),
                    default,
                )
                return default

        def _int(key: str, default: int) -> int:
            if key not in raw:
                return default
            try:
                value = int(raw[key])
            except (TypeError, ValueError):
                logger.warning(
                    "config: invalid stream_replay.%s=%r, using default %s",
                    key,
                    raw.get(key),
                    default,
                )
                return default
            return max(1, value)

        return cls(
            enabled=bool(raw.get("enabled", True)),
            ttl_seconds=_num("ttl_seconds", 300.0, minimum=1.0),
            max_events=_int("max_events", 20_000),
            max_sessions=_int("max_sessions", 2_000),
            persist=bool(raw.get("persist", True)),
            poll_interval=_num("poll_interval_ms", 0.05, minimum=0.005) / 1000.0
            if "poll_interval_ms" in raw
            else _num("poll_interval", 0.05, minimum=0.005),
            orphan_timeout=_num("orphan_timeout", 30.0, minimum=1.0),
            heartbeat_interval=_num("heartbeat_interval", 5.0, minimum=0.5),
        )


# ---------------------------------------------------------------------------
# Cursor parsing / message encoding
# ---------------------------------------------------------------------------


def new_session_id() -> str:
    """Generate an opaque replay session id."""
    return f"{SESSION_PREFIX}{uuid.uuid4().hex}"


def parse_resume_cursor(headers: Any) -> tuple[str, int] | None:
    """Extract a ``(session_id, cursor)`` pair from request headers.

    Accepts either the standard SSE ``Last-Event-ID: <sid>:<seq>`` form or
    the explicit ``X-Stream-Replay-Id`` + ``X-Stream-Replay-Cursor`` pair.

    Returns:
        ``None`` when no replay headers are present (a normal request).
        Raises :class:`ReplayRejected` (HTTP 400) when replay headers are
        present but malformed.
    """
    last_event = headers.get(HEADER_LAST_EVENT_ID)
    if last_event:
        sid, sep, raw_seq = last_event.strip().rpartition(":")
        if not sep or not sid:
            raise ReplayRejected(
                400,
                "invalid_cursor",
                "Malformed Last-Event-ID header; expected '<session-id>:<sequence>'",
            )
        return sid, _parse_seq(raw_seq)

    sid = headers.get(HEADER_REPLAY_ID)
    raw_cursor = headers.get(HEADER_REPLAY_CURSOR)
    if sid and raw_cursor is not None:
        return sid.strip(), _parse_seq(raw_cursor)
    if sid or raw_cursor is not None:
        raise ReplayRejected(
            400,
            "invalid_cursor",
            "Resume requires both X-Stream-Replay-Id and X-Stream-Replay-Cursor",
        )
    return None


def _parse_seq(raw: Any) -> int:
    try:
        seq = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ReplayRejected(
            400, "invalid_cursor", f"Malformed replay cursor: {raw!r}"
        ) from None
    if seq < 0:
        raise ReplayRejected(
            400, "invalid_cursor", f"Replay cursor must be >= 0, got {seq}"
        )
    return seq


def encode_message(session_id: str, seq: int, payload: str) -> str:
    """Prefix a formatted SSE message with its standard ``id:`` field."""
    return f"id: {session_id}:{seq}\n{payload}"


def replay_headers(
    session_id: str, cursor: int, ttl: float, *, resumed: bool = False
) -> dict[str, str]:
    """Build response headers advertising (and describing) a replay stream."""
    headers = {
        HEADER_REPLAY_ID: session_id,
        HEADER_REPLAY_CURSOR: str(cursor),
        HEADER_REPLAY_TTL: str(int(ttl)),
    }
    if resumed:
        headers[HEADER_REPLAY] = "resumed"
    return headers


# ---------------------------------------------------------------------------
# Store records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionRecord:
    """A replay session's metadata row."""

    id: str
    source_provider: str
    model: str
    state: str
    last_seq: int
    created_at: float
    expires_at: float | None
    heartbeat: float
    truncated: bool = False


@dataclass(slots=True)
class StoredEvent:
    """One cached SSE message (raw text, without the ``id:`` prefix)."""

    seq: int
    payload: str


class ReplayStore(Protocol):
    """Storage backend for replay sessions and their events."""

    def create_session(self, record: SessionRecord) -> None: ...

    def get_session(self, session_id: str) -> SessionRecord | None: ...

    def append_events(
        self,
        session_id: str,
        events: list[tuple[int, str]],
        *,
        heartbeat: float,
        truncated: bool,
    ) -> None: ...

    def snapshot(
        self, session_id: str, after_seq: int
    ) -> tuple[SessionRecord | None, list[StoredEvent], int]: ...

    def finish(
        self,
        session_id: str,
        state: str,
        *,
        expires_at: float,
        heartbeat: float,
    ) -> None: ...

    def claim_orphan(
        self,
        session_id: str,
        *,
        expected_last_seq: int,
        cutoff: float,
        events: list[tuple[int, str]],
        expires_at: float,
        heartbeat: float,
    ) -> bool: ...

    def touch_heartbeats(self, session_ids: list[str], heartbeat: float) -> None: ...

    def prune_expired(self, now: float) -> None: ...

    def count_sessions(self) -> int: ...

    def oldest_terminal_session(self) -> str | None: ...

    def delete_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store (single process)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MemoryRow:
    record: SessionRecord
    events: deque[StoredEvent]


class MemoryReplayStore:
    """Process-local replay storage.  Used when SQLite persistence is off."""

    def __init__(self, *, max_events: int) -> None:
        self._rows: dict[str, _MemoryRow] = {}
        self._max_events = max_events

    def create_session(self, record: SessionRecord) -> None:
        self._rows[record.id] = _MemoryRow(record=record, events=deque())

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._rows.get(session_id)
        return row.record if row is not None else None

    def append_events(
        self,
        session_id: str,
        events: list[tuple[int, str]],
        *,
        heartbeat: float,
        truncated: bool,
    ) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        for seq, payload in events:
            row.events.append(StoredEvent(seq=seq, payload=payload))
        while len(row.events) > self._max_events:
            row.events.popleft()
        row.record.last_seq = events[-1][0] if events else row.record.last_seq
        row.record.heartbeat = heartbeat
        if truncated:
            row.record.truncated = True

    def snapshot(
        self, session_id: str, after_seq: int
    ) -> tuple[SessionRecord | None, list[StoredEvent], int]:
        row = self._rows.get(session_id)
        if row is None:
            return None, [], 0
        events = [
            StoredEvent(e.seq, e.payload) for e in row.events if e.seq > after_seq
        ]
        min_seq = row.events[0].seq if row.events else row.record.last_seq
        return row.record, events, min_seq

    def finish(
        self,
        session_id: str,
        state: str,
        *,
        expires_at: float,
        heartbeat: float,
    ) -> None:
        row = self._rows.get(session_id)
        if row is None:
            return
        row.record.state = state
        row.record.expires_at = expires_at
        row.record.heartbeat = heartbeat

    def claim_orphan(
        self,
        session_id: str,
        *,
        expected_last_seq: int,
        cutoff: float,
        events: list[tuple[int, str]],
        expires_at: float,
        heartbeat: float,
    ) -> bool:
        row = self._rows.get(session_id)
        if row is None:
            return False
        rec = row.record
        if rec.state != STATE_STREAMING or rec.heartbeat >= cutoff:
            return False
        if rec.last_seq != expected_last_seq:
            return False
        for seq, payload in events:
            row.events.append(StoredEvent(seq=seq, payload=payload))
        while len(row.events) > self._max_events:
            row.events.popleft()
        rec.last_seq = events[-1][0] if events else rec.last_seq
        rec.state = STATE_INTERRUPTED
        rec.expires_at = expires_at
        rec.heartbeat = heartbeat
        return True

    def touch_heartbeats(self, session_ids: list[str], heartbeat: float) -> None:
        for sid in session_ids:
            row = self._rows.get(sid)
            if row is not None and row.record.state == STATE_STREAMING:
                row.record.heartbeat = heartbeat

    def prune_expired(self, now: float) -> None:
        expired = [
            sid
            for sid, row in self._rows.items()
            if row.record.expires_at is not None and row.record.expires_at < now
        ]
        for sid in expired:
            del self._rows[sid]

    def count_sessions(self) -> int:
        return len(self._rows)

    def oldest_terminal_session(self) -> str | None:
        candidates = [
            (row.record.expires_at, sid)
            for sid, row in self._rows.items()
            if row.record.state in _TERMINAL_STATES
            and row.record.expires_at is not None
        ]
        return min(candidates)[1] if candidates else None

    def delete_session(self, session_id: str) -> None:
        self._rows.pop(session_id, None)

    def close(self) -> None:
        self._rows.clear()


# ---------------------------------------------------------------------------
# SQLite store (cross-process, survives restarts)
# ---------------------------------------------------------------------------


class SqliteReplayStore:
    """SQLite-backed replay storage using WAL journal mode."""

    def __init__(self, data_dir: str, *, max_events: int) -> None:
        self._max_events = max_events
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._data_dir / _DB_FILENAME), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_tables()

    @property
    def db_path(self) -> Path:
        return self._data_dir / _DB_FILENAME

    def _init_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_sessions (
                session_id      TEXT PRIMARY KEY,
                source_provider TEXT NOT NULL,
                model           TEXT NOT NULL DEFAULT '',
                state           TEXT NOT NULL,
                last_seq        INTEGER NOT NULL DEFAULT 0,
                created_at      REAL NOT NULL,
                expires_at      REAL,
                heartbeat       REAL NOT NULL,
                truncated       INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS replay_events (
                session_id TEXT NOT NULL,
                seq        INTEGER NOT NULL,
                payload    TEXT NOT NULL,
                PRIMARY KEY (session_id, seq)
            ) WITHOUT ROWID;
            """
        )
        self._conn.commit()

    @staticmethod
    def _row_to_record(row: Any) -> SessionRecord:
        return SessionRecord(
            id=row[0],
            source_provider=row[1],
            model=row[2],
            state=row[3],
            last_seq=row[4],
            created_at=row[5],
            expires_at=row[6],
            heartbeat=row[7],
            truncated=bool(row[8]),
        )

    def create_session(self, record: SessionRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO replay_sessions
                (session_id, source_provider, model, state, last_seq,
                 created_at, expires_at, heartbeat, truncated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                record.id,
                record.source_provider,
                record.model,
                record.state,
                record.last_seq,
                record.created_at,
                record.expires_at,
                record.heartbeat,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> SessionRecord | None:
        cur = self._conn.execute(
            "SELECT session_id, source_provider, model, state, last_seq, "
            "created_at, expires_at, heartbeat, truncated "
            "FROM replay_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    def append_events(
        self,
        session_id: str,
        events: list[tuple[int, str]],
        *,
        heartbeat: float,
        truncated: bool,
    ) -> None:
        if not events:
            return
        last_seq = events[-1][0]
        try:
            self._conn.executemany(
                "INSERT INTO replay_events (session_id, seq, payload) VALUES (?, ?, ?)",
                [(session_id, seq, payload) for seq, payload in events],
            )
            self._conn.execute(
                "UPDATE replay_sessions SET last_seq = ?, heartbeat = ?, "
                "truncated = ? WHERE session_id = ?",
                (last_seq, heartbeat, 1 if truncated else 0, session_id),
            )
            low_water = last_seq - self._max_events
            if low_water > 0:
                self._conn.execute(
                    "DELETE FROM replay_events WHERE session_id = ? AND seq <= ?",
                    (session_id, low_water),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def snapshot(
        self, session_id: str, after_seq: int
    ) -> tuple[SessionRecord | None, list[StoredEvent], int]:
        rec = self.get_session(session_id)
        if rec is None:
            return None, [], 0
        cur = self._conn.execute(
            "SELECT seq, payload FROM replay_events "
            "WHERE session_id = ? AND seq > ? ORDER BY seq",
            (session_id, after_seq),
        )
        events = [StoredEvent(seq=row[0], payload=row[1]) for row in cur.fetchall()]
        min_cur = self._conn.execute(
            "SELECT MIN(seq) FROM replay_events WHERE session_id = ?",
            (session_id,),
        )
        min_seq = min_cur.fetchone()[0]
        if min_seq is None:
            min_seq = rec.last_seq
        return rec, events, min_seq

    def finish(
        self,
        session_id: str,
        state: str,
        *,
        expires_at: float,
        heartbeat: float,
    ) -> None:
        self._conn.execute(
            "UPDATE replay_sessions SET state = ?, expires_at = ?, heartbeat = ? "
            "WHERE session_id = ?",
            (state, expires_at, heartbeat, session_id),
        )
        self._conn.commit()

    def claim_orphan(
        self,
        session_id: str,
        *,
        expected_last_seq: int,
        cutoff: float,
        events: list[tuple[int, str]],
        expires_at: float,
        heartbeat: float,
    ) -> bool:
        """Atomically claim a stale streaming session and append its tail.

        Returns True only for the single winning caller (state was
        streaming, heartbeat older than *cutoff*, and last_seq matched).
        """
        try:
            cur = self._conn.execute(
                "UPDATE replay_sessions SET state = ?, expires_at = ?, "
                "heartbeat = ?, last_seq = ? "
                "WHERE session_id = ? AND state = ? AND heartbeat < ? AND last_seq = ?",
                (
                    STATE_INTERRUPTED,
                    expires_at,
                    heartbeat,
                    events[-1][0] if events else expected_last_seq,
                    session_id,
                    STATE_STREAMING,
                    cutoff,
                    expected_last_seq,
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return False
            self._conn.executemany(
                "INSERT INTO replay_events (session_id, seq, payload) VALUES (?, ?, ?)",
                [(session_id, seq, payload) for seq, payload in events],
            )
            low_water = (
                events[-1][0] if events else expected_last_seq
            ) - self._max_events
            if low_water > 0:
                self._conn.execute(
                    "DELETE FROM replay_events WHERE session_id = ? AND seq <= ?",
                    (session_id, low_water),
                )
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def touch_heartbeats(self, session_ids: list[str], heartbeat: float) -> None:
        if not session_ids:
            return
        self._conn.executemany(
            "UPDATE replay_sessions SET heartbeat = ? "
            "WHERE session_id = ? AND state = ?",
            [(heartbeat, sid, STATE_STREAMING) for sid in session_ids],
        )
        self._conn.commit()

    def prune_expired(self, now: float) -> None:
        cur = self._conn.execute(
            "SELECT session_id FROM replay_sessions "
            "WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        ids = [row[0] for row in cur.fetchall()]
        for sid in ids:
            self._conn.execute("DELETE FROM replay_events WHERE session_id = ?", (sid,))
            self._conn.execute(
                "DELETE FROM replay_sessions WHERE session_id = ?", (sid,)
            )
        self._conn.commit()

    def count_sessions(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM replay_sessions")
        return int(cur.fetchone()[0])

    def oldest_terminal_session(self) -> str | None:
        cur = self._conn.execute(
            "SELECT session_id FROM replay_sessions "
            "WHERE state IN (?, ?, ?) AND expires_at IS NOT NULL "
            "ORDER BY expires_at ASC LIMIT 1",
            (STATE_COMPLETE, STATE_FAILED, STATE_INTERRUPTED),
        )
        row = cur.fetchone()
        return row[0] if row is not None else None

    def delete_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM replay_events WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "DELETE FROM replay_sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Live (in-process) pump coordination
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LiveSession:
    """Condition variable for subscribers attached to a local pump."""

    condition: asyncio.Condition
    seq: int = 0
    state: str = STATE_STREAMING


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class StreamReplayManager:
    """Coordinates replay sessions across a store and local pump tasks.

    Args:
        settings: Replay settings.
        data_dir: Directory for ``stream_replay.db``.  When omitted or when
            ``settings.persist`` is False, an in-memory store is used
            (same-process replays only).
    """

    def __init__(
        self,
        settings: StreamReplaySettings | None = None,
        *,
        data_dir: str | None = None,
    ) -> None:
        self.settings = settings or StreamReplaySettings()
        if self.settings.persist and data_dir:
            self._store: ReplayStore = SqliteReplayStore(
                data_dir, max_events=self.settings.max_events
            )
            self.persistent = True
        else:
            if self.settings.persist:
                logger.info(
                    "stream replay: no data directory configured; "
                    "using an in-memory replay cache (single process only)"
                )
            self._store = MemoryReplayStore(max_events=self.settings.max_events)
            self.persistent = False
        self._live: dict[str, _LiveSession] = {}
        self._maintenance_task: asyncio.Task[None] | None = None
        self._pumps: set[asyncio.Task[None]] = set()
        self._closed = False
        # One sqlite3.Connection is shared across to_thread workers;
        # serialise store access so concurrent pumps/subscribers do not
        # interleave cursors on the same connection.
        self._store_lock = asyncio.Lock()

    async def _store_call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous store method under the serialisation lock."""
        async with self._store_lock:
            return await asyncio.to_thread(func, *args, **kwargs)

    # -- lifecycle -------------------------------------------------------

    def track_pump(self, task: asyncio.Task[None]) -> None:
        """Register a detached upstream pump task for shutdown cancellation."""
        self._pumps.add(task)
        task.add_done_callback(self._pumps.discard)

    async def start(self) -> None:
        """Start the heartbeat/pruning maintenance loop."""
        if self._maintenance_task is not None:
            return
        self._maintenance_task = asyncio.create_task(self._maintain())

    async def aclose(self) -> None:
        """Cancel pumps/maintenance and close the backing store."""
        self._closed = True
        for task in self._pumps:
            task.cancel()
        if self._pumps:
            await asyncio.gather(*self._pumps, return_exceptions=True)
        self._pumps.clear()
        task = self._maintenance_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        self._store.close()

    async def _maintain(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self.settings.heartbeat_interval)
                now = time.time()
                live_ids = list(self._live.keys())
                await self._store_call(self._store.touch_heartbeats, live_ids, now)
                await self._store_call(self._store.prune_expired, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("stream replay maintenance failed", exc_info=True)

    # -- pump-side API ---------------------------------------------------

    async def create_session(self, source_provider: str, model: str) -> str | None:
        """Register a new streaming session.

        Returns the session id, or ``None`` when the session cap is reached
        even after evicting expired/old terminal sessions.  ``None`` tells
        the caller to fall back to legacy (non-replayable) streaming.
        """
        now = time.time()
        try:
            async with self._store_lock:
                await asyncio.to_thread(self._store.prune_expired, now)
                count = await asyncio.to_thread(self._store.count_sessions)
                if count >= self.settings.max_sessions:
                    oldest = await asyncio.to_thread(
                        self._store.oldest_terminal_session
                    )
                    if oldest is not None:
                        await asyncio.to_thread(self._store.delete_session, oldest)
                    else:
                        logger.warning(
                            "stream replay session cap (%d) reached; "
                            "falling back to non-replayable stream",
                            self.settings.max_sessions,
                        )
                        return None
                session_id = new_session_id()
                record = SessionRecord(
                    id=session_id,
                    source_provider=source_provider,
                    model=model,
                    state=STATE_STREAMING,
                    last_seq=0,
                    created_at=now,
                    expires_at=None,
                    heartbeat=now,
                )
                await asyncio.to_thread(self._store.create_session, record)
            self._live[session_id] = _LiveSession(condition=asyncio.Condition())
            return session_id
        except Exception:
            logger.warning("failed to create stream replay session", exc_info=True)
            return None

    async def append_messages(self, session_id: str, messages: list[str]) -> None:
        """Persist and publish formatted SSE messages from the pump."""
        live = self._live.get(session_id)
        if live is None or not messages:
            return
        async with live.condition:
            assigned: list[tuple[int, str]] = []
            for payload in messages:
                live.seq += 1
                assigned.append((live.seq, payload))
            truncated = live.seq > self.settings.max_events
        await self._store_call(
            self._store.append_events,
            session_id,
            assigned,
            heartbeat=time.time(),
            truncated=truncated,
        )
        async with live.condition:
            live.condition.notify_all()

    async def finish(self, session_id: str, *, state: str = STATE_COMPLETE) -> None:
        """Mark a pumped session terminal and publish the final state."""
        live = self._live.get(session_id)
        now = time.time()
        try:
            await self._store_call(
                self._store.finish,
                session_id,
                state,
                expires_at=now + self.settings.ttl_seconds,
                heartbeat=now,
            )
        except Exception:
            logger.debug("failed to finalize replay session %s", session_id)
        if live is not None:
            async with live.condition:
                live.state = state
                live.condition.notify_all()
            self._live.pop(session_id, None)

    # -- subscriber-side API ---------------------------------------------

    async def validate_resume(self, session_id: str, cursor: int) -> SessionRecord:
        """Validate a resume request before any SSE bytes are sent.

        Raises :class:`ReplayRejected` with an appropriate HTTP status.
        """
        now = time.time()
        record = await self._store_call(self._store.get_session, session_id)
        if record is None:
            raise ReplayRejected(
                410,
                "session_not_found",
                "Stream replay session not found or expired; "
                "please start a new request",
            )

        # A foreign streaming session with a stale heartbeat belongs to a
        # dead pump process.  Try to atomically claim it (synthesizing the
        # interrupted tail) before range checks, so cursors at the new tip
        # are evaluated against the final record.
        if (
            record.state == STATE_STREAMING
            and session_id not in self._live
            and now - record.heartbeat > self.settings.orphan_timeout
        ):
            record = await self._claim_orphan_session(session_id, record, now)

        if record.state in _TERMINAL_STATES and record.expires_at is not None:
            if record.expires_at < now:
                await self._store_call(self._store.delete_session, session_id)
                raise ReplayRejected(
                    410,
                    "session_expired",
                    "Stream replay cache has expired; please start a new request",
                )

        if cursor > record.last_seq:
            raise ReplayRejected(
                410,
                "cursor_out_of_range",
                f"Replay cursor {cursor} is beyond the stream length "
                f"{record.last_seq}; please start a new request",
            )

        if cursor == record.last_seq and record.state in _TERMINAL_STATES:
            raise ReplayRejected(
                410,
                "session_complete",
                "Stream replay session already completed at this cursor; "
                "please start a new request",
            )

        _, _, min_seq = await self._store_call(self._store.snapshot, session_id, cursor)
        # The ring window already rolled past the requested cursor: events
        # the client is missing no longer exist, so a replay would have a
        # gap. ``min_seq`` equals ``last_seq`` when nothing is buffered yet.
        if record.last_seq > 0 and min_seq > cursor + 1:
            raise ReplayRejected(
                410,
                "cursor_evicted",
                "Events around the requested replay cursor were evicted; "
                "please start a new request",
            )
        return record

    def subscribe(self, session_id: str, cursor: int) -> Any:
        """Create an async iterator yielding encoded SSE messages after *cursor*.

        Callers must validate via :meth:`validate_resume` first.  The
        iterator terminates cleanly once the session reaches a terminal
        state.  Yields fully formatted SSE text (with the ``id:`` prefix).

        Implemented as a native async generator so that disconnecting
        clients (``aclose()``) cancel an in-flight condition wait cleanly.
        """
        return self._subscribe(session_id, cursor)

    async def _subscribe(self, session_id: str, cursor: int) -> Any:
        pending: deque[StoredEvent] = deque()
        seq = cursor
        while True:
            while not pending:
                record, events = await self._next_snapshot(session_id, seq)
                if record is None:
                    return

                if events:
                    pending.extend(events)
                    break

                if record.state in _TERMINAL_STATES:
                    return

                live = self._live.get(session_id)
                if live is not None:
                    # Attached to the local pump: wake immediately on append.
                    async with live.condition:
                        if live.seq == seq and live.state == STATE_STREAMING:
                            await live.condition.wait()
                    continue

                # Pumped by another process (or the process died): poll the
                # shared store.  An orphan claim converts the session to a
                # terminal state, which the next snapshot observes.
                now = time.time()
                if now - record.heartbeat > self.settings.orphan_timeout:
                    await self._claim_orphan_session(session_id, record, now)
                else:
                    await asyncio.sleep(self.settings.poll_interval)

            event = pending.popleft()
            seq = event.seq
            yield encode_message(session_id, event.seq, event.payload)

    async def _claim_orphan_session(
        self, session_id: str, record: SessionRecord, now: float
    ) -> SessionRecord:
        """Synthesize an interrupted tail and atomically claim an orphan.

        Only one process wins the conditional update; losers re-read the
        record claimed by the winner.
        """
        messages = _build_interruption_messages(record.source_provider)
        events = [
            (record.last_seq + offset, payload)
            for offset, payload in enumerate(messages, start=1)
        ]
        expires_at = now + self.settings.ttl_seconds
        async with self._store_lock:
            claimed = await asyncio.to_thread(
                self._store.claim_orphan,
                session_id,
                expected_last_seq=record.last_seq,
                cutoff=now - self.settings.orphan_timeout,
                events=events,
                expires_at=expires_at,
                heartbeat=now,
            )
            fresh = await asyncio.to_thread(self._store.get_session, session_id)
        if claimed:
            logger.info(
                "Claimed orphaned stream replay session %s (last heartbeat %ss ago)",
                session_id,
                round(now - record.heartbeat, 1),
            )
        return fresh or record

    # -- internals -------------------------------------------------------

    async def _next_snapshot(
        self, session_id: str, cursor: int
    ) -> tuple[SessionRecord | None, list[StoredEvent]]:
        record, events, _min_seq = await self._store_call(
            self._store.snapshot, session_id, cursor
        )
        return record, events


def _build_interruption_messages(source_provider: str) -> list[str]:
    """Format the terminal notice appended when a pump process died."""
    formatter = SSE_FORMATTERS.get(source_provider)
    if formatter is None:
        return []
    reason = "stream interrupted: gateway process ended before completion"
    out = [
        formatter(event) for event in build_stream_error_events(source_provider, reason)
    ]
    if source_provider in ("openai_chat", "openai_responses", "open_responses"):
        out.append(format_sse_done())
    return out

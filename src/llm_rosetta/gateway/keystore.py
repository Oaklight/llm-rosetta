"""SQLite-backed API key storage with hash-based validation.

Keys are stored as SHA-256 hashes — plaintext is never persisted.
An in-memory cache (hash → KeyContext) keeps auth lookups at O(1)
without hitting SQLite on every request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_rosetta._vendor import aiosqlite

logger = logging.getLogger("llm-rosetta.keystore")

_DB_FILENAME = "keys.db"


@dataclass(frozen=True)
class KeyContext:
    """Per-request auth context attached via ContextVar after validation."""

    label: str
    allowed_shims: frozenset[str]
    key_hash: str = ""


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _generate_key() -> str:
    return f"rsk-{secrets.token_hex(24)}"


def _generate_id() -> str:
    return uuid.uuid4().hex[:8]


class KeyStore:
    """Async SQLite-backed API key store with in-memory validation cache.

    Use the async :meth:`create` classmethod to construct instances::

        ks = await KeyStore.create("/var/data/keys.db")

    ``validate()`` and ``has_keys()`` remain synchronous (pure cache
    lookups) so they can be called from the hot auth path without await.
    """

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        self._db_path: Path = Path()
        self._cache: dict[str, tuple[str, KeyContext]] = {}
        self._last_touch: dict[str, float] = {}

    @classmethod
    async def create(cls, db_path: str | Path) -> KeyStore:
        """Async factory: create a KeyStore with an open connection."""
        instance = cls()
        instance._db_path = Path(db_path)
        instance._db_path.parent.mkdir(parents=True, exist_ok=True)
        instance._conn = await aiosqlite.connect(str(instance._db_path))
        await instance._conn.execute("PRAGMA journal_mode=WAL")
        await instance._conn.execute("PRAGMA synchronous=NORMAL")
        await instance._conn.execute(
            """CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                key_hash    TEXT NOT NULL UNIQUE,
                label       TEXT NOT NULL DEFAULT '',
                allowed_shims TEXT NOT NULL DEFAULT '["*"]',
                created     TEXT NOT NULL,
                rotated     TEXT
            )"""
        )
        await instance._conn.commit()
        await instance._migrate_last_used()
        await instance._refresh_cache()
        return instance

    async def _migrate_last_used(self) -> None:
        cursor = await self._conn.execute("PRAGMA table_info(api_keys)")
        cols = {r[1] for r in await cursor.fetchall()}
        if "last_used" not in cols:
            await self._conn.execute("ALTER TABLE api_keys ADD COLUMN last_used TEXT")
            await self._conn.commit()

    async def _refresh_cache(self) -> None:
        """Rebuild the in-memory hash → (id, KeyContext) lookup."""
        cursor = await self._conn.execute(
            "SELECT id, key_hash, label, allowed_shims FROM api_keys"
        )
        rows = await cursor.fetchall()
        cache: dict[str, tuple[str, KeyContext]] = {}
        for row_id, key_hash, label, shims_json in rows:
            try:
                shims = frozenset(json.loads(shims_json))
            except (json.JSONDecodeError, TypeError):
                shims = frozenset({"*"})
            cache[key_hash] = (
                row_id,
                KeyContext(label=label, allowed_shims=shims),
            )
        self._cache = cache

    def validate(self, raw_key: str) -> tuple[str, KeyContext] | None:
        """Validate a raw API key and return ``(key_id, context)``, or None.

        Synchronous — uses the in-memory cache only.
        """
        return self._cache.get(_hash_key(raw_key))

    def touch(self, key_id: str, interval: float = 300.0) -> None:
        """Record a last-used timestamp, throttled to one write per *interval* seconds.

        Remains synchronous — schedules the async write as a fire-and-forget
        task so it can be called from the sync auth-hook hot path without await.
        """
        now = time.monotonic()
        if now - self._last_touch.get(key_id, 0) < interval:
            return
        self._last_touch[key_id] = now

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._touch_write(key_id))

    async def _touch_write(self, key_id: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            await self._conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE id = ?", (ts, key_id)
            )
            await self._conn.commit()
        except Exception as exc:
            logger.debug("touch write failed for key %s: %s", key_id, exc)

    async def backfill_last_used(self, request_log_db_path: str | Path) -> int:
        """Backfill last_used from request log for keys that have no value yet."""
        import sqlite3 as _sqlite3

        try:
            log_conn = _sqlite3.connect(str(request_log_db_path))
        except Exception:
            return 0
        updated = 0
        try:
            cursor = await self._conn.execute(
                "SELECT id, label, last_used FROM api_keys"
            )
            rows = await cursor.fetchall()
            for row_id, label, last_used in rows:
                if last_used or not label:
                    continue
                r = log_conn.execute(
                    "SELECT MAX(timestamp) FROM request_log WHERE api_key_label = ?",
                    (label,),
                ).fetchone()
                if r and r[0]:
                    await self._conn.execute(
                        "UPDATE api_keys SET last_used = ? WHERE id = ?",
                        (r[0], row_id),
                    )
                    updated += 1
            if updated:
                await self._conn.commit()
        finally:
            log_conn.close()
        return updated

    def has_keys(self) -> bool:
        """Check if any keys exist (synchronous cache check)."""
        return bool(self._cache)

    async def create_key(
        self,
        label: str = "",
        allowed_shims: list[str] | None = None,
        manual_key: str | None = None,
    ) -> tuple[str, str]:
        """Create a new API key.

        Returns:
            (id, raw_key) — the raw key is shown once and never stored.
        """
        key_id = _generate_id()
        raw_key = manual_key or _generate_key()
        key_hash = _hash_key(raw_key)
        shims = json.dumps(allowed_shims or ["*"])
        created = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO api_keys (id, key_hash, label, allowed_shims, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_id, key_hash, label, shims, created),
        )
        await self._conn.commit()
        await self._refresh_cache()
        return key_id, raw_key

    async def list_keys(self) -> list[dict[str, Any]]:
        """List all keys without secrets."""
        cursor = await self._conn.execute(
            "SELECT id, label, allowed_shims, created, rotated, last_used FROM api_keys"
        )
        rows = await cursor.fetchall()
        result = []
        for row_id, label, shims_json, created, rotated, last_used in rows:
            try:
                shims = json.loads(shims_json)
            except (json.JSONDecodeError, TypeError):
                shims = ["*"]
            entry: dict[str, Any] = {
                "id": row_id,
                "label": label,
                "allowed_shims": shims,
                "created": created,
            }
            if rotated:
                entry["rotated"] = rotated
            if last_used:
                entry["last_used"] = last_used
            result.append(entry)
        return result

    async def update(
        self,
        key_id: str,
        label: str | None = None,
        allowed_shims: list[str] | None = None,
    ) -> bool:
        """Update label and/or allowed_shims for a key.  Returns True if found."""
        parts: list[str] = []
        params: list[Any] = []
        if label is not None:
            parts.append("label = ?")
            params.append(label)
        if allowed_shims is not None:
            parts.append("allowed_shims = ?")
            params.append(json.dumps(allowed_shims))
        if not parts:
            return await self._key_exists(key_id)
        params.append(key_id)
        cur = await self._conn.execute(
            f"UPDATE api_keys SET {', '.join(parts)} WHERE id = ?", params
        )
        await self._conn.commit()
        if cur.rowcount == 0:
            return False
        await self._refresh_cache()
        return True

    async def delete(self, key_id: str) -> bool:
        """Delete a key by id.  Returns True if found."""
        cur = await self._conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        await self._conn.commit()
        if cur.rowcount == 0:
            return False
        await self._refresh_cache()
        return True

    async def rotate(self, key_id: str) -> str | None:
        """Rotate a key: generate new raw key, update hash.  Returns new raw key."""
        row = await self._conn.execute_fetchone(
            "SELECT id FROM api_keys WHERE id = ?", (key_id,)
        )
        if not row:
            return None
        new_key = _generate_key()
        new_hash = _hash_key(new_key)
        rotated = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "UPDATE api_keys SET key_hash = ?, rotated = ? WHERE id = ?",
            (new_hash, rotated, key_id),
        )
        await self._conn.commit()
        await self._refresh_cache()
        return new_key

    async def import_from_config(self, config_keys: list[dict[str, str]]) -> int:
        """Import plaintext keys from config into SQLite (idempotent)."""
        imported = 0
        for entry in config_keys:
            raw_key = entry.get("key", "")
            if not raw_key:
                continue
            key_hash = _hash_key(raw_key)
            await self._conn.execute(
                "INSERT OR IGNORE INTO api_keys "
                "(id, key_hash, label, allowed_shims, created) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entry.get("id", _generate_id()),
                    key_hash,
                    entry.get("label", ""),
                    '["*"]',
                    entry.get("created", ""),
                ),
            )
            cursor = await self._conn.execute("SELECT changes()")
            row = await cursor.fetchone()
            if row and row[0]:
                imported += 1
        if imported:
            await self._conn.commit()
            await self._refresh_cache()
        return imported

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass

    async def _key_exists(self, key_id: str) -> bool:
        row = await self._conn.execute_fetchone(
            "SELECT 1 FROM api_keys WHERE id = ?", (key_id,)
        )
        return row is not None

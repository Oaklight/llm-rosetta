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
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm-rosetta.keystore")

_DB_FILENAME = "keys.db"


@dataclass(frozen=True)
class KeyContext:
    """Per-request auth context attached via ContextVar after validation."""

    label: str
    allowed_shims: frozenset[str]


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _generate_key() -> str:
    return f"rsk-{secrets.token_hex(24)}"


def _generate_id() -> str:
    return uuid.uuid4().hex[:8]


class KeyStore:
    """SQLite-backed API key store with in-memory validation cache.

    Args:
        db_path: Path to the SQLite database file.  Created if missing.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS api_keys (
                id          TEXT PRIMARY KEY,
                key_hash    TEXT NOT NULL UNIQUE,
                label       TEXT NOT NULL DEFAULT '',
                allowed_shims TEXT NOT NULL DEFAULT '["*"]',
                created     TEXT NOT NULL,
                rotated     TEXT
            )"""
        )
        self._conn.commit()
        self._cache: dict[str, tuple[str, KeyContext]] = {}
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Rebuild the in-memory hash → (id, KeyContext) lookup."""
        rows = self._conn.execute(
            "SELECT id, key_hash, label, allowed_shims FROM api_keys"
        ).fetchall()
        cache: dict[str, tuple[str, KeyContext]] = {}
        for row_id, key_hash, label, shims_json in rows:
            try:
                shims = frozenset(json.loads(shims_json))
            except (json.JSONDecodeError, TypeError):
                shims = frozenset({"*"})
            cache[key_hash] = (row_id, KeyContext(label=label, allowed_shims=shims))
        self._cache = cache

    def validate(self, raw_key: str) -> KeyContext | None:
        """Validate a raw API key and return its context, or None."""
        entry = self._cache.get(_hash_key(raw_key))
        return entry[1] if entry else None

    def has_keys(self) -> bool:
        return bool(self._cache)

    def create(
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
        from datetime import datetime, timezone

        created = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO api_keys (id, key_hash, label, allowed_shims, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_id, key_hash, label, shims, created),
        )
        self._conn.commit()
        self._refresh_cache()
        return key_id, raw_key

    def list_keys(self) -> list[dict[str, Any]]:
        """List all keys without secrets."""
        rows = self._conn.execute(
            "SELECT id, label, allowed_shims, created, rotated FROM api_keys"
        ).fetchall()
        result = []
        for row_id, label, shims_json, created, rotated in rows:
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
            result.append(entry)
        return result

    def update(
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
            return self._key_exists(key_id)
        params.append(key_id)
        cur = self._conn.execute(
            f"UPDATE api_keys SET {', '.join(parts)} WHERE id = ?", params
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return False
        self._refresh_cache()
        return True

    def delete(self, key_id: str) -> bool:
        """Delete a key by id.  Returns True if found."""
        cur = self._conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            return False
        self._refresh_cache()
        return True

    def rotate(self, key_id: str) -> str | None:
        """Rotate a key: generate new raw key, update hash.  Returns new raw key or None."""
        row = self._conn.execute(
            "SELECT id FROM api_keys WHERE id = ?", (key_id,)
        ).fetchone()
        if not row:
            return None
        new_key = _generate_key()
        new_hash = _hash_key(new_key)
        from datetime import datetime, timezone

        rotated = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE api_keys SET key_hash = ?, rotated = ? WHERE id = ?",
            (new_hash, rotated, key_id),
        )
        self._conn.commit()
        self._refresh_cache()
        return new_key

    def import_from_config(self, config_keys: list[dict[str, str]]) -> int:
        """Import plaintext keys from config into SQLite (idempotent).

        Returns the number of keys newly imported.
        """
        imported = 0
        for entry in config_keys:
            raw_key = entry.get("key", "")
            if not raw_key:
                continue
            key_hash = _hash_key(raw_key)
            try:
                self._conn.execute(
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
                if self._conn.execute("SELECT changes()").fetchone()[0]:
                    imported += 1
            except sqlite3.IntegrityError:
                pass
        if imported:
            self._conn.commit()
            self._refresh_cache()
        return imported

    def close(self) -> None:
        self._conn.close()

    def _key_exists(self, key_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            is not None
        )

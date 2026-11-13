"""The keystore's last-used backfill must not read through a cleared range.

The backfill opens its own read-only connection to the gateway database, so it
never goes through :class:`VisibilityFilter`; this pins the clause it builds by
hand.  A key stamped with a request the operator already removed is a small
leak, but it is a leak, and it is the one read outside the persistence layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.observability.persistence import PersistenceManager


def _ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _entry(entry_id: str, timestamp: str, label: str) -> dict:
    return {
        "id": entry_id,
        "timestamp": timestamp,
        "model": "m",
        "source_provider": "openai",
        "target_provider": "messages",
        "is_stream": False,
        "status_code": 200,
        "duration_ms": 1.0,
        "api_key_label": label,
    }


@pytest.fixture
async def stores(tmp_path):
    persistence = await PersistenceManager.create(str(tmp_path), success_max=1000)
    keystore = await KeyStore.create(tmp_path / "keys.db")
    yield persistence, keystore
    await persistence.close()
    await keystore.close()


async def test_backfill_ignores_a_hidden_request(stores):
    persistence, keystore = stores
    await keystore.create_key(label="k1")
    visible = _ago(minutes=50)
    hidden = _ago(minutes=25)
    await persistence.insert_log_entries(
        [_entry("old", visible, "k1"), _entry("new", hidden, "k1")]
    )
    await persistence.add_hidden_range(
        "request_log", _ago(minutes=30), _ago(minutes=20), job_id="j1"
    )

    updated = await keystore.backfill_last_used(persistence.db_path)

    assert updated == 1
    keys = await keystore.list_keys()
    entry = next(key for key in keys if key["label"] == "k1")
    assert entry["last_used"] == visible

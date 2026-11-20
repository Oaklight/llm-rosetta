"""Rows that still carry a retired provider tag after the aliases were removed.

``STANDARD_SHAPES`` no longer lists ``anthropic`` and its siblings, so the total
repair cannot read such a row.  The gateway keeps serving and says so: a warning
names the retired tags and ``llm-rosetta-migrate --db``.
"""

from __future__ import annotations

import logging

import pytest

from llm_rosetta.observability.usage import (
    STANDARD_SHAPES,
    cache_is_additive,
    shape_for,
)
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.observability.request_log import RequestLogEntry

LOGGER = "llm-rosetta.observability"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture()
def warnings_seen():
    handler = _Records()
    logger = logging.getLogger(LOGGER)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


def _entry(target: str) -> dict:
    return RequestLogEntry.create(
        model="m",
        source_provider="chat_completions",
        target_provider=target,
        is_stream=False,
        status_code=200,
        duration_ms=1.0,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=100,
        total_tokens=15,
    ).to_dict()


@pytest.mark.parametrize(
    "old", ["openai_chat", "openai_responses", "anthropic", "google"]
)
def test_a_retired_tag_is_not_a_usage_standard(old: str) -> None:
    assert old not in STANDARD_SHAPES
    assert shape_for(old) is None
    assert cache_is_additive(old) is False


@pytest.mark.asyncio
async def test_an_unmigrated_row_is_left_alone_and_reported(tmp_path, warnings_seen):
    pm = await PersistenceManager.create(str(tmp_path))
    try:
        await pm.insert_log_entries([_entry("anthropic"), _entry("messages")])
        await pm.backfill_total_tokens()
        rows = await pm._conn.execute_fetchall(
            "SELECT target_provider, total_tokens FROM request_log "
            "ORDER BY target_provider"
        )
    finally:
        await pm.close()

    totals = dict((row[0], row[1]) for row in rows)
    # The current tag's derived total is repaired to include the cache; the
    # retired tag's is not, which is why the migration has to run.
    assert totals["messages"] == 115
    assert totals["anthropic"] == 15
    message = next(m for m in warnings_seen if "retired provider tag" in m)
    assert "1 request_log rows" in message
    assert "llm-rosetta-migrate --db" in message


@pytest.mark.asyncio
async def test_a_migrated_ledger_is_silent(tmp_path, warnings_seen):
    pm = await PersistenceManager.create(str(tmp_path))
    try:
        await pm.insert_log_entries([_entry("messages")])
        await pm.backfill_total_tokens()
    finally:
        await pm.close()

    assert not [m for m in warnings_seen if "retired provider tag" in m]


@pytest.mark.asyncio
async def test_the_embedding_provider_tag_is_reported_too(tmp_path, warnings_seen):
    """Typed requests were logged under their model type; ``embedding`` is retired."""
    pm = await PersistenceManager.create(str(tmp_path))
    try:
        await pm.insert_log_entries([_entry("embedding"), _entry("messages")])
        count = await pm._warn_about_retired_tags()
    finally:
        await pm.close()

    assert count == 1
    message = next(m for m in warnings_seen if "retired provider tag" in m)
    assert "embedding" in message
    assert "llm-rosetta-migrate --db" in message


@pytest.mark.asyncio
async def test_a_migrated_embeddings_row_is_not_reported(tmp_path, warnings_seen):
    pm = await PersistenceManager.create(str(tmp_path))
    try:
        await pm.insert_log_entries([_entry("embeddings")])
        count = await pm._warn_about_retired_tags()
    finally:
        await pm.close()

    assert count == 0

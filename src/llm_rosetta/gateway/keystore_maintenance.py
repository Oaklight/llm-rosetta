"""Bring a ``keys.db`` up to the schema the code expects, as maintenance steps.

This is the second store, and the one that proves the per-store rule is real:
``api_keys`` is a different SQLite file from the request log, opened by a
different class, with its own registry and its own ledger.  Nothing here reads
the request log's store, and nothing there reads this one.

What it replaces is four decisions made by introspection and recorded nowhere:

- ``_migrate_last_used`` and ``_migrate_usage_bucket`` asked
  ``PRAGMA table_info(api_keys)`` whether their column was present, and had no
  way to say "already done" other than "the column is there";
- the bucket backfill lived at the tail of the second of those, so a DDL helper
  was also a data repair and neither was reported;
- nothing logged any of it, so a store that had been brought forward looked
  exactly like one that had not.

Now each is a step under :attr:`Policy.MANDATORY <...maintenance.Policy>` — the
whole open is mandatory here, as it was before: an exception out of ``create()``
has always aborted startup, and a store whose schema is not what this code
expects is one the gateway must not authenticate against.

The ledger is a ``schema_version`` table in ``keys.db``, the same shape the
request log uses.  A file written before the ledger existed has no rows in it, so
it is adopted: the columns it already has say which versions it already has.
That inference cannot be replaced by a record — a store that predates the record
is exactly the case where there is none — but it is now one named step with a
result, and it only ever runs for a store that has no ledger at all.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any

from llm_rosetta.observability.maintenance import (
    Policy,
    Registry,
    Step,
    StepContext,
)

logger = logging.getLogger(__name__)

#: What the admin surface and the log lines call this registry.
NAME = "api_keys"

#: Which column came with which version: the evidence the adoption step reads.
#: The last column of a version, exactly as the request log's adoption step
#: checks v1's last column.  ``test_every_version_has_adoption_evidence`` fails
#: if a version is added to ``KeyStore.VERSIONS`` without an entry here.
COLUMN_OF: dict[int, str] = {
    1: "last_used",
    2: "usage_bucket",
}

_CREATE_LEDGER = """CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)"""


async def _recorded(conn: Any, version: int) -> bool:
    """Whether this store has recorded one version.

    Per version, not "up to": the adoption step below can legitimately record a
    *higher* version than one that is still missing — a ``keys.db`` with
    ``usage_bucket`` but without ``last_used`` is exactly that store — so a step
    asking ``version <= MAX(version)`` would skip the migration whose column its
    file does not have and then report the store as current.  A ladder is honest
    only when the versions below it are known to be there.

    Measured on that store, the two readings side by side::

        per version   versions=[1, 2]   v1 changed 2   last_used present
        as a ladder   versions=[2]      v1 changed 0   last_used still absent

    The second row is a store reported as current with a column missing.
    """
    row = await conn.execute_fetchone(
        "SELECT 1 FROM schema_version WHERE version = ?", (version,)
    )
    return row is not None


async def _has_ledger(conn: Any) -> bool:
    """Whether this store records anything at all."""
    row = await conn.execute_fetchone("SELECT 1 FROM schema_version LIMIT 1")
    return row is not None


async def _adopt_pre_registry_schema(context: StepContext) -> int:
    """Record the versions whose columns a pre-ledger store already has.

    ``keys.db`` was written for a long time without any record of what had been
    applied to it, so a file from that era is recognised by its columns: the
    presence of ``last_used`` means v1's work is in it, ``usage_bucket`` means
    v2's.  The alternative — running every migration and swallowing the
    duplicate-column error — would work by luck and would leave the ledger
    empty, which is what this step exists to prevent.

    Returns the number of versions recorded, so a store that was adopted says
    so in its own result.
    """
    conn = context.keystore._conn
    await conn.execute(_CREATE_LEDGER)
    if await _has_ledger(conn):
        return 0
    columns = await _columns(conn)
    recorded = 0
    for version, column in sorted(COLUMN_OF.items()):
        if column in columns:
            await conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) "
                "VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            recorded += 1
    if recorded:
        await conn.commit()
        logger.debug("Adopted a pre-ledger keys.db at version %d", recorded)
    return recorded


async def _apply_migration(
    context: StepContext,
    *,
    version: int,
    description: str,
    statements: list[str],
) -> int:
    """Apply one versioned migration, and say what it took to do it.

    The ledger decides *whether* this runs, one version at a time; the column
    check below keeps it safe for a store that has the column without the record
    — one adopted by an earlier release, or edited by hand.  A statement skipped
    that way did not run and is not counted, so a migration that only skips
    reports 1: the version row.

    A migration already recorded returns 0 without touching the store, which is
    what makes a second open a no-op.
    """
    conn = context.keystore._conn
    if await _recorded(conn, version):
        return 0
    logger.info("Applying keys.db migration v%d: %s", version, description)
    columns = await _columns(conn)
    executed = 0
    for sql in statements:
        # ``ADD COLUMN`` has no ``IF NOT EXISTS``; the columns are read once
        # rather than per statement, so the guard costs one query.
        if any(f"ADD COLUMN {column}" in sql for column in columns):
            continue
        await conn.execute(sql)
        executed += 1
    await conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, datetime.now(timezone.utc).isoformat()),
    )
    await conn.commit()
    logger.info("Keys.db migration v%d applied", version)
    return executed + 1


async def _backfill_usage_bucket(context: StepContext) -> int:
    """Give every key the usage bucket its hash implies.

    Rows written before the bucket column existed carry no bucket, and the
    bucket is what the request log attributes usage by.  Idempotent by its own
    ``WHERE`` clause: once every row has one, this matches nothing.
    """
    from .keystore import key_bucket

    conn = context.keystore._conn
    cursor = await conn.execute(
        "SELECT id, key_hash FROM api_keys WHERE usage_bucket IS NULL"
    )
    rows = await cursor.fetchall()
    for row_id, key_hash in rows:
        await conn.execute(
            "UPDATE api_keys SET usage_bucket = ? WHERE id = ?",
            (key_bucket(key_hash), row_id),
        )
    if rows:
        await conn.commit()
    return len(rows)


async def _columns(conn: Any) -> set[str]:
    """The column names ``api_keys`` has, as a set."""
    cursor = await conn.execute("PRAGMA table_info(api_keys)")
    return {row[1] for row in await cursor.fetchall()}


def build_registry() -> Registry:
    """This store's steps, under the policy a schema failure deserves.

    A fresh registry per open, because a registry keeps the results of its run
    and those belong to one ``KeyStore``: two managers, or a test and the app,
    must not read each other's.

    The version table and the file name are read from the store here rather than
    at module scope: this module is imported by :mod:`~llm_rosetta.gateway.keystore`,
    so naming them at the top would be a cycle.
    """
    from .keystore import VERSIONS, _DB_FILENAME

    steps = [
        Step(
            name="api_keys.adopt_pre_registry_schema",
            description=(
                "record the versions a keys.db already has, for one written "
                "before there was a ledger"
            ),
            store=_DB_FILENAME,
            run=_adopt_pre_registry_schema,
        )
    ]
    steps.extend(
        Step(
            name=f"api_keys.v{version}",
            description=description,
            store=_DB_FILENAME,
            run=partial(
                _apply_migration,
                version=version,
                description=description,
                statements=statements,
            ),
        )
        for version, description, statements in VERSIONS
    )
    steps.append(
        Step(
            name="api_keys.backfill_usage_bucket",
            description=(
                "give every key the usage bucket its hash implies, for rows "
                "written before the column existed"
            ),
            store=_DB_FILENAME,
            run=_backfill_usage_bucket,
        )
    )
    return Registry(name=NAME, steps=steps, policy=Policy.MANDATORY)

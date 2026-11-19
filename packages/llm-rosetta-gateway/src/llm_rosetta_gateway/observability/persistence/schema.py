"""Schema creation, versioned migrations and schema-version reads."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from typing import Any

from ..maintenance import Policy, Registry, Step, StepContext
from ._common import (
    _DB_FILENAME,
    logger,
)
from ._state import _StoreState


class _SchemaMixin(_StoreState):
    """Schema creation, versioned migrations and schema-version reads."""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _init_tables(self) -> None:
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_log (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                model           TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                is_stream       INTEGER NOT NULL,
                status_code     INTEGER NOT NULL,
                duration_ms     REAL NOT NULL,
                error_detail    TEXT,
                api_key_label   TEXT,
                target_provider_name TEXT,
                client_ip       TEXT,
                user_agent      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rl_timestamp
                ON request_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_rl_status
                ON request_log(status_code);
            CREATE INDEX IF NOT EXISTS idx_rl_success_ts
                ON request_log(timestamp ASC) WHERE status_code < 400;
            CREATE TABLE IF NOT EXISTS metrics (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dump_bodies (
                hash        TEXT PRIMARY KEY,
                data        BLOB NOT NULL,
                orig_bytes  INTEGER NOT NULL,
                created     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS error_dumps (
                id                  TEXT PRIMARY KEY,
                request_log_id      TEXT REFERENCES request_log(id) ON DELETE SET NULL,
                timestamp           TEXT NOT NULL,
                model               TEXT,
                source_provider     TEXT,
                target_provider     TEXT,
                provider_name       TEXT,
                status_code         INTEGER,
                error_phase         TEXT,
                body_hash           TEXT,
                response_text       TEXT,
                upstream_url        TEXT,
                converted_body_hash TEXT,
                tool_name_collisions TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ed_timestamp
                ON error_dumps(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_ed_request_log
                ON error_dumps(request_log_id);

            CREATE TABLE IF NOT EXISTS ops_log (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                severity    TEXT NOT NULL,
                message     TEXT NOT NULL,
                details     TEXT,
                source      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ol_timestamp
                ON ops_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_ol_event_type
                ON ops_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_ol_severity_ts
                ON ops_log(severity, timestamp DESC);

            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT NOT NULL
            );
        """)

    # ------------------------------------------------------------------
    # Versioned migrations
    # ------------------------------------------------------------------

    # Each entry: (version, description, sql_statements)
    # Migrations are applied in order; each bumps the schema version.
    _MIGRATIONS: list[tuple[int, str, list[str]]] = [
        (
            1,
            "add request_log extra columns",
            [
                "ALTER TABLE request_log ADD COLUMN target_provider_name TEXT",
                "ALTER TABLE request_log ADD COLUMN client_ip TEXT",
                "ALTER TABLE request_log ADD COLUMN profile TEXT",
                "ALTER TABLE request_log ADD COLUMN input_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN output_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN total_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN cache_read_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN cache_creation_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN reasoning_tokens INTEGER",
            ],
        ),
        (
            2,
            "add error_dumps indexes on model and error_phase",
            [
                "CREATE INDEX IF NOT EXISTS idx_ed_model ON error_dumps(model)",
                "CREATE INDEX IF NOT EXISTS idx_ed_error_phase ON error_dumps(error_phase)",
            ],
        ),
        (
            3,
            "add error_dumps.tool_name_collisions",
            [
                "ALTER TABLE error_dumps ADD COLUMN tool_name_collisions TEXT",
            ],
        ),
        (
            4,
            "add request_log.api_key_bucket for per-key usage attribution",
            [
                "ALTER TABLE request_log ADD COLUMN api_key_bucket TEXT",
                "CREATE INDEX IF NOT EXISTS idx_rl_api_key_bucket "
                "ON request_log(api_key_bucket, timestamp)",
            ],
        ),
        (
            5,
            "index request_log.api_key_label",
            [
                "CREATE INDEX IF NOT EXISTS idx_rl_api_key_label "
                "ON request_log(api_key_label)",
            ],
        ),
        (
            6,
            "add request_log.model_type for the non-chat model types",
            [
                # The default is what keeps every row that predates the column
                # meaning chat, and SQLite fills existing rows from it on read,
                # so the hottest table is not rewritten to add it.
                "ALTER TABLE request_log ADD COLUMN model_type TEXT DEFAULT 'llm'",
            ],
        ),
        (
            7,
            "add request_log.request_id so an X-Request-Id resolves to a row",
            [
                # ``id`` is the entry id, not the id the client sees in
                # ``X-Request-Id``; without this column the only way to find a
                # failing request was grepping the container log (#1217).
                "ALTER TABLE request_log ADD COLUMN request_id TEXT",
                "CREATE INDEX IF NOT EXISTS idx_rl_request_id "
                "ON request_log(request_id)",
            ],
        ),
        (
            8,
            "add request_log.ttft_ms so a stream's generation rate is measurable",
            [
                # The stream path already measured time to first chunk; it
                # kept it in the profile JSON, where no query could reach it
                # without parsing every row.
                "ALTER TABLE request_log ADD COLUMN ttft_ms REAL",
            ],
        ),
        (
            9,
            "rename request_log.model_type 'embedding' to 'embeddings'",
            [
                # The category is spelled like its endpoint now, and the
                # usage report groups by this column: without the rewrite one
                # model type would report as two rows, split at the upgrade.
                "UPDATE request_log SET model_type = 'embeddings' "
                "WHERE model_type = 'embedding'",
            ],
        ),
        (
            10,
            "rename request_log.model_type 'decision' to 'decisions'",
            [
                "UPDATE request_log SET model_type = 'decisions' "
                "WHERE model_type = 'decision'",
            ],
        ),
        (
            11,
            "add request_log.user_agent so a decision names its client",
            [
                # The routing panel's decision stream shows who called; a key
                # label is one name per key, while the agent CLI behind it is
                # only visible in the header the client sends.
                "ALTER TABLE request_log ADD COLUMN user_agent TEXT",
            ],
        ),
        (
            12,
            "add request_log.conversation_id so the routing page groups by "
            "conversation",
            [
                # The panel folded rows by (client, User-Agent), so concurrent
                # sessions on one key collapsed into one line.  The id is the
                # digest the routing stages already derive for the upstream
                # session header; historical rows stay NULL and are read the
                # old way.  No index: the panel folds the newest few hundred
                # rows of one model, which the timestamp index already serves.
                "ALTER TABLE request_log ADD COLUMN conversation_id TEXT",
            ],
        ),
        (
            13,
            "add request_log.upstream_model so a row names the lane that served",
            [
                # One model name may route to several upstream models of one
                # provider (#1430); without this column two lanes are one
                # provider name in every row, and the operator cannot tell the
                # cheap one from the fast one.  The client's own ``model`` is
                # unchanged; this is what the request was sent as upstream.
                # NULL on a row written before the column existed, and on a
                # lane that names no alias.
                "ALTER TABLE request_log ADD COLUMN upstream_model TEXT",
            ],
        ),
        (
            14,
            "add hidden_ranges so a cleared range disappears before it is deleted",
            [
                # A clear is O(1): record the range, stop returning its rows,
                # and let a background job delete them.  The bounds use a
                # sentinel for "open" rather than a NULL, so the predicate
                # needs no COALESCE and the planner keeps the timestamp index.
                "CREATE TABLE IF NOT EXISTS hidden_ranges ("
                "    id         TEXT PRIMARY KEY,"
                "    table_name TEXT NOT NULL,"
                "    start_ts   TEXT NOT NULL,"
                "    end_ts     TEXT NOT NULL,"
                "    job_id     TEXT NOT NULL,"
                "    created_at TEXT NOT NULL"
                ")",
                "CREATE INDEX IF NOT EXISTS idx_hidden_ranges_table "
                "ON hidden_ranges(table_name, start_ts, end_ts)",
                "CREATE INDEX IF NOT EXISTS idx_hidden_ranges_job "
                "ON hidden_ranges(job_id)",
            ],
        ),
        (
            15,
            "add jobs so a destructive admin operation can run in the background",
            [
                # One envelope for every kind of background work: the data
                # operations now, Batch API submissions later.  A lease rather
                # than a boolean "running" is what makes a second process safe
                # when one exists; the dedupe index keeps a repeated admin
                # click from queueing the same cleanup twice.
                "CREATE TABLE IF NOT EXISTS jobs ("
                "    id                TEXT PRIMARY KEY,"
                "    kind              TEXT NOT NULL,"
                "    payload           TEXT NOT NULL,"
                "    lane              TEXT NOT NULL DEFAULT 'sqlite',"
                "    status            TEXT NOT NULL,"
                "    priority          INTEGER NOT NULL DEFAULT 100,"
                "    created_at        TEXT NOT NULL,"
                "    started_at        TEXT,"
                "    finished_at       TEXT,"
                "    attempts          INTEGER NOT NULL DEFAULT 0,"
                "    max_attempts      INTEGER NOT NULL DEFAULT 3,"
                "    lease_owner       TEXT,"
                "    lease_expires_at  TEXT,"
                "    heartbeat_at      TEXT,"
                "    cancel_requested  INTEGER NOT NULL DEFAULT 0,"
                "    progress_done     INTEGER NOT NULL DEFAULT 0,"
                "    progress_total    INTEGER,"
                "    progress_unit     TEXT,"
                "    result            TEXT,"
                "    error             TEXT,"
                "    hidden_range_ids  TEXT NOT NULL DEFAULT '[]',"
                "    dedupe_key        TEXT"
                ")",
                "CREATE INDEX IF NOT EXISTS idx_jobs_claim "
                "ON jobs(status, lane, priority, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC)",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedupe "
                "ON jobs(dedupe_key) WHERE dedupe_key IS NOT NULL "
                "AND status IN ('queued', 'running')",
            ],
        ),
        (
            # The usage report aggregates the widest window and then folds
            # dozens of conditional sums.  A covering index lets SQLite read the
            # whole answer from the index instead of the table: measured on a
            # 200 000-row log, 429 ms became 226 ms (#1522).
            16,
            "cover the per-key usage aggregate with one index",
            [
                "CREATE INDEX IF NOT EXISTS idx_rl_usage ON request_log("
                "timestamp, api_key_bucket, api_key_label, model_type, "
                "input_tokens, output_tokens, total_tokens)",
            ],
        ),
    ]

    async def _run_migrations(self) -> None:
        """Apply pending versioned migrations, through the maintenance registry.

        Every migration is a :class:`~llm_rosetta_gateway.observability.maintenance.Step`
        under ``Policy.MANDATORY``: the SQL, the order and the
        ``schema_version`` bookkeeping are what they were, and a step that fails
        still raises out of :meth:`create`, which is what aborts a startup that
        would otherwise serve from a store of unknown shape.

        The registry stays on the manager (:attr:`maintenance`), so what the
        open actually did is readable afterwards instead of only logged.
        """
        self.maintenance = self.migration_registry()
        await self.maintenance.run(StepContext(persistence=self))

    @classmethod
    def migration_registry(cls) -> Registry:
        """A fresh registry over this store's migration steps.

        Fresh rather than shared: a registry keeps the results of its last run,
        and those belong to one store — two managers, or a test and the app,
        must not read each other's.
        """
        return Registry(
            name="request_log",
            steps=_migration_steps(cls._MIGRATIONS),
            policy=Policy.MANDATORY,
        )

    async def schema_version(self) -> int:
        """Return the current schema version number, 0 when it cannot be read."""
        version = await self.read_schema_version()
        return version if version is not None else 0

    async def read_schema_version(self) -> int | None:
        """The store's schema version, or ``None`` when there is none to read.

        :meth:`schema_version` reports 0 for both "a database with no
        ``schema_version`` table" and "a database that could not be read at
        all", which is fine for a caller that only compares numbers and wrong
        for one that must explain itself: a store whose write-ahead log cannot
        be attached reads as *empty*, so it looks like a store that was never
        migrated.  A caller about to refuse is the one that needs the
        difference.
        """
        try:
            row = await self._conn.execute_fetchone(
                "SELECT MAX(version) FROM schema_version"
            )
        except Exception:
            return None
        return row[0] if row and row[0] is not None else 0


def _migration_steps(
    migrations: Sequence[tuple[int, str, list[str]]],
) -> list[Step]:
    """This store's versioned migrations, one step each, in version order.

    The SQL is not restated here: ``PersistenceManager._MIGRATIONS`` stays the
    table and a step is what wraps one of its rows, so the two cannot drift.
    The first step is not in that table — a store written before the registry
    existed carries v1's columns and no version row, and adopting it is a step
    too (:func:`_adopt_pre_registry_schema`).
    """
    steps = [
        Step(
            name="request_log.adopt_pre_registry_schema",
            description=(
                "record v1 for a store whose columns predate the version table"
            ),
            store=_DB_FILENAME,
            run=_adopt_pre_registry_schema,
        )
    ]
    steps.extend(
        Step(
            name=f"request_log.v{version}",
            description=description,
            store=_DB_FILENAME,
            run=partial(
                _apply_migration,
                version=version,
                description=description,
                statements=statements,
            ),
        )
        for version, description, statements in migrations
    )
    return steps


async def _recorded_version(conn: Any) -> int:
    """The version the store has recorded, 0 when it has recorded none."""
    row = await conn.execute_fetchone("SELECT MAX(version) FROM schema_version")
    return row[0] if row and row[0] is not None else 0


async def _adopt_pre_registry_schema(context: StepContext) -> int:
    """Record v1 for a database whose v1 columns were added before the registry.

    The old ``_migrate_add_columns`` added the columns and recorded nothing, so
    a database it touched looks like a store at version 0 with a version 1
    schema — and every migration after it would then try to add columns that
    are already there.  The presence of v1's last column is the evidence, and
    this step is where that inference lives until a recorded step replaces it.

    Returns 1 when the version was recorded, 0 when the store had already
    recorded one or has no v1 columns to adopt.
    """
    conn = context.persistence._conn
    if await _recorded_version(conn) != 0:
        return 0
    cursor = await conn.execute("PRAGMA table_info(request_log)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "reasoning_tokens" not in columns:
        return 0
    await conn.execute(
        "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
        (1, datetime.now(timezone.utc).isoformat()),
    )
    await conn.commit()
    logger.debug("Adopted a pre-registry store as schema version 1")
    return 1


async def _apply_migration(
    context: StepContext,
    *,
    version: int,
    description: str,
    statements: Sequence[str],
) -> int:
    """Apply one versioned migration, and say what it took to do it.

    ``changed`` counts the statements that ran and the version row that records
    them; a statement skipped by the duplicate-column tolerance did not run and
    is not counted, so a migration that only skips reports 1 — the row.  A
    migration already recorded returns 0 without touching the store, which is
    what makes the registry's second run a no-op.
    """
    conn = context.persistence._conn
    if version <= await _recorded_version(conn):
        return 0
    logger.info("Applying migration v%d: %s", version, description)
    executed = 0
    for sql in statements:
        try:
            await conn.execute(sql)
        except Exception as exc:
            if "duplicate column" in str(exc).lower():
                # A column a release added without recording the version: the
                # schema is already what this migration wants, which is not the
                # same as the migration having run.
                continue
            raise
        executed += 1
    await conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, datetime.now(timezone.utc).isoformat()),
    )
    await conn.commit()
    logger.info("Migration v%d applied", version)
    return executed + 1

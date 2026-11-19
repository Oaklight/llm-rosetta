"""Error-dump rows and their deduplicated bodies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ._common import (
    _DUMP_COLUMN_NAMES,
    _DUMP_COLUMNS,
    _DUMP_ORDER,
    _RP_DEFAULTS,
    _and,
    _like_needle,
    _where,
)
from ._state import _StoreState


class _DumpsMixin(_StoreState):
    """Error-dump rows and their deduplicated bodies."""

    # ------------------------------------------------------------------
    # Error dumps
    # ------------------------------------------------------------------

    # Default retention cap for error_dumps rows.
    DEFAULT_DUMP_MAX = _RP_DEFAULTS.dump_max

    async def insert_dump_body(
        self, body_hash: str, data: bytes, orig_bytes: int
    ) -> None:
        """Insert a compressed body blob, deduplicating by hash.

        If the hash already exists the row is silently skipped.
        """

        await self._conn.execute(
            "INSERT OR IGNORE INTO dump_bodies (hash, data, orig_bytes, created) "
            "VALUES (?, ?, ?, ?)",
            (body_hash, data, orig_bytes, datetime.now(timezone.utc).isoformat()),
        )
        await self._conn.commit()

    async def insert_error_dump(
        self,
        *,
        dump_id: str,
        request_log_id: str | None,
        timestamp: str,
        model: str | None,
        source_provider: str | None,
        target_provider: str | None,
        provider_name: str | None,
        status_code: int | None,
        error_phase: str | None,
        body_hash: str | None,
        response_text: str | None,
        upstream_url: str | None,
        converted_body_hash: str | None = None,
        tool_name_collisions: str | None = None,
    ) -> None:
        """Insert an error dump record and prune if over capacity."""
        await self._conn.execute(
            "INSERT OR IGNORE INTO error_dumps "
            "(id, request_log_id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, error_phase, "
            "body_hash, response_text, upstream_url, converted_body_hash, "
            "tool_name_collisions) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dump_id,
                request_log_id,
                timestamp,
                model,
                source_provider,
                target_provider,
                provider_name,
                status_code,
                error_phase,
                body_hash,
                response_text,
                upstream_url,
                converted_body_hash,
                tool_name_collisions,
            ),
        )
        await self._conn.commit()
        await self._prune_error_dumps()

    def _dump_filter_sql(
        self,
        *,
        model: str | None = None,
        model_search: str | None = None,
        error_phase: str | None = None,
        provider: str | None = None,
        status_min: int | None = None,
        status_max: int | None = None,
        start: str | None = None,
        end: str | None = None,
        q: str | None = None,
        ids: Sequence[str] | None = None,
        before: tuple[str, str] | None = None,
    ) -> tuple[str, list[Any]]:
        """Build the ``WHERE`` clause shared by every error-dump read.

        One builder, so the page, the id list and any future read agree on
        what a filter means.  ``LIKE`` needles escape the caller's own
        wildcards: a search for ``100%`` must not match every row.

        This is also the one place a row-visibility predicate belongs once
        the store has one: every read here goes through this clause, so
        adding it in this method covers the page and the id list at once
        (``error_dump_facets`` queries the table directly and needs its own).

        *before* is a ``(timestamp, id)`` keyset cursor: it keeps the rows
        older than that position, which is a stable page boundary unlike an
        ``OFFSET`` on a table that is still being written.

        Returns:
            A ``(where_sql, params)`` tuple; ``where_sql`` is empty when no
            filter is active.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if ids is not None:
            if not ids:
                # An empty id list matches nothing.  Without this branch the
                # ``IN ()`` clause would be a syntax error instead.
                return "WHERE 0", []
            placeholders = ", ".join("?" for _ in ids)
            clauses.append(f"id IN ({placeholders})")
            params.extend(ids)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if model_search:
            clauses.append(r"model LIKE ? ESCAPE '\'")
            params.append(_like_needle(model_search))
        if error_phase:
            clauses.append("error_phase = ?")
            params.append(error_phase)
        if provider:
            clauses.append("(provider_name = ? OR target_provider = ?)")
            params.extend([provider, provider])
        if status_min is not None:
            clauses.append("status_code >= ?")
            params.append(status_min)
        if status_max is not None:
            clauses.append("status_code <= ?")
            params.append(status_max)
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)
        if before is not None:
            # Row-value comparison, spelled out for SQLite: the row is past
            # the cursor when its timestamp is older, or equal and its id
            # sorts lower — the same order ``_DUMP_ORDER`` names.
            clauses.append("(timestamp < ? OR (timestamp = ? AND id < ?))")
            params.extend([before[0], before[0], before[1]])
        if q:
            needle = _like_needle(q)
            clauses.append(
                r"(id LIKE ? ESCAPE '\' OR request_log_id LIKE ? ESCAPE '\')"
            )
            params.extend([needle, needle])

        vis_sql, vis_params = self._visible("error_dumps")
        if vis_sql:
            clauses.append(vis_sql)
            params.extend(vis_params)

        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params

    async def query_error_dumps(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        model_search: str | None = None,
        error_phase: str | None = None,
        provider: str | None = None,
        status_min: int | None = None,
        status_max: int | None = None,
        start: str | None = None,
        end: str | None = None,
        q: str | None = None,
        ids: Sequence[str] | None = None,
        before: tuple[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query error dumps with optional filters, newest first.

        The filters are applied by SQLite, so the page and its ``total``
        describe the whole store rather than the rows a browser happened to
        load.  ``start``/``end`` are inclusive UTC ISO 8601 bounds compared
        against the stored ``isoformat()`` strings.

        ``before`` pages by keyset instead of by offset: pass the last row's
        ``(timestamp, id)`` to get the rows after it.  ``total`` still
        describes the filtered set, not the remainder.

        Returns:
            A ``(entries, total)`` tuple.
        """
        where_sql, params = self._dump_filter_sql(
            model=model,
            model_search=model_search,
            error_phase=error_phase,
            provider=provider,
            status_min=status_min,
            status_max=status_max,
            start=start,
            end=end,
            q=q,
            ids=ids,
            before=before,
        )

        count_row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM error_dumps {where_sql}", params
        )
        total = count_row[0] if count_row else 0

        rows = await self._conn.execute_fetchall(
            f"SELECT {_DUMP_COLUMNS} FROM error_dumps {where_sql} "
            f"{_DUMP_ORDER} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        entries: list[dict[str, Any]] = [
            {k: v for k, v in zip(_DUMP_COLUMN_NAMES, row) if v is not None}
            for row in rows
        ]
        return entries, total

    async def query_error_dump_ids(
        self,
        *,
        model: str | None = None,
        model_search: str | None = None,
        error_phase: str | None = None,
        provider: str | None = None,
        status_min: int | None = None,
        status_max: int | None = None,
        start: str | None = None,
        end: str | None = None,
        q: str | None = None,
    ) -> list[str]:
        """Return the ids of every dump the filters keep, newest first.

        *Select all filtered* and *invert* act on the whole filtered set, and
        the list endpoint only ever returns one page of it.  This is the one
        read that spans the set, and it carries ids only — no bodies, no
        previews — so a 10 000-row match stays a few hundred kilobytes.
        """
        where_sql, params = self._dump_filter_sql(
            model=model,
            model_search=model_search,
            error_phase=error_phase,
            provider=provider,
            status_min=status_min,
            status_max=status_max,
            start=start,
            end=end,
            q=q,
        )
        rows = await self._conn.execute_fetchall(
            f"SELECT id FROM error_dumps {where_sql} {_DUMP_ORDER}",
            params,
        )
        return [row[0] for row in rows]

    async def error_dump_facets(self) -> dict[str, list[str]]:
        """Return the provider and model values present in the dump store.

        The toolbar's dropdowns list what the store has ever recorded, not
        what the current filter kept — otherwise choosing a provider would
        empty the list it was chosen from.

        This reads the table directly rather than through
        ``_dump_filter_sql``, so a row-visibility predicate has to be applied
        here as well as there.
        """
        vis_sql, vis_params = self._visible("error_dumps")
        provider_where = _and(vis_sql) if vis_sql else ""
        providers = await self._conn.execute_fetchall(
            "SELECT DISTINCT provider_name FROM error_dumps "
            f"WHERE provider_name IS NOT NULL{provider_where} "
            "ORDER BY provider_name",
            list(vis_params),
        )
        models = await self._conn.execute_fetchall(
            "SELECT DISTINCT model FROM error_dumps "
            f"WHERE model IS NOT NULL{provider_where} ORDER BY model",
            list(vis_params),
        )
        return {
            "providers": [row[0] for row in providers],
            "models": [row[0] for row in models],
        }

    async def get_error_dump(self, dump_id: str) -> dict[str, Any] | None:
        """Return a single error dump by ID, or ``None``."""
        vis_sql, vis_params = self._visible("error_dumps")
        row = await self._conn.execute_fetchone(
            f"SELECT {_DUMP_COLUMNS} FROM error_dumps WHERE id = ?{_and(vis_sql)}",
            (dump_id, *vis_params),
        )
        if row is None:
            return None
        entry: dict[str, Any] = {
            k: v for k, v in zip(_DUMP_COLUMN_NAMES, row) if v is not None
        }
        return entry

    async def get_dump_body(self, body_hash: str) -> bytes | None:
        """Return the compressed body blob for a hash, or ``None``."""
        row = await self._conn.execute_fetchone(
            "SELECT data FROM dump_bodies WHERE hash = ?", (body_hash,)
        )
        return row[0] if row else None

    async def count_error_dumps(self, *, include_hidden: bool = False) -> int:
        """Return the total number of error dump entries."""
        vis_sql, vis_params = (
            ("", []) if include_hidden else self._visible("error_dumps")
        )
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM error_dumps{_where(vis_sql)}", vis_params
        )
        return row[0] if row else 0

    async def delete_error_dump(self, dump_id: str) -> bool:
        """Delete a single error dump by ID and clean up orphaned bodies."""
        cur = await self._conn.execute(
            "DELETE FROM error_dumps WHERE id = ?", (dump_id,)
        )
        if cur.rowcount == 0:
            return False
        await self._delete_orphan_bodies()
        await self._conn.commit()
        return True

    async def clear_error_dumps(self) -> None:
        """Delete all error dumps and orphaned bodies."""
        await self._conn.execute("DELETE FROM error_dumps")
        await self._conn.execute(
            "DELETE FROM dump_bodies WHERE hash NOT IN "
            "(SELECT body_hash FROM error_dumps WHERE body_hash IS NOT NULL "
            " UNION SELECT converted_body_hash FROM error_dumps "
            " WHERE converted_body_hash IS NOT NULL)"
        )
        await self._conn.commit()

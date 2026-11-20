"""Usage aggregates, backfills and usage-column updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from llm_rosetta.formats import canonical_model_type
from llm_rosetta.observability.usage import (
    DEFAULT_MODEL_TYPE,
    LEDGER_COLUMNS,
    USAGE_WINDOWS,
)

from ._common import (
    _CACHE_ADDITIVE_TARGETS,
    _and,
    logger,
)
from ._state import _StoreState


class _UsageMixin(_StoreState):
    """Usage aggregates, backfills and usage-column updates."""

    async def backfill_provider_names(
        self, model_to_provider: Mapping[str, str]
    ) -> int:
        """Backfill target_provider_name for old entries using the model->provider mapping.

        Only updates rows where target_provider_name is NULL and the
        model exists in the current config.

        Args:
            model_to_provider: Mapping from model name to provider name
                (e.g. ``{"argo:claude-opus-4.6": "Argo Claude"}``).

        Returns:
            Number of rows updated.
        """
        if not model_to_provider:
            return 0
        total = 0
        for model_name, provider_name in model_to_provider.items():
            cursor = await self._conn.execute(
                "UPDATE request_log SET target_provider_name = ? "
                "WHERE model = ? AND target_provider_name IS NULL",
                (provider_name, model_name),
            )
            total += cursor.rowcount
        if total:
            await self._conn.commit()
        return total

    async def backfill_total_tokens(self) -> int:
        """Fill totals the writer could not, and repair derived Anthropic ones.

        The writer stores :func:`~llm_rosetta.observability.usage.token_total`
        — the provider's own total when it sends one, otherwise the additive
        sum.  Two kinds of historical row are missing that value, and only
        those two are touched here:

        - a row with no total at all, which no view can count; it is filled
          additively, the same fallback the writer applies.
        - a row from a Messages-API upstream that reports cache tokens, where
          the old writer stored input + output and left the cache out.

        A total a provider *did* report is never rewritten.  For the
        OpenAI-shaped standards that number is legitimately below the additive
        sum — their cached counters sit inside the prompt count — so adding the
        cache columns to it would inflate it.  Rewriting by that rule is what
        made the column mean one thing for the rows a restart had touched and
        another for the rest; the rule now matches the writer's, so a row's
        total survives any number of restarts unchanged.

        Returns:
            Number of rows updated.
        """
        additive = (
            "COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0) "
            "+ COALESCE(cache_read_tokens, 0) + COALESCE(cache_creation_tokens, 0)"
        )
        updated = 0

        # 1. Nothing was recorded at all.  Runs first so that step 2 sees the
        #    rows it filled rather than a NULL it would have to special-case.
        cursor = await self._conn.execute(
            f"UPDATE request_log SET total_tokens = {additive} "
            "WHERE total_tokens IS NULL "
            f"  AND ({additive}) > 0"
        )
        updated += cursor.rowcount or 0

        # 2. A derived total that left the cache out.  Restricted to the
        #    standards whose cache counters are additive; for the others a
        #    total below this sum is the provider's own answer, not a mistake.
        placeholders = ", ".join("?" * len(_CACHE_ADDITIVE_TARGETS))
        cursor = await self._conn.execute(
            f"UPDATE request_log SET total_tokens = {additive} "
            f"WHERE total_tokens < {additive} "
            "  AND (cache_read_tokens > 0 OR cache_creation_tokens > 0) "
            f"  AND target_provider IN ({placeholders})",
            _CACHE_ADDITIVE_TARGETS,
        )
        updated += cursor.rowcount or 0

        if updated:
            await self._conn.commit()
            logger.info("Backfilled total_tokens for %d rows", updated)
        return updated

    async def query_token_usage_by_day(
        self,
        *,
        days: int = 7,
        api_key_label: str | None = None,
        model_type: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate token usage grouped by day, split by model type.

        A day carries the counters over every type it saw plus the same
        counters for each type, so the panel shows the breakdown without a
        second query.  Rows are grouped by ``(date, model_type)`` and folded
        once each: a day's scalars are summed from the raw rows rather than
        from its own split, so the two readings are independently right.

        Args:
            days: Number of days to look back.
            api_key_label: Optional filter by API key label.
            model_type: Optional filter by model type.

        Returns:
            ``{"days": [...], "totals": {...}}``.  A day holds ``day``,
            ``request_count``, the token columns and ``by_model_type`` — the
            split holding only the types that day saw.  ``totals`` holds the
            counters summed over every day and that same split summed over
            every day.  Without a filter the counters cover every type, and
            with one they cover the filtered type.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        sums = ", ".join(f"COALESCE(SUM({col}), 0)" for col in LEDGER_COLUMNS)
        sql = (
            "SELECT date(timestamp) as day, "
            "model_type, "
            "COUNT(*) as request_count, "
            f"{sums} "
            "FROM request_log WHERE timestamp >= ?"
        )
        params: list[Any] = [cutoff]
        if api_key_label:
            sql += " AND api_key_label = ?"
            params.append(api_key_label)
        if model_type:
            sql += " AND model_type = ?"
            params.append(canonical_model_type(model_type))
        vis_sql, vis_params = self._visible("request_log", since=cutoff)
        if vis_sql:
            sql += _and(vis_sql)
            params.extend(vis_params)
        sql += " GROUP BY date(timestamp), model_type ORDER BY day DESC"

        rows = await self._conn.execute_fetchall(sql, params)
        # ``request_count`` first, then the token columns in the order the
        # select list was built from, which is the order they are read in.
        counters = ("request_count", *LEDGER_COLUMNS)
        # The per-type splits keep their own typed dicts: a day's entry mixes
        # them with its scalar counters, and a reader that has to guess which
        # is which is a reader that can be wrong.
        type_totals: dict[str, dict[str, int]] = {}
        totals: dict[str, Any] = {
            "by_model_type": type_totals,
            **dict.fromkeys(counters, 0),
        }
        # Insertion order is the row order, and the rows arrive newest first,
        # so the days stay newest first.
        days_by_date: dict[str, dict[str, Any]] = {}
        for r in rows:
            day, row_type = r[0], r[1] or DEFAULT_MODEL_TYPE
            split = dict(zip(counters, r[2:]))
            entry: dict[str, Any] | None = days_by_date.get(day)
            if entry is None:
                entry = {
                    "day": day,
                    "by_model_type": {},
                    **dict.fromkeys(counters, 0),
                }
                days_by_date[day] = entry
            day_split: dict[str, dict[str, int]] = entry["by_model_type"]
            for col in counters:
                entry[col] += split[col]
                totals[col] += split[col]
            for bucket in (day_split, type_totals):
                counters_of_type = bucket.get(row_type)
                if counters_of_type is None:
                    counters_of_type = dict.fromkeys(counters, 0)
                    bucket[row_type] = counters_of_type
                for col in counters:
                    counters_of_type[col] += split[col]
        return {"days": list(days_by_date.values()), "totals": totals}

    async def query_rolling_24h_tokens(self) -> dict[str, int]:
        """Sum token usage over the last 24 hours.

        ``total_tokens_24h`` is the canonical figure every view displays; the
        subset columns ride along as the breakdown behind it.  They do not add
        up to it — reasoning is part of the output, and cache tokens are a
        subset of the input for some providers and additive for others.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        vis_sql, vis_params = self._visible("request_log", since=cutoff)
        row = await self._conn.execute_fetchone(
            "SELECT COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(cache_read_tokens), 0), "
            "COALESCE(SUM(cache_creation_tokens), 0), "
            "COALESCE(SUM(reasoning_tokens), 0), "
            "COALESCE(SUM(total_tokens), 0) "
            f"FROM request_log WHERE timestamp >= ?{_and(vis_sql)}",
            (cutoff, *vis_params),
        )
        return {
            "input_tokens_24h": row[0],
            "output_tokens_24h": row[1],
            "cache_read_tokens_24h": row[2],
            "cache_creation_tokens_24h": row[3],
            "reasoning_tokens_24h": row[4],
            "total_tokens_24h": row[5],
        }

    async def query_request_histogram(
        self, minutes: int = 300, buckets: int = 60
    ) -> dict[str, Any]:
        """Count requests per time bucket over the retained request log.

        The in-memory series holds five minutes at one-second resolution, so
        anything longer has to come from the log.  The window is clipped to
        the retention caps: a bucket older than the oldest kept row reads zero
        because the row is gone, not because no request arrived.

        Args:
            minutes: Window length, at most seven days.
            buckets: Number of equal buckets in the window, at most 240.

        Returns:
            A mapping with the window description and one entry per bucket:
            ``count``, ``errors``, ``output_tokens`` and ``duration_ms``.
        """
        from datetime import datetime, timedelta, timezone

        minutes = max(1, min(int(minutes), 7 * 24 * 60))
        buckets = max(1, min(int(buckets), 240))
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=minutes)
        start_epoch = int(start.timestamp())
        width = max(1, int(minutes * 60 / buckets))

        vis_sql, vis_params = self._visible("request_log", since=start.isoformat())
        rows = await self._conn.execute_fetchall(
            "SELECT CAST((CAST(strftime('%s', timestamp) AS INTEGER) - ?) / ? "
            "AS INTEGER) AS b, "
            "COUNT(*), "
            "COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(duration_ms), 0) "
            f"FROM request_log WHERE timestamp >= ?{_and(vis_sql)} "
            "GROUP BY b ORDER BY b",
            (start_epoch, width, start.isoformat(), *vis_params),
        )
        # A row in the sliver after the last whole bucket (a window that does
        # not divide evenly) still belongs to the chart: fold it into the last
        # bucket rather than drop it.
        by_index: dict[int, Any] = {}
        for row in rows:
            if row[0] is None:
                continue
            idx = min(max(int(row[0]), 0), buckets - 1)
            prev = by_index.get(idx)
            if prev is None:
                by_index[idx] = row
            else:
                by_index[idx] = (
                    idx,
                    prev[1] + row[1],
                    prev[2] + row[2],
                    prev[3] + row[3],
                    prev[4] + row[4],
                )

        series = []
        total = 0
        for i in range(buckets):
            row = by_index.get(i)
            count = int(row[1]) if row else 0
            total += count
            series.append(
                {
                    "t": (start + timedelta(seconds=i * width)).isoformat(),
                    "count": count,
                    "errors": int(row[2]) if row else 0,
                    "output_tokens": int(row[3]) if row else 0,
                    "duration_ms": float(row[4]) if row else 0.0,
                }
            )

        return {
            "minutes": minutes,
            "bucket_seconds": width,
            "start": start.isoformat(),
            "end": now.isoformat(),
            "total": total,
            "buckets": series,
        }

    async def query_usage_groups(self) -> dict[str, Any]:
        """Aggregate token usage per credential bucket, label and model type.

        Every window is relative to now (``now - hours``).  A single indexed
        range scan over the widest window feeds the conditional-aggregation
        sums, so the whole report costs one pass over the log.

        Rows are grouped by ``(bucket, label, model_type)`` rather than by
        bucket alone.  The caller knows which buckets belong to a live
        credential and needs the label split to account for the rest, so the
        split is done here once and the classification is left to it; the model
        type is split for the same reason — a caller that wants "how much was
        embedding" should not have to re-query.  Rows carrying no bucket —
        anonymous requests, and everything logged before the column existed —
        are reported under the ``None`` bucket instead of being dropped, so
        the report covers every request the dashboard counts.

        Returns:
            ``{"windows": [label, ...], "groups": [...]}``, each group being
            ``{"bucket": str | None, "label": str | None, "model_type": str,
            "windows": {...}}`` with each per-window dict holding
            ``request_count`` plus the token totals.  Buckets with no traffic
            in the widest window are absent.
        """
        select, params, since = self._usage_window_select(
            ("api_key_bucket", "api_key_label", "model_type")
        )
        vis_sql, vis_params = self._visible("request_log", since=since)
        rows = await self._conn.execute_fetchall(
            f"SELECT {select} FROM request_log "
            f"WHERE timestamp >= ?{_and(vis_sql)} "
            "GROUP BY api_key_bucket, api_key_label, model_type",
            [*params, since, *vis_params],
        )

        # One COUNT plus the token columns per window; the three leading columns
        # are the bucket, the label and the model type.
        groups = self._fold_window_rows(rows, ("bucket", "label", "model_type"))
        for group in groups:
            group["model_type"] = group["model_type"] or DEFAULT_MODEL_TYPE
        return {
            "windows": [name for name, _hours in USAGE_WINDOWS],
            "groups": groups,
        }

    @staticmethod
    def _usage_window_select(
        leading: Sequence[str],
    ) -> tuple[str, list[Any], str]:
        """The conditional-aggregation select for one usage read.

        Several reads ask the same question of the same table — "how much, in
        each rolling window" — and differ only in what they group by, so the
        select is built here rather than spelled out once per read with a
        chance to drift apart each time.  The caller appends its own ``GROUP
        BY`` and ``ORDER BY``.

        Args:
            leading: The columns to group by, in order.

        Returns:
            The select list, its parameters (one cutoff per counter, in the
            order they appear), and the widest window's cutoff — the bound the
            caller puts on ``timestamp`` so a read that only reports the narrow
            windows still skips the rows no window can see.
        """
        now = datetime.now(timezone.utc)
        parts = list(leading)
        params: list[Any] = []
        for _name, hours in USAGE_WINDOWS:
            cutoff = (now - timedelta(hours=hours)).isoformat()
            parts.append("COUNT(CASE WHEN timestamp >= ? THEN 1 END)")
            params.append(cutoff)
            for col in LEDGER_COLUMNS:
                parts.append(
                    f"COALESCE(SUM(CASE WHEN timestamp >= ? THEN {col} ELSE 0 END), 0)"
                )
                params.append(cutoff)
        return (
            ", ".join(parts),
            params,
            (now - timedelta(hours=USAGE_WINDOWS[-1][1])).isoformat(),
        )

    @staticmethod
    def _fold_window_rows(
        rows: Sequence[Sequence[Any]], leading: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Fold a usage read's rows into one dict per group.

        The columns come back in the order :meth:`_usage_window_select` built
        them: the leading columns, then for each window a count and the ledger
        columns in order.
        """
        stride = 1 + len(LEDGER_COLUMNS)
        folded: list[dict[str, Any]] = []
        for row in rows:
            windows: dict[str, dict[str, int]] = {}
            idx = len(leading)
            for name, _hours in USAGE_WINDOWS:
                window: dict[str, int] = {"request_count": row[idx]}
                for offset, col in enumerate(LEDGER_COLUMNS, start=1):
                    window[col] = row[idx + offset]
                windows[name] = window
                idx += stride
            folded.append({**dict(zip(leading, row)), "windows": windows})
        return folded

    async def query_usage_by_model(self) -> dict[str, Any]:
        """Aggregate token usage per model and model type.

        The cut a caller wants when the question is "which models moved the
        tokens" rather than "who spent them": the model name is a column of the
        row, not an attribution, so it is its own read rather than a dim of
        :meth:`query_usage_groups`.  Rows are grouped by ``(model, model_type)``
        so one model's chat and embedding traffic stay distinguishable.

        Returns:
            ``{"windows": [...], "groups": [...]}``, each group being
            ``{"model": str, "model_type": str, "windows": {...}}``, widest
            window's total first.
        """
        select, params, since = self._usage_window_select(("model", "model_type"))
        # The rows are already bounded to the widest window, so the ordering
        # sum is that window's total for the model.
        vis_sql, vis_params = self._visible("request_log", since=since)
        rows = await self._conn.execute_fetchall(
            f"SELECT {select} FROM request_log WHERE timestamp >= ?"
            f"{_and(vis_sql)} "
            "GROUP BY model, model_type ORDER BY SUM(total_tokens) DESC, model",
            [*params, since, *vis_params],
        )
        groups = self._fold_window_rows(rows, ("model", "model_type"))
        for group in groups:
            group["model_type"] = group["model_type"] or DEFAULT_MODEL_TYPE
        return {"windows": [name for name, _hours in USAGE_WINDOWS], "groups": groups}

    async def backfill_usage_buckets(self, label_to_bucket: Mapping[str, str]) -> int:
        """Attribute logged requests that carry no credential bucket.

        Runs at startup.  A request logged before the bucket column existed —
        or by a credential that never had one — is matched to its bucket
        through the label on the row.  Callers pass only unambiguous labels:
        a label shared by several live keys, or claimed by none, is left NULL
        and stays in the panel's unattributed remainder.

        Args:
            label_to_bucket: Label to bucket, already disambiguated.

        Returns:
            Number of rows updated.
        """
        updated = 0
        for label, bucket in label_to_bucket.items():
            cursor = await self._conn.execute(
                "UPDATE request_log SET api_key_bucket = ? "
                "WHERE api_key_bucket IS NULL AND api_key_label = ?",
                (bucket, label),
            )
            updated += cursor.rowcount or 0
        if updated:
            await self._conn.commit()
            logger.info("Backfilled the usage bucket of %d log rows", updated)
        return updated

    async def update_entry_usage(
        self,
        entry_id: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        *,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Write back token usage for an existing log entry.

        Used by the streaming path to record usage extracted from the
        final stream event.
        """
        await self._conn.execute(
            "UPDATE request_log SET input_tokens = ?, output_tokens = ?, "
            "total_tokens = ?, cache_read_tokens = ?, "
            "cache_creation_tokens = ?, reasoning_tokens = ? WHERE id = ?",
            (
                input_tokens,
                output_tokens,
                total_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                reasoning_tokens,
                entry_id,
            ),
        )
        await self._conn.commit()

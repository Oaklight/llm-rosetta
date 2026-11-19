"""Error-dump export (streamed archives)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from ._common import (
    _DUMP_COLUMN_NAMES,
    _DUMP_COLUMNS,
    _DUMP_EXPORT_PAGE,
    _DUMP_ORDER,
    _ChunkSink,
    _dump_metadata_member,
)
from ._state import _StoreState


class _DumpExportMixin(_StoreState):
    """Error-dump export (streamed archives)."""

    async def iter_export_error_dumps(
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
        page_size: int = _DUMP_EXPORT_PAGE,
    ) -> AsyncIterator[bytes]:
        """Stream the dumps the filters keep as a tar.gz archive.

        The same builder the list uses, so an export of a filtered list holds
        exactly the rows that list shows — a 206-row status filter exports
        206 dumps, not the whole store.

        The rows are read in ``(timestamp, id)`` keyset pages, every dump is
        written as its own ``metadata/<id>.json`` member, and the archive is
        drained after each member.  Nothing in the archive depends on a pass
        over the whole store, so an export holds one page of rows, one entry,
        one body and one member's compressed output at a time — not the entry
        list, and not the whole ``metadata.json`` either.

        A drain is not a chunk per member: the gzip encoder buffers, so a
        drain can return nothing and several members can share one chunk.

        Args:
            model/model_search/error_phase/provider/status_min/status_max:
                The list's filters.
            start: ISO timestamp lower bound (inclusive). None = no lower bound.
            end: ISO timestamp upper bound (inclusive). None = no upper bound.
            q: Free text over the id and the request-log id.
            page_size: Rows read per metadata page.

        Yields:
            ``tar.gz`` bytes: one ``metadata/<id>.json`` per dump, in
            newest-first order, then one ``bodies/<hash>.bin`` per distinct
            body.  Joined, the chunks are the archive the in-memory export
            returns.
        """
        import io
        import tarfile

        filters: dict[str, Any] = {
            "model": model,
            "model_search": model_search,
            "error_phase": error_phase,
            "provider": provider,
            "status_min": status_min,
            "status_max": status_max,
            "start": start,
            "end": end,
            "q": q,
        }
        sink = _ChunkSink()
        # The sink is write-only, and "w:gz" only ever writes to it; the
        # stub's file-object protocol also asks for read/seek/close.
        archive = tarfile.open(fileobj=cast(Any, sink), mode="w:gz")
        try:
            hashes: set[str] = set()
            async for entry in self._iter_dump_entries(filters, page_size):
                name, payload = _dump_metadata_member(entry)
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                for key in ("body_hash", "converted_body_hash"):
                    value = entry.get(key)
                    if value:
                        hashes.add(value)
                # Usually empty: the gzip encoder holds its output until it
                # has enough, so a member is freed before it is sent.
                chunk = sink.drain()
                if chunk:
                    yield chunk

            for body_hash in sorted(hashes):
                row = await self._conn.execute_fetchone(
                    "SELECT data FROM dump_bodies WHERE hash = ?", (body_hash,)
                )
                if not row or not row[0]:
                    continue
                data = row[0]
                info = tarfile.TarInfo(name=f"bodies/{body_hash}.bin")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
                chunk = sink.drain()
                if chunk:
                    yield chunk
        finally:
            # Also runs when the consumer stops early: closing writes the
            # archive trailer into the sink, which nobody drains then, and
            # releases the gzip encoder.
            archive.close()

        # Only a completed export reaches this.  The chunk holds the gzip
        # trailer, and without it the archive is truncated.
        chunk = sink.drain()
        if chunk:
            yield chunk

    async def _iter_dump_entries(
        self,
        filters: dict[str, Any],
        page_size: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield the filtered dumps newest first, one keyset page at a time.

        The cursor is the row's own ``(timestamp, id)``, so a page boundary
        stays the same position while the table is being written.  Rows that
        share a timestamp are carried by the id half; without it the walk
        would stop at the end of the first page.
        """
        before: tuple[str, str] | None = None
        while True:
            where_sql, params = self._dump_filter_sql(**filters, before=before)
            rows = await self._conn.execute_fetchall(
                f"SELECT {_DUMP_COLUMNS} FROM error_dumps {where_sql} "
                f"{_DUMP_ORDER} LIMIT ?",
                [*params, page_size],
            )
            if not rows:
                return
            for row in rows:
                yield {k: v for k, v in zip(_DUMP_COLUMN_NAMES, row) if v is not None}
            if len(rows) < page_size:
                return
            # The row is past the cursor when its timestamp is older, or
            # equal and its id sorts lower — the order ``_DUMP_ORDER`` names.
            position = (rows[-1][2], rows[-1][0])
            if position == before:
                # A cursor that does not move would read this page forever.
                return
            before = position

    async def export_error_dumps(
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
    ) -> bytes:
        """Export the dumps the filters keep as tar.gz bytes.

        The in-memory form of :meth:`iter_export_error_dumps`, for callers
        that need the archive as one value.  It holds everything the stream
        avoids holding — the whole archive — so a caller that can write
        chunks should use the iterator.  Nothing in the gateway calls it:
        the route and the CLI both take the iterator, and this exists for
        tests and for callers outside the repository.

        Args:
            model/model_search/error_phase/provider/status_min/status_max:
                The list's filters.
            start: ISO timestamp lower bound (inclusive). None = no lower bound.
            end: ISO timestamp upper bound (inclusive). None = no upper bound.
            q: Free text over the id and the request-log id.

        Returns:
            In-memory tar.gz archive: one ``metadata/<id>.json`` per dump and
            one ``bodies/<hash>.bin`` per distinct body.
        """
        # A bytearray, not a list joined at the end: the chunks and the
        # joined copy would both be live across the join, and the archive is
        # the largest thing this method holds.
        archive = bytearray()
        async for chunk in self.iter_export_error_dumps(
            model=model,
            model_search=model_search,
            error_phase=error_phase,
            provider=provider,
            status_min=status_min,
            status_max=status_max,
            start=start,
            end=end,
            q=q,
        ):
            archive += chunk
        return bytes(archive)

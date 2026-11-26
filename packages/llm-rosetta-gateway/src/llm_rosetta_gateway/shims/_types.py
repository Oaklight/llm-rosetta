"""Constants, errors and the status/report records of the shim-source loader."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


# The literal keeps log records under the name they have always had (the flat
# ``shims.sources`` module's ``__name__``); log filters key on it.  Do not
# change it to ``__name__``.
logger = logging.getLogger("llm_rosetta_gateway.shims.sources")

#: Download cap for a vendor archive.
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
#: Default re-fetch interval for an unpinned remote (seconds).
REMOTE_TTL_SECONDS = 24 * 3600
#: HTTP timeout for fetch + ref resolution (seconds).
HTTP_TIMEOUT = 30.0
#: Default cache directory name, resolved next to the config file by the gateway.
DEFAULT_CACHE_DIRNAME = "shims-cache"

HttpGet = Callable[..., bytes]


class ShimSourceError(RuntimeError):
    """Raised when a declared shim source cannot be loaded (``strict`` mode)."""


class ShimArchiveMismatch(ShimSourceError):
    """The downloaded archive does not match the configured ``sha256``.

    Distinct from a transient fetch failure: the bytes are *wrong*, not
    missing, so the stale-cache fallback must not paper over it.
    """


class ShimArchiveUnusable(ShimSourceError):
    """A downloaded body is not an archive the loader can install.

    A corruption signal, not a network symptom: the server answered and the
    bytes arrived, they just are not the source.  A body the reader aborts for
    crossing :data:`MAX_ARCHIVE_BYTES` is the same verdict reached earlier, so
    it is raised while the stream is still being read, not only after the
    download.  Like a digest mismatch, this is never masked with the stale
    cache.
    """


class ShimSourceUnreachable(ShimSourceError):
    """No configured leg could be reached (a transport failure on every leg).

    A transport failure — connection refused, DNS failure, timeout — is the
    only condition that makes the loader try a declared mirror.  A server that
    answers (even with an error) or bytes that do not match their pin are the
    source's verdict, not a network symptom, so they raise a plain
    :class:`ShimSourceError` instead.
    """


@dataclass(frozen=True)
class ShimSourceStatus:
    """Outcome of loading one declared shim source.

    Attributes:
        kind: Always ``"remote"``: a source is fetched from an extension index.
        location: The declared source URL.
        name: Remote name (derived when not given).
        ok: Whether the source loaded.  A stale-cache fallback counts as ok.
        shim_count: Number of shims registered from this source.
        shim_names: Names of the shims registered from this source.
        resolved_dir: Absolute directory that was scanned.
        error: Failure reason when ``ok`` is ``False``.
        from_cache: Whether the remote was served from the cache.
        stale: Whether a stale cache was used because the fetch failed.
        fetched_at: Unix time of the (cached or fresh) archive.
        resolved_ref: Concrete ref the archive was fetched at (a HEAD fetch is
            pinned to its commit SHA here).
        checksum_ok: ``True``/``False`` when a ``sha256`` pin was checked.
        leg: Which URL served the archive: ``"primary"`` or ``"mirror"``.
            ``None`` when the source never resolved.
        archive_url: The concrete URL the archive was fetched from (the
            expanded mirror template when ``leg`` is ``"mirror"``).
        unreachable: Whether the source failed because no leg could be
            reached, as opposed to a server answer or a bad archive.
        manifest_shims: The shim names the installed index entry declares.
        source: The manifest source a vendor install came from (``contrib``,
            ``argonne``, or the id a manifest URL declares).  ``None`` for a
            plain remote.
        vendor: The vendor a manifest install covers.  ``None`` otherwise.
        release: The manifest record of the installed version (version,
            commit, sha256, requires, shims, tags, maturity, ...), as the
            manifest spells it.  ``None`` when nothing is installed.
        skipped: Why a selected vendor was not installed (promoted to core,
            core ships the name, no compatible version) — a decision, not a
            failure, so ``ok`` stays true.
        extension: The id of the installed extension a vendor install carries.
            ``None`` otherwise.
        state: What loading the extension did: ``"loaded"``, ``"skipped"`` or
            ``"failed"`` (core's own words); ``None`` before it was loaded.
        note: A sentence the operator should read though nothing failed, such
            as the required extensions that are not installed.
    """

    kind: Literal["remote"]
    location: str
    name: str | None = None
    ok: bool = True
    shim_count: int = 0
    shim_names: tuple[str, ...] = ()
    resolved_dir: str | None = None
    error: str | None = None
    from_cache: bool = False
    stale: bool = False
    fetched_at: float | None = None
    resolved_ref: str | None = None
    checksum_ok: bool | None = None
    leg: Literal["primary", "mirror"] | None = None
    archive_url: str | None = None
    unreachable: bool = False
    manifest_shims: tuple[str, ...] = ()
    source: str | None = None
    vendor: str | None = None
    release: dict[str, Any] | None = None
    skipped: str | None = None
    extension: str | None = None
    state: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view for the admin API."""
        return {
            "kind": self.kind,
            "location": self.location,
            "name": self.name,
            "ok": self.ok,
            "shim_count": self.shim_count,
            "shim_names": list(self.shim_names),
            "resolved_dir": self.resolved_dir,
            "error": self.error,
            "from_cache": self.from_cache,
            "stale": self.stale,
            "fetched_at": self.fetched_at,
            "resolved_ref": self.resolved_ref,
            "checksum_ok": self.checksum_ok,
            "leg": self.leg,
            "archive_url": self.archive_url,
            "unreachable": self.unreachable,
            "manifest_shims": list(self.manifest_shims),
            "source": self.source,
            "vendor": self.vendor,
            "release": dict(self.release) if self.release else None,
            "skipped": self.skipped,
            "extension": self.extension,
            "state": self.state,
            "note": self.note,
        }


@dataclass
class ShimLoadReport:
    """Result of a :func:`rebuild_shims` call."""

    sources: list[ShimSourceStatus] = field(default_factory=list)
    total_shims: int = 0
    rebuilt_at: float = 0.0

    def failures(self) -> list[ShimSourceStatus]:
        """Return the sources that failed to load."""
        return [s for s in self.sources if not s.ok]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view for the admin API."""
        return {
            "sources": [s.to_dict() for s in self.sources],
            "total_shims": self.total_shims,
            "rebuilt_at": self.rebuilt_at,
            "failure_count": len(self.failures()),
        }

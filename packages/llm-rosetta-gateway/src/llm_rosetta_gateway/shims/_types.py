"""Constants, errors and the status/report records of the shim-source loader."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


# The literal keeps log records under the pre-split name: the flat module used
# ``getLogger(__name__)``, which resolved to ``llm_rosetta_gateway.shims.sources``,
# and ``__name__`` here would be ``...shims._types``.  Do not change it back to
# ``__name__``; log filters key on the old name.
#
# Pickles: the status/report/error classes below and ``_RemoteSpec`` (in
# ``_spec``) report a new ``__module__`` after the split, but ``sources``
# re-exports every one of them, so a pickle written before the split still
# loads: unpickling imports the old module path and reads the name from it.
# Keep those re-exports in ``sources`` for as long as old pickles may exist.
logger = logging.getLogger("llm_rosetta_gateway.shims.sources")

#: Download cap for a remote archive.
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
#: Cap on the total uncompressed size written to disk.
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
#: Cap on the number of archive members.
MAX_ARCHIVE_MEMBERS = 20_000
#: Default re-fetch interval for an unpinned remote (seconds).
REMOTE_TTL_SECONDS = 24 * 3600
#: HTTP timeout for fetch + ref resolution (seconds).
HTTP_TIMEOUT = 30.0
#: Default cache directory name, resolved next to the config file by the gateway.
DEFAULT_CACHE_DIRNAME = "shims-cache"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:$")
_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

RefreshMode = Literal["auto", "always", "never"]
_REFRESH_MODES: tuple[RefreshMode, ...] = ("auto", "always", "never")
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
        kind: ``"path"`` for a local directory, ``"remote"`` for a fetched
            archive.
        location: The declared directory path or remote URL.
        name: Remote name (derived when not given).  ``None`` for paths.
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
        ref: The ref the declaration named (``None`` for a HEAD fetch), so the
            update check can ask about the same ref it fetched.
        token_env: The environment variable the declaration names for its
            credential (the name, never the token), so a private repository is
            checked with the same credential it was fetched with.
        leg: Which URL served the archive: ``"primary"`` or ``"mirror"``.
            ``None`` when the source never resolved.
        archive_url: The concrete URL the archive was fetched from (the
            expanded mirror template when ``leg`` is ``"mirror"``).
        unreachable: Whether the source failed because no leg could be
            reached, as opposed to a server answer or a bad archive.
        include: The ``include`` selection the source's declaration carried,
            as written (exact names and ``*`` globs).
        exclude: The ``exclude`` selection, as written.
        unmatched: ``include`` patterns that selected nothing from what this
            source ships.  A name the package stopped shipping lands here,
            which is how a stale config is visible instead of silently empty.
        unmatched_excludes: ``exclude`` patterns that matched nothing the
            source ships.  An exclude is measured against the shipped set, not
            the loaded one, so a pattern that hit a shim already gone is told
            apart from one that hit a shim the selection dropped.
        emptied_includes: ``include`` patterns whose every hit the ``exclude``
            dropped, when that left the source with nothing loaded.  Every
            pattern "matched", so neither of the fields above reports it, and
            yet the outcome is the one the include warning exists for: the
            operator believes a shim is running and nothing loads.
        manifest_shims: The shim names the source's ``shims.yaml`` declares.
            Empty when it ships no manifest, which is the legacy scan layout.
        declined: The names the selection dropped — a shim the source ships, or
            a leaf it refused before validation could have failed it.  A loaded
            name is never here; ``shim_names`` plus this is the set a pattern
            can match, which is what makes an ``exclude`` that hit nothing
            visible at all.
        source: The manifest source a vendor install came from (``contrib``,
            ``argonne``, or the id a manifest URL declares).  ``None`` for a
            directory or a plain remote.
        vendor: The vendor a manifest install covers.  ``None`` otherwise.
        release: The manifest record of the installed version (version,
            commit, sha256, requires, shims, tags, maturity, ...), as the
            manifest spells it.  ``None`` when nothing is installed.
        skipped: Why a selected vendor was not installed (promoted to core,
            core ships the name, no compatible version) — a decision, not a
            failure, so ``ok`` stays true.
    """

    kind: Literal["path", "remote"]
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
    ref: str | None = None
    token_env: str | None = None
    leg: Literal["primary", "mirror"] | None = None
    archive_url: str | None = None
    unreachable: bool = False
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()
    unmatched_excludes: tuple[str, ...] = ()
    emptied_includes: tuple[str, ...] = ()
    manifest_shims: tuple[str, ...] = ()
    declined: tuple[str, ...] = ()
    source: str | None = None
    vendor: str | None = None
    release: dict[str, Any] | None = None
    skipped: str | None = None

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
            "ref": self.ref,
            "token_env": self.token_env,
            "leg": self.leg,
            "archive_url": self.archive_url,
            "unreachable": self.unreachable,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "unmatched": list(self.unmatched),
            "unmatched_excludes": list(self.unmatched_excludes),
            "emptied_includes": list(self.emptied_includes),
            "manifest_shims": list(self.manifest_shims),
            "declined": list(self.declined),
            "source": self.source,
            "vendor": self.vendor,
            "release": dict(self.release) if self.release else None,
            "skipped": self.skipped,
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

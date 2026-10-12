"""External shim sources — load provider shims from directories and archives.

Built-in shims are registered at import time from the bundled ``providers/``
directory (see :mod:`llm_rosetta.shims.providers`).  This module adds a second,
*declarative* source of shims: extra directories, and remote archives fetched
into a local cache.  The gateway exposes both through its ``shims`` config
section; this module is the config-agnostic core.

Public entry points:

- :func:`rebuild_shims` — full registry rebuild (built-ins + entry points +
  ``extra_dirs`` + ``extra_remotes``).  Idempotent.  This is the programmatic
  entry point for callers that want to load shims before calling
  :func:`llm_rosetta.convert`.
- :func:`resolve_remote` — fetch and extract a single remote archive, returning
  a :class:`ShimSourceStatus` whose ``resolved_dir`` is the directory to scan.

Remote archives are fetched over HTTPS with the vendored synchronous HTTP client
(no ``git``, no ``pip``).  GitHub repository URLs are translated to their
codeload tarball endpoint; any other host works by giving an explicit archive
URL (optionally containing ``{ref}``), so GitLab and self-hosted forges are
supported without a per-host code path.

Security
--------
A shim directory may ship a ``transforms.py`` that is executed on load, so a
declared source is **code execution** — the same trust level as ``pip
install``ing a plugin package.  Only add sources you trust.  Extraction rejects
absolute paths, ``..`` traversal, symlink/hardlink and device members, and both
the download and the extracted tree are size-capped.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from .provider_shim import (
    _clear_registry_caches,
    list_shims,
    register_shim,
    staging_registry,
)
from .providers import load_providers, load_providers_from_dir

logger = logging.getLogger(__name__)

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
_DRIVE_RE = re.compile(r"^[A-Za-z]:$")
_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")

RefreshMode = Literal["auto", "always", "never"]
_REFRESH_MODES: tuple[RefreshMode, ...] = ("auto", "always", "never")
HttpGet = Callable[..., bytes]

# In-process guard: a rebuild must not be observed half-empty by another thread.
_rebuild_lock = threading.Lock()

#: Shim names installed by the most recent rebuild from ``extra_dirs`` /
#: ``extra_remotes``.  Tracked so a later rebuild can drop exactly those while
#: leaving shims registered in-process via ``register_shim`` alone.
_source_shims: set[str] = set()


class ShimSourceError(RuntimeError):
    """Raised when a declared shim source cannot be loaded (``strict`` mode)."""


class ShimArchiveMismatch(ShimSourceError):
    """The downloaded archive does not match the configured ``sha256``.

    Distinct from a transient fetch failure: the bytes are *wrong*, not
    missing, so the stale-cache fallback must not paper over it.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rebuild_shims(
    extra_dirs: Sequence[str | os.PathLike[str]] = (),
    extra_remotes: Sequence[Mapping[str, Any]] = (),
    *,
    cache_root: str | os.PathLike[str] | None = None,
    strict: bool = False,
    http_get: HttpGet | None = None,
    now: Callable[[], float] | None = None,
    ttl: int = REMOTE_TTL_SECONDS,
    proxy: str | None = None,
) -> ShimLoadReport:
    """Rebuild the shim registry from built-ins, entry points and sources.

    The registry is cleared and repopulated in a fixed order — built-ins, then
    entry-point plugins, then ``extra_dirs`` in order, then ``extra_remotes`` in
    order — so a later source overrides an earlier one of the same shim name.
    Idempotent: calling twice with the same inputs yields the same registry.

    Args:
        extra_dirs: Local directories to scan for ``provider.yaml`` (flat or
            grouped layout, as :func:`load_providers_from_dir` accepts).
        extra_remotes: Remote archive specs; see :func:`resolve_remote`.
        cache_root: Directory for fetched archives.  Required when
            ``extra_remotes`` is non-empty.
        strict: When ``True``, raise :class:`ShimSourceError` if any declared
            source fails; otherwise failures are reported in the result.
        http_get: Injectable ``(url, *, headers, timeout, proxy) -> bytes``.
            Defaults to the vendored synchronous HTTP client.
        now: Injectable clock (defaults to :func:`time.time`).
        ttl: Re-fetch interval for unpinned remotes (seconds).
        proxy: Upstream proxy URL for fetches.

    Returns:
        A :class:`ShimLoadReport` describing each source and the final registry
        size.

    Raises:
        ShimSourceError: Only when ``strict`` is set and a source failed.
    """
    now_fn = now or time.time
    statuses: list[ShimSourceStatus] = []

    with _rebuild_lock:
        # Slow, network-bound work first: fetch/extract every remote *before*
        # the registry is touched, so shims are never missing from the visible
        # registry for the duration of a fetch.
        prefetched = [
            _resolve_remote_source(
                spec,
                cache_root=cache_root,
                http_get=http_get,
                now=now_fn,
                ttl=ttl,
                proxy=proxy,
            )
            for spec in extra_remotes
        ]

        # The registry is rebuilt off to the side and swapped in with a single
        # reference assignment (see ``staging_registry``), so a reader on any
        # other thread — notably the gateway's event loop, which resolves each
        # request's shim by name against this registry — always sees either the
        # old registry or the complete new one, never a partially-built one.
        # The caches derived from it are dropped first, outside the staging
        # block, so the new registry starts from a clean slate.
        _clear_registry_caches()
        out_of_band = {s.name: s for s in list_shims() if s.name not in _source_shims}

        with staging_registry():
            load_providers()
            known = {s.name for s in list_shims()}
            for name, shim in out_of_band.items():
                if name not in known:
                    register_shim(shim)

            for directory in extra_dirs:
                statuses.append(_load_local_dir(directory))

            for status in prefetched:
                statuses.append(_scan_status(status))

        _source_shims.clear()
        _source_shims.update(name for status in statuses for name in status.shim_names)
        total = len(list_shims())

    report = ShimLoadReport(sources=statuses, total_shims=total, rebuilt_at=now_fn())

    failures = report.failures()
    for status in failures:
        logger.warning(
            "Shim source failed (%s %s): %s",
            status.kind,
            status.location,
            status.error,
        )
    if strict and failures:
        detail = "; ".join(f"{s.kind} {s.location}: {s.error}" for s in failures)
        raise ShimSourceError(f"failed to load shim source(s): {detail}")

    return report


def resolve_remote(
    spec: Mapping[str, Any],
    *,
    cache_root: str | os.PathLike[str] | None,
    http_get: HttpGet | None = None,
    now: Callable[[], float] | None = None,
    ttl: int = REMOTE_TTL_SECONDS,
    proxy: str | None = None,
) -> ShimSourceStatus:
    """Fetch, cache and scan one remote shim archive.

    The spec keys are: ``url`` (required), ``name``, ``ref``, ``path``,
    ``archive`` (``auto`` / ``tar.gz`` / ``zip``), ``sha256``, ``token_env``
    and ``refresh`` (``auto`` / ``always`` / ``never``).

    When ``ref`` is omitted and ``url`` is a GitHub repository URL, ``HEAD`` is
    resolved to a concrete commit SHA via the GitHub API and recorded as
    ``resolved_ref`` in the cache stamp, so later runs reuse the same commit
    without re-resolving.  The user's config file is never rewritten.

    On a *transient* fetch failure with an existing (even stale) cache, the
    cache is used and the status is marked ``stale``; the source only fails when
    there is no cache to fall back on.  A ``sha256`` mismatch is **not**
    transient — the bytes are wrong, not missing — so it always fails, even
    with a warm cache.

    Args:
        spec: The remote source mapping.
        cache_root: Directory under which the per-source cache lives.
        http_get: Injectable HTTP getter (see :func:`rebuild_shims`).
        now: Injectable clock.
        ttl: Re-fetch interval for unpinned remotes (seconds).
        proxy: Upstream proxy URL for fetches.

    Returns:
        A :class:`ShimSourceStatus`.  Failures are returned, not raised.
    """
    return _scan_status(
        _resolve_remote_source(
            spec,
            cache_root=cache_root,
            http_get=http_get,
            now=now,
            ttl=ttl,
            proxy=proxy,
        )
    )


def _resolve_remote_source(
    spec: Mapping[str, Any],
    *,
    cache_root: str | os.PathLike[str] | None,
    http_get: HttpGet | None = None,
    now: Callable[[], float] | None = None,
    ttl: int = REMOTE_TTL_SECONDS,
    proxy: str | None = None,
) -> ShimSourceStatus:
    """Fetch/extract one remote archive without touching the shim registry.

    Split out of :func:`resolve_remote` so a caller can do all the slow network
    work *before* mutating the global registry (see :func:`rebuild_shims`).
    The returned status carries ``resolved_dir`` but no shim counts; call
    :func:`_scan_status` to load and count.
    """
    now_fn = now or time.time
    location = str(spec.get("url", "")) if isinstance(spec, Mapping) else str(spec)
    name: str | None = None
    try:
        parsed = _parse_remote_spec(spec, cache_root)
        name = parsed.name
        location = parsed.url

        cached = _read_stamp(parsed.stamp_path)
        has_cache = parsed.source_dir.is_dir() and bool(cached)
        need_fetch = _needs_fetch(parsed, cached, has_cache, now_fn(), ttl)
        stale = False
        from_cache = not need_fetch

        if need_fetch:
            try:
                cached = _fetch_remote(
                    parsed,
                    headers=_auth_headers(parsed.token_env),
                    http_get=http_get or _default_http_get,
                    now_fn=now_fn,
                    proxy=proxy,
                )
            except Exception as exc:
                # A digest mismatch means the bytes are wrong, not missing —
                # never mask it with the stale cache.  Everything else may
                # degrade to the cache when one exists.
                if not has_cache or isinstance(exc, ShimArchiveMismatch):
                    raise
                logger.warning(
                    "Shim remote %s: fetch failed (%s); using stale cache",
                    location,
                    exc,
                )
                from_cache, stale = True, True

        return ShimSourceStatus(
            kind="remote",
            location=location,
            name=name,
            ok=True,
            resolved_dir=str(_remote_subdir(parsed)),
            from_cache=from_cache,
            stale=stale,
            fetched_at=cached.get("fetched_at"),
            resolved_ref=cached.get("resolved_ref") or parsed.ref,
        )
    except Exception as exc:
        logger.warning("Shim remote %s: %s", location, exc)
        return _remote_failure(location, name, exc)


def _scan_status(status: ShimSourceStatus) -> ShimSourceStatus:
    """Load the provider directory a resolved remote status points at."""
    if not status.ok or not status.resolved_dir:
        return status
    return _scan_dir(
        Path(status.resolved_dir),
        kind=status.kind,
        location=status.location,
        name=status.name,
        from_cache=status.from_cache,
        stale=status.stale,
        fetched_at=status.fetched_at,
        resolved_ref=status.resolved_ref,
    )


@dataclass(frozen=True)
class _RemoteSpec:
    """A normalized remote spec plus its resolved cache paths."""

    url: str
    name: str
    ref: str | None
    subpath: str
    archive: str
    sha256: str | None
    token_env: str | None
    refresh: RefreshMode
    github: tuple[str, str] | None
    source_dir: Path
    stamp_path: Path


def _parse_remote_spec(
    spec: Mapping[str, Any], cache_root: str | os.PathLike[str] | None
) -> _RemoteSpec:
    """Validate and normalize a remote spec, resolving its cache paths.

    Raises:
        ShimSourceError: On a missing/blank ``url``, an unknown ``archive`` or
            ``refresh`` value, or when no cache directory is available.
    """
    if not isinstance(spec, Mapping):
        raise ShimSourceError("remote source must be a mapping")
    url = spec.get("url")
    if not isinstance(url, str) or not url:
        raise ShimSourceError("remote source requires a non-empty 'url'")

    archive = spec.get("archive") or "auto"
    if archive not in ("auto", "tar.gz", "zip"):
        raise ShimSourceError(f"invalid archive type {archive!r}")
    refresh_raw = spec.get("refresh") or "auto"
    if refresh_raw not in _REFRESH_MODES:
        raise ShimSourceError(
            f"invalid refresh mode {refresh_raw!r} (expected auto/always/never)"
        )
    refresh = cast("RefreshMode", refresh_raw)
    if cache_root is None:
        raise ShimSourceError("remote sources require a cache directory")

    _validate_url_scheme(url)
    github = _parse_github(url)
    name = spec.get("name") or _derive_name(url, github)
    ref = spec.get("ref") or None
    root = Path(cache_root)
    key = _cache_key(name, url, ref)
    return _RemoteSpec(
        url=url,
        name=name,
        ref=ref,
        subpath=spec.get("path") or "",
        archive=archive,
        sha256=spec.get("sha256") or None,
        token_env=spec.get("token_env") or None,
        refresh=refresh,
        github=github,
        source_dir=root / key,
        stamp_path=root / f"{key}.stamp.json",
    )


def _needs_fetch(
    spec: _RemoteSpec,
    cached: Mapping[str, Any],
    has_cache: bool,
    now: float,
    ttl: int,
) -> bool:
    """Decide whether *spec* must be (re)fetched.

    A ``sha256`` pin is re-checked against the *cached* archive, not just at
    fetch time: the cache satisfies the pin only when the stamp records the same
    digest, so a rotated or tampered upstream source is still detected on a warm
    cache.  A mismatch always forces a re-fetch (which then fails loudly) even
    under ``refresh: "never"`` — serving bytes that are not the pinned ones is
    never the intent of a pin.
    """
    if has_cache and spec.sha256 is not None:
        cached_digest = str(cached.get("sha256") or "")
        if cached_digest.lower() != spec.sha256.lower():
            return True
    if spec.refresh == "never":
        return not has_cache
    if spec.refresh == "always":
        return True
    if spec.sha256 is not None:
        return not has_cache
    if spec.ref and _SHA_RE.match(spec.ref):
        return not has_cache
    fetched_at = cached.get("fetched_at", 0.0) if cached else 0.0
    return not has_cache or (now - fetched_at) > ttl


def _remote_subdir(spec: _RemoteSpec) -> Path:
    """Return the cache subdirectory that holds the provider directories."""
    return spec.source_dir / spec.subpath if spec.subpath else spec.source_dir


def _remote_failure(
    location: str, name: str | None, exc: Exception
) -> ShimSourceStatus:
    """Build a failed remote status from an exception."""
    return ShimSourceStatus(
        kind="remote",
        location=location,
        name=name,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
    )


# ---------------------------------------------------------------------------
# Local directories
# ---------------------------------------------------------------------------


def _load_local_dir(directory: str | os.PathLike[str]) -> ShimSourceStatus:
    """Scan one local directory, reporting failure instead of raising."""
    return _scan_dir(Path(directory), kind="path", location=str(directory))


def _scan_dir(
    directory: Path,
    *,
    kind: Literal["path", "remote"],
    location: str,
    name: str | None = None,
    from_cache: bool = False,
    stale: bool = False,
    fetched_at: float | None = None,
    resolved_ref: str | None = None,
) -> ShimSourceStatus:
    """Run the loader over *directory* and fold the result into a status."""
    if not directory.is_dir():
        return ShimSourceStatus(
            kind=kind,
            location=location,
            name=name,
            ok=False,
            resolved_dir=str(directory),
            from_cache=from_cache,
            stale=stale,
            fetched_at=fetched_at,
            resolved_ref=resolved_ref,
            error=f"not a directory: {directory}",
        )
    try:
        shims = load_providers_from_dir(directory, on_error="warn")
    except Exception as exc:
        return ShimSourceStatus(
            kind=kind,
            location=location,
            name=name,
            ok=False,
            resolved_dir=str(directory),
            from_cache=from_cache,
            stale=stale,
            fetched_at=fetched_at,
            resolved_ref=resolved_ref,
            error=f"{type(exc).__name__}: {exc}",
        )
    names = tuple(s.name for s in shims)
    return ShimSourceStatus(
        kind=kind,
        location=location,
        name=name,
        ok=True,
        shim_count=len(names),
        shim_names=names,
        resolved_dir=str(directory),
        from_cache=from_cache,
        stale=stale,
        fetched_at=fetched_at,
        resolved_ref=resolved_ref,
    )


# ---------------------------------------------------------------------------
# Remote fetch + extract
# ---------------------------------------------------------------------------


def _fetch_remote(
    spec: _RemoteSpec,
    *,
    headers: dict[str, str],
    http_get: HttpGet,
    now_fn: Callable[[], float],
    proxy: str | None,
) -> dict[str, Any]:
    """Fetch, verify and atomically install one archive; return its stamp."""
    ref = spec.ref
    resolved_ref = ref
    if ref is None and spec.github is not None:
        resolved_ref = _resolve_github_head(
            spec.github[0],
            spec.github[1],
            headers=headers,
            http_get=http_get,
            proxy=proxy,
        )

    archive_url = _archive_url(
        spec.url, resolved_ref if spec.github else ref, spec.github
    )

    with _file_lock(spec.source_dir.parent / f"{spec.source_dir.name}.lock"):
        data = http_get(archive_url, headers=headers, timeout=HTTP_TIMEOUT, proxy=proxy)
        if not isinstance(data, (bytes, bytearray)):
            raise ShimSourceError("http_get must return bytes")
        data = bytes(data)
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ShimSourceError(
                f"archive exceeds {MAX_ARCHIVE_BYTES} bytes ({len(data)})"
            )
        if spec.sha256:
            digest = hashlib.sha256(data).hexdigest()
            if digest.lower() != spec.sha256.lower():
                raise ShimArchiveMismatch(
                    f"sha256 mismatch: expected {spec.sha256}, got {digest}"
                )

        tmp_root = spec.source_dir.parent / ".tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / f"extract-{uuid.uuid4().hex}"
        try:
            tmp_dir.mkdir(parents=False)
            _extract_archive(
                data,
                tmp_dir,
                archive=spec.archive,
                url=archive_url,
                declared_path=spec.subpath,
            )
            _atomic_replace(tmp_dir, spec.source_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        stamp = {
            "url": spec.url,
            "ref": ref,
            "resolved_ref": resolved_ref,
            "subpath": spec.subpath,
            "fetched_at": now_fn(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        _write_stamp(spec.stamp_path, stamp)
        return stamp


def _archive_url(url: str, ref: str | None, github: tuple[str, str] | None) -> str:
    """Translate a declared URL into the archive URL to GET."""
    if github is not None:
        owner, repo = github
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref or 'HEAD'}"
    if "{ref}" in url:
        return url.replace("{ref}", ref or "HEAD")
    if url.lower().endswith(_ARCHIVE_SUFFIXES):
        return url
    raise ShimSourceError(
        "not a GitHub repository URL and not an archive URL; give an explicit "
        "archive URL (optionally containing '{ref}') or a GitHub repo URL"
    )


def _resolve_github_head(
    owner: str,
    repo: str,
    *,
    headers: dict[str, str],
    http_get: HttpGet,
    proxy: str | None,
) -> str:
    """Resolve a GitHub repo's HEAD to a concrete commit SHA."""
    api = f"https://api.github.com/repos/{owner}/{repo}/commits/HEAD"
    data = http_get(api, headers=headers, timeout=HTTP_TIMEOUT, proxy=proxy)
    payload = json.loads(bytes(data).decode("utf-8"))
    sha = payload.get("sha")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise ShimSourceError("could not resolve HEAD to a commit SHA")
    return sha


def _extract_archive(
    data: bytes, dest: Path, *, archive: str, url: str, declared_path: str = ""
) -> None:
    """Dispatch to the tar/zip extractor based on the declared or sniffed type."""
    kind = archive
    if kind == "auto":
        kind = _sniff_archive(data, url)
    if kind == "zip":
        _extract_zip(data, dest, declared_path=declared_path)
    else:
        _extract_tar(data, dest, declared_path=declared_path)


def _sniff_archive(data: bytes, url: str) -> Literal["tar.gz", "zip"]:
    """Guess the archive kind from magic bytes, falling back to the URL."""
    if data[:2] == b"\x1f\x8b" or data[:2] == b"BZ" or data[:6] == b"\xfd7zXZ\x00":
        return "tar.gz"
    if data[:4] == b"PK\x03\x04":
        return "zip"
    if url.lower().endswith(".zip"):
        return "zip"
    return "tar.gz"


def _extract_tar(data: bytes, dest: Path, *, declared_path: str = "") -> None:
    """Safely extract a tar archive into *dest*."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
    except tarfile.TarError as exc:
        raise ShimSourceError(f"not a valid tar archive: {exc}") from exc
    with tar:
        members = tar.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ShimSourceError("archive has too many members")
        strip = _common_root_prefix([m.name for m in members], declared_path)
        total = 0
        for member in members:
            # Validate the *original* name: the common-root strip must not be
            # able to launder a drive prefix or a traversal component.
            _validate_member_name(member.name)
            name = _strip_prefix(member.name, strip)
            if not name:
                continue
            if member.issym() or member.islnk():
                raise ShimSourceError(f"archive contains a link member: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ShimSourceError(
                    f"archive contains a special member: {member.name}"
                )
            target = dest / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > MAX_EXTRACTED_BYTES:
                raise ShimSourceError("extracted archive exceeds size cap")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024)


def _extract_zip(data: bytes, dest: Path, *, declared_path: str = "") -> None:
    """Safely extract a zip archive into *dest*."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ShimSourceError(f"not a valid zip archive: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ShimSourceError("archive has too many members")
        strip = _common_root_prefix([i.filename for i in infos], declared_path)
        total = 0
        for info in infos:
            _validate_member_name(info.filename)
            name = _strip_prefix(info.filename, strip)
            if not name:
                continue
            mode = info.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise ShimSourceError(
                    f"archive contains a symlink member: {info.filename}"
                )
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += info.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ShimSourceError("extracted archive exceeds size cap")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 64 * 1024)


def _common_root_prefix(names: Sequence[str], declared_path: str = "") -> str:
    """Return the single top-level directory all *names* share, else ``""``.

    GitHub/GitLab archives wrap everything in one ``<repo>-<ref>/`` directory
    whose name varies by ref; stripping it makes a declared ``path`` subpath
    stable across refs.

    A declared *declared_path* wins over the guess: if it already names the
    shared root (or a path underneath it), nothing is stripped, otherwise the
    heuristic would remove the very directory the caller pointed at.
    """
    roots = {n.replace("\\", "/").split("/", 1)[0] for n in names if n}
    if len(roots) != 1:
        return ""
    root = next(iter(roots))
    if root in ("", ".", ".."):
        return ""
    declared = declared_path.replace("\\", "/").strip("/")
    if declared == root or declared.startswith(root + "/"):
        return ""
    return root


def _strip_prefix(name: str, prefix: str) -> str:
    """Drop *prefix* and the following slash from *name* (no-op if absent)."""
    if not prefix:
        return name
    if name == prefix:
        return ""
    if name.startswith(prefix + "/"):
        return name[len(prefix) + 1 :]
    return name


def _validate_member_name(name: str) -> None:
    """Reject archive member names that could escape the extraction root.

    Backslashes are normalised first, so a Windows-style ``..\\`` traversal or
    a drive-prefixed path cannot slip past the checks (and cannot be laundered
    by the common-root strip, which runs after validation).
    """
    if "\x00" in name:
        raise ShimSourceError("archive member name contains NUL")
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if normalized.startswith("/") or (parts and _DRIVE_RE.match(parts[0])):
        raise ShimSourceError(f"archive contains an absolute path: {name}")
    if ".." in parts:
        raise ShimSourceError(f"archive contains a traversal path: {name}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(name: str, url: str, ref: str | None) -> str:
    """Return a stable, filesystem-safe cache directory name."""
    digest = hashlib.sha256(f"{url}@{ref or 'HEAD'}".encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.") or "source"
    return f"{slug}-{digest}"


def _read_stamp(path: Path) -> dict[str, Any]:
    """Read a cache stamp, returning ``{}`` when missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_stamp(path: Path, stamp: Mapping[str, Any]) -> None:
    """Write a cache stamp atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(stamp), f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _atomic_replace(src: Path, dest: Path) -> None:
    """Replace directory *dest* with *src* as atomically as the OS allows."""
    if dest.exists():
        old = dest.with_name(f"{dest.name}.old-{uuid.uuid4().hex}")
        os.replace(dest, old)
        try:
            os.replace(src, dest)
        finally:
            shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(src, dest)


@contextlib.contextmanager
def _file_lock(path: Path):
    """Best-effort cross-process advisory lock (POSIX ``flock``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


# ---------------------------------------------------------------------------
# URL / HTTP helpers
# ---------------------------------------------------------------------------


def _parse_github(url: str) -> tuple[str, str] | None:
    """Return ``(owner, repo)`` for a GitHub repository URL, else ``None``."""
    m = _GITHUB_RE.match(url)
    if m is None:
        return None
    return m.group("owner"), m.group("repo")


def _derive_name(url: str, github: tuple[str, str] | None) -> str:
    """Derive a source name from a URL when none is given."""
    if github is not None:
        return github[1]
    tail = url.rstrip("/").split("/")[-1]
    return tail or "source"


def _validate_url_scheme(url: str) -> None:
    """Require HTTPS, except loopback HTTP (local test servers)."""
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme == "https":
        return
    if scheme == "http":
        host = url.split("://", 1)[1].split("/", 1)[0].split("@")[-1]
        hostname = host.split(":", 1)[0]
        if hostname in ("127.0.0.1", "localhost", "::1", "[::1]"):
            return
        raise ShimSourceError(
            "remote shim sources must use https:// (loopback may use http)"
        )
    raise ShimSourceError(f"unsupported URL scheme in {url!r}")


def _auth_headers(token_env: str | None) -> dict[str, str]:
    """Build request headers, adding a bearer token when configured."""
    headers = {"User-Agent": "llm-rosetta-shim-loader", "Accept": "*/*"}
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise ShimSourceError(
                f"token_env {token_env!r} is not set in the environment"
            )
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _default_http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = HTTP_TIMEOUT,
    proxy: str | None = None,
) -> bytes:
    """Fetch *url* with the vendored synchronous HTTP client.

    The body is read incrementally and the read is aborted as soon as it would
    exceed :data:`MAX_ARCHIVE_BYTES`, so the cap is a real memory bound for the
    default getter rather than a post-hoc sanity check.  (An injected
    ``http_get`` still only gets the backstop check in :func:`_fetch_remote`.)
    """
    from llm_rosetta._vendor.httpclient import Client, Response

    chunk_size = 64 * 1024
    with Client(headers=headers, timeout=timeout, proxy=proxy) as client:
        response = client.get(url, stream=True)
        if isinstance(response, Response):  # pragma: no cover - stream=True
            raise ShimSourceError("expected a streaming response")
        with response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(chunk_size):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise ShimSourceError(
                        f"archive exceeds {MAX_ARCHIVE_BYTES} bytes; download aborted"
                    )
                chunks.append(chunk)
            return b"".join(chunks)


__all__ = [
    "ShimSourceError",
    "ShimArchiveMismatch",
    "ShimSourceStatus",
    "ShimLoadReport",
    "rebuild_shims",
    "resolve_remote",
    "DEFAULT_CACHE_DIRNAME",
    "REMOTE_TTL_SECONDS",
]

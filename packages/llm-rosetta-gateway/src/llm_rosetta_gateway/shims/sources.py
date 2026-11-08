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
from urllib.parse import quote, urlsplit
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from llm_rosetta.shims.provider_shim import (
    _clear_registry_caches,
    list_shims,
    register_shim,
    staging_registry,
    unregister_shim,
)
from llm_rosetta.shims.providers import load_providers, load_providers_from_dir

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
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
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

#: The leg that last served each remote source, keyed by its cache directory
#: name, so a working mirror is not re-probed on every fetch.  Process-local:
#: a restart costs one failed connect, which is the documented cost of the
#: lazy (non-pre-probing) selection.
_chosen_legs: dict[str, str] = {}
_chosen_legs_lock = threading.Lock()


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
    disabled: Sequence[str] = (),
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
        disabled: Shim names to leave out of the registry once everything has
            loaded.  A name is dropped after its source registers it, so the
            sources stay the one account of what a configuration declares.

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

            # Disabled names go last: every source has registered by now, so a
            # name is dropped whatever source would have provided it.
            for name in disabled:
                unregister_shim(name)

        if disabled:
            off = set(disabled)
            statuses = [
                replace(
                    status,
                    shim_names=tuple(n for n in status.shim_names if n not in off),
                    shim_count=len([n for n in status.shim_names if n not in off]),
                )
                for status in statuses
            ]

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
    ``archive`` (``auto`` / ``tar.gz`` / ``zip``), ``sha256``, ``token_env``,
    ``refresh`` (``auto`` / ``always`` / ``never``) and ``mirror``.

    A ``mirror`` is a second URL for the same source — a template string or an
    object with ``url_template`` and its own ``sha256``.  It is consulted only
    when the primary fails with a transport error (connection, DNS, timeout);
    an HTTP error or a bad archive from the primary is the source's verdict
    and does not fall back.

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
                # A digest mismatch means the bytes are wrong, not missing,
                # and an unusable archive means the server answered with
                # something that is not the source: neither is masked with the
                # stale cache.  A transport failure still is.
                if not has_cache or isinstance(
                    exc, (ShimArchiveMismatch, ShimArchiveUnusable)
                ):
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
            resolved_dir=str(_remote_subdir(parsed, _stamp_leg(cached))),
            from_cache=from_cache,
            stale=stale,
            fetched_at=cached.get("fetched_at"),
            resolved_ref=cached.get("resolved_ref") or parsed.ref,
            ref=parsed.ref,
            token_env=parsed.token_env,
            leg=_stamp_leg(cached),
            archive_url=cached.get("archive_url"),
        )
    except Exception as exc:
        logger.warning("Shim remote %s: %s", location, exc)
        return _remote_failure(location, name, exc)


def _scan_status(status: ShimSourceStatus) -> ShimSourceStatus:
    """Load the provider directory a resolved remote status points at."""
    if not status.ok or not status.resolved_dir:
        return status
    scanned = _scan_dir(
        Path(status.resolved_dir),
        kind=status.kind,
        location=status.location,
        name=status.name,
        from_cache=status.from_cache,
        stale=status.stale,
        fetched_at=status.fetched_at,
        resolved_ref=status.resolved_ref,
        ref=status.ref,
        token_env=status.token_env,
    )
    return replace(scanned, leg=status.leg, archive_url=status.archive_url)


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
    mirror_url: str | None = None
    mirror_sha256: str | None = None
    mirror_token_env: str | None = None
    #: Scan directory inside the *mirror's* archive.  ``None`` inherits
    #: ``subpath``: the legs need different paths only when the archives they
    #: serve are laid out differently.
    mirror_subpath: str | None = None


def normalize_subpath(value: Any, *, where: str) -> str:
    """Normalize a declared scan directory, refusing anything that can escape.

    Both a source's ``path`` and a mirror's ``path`` name a directory inside an
    extracted archive.  ``None`` and ``""`` mean the extraction root, as does
    ``"."``; anything else must be a relative path of ordinary components, so a
    declaration cannot point at an absolute location or walk out of the cache
    directory (the extractor already refuses such members; this refuses the
    *declaration*).

    Args:
        value: The declared value.
        where: The config key, for the error message.

    Returns:
        The normalized relative path, or ``""`` for the archive root.

    Raises:
        ValueError: On a non-string, an absolute path, a backslash spelling, or
            a ``..`` component.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string path")
    if "\\" in value:
        raise ValueError(f"{where} must use '/' separators")
    text = value
    if text in ("", ".", "./"):
        return ""
    if text.startswith("/"):
        raise ValueError(f"{where} must be a relative path inside the archive")
    if len(text) > 1 and text[1] == ":":
        raise ValueError(f"{where} must be a relative path inside the archive")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        return ""
    if any(part == ".." for part in parts):
        raise ValueError(f"{where} must not contain '..'")
    return "/".join(parts)


def normalize_mirror(
    value: Any,
) -> tuple[str, str | None, str | None, str | None] | None:
    """Validate a ``mirror`` declaration and return its parts.

    A mirror is a second URL for the same source, used only when the primary
    is unreachable.  It is an ordinary archive URL template, so any host can
    serve it — a public CDN and an internal mirror are declared the same way.

    Accepted shapes::

        "mirror": "https://mirror.example/shims@{ref}.tar.gz"
        "mirror": {"url_template": "https://cdn.example/gh/o/r@{ref}/dist/shims.tar.gz",
                   "sha256": "0123…abcd",
                   "token_env": "MIRROR_TOKEN",
                   "path": "."}

    The bytes a mirror serves need not be the primary's bytes (a CDN serving a
    committed ``dist/<name>.tar.gz`` differs from the forge's repository
    tarball), so the mirror carries its own ``sha256``.

    A mirror never receives the primary's ``token_env``: that credential
    belongs to the primary's origin.  A mirror that needs its own declares it
    here.

    ``path`` is the directory to scan inside the mirror's archive.  Omitted, it
    inherits the source's ``path`` — the common case, where both legs serve the
    same tree.  It exists because a committed ``dist`` archive is laid out from
    the providers directory (``providers/`` at its root) while the forge
    tarball keeps the repository's ``src/<package>/providers``: one path cannot
    address both, and the mirror descriptor is what describes the mirror's
    bytes.

    Args:
        value: The declared ``mirror``: ``None``, a template string, or an
            object with ``url_template`` and optional ``sha256`` /
            ``token_env`` / ``path``.

    Returns:
        ``(url_template, sha256 or None, token_env or None, path or None)``,
        or ``None`` when no mirror is declared.

    Raises:
        ValueError: On an unknown shape, an unknown key, a template that is
            not an archive URL the loader accepts, a malformed digest, or a
            ``path`` that is not a relative directory inside the archive.
    """
    if value is None:
        return None
    pin: Any = None
    token_env: Any = None
    subpath: Any = None
    if isinstance(value, str):
        template = value
    elif isinstance(value, Mapping):
        unknown = set(value) - {"url_template", "sha256", "token_env", "path"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(
                f"unknown mirror key(s): {names} "
                "(expected url_template, sha256, token_env, path)"
            )
        template = value.get("url_template")
        pin = value.get("sha256")
        token_env = value.get("token_env")
        subpath = value.get("path")
        if not isinstance(template, str) or not template:
            raise ValueError("mirror requires a non-empty 'url_template'")
    else:
        raise ValueError(
            "mirror must be a URL template string or an object with 'url_template'"
        )
    if not template:
        raise ValueError("mirror requires a non-empty 'url_template'")
    if pin is not None and (not isinstance(pin, str) or not _SHA256_RE.match(pin)):
        raise ValueError("mirror 'sha256' must be 64 hexadecimal characters")
    if token_env is not None and (not isinstance(token_env, str) or not token_env):
        raise ValueError("mirror 'token_env' must be an environment variable name")
    if not is_supported_archive_url(template):
        raise ValueError(
            "mirror 'url_template' must be an archive URL "
            "(.tar.gz/.tgz/.tar/.zip), a GitHub repository URL, or contain "
            "'{ref}'"
        )
    try:
        _validate_url_scheme(template)
    except ShimSourceError as exc:
        raise ValueError(str(exc)) from None
    # An absent ``path`` inherits the source's (``None``); a declared one — even
    # ``"."`` or ``""`` — names the mirror's own scan directory.
    declared_path = (
        None if subpath is None else normalize_subpath(subpath, where="mirror 'path'")
    )
    return template, pin, token_env, declared_path


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
    try:
        subpath = normalize_subpath(spec.get("path"), where="'path'")
        mirror = normalize_mirror(spec.get("mirror"))
    except ValueError as exc:
        raise ShimSourceError(str(exc)) from None
    mirror_url, mirror_sha256, mirror_token_env, mirror_subpath = mirror or (
        None,
        None,
        None,
        None,
    )
    if mirror_url and spec.get("sha256") and not mirror_sha256:
        # A pinned source relies on the pin to notice bytes it did not approve.
        # The mirror is the path an on-path attacker can force (make the
        # primary unreachable, serve whatever the mirror answers), so an
        # unpinned mirror would quietly undo the pin it was declared beside.
        raise ShimSourceError(
            "a mirror must declare its own 'sha256' when the source pins its "
            "primary: the two legs serve different bytes, and an unpinned "
            "mirror would serve bytes the pin never approved"
        )
    name = spec.get("name") or _derive_name(url, github)
    ref = spec.get("ref") or None
    root = Path(cache_root)
    key = _cache_key(name, url, ref)
    return _RemoteSpec(
        url=url,
        name=name,
        ref=ref,
        subpath=subpath,
        archive=archive,
        sha256=spec.get("sha256") or None,
        token_env=spec.get("token_env") or None,
        refresh=refresh,
        github=github,
        source_dir=root / key,
        stamp_path=root / f"{key}.stamp.json",
        mirror_url=mirror_url,
        mirror_sha256=mirror_sha256,
        mirror_token_env=mirror_token_env,
        mirror_subpath=mirror_subpath,
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

    The pin checked is the one belonging to the leg that produced the cache: a
    cache fetched through the mirror is judged by the mirror's pin, because the
    two legs legitimately serve different bytes.  A stamp written before legs
    were recorded came from the primary.
    """
    pinned = spec.mirror_sha256 if _stamp_leg(cached) == "mirror" else spec.sha256
    if has_cache and pinned is not None:
        cached_digest = str(cached.get("sha256") or "")
        if cached_digest.lower() != pinned.lower():
            return True
    if spec.refresh == "never":
        return not has_cache
    if has_cache and spec.github is not None and pinned is None:
        # Stamps written before refs were resolved — or written while the API
        # was unreachable — hold the literal ref; a commit is what the update
        # check can compare against.  Checked after ``never``: the operator's
        # schedule wins, and the pin is re-resolved on the next fetch it allows.
        #
        # A digest pin is exempt: the bytes are the identity, the commit adds
        # nothing to the verdict, and re-pinning is what made a warm cache
        # cold — one API attempt on every reload, in exactly the environment
        # the mirror exists for (#1433).
        if not _SHA_RE.match(str(cached.get("resolved_ref") or "")):
            return True
    if spec.refresh == "always":
        return True
    if spec.sha256 is not None or spec.mirror_sha256 is not None:
        return not has_cache
    if spec.ref and _SHA_RE.match(spec.ref):
        return not has_cache
    fetched_at = cached.get("fetched_at", 0.0) if cached else 0.0
    return not has_cache or (now - fetched_at) > ttl


def _leg_subpath(spec: _RemoteSpec, leg: str) -> str:
    """Return the scan directory for *leg* inside its archive.

    A mirror that declares its own ``path`` is described by it; otherwise the
    two legs share the source's path, which is the case when both serve the
    same tree.
    """
    if leg == "mirror" and spec.mirror_subpath is not None:
        return spec.mirror_subpath
    return spec.subpath


def _remote_subdir(spec: _RemoteSpec, leg: str) -> Path:
    """Return the cache subdirectory that holds the provider directories."""
    subpath = _leg_subpath(spec, leg)
    return spec.source_dir / subpath if subpath else spec.source_dir


def _stamp_leg(stamp: Mapping[str, Any]) -> Literal["primary", "mirror"]:
    """Return the leg a cache stamp records; pre-leg stamps were primary."""
    return "mirror" if stamp.get("leg") == "mirror" else "primary"


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
        unreachable=isinstance(exc, ShimSourceUnreachable),
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
    ref: str | None = None,
    token_env: str | None = None,
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
            ref=ref,
            token_env=token_env,
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
            ref=ref,
            token_env=token_env,
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
        ref=ref,
        token_env=token_env,
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
    """Fetch, verify and atomically install one archive; return its stamp.

    The primary URL is attempted first.  A transport failure (connection
    refused, DNS failure, timeout) moves on to the declared mirror, and so does
    any failure of the mirror itself — it is a stand-in, not an authority, so
    its answer never decides for a primary nobody asked.  On the primary, a
    server answer (an HTTP error, a digest mismatch) or a corrupt archive is
    the source's verdict and never triggers the fallback.  The leg that last
    worked is remembered per source, so a working mirror costs one failed
    connect for the process lifetime, not one per fetch.
    """
    ref = spec.ref
    resolved_ref = ref
    if spec.github is not None:
        # A declared ref ("main", "v1.0.0") is resolved to the commit it points
        # at *now*, so the stamp records a commit and the update check compares
        # commit with commit.  Codeload accepts either spelling for the archive.
        try:
            resolved_ref = _resolve_github_head(
                spec.github[0],
                spec.github[1],
                ref=ref,
                headers=headers,
                http_get=http_get,
                proxy=proxy,
            )
        except Exception as exc:  # noqa: BLE001 - the pin must not gate the install
            # The commit is for the update check, not a precondition for
            # installing the source: an anonymous rate limit, or a network that
            # allows codeload but not api.github.com, still gets the archive.
            # The stamp then holds the ref itself, and _needs_fetch re-pins it
            # on the next fetch.
            logger.warning(
                "Shim remote %s: cannot resolve %s to a commit (%s); "
                "pinning the ref itself",
                spec.url,
                ref or "HEAD",
                exc,
            )
            resolved_ref = ref

    effective_ref = resolved_ref if spec.github else ref
    primary_url = _archive_url(spec.url, effective_ref, spec.github)
    # The mirror is addressed by the ref the operator declared, not by the
    # commit the forge resolved it to: a mirror publishes one directory per
    # release (``...@dist/v0.1.0/...``, #1412), and its URL must not depend on
    # api.github.com answering — an anonymous rate limit would otherwise turn
    # a working fallback into a 404.  Its bytes are pinned by ``sha256``, so
    # the release spelling is what the pin is about.
    legs = _leg_plan(spec, primary_url, ref=effective_ref, mirror_ref=ref)

    with _file_lock(spec.source_dir.parent / f"{spec.source_dir.name}.lock"):
        return _fetch_installable(
            spec,
            legs,
            ref=ref,
            resolved_ref=resolved_ref,
            headers=headers,
            http_get=http_get,
            proxy=proxy,
            now_fn=now_fn,
        )


def _fetch_installable(
    spec: _RemoteSpec,
    legs: Sequence[tuple[str, str, str | None]],
    *,
    ref: str | None,
    resolved_ref: str | None,
    headers: dict[str, str],
    http_get: HttpGet,
    proxy: str | None,
    now_fn: Callable[[], float],
) -> dict[str, Any]:
    """Install the first leg whose body is usable, and stamp what it produced.

    The stamp records the leg that served, its URL and the digest of its
    bytes — not the primary's — so a cache fetched through the mirror keeps
    being judged by the mirror's pin.
    """
    leg, archive_url, _pin, digest = _fetch_first_usable(
        legs, spec, headers=headers, http_get=http_get, proxy=proxy
    )
    stamp = {
        "url": spec.url,
        "ref": ref,
        "resolved_ref": resolved_ref,
        "subpath": spec.subpath,
        "fetched_at": now_fn(),
        "sha256": digest,
        "leg": leg,
        "archive_url": archive_url,
    }
    _write_stamp(spec.stamp_path, stamp)
    with _chosen_legs_lock:
        _chosen_legs[spec.source_dir.name] = leg
    return stamp


def _install_archive(
    spec: _RemoteSpec,
    data: bytes,
    *,
    leg: str,
    archive_url: str,
    pin: str | None,
) -> str:
    """Verify, extract and atomically install one downloaded archive.

    Returns:
        The archive's sha256, for the stamp.

    Raises:
        ShimArchiveMismatch: The bytes do not match the leg's pin.
        ShimArchiveUnusable: The body is oversized, not an archive, or unsafe.
        OSError: This host could not create or write the extracted tree — a
            local condition, not a verdict on the body.
    """
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ShimArchiveUnusable(
            f"archive exceeds {MAX_ARCHIVE_BYTES} bytes ({len(data)})"
        )
    digest = hashlib.sha256(data).hexdigest()
    if pin and digest.lower() != pin.lower():
        raise ShimArchiveMismatch(
            f"sha256 mismatch for source {spec.url} via {leg} "
            f"({archive_url}): expected {pin}, got {digest}"
        )

    tmp_root = spec.source_dir.parent / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / f"extract-{uuid.uuid4().hex}"
    try:
        tmp_dir.mkdir(parents=False)
        try:
            _extract_archive(
                data,
                tmp_dir,
                archive=spec.archive,
                url=archive_url,
                declared_path=_leg_subpath(spec, leg),
            )
        except ShimSourceError as exc:
            # The bytes arrived and are not the source: a corruption signal
            # that must not be softened into the stale cache.
            raise ShimArchiveUnusable(str(exc)) from exc
        _atomic_replace(tmp_dir, spec.source_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return digest


def _leg_plan(
    spec: _RemoteSpec,
    primary_url: str,
    *,
    ref: str | None,
    mirror_ref: str | None = None,
) -> list[tuple[str, str, str | None]]:
    """Return the legs to try, in order, as ``(leg, url, sha256 pin)``.

    *ref* fills the primary's ``{ref}`` (the resolved commit when the forge
    resolved one); *mirror_ref* fills the mirror's, defaulting to the same
    value.  The caller passes the declared ref for the mirror so its URL does
    not depend on the forge's API.

    The leg that last served this source is tried first, so a mirror chosen
    because the primary was unreachable is not re-probed on every fetch.  If
    the remembered leg fails with a transport error, the other leg is still
    tried — the mirror may be down while the primary has recovered.
    """
    legs: list[tuple[str, str, str | None]] = [("primary", primary_url, spec.sha256)]
    if spec.mirror_url:
        mirror_slot = ref if mirror_ref is None else mirror_ref
        legs.append(
            (
                "mirror",
                spec.mirror_url.replace("{ref}", mirror_slot or "HEAD"),
                spec.mirror_sha256,
            )
        )
    with _chosen_legs_lock:
        chosen = _chosen_legs.get(spec.source_dir.name)
    if chosen == "mirror" and len(legs) == 2:
        legs.reverse()
    return legs


def _fetch_first_usable(
    legs: Sequence[tuple[str, str, str | None]],
    spec: _RemoteSpec,
    *,
    headers: dict[str, str],
    http_get: HttpGet,
    proxy: str | None,
) -> tuple[str, str, str | None, str]:
    """Fetch the first leg whose body can be installed.

    A leg is judged whole: the request, the size, the pin and the extraction
    happen inside this loop, so a leg that answers with something unusable has
    not served the source and the next leg is still asked.  That falls out of
    the same rule as a transport failure for the mirror — it stands in for the
    primary and is not an authority — while the primary's own answer, a pin
    mismatch and a corrupt primary body stay final.

    Returns:
        ``(leg, archive_url, pin, digest)``.

    Raises:
        ShimArchiveMismatch: A leg served bytes that do not match its pin.
        ShimSourceUnreachable: No leg answered (a transport failure on each).
        ShimSourceError: A leg answered with an error, or with a body that
            cannot be installed (:class:`ShimArchiveUnusable`).
    """
    failures: list[str] = []
    all_unreachable = True
    unusable = False
    primary_url = next((url for name, url, _ in legs if name == "primary"), legs[0][1])
    for index, (leg, url, pin) in enumerate(legs):
        try:
            data = http_get(
                url,
                headers=_leg_headers(
                    headers, leg=leg, url=url, primary_url=primary_url, spec=spec
                ),
                timeout=HTTP_TIMEOUT,
                proxy=proxy,
            )
        except Exception as exc:
            unreachable = _is_unreachable(exc)
            # A getter that raises the unusable class — the default one does
            # when the stream crosses the size cap — made the same judgement
            # the install step makes: the body is not the source.  Carry it,
            # so the error this loop returns keeps the class the cache reads
            # as "do not mask".
            unusable = unusable or isinstance(exc, ShimArchiveUnusable)
            all_unreachable = all_unreachable and unreachable
            failures.append(f"{leg} {url}: {type(exc).__name__}: {exc}")
            # A transport failure moves on to the other leg.  So does any
            # failure of the mirror: it stands in for the primary and is not
            # an authority, so its error must not decide for a primary nobody
            # asked — a CDN that has not got the version yet says nothing about
            # the forge.  The primary's own verdicts stay final.
            if index + 1 < len(legs) and (unreachable or leg == "mirror"):
                _log_leg_failure(spec, url, exc, unreachable, legs[index + 1])
                continue
            raise _exhausted_error(
                exc,
                spec=spec,
                leg=leg,
                url=url,
                failures=failures,
                all_unreachable=all_unreachable,
                unusable=unusable,
            ) from exc
        if not isinstance(data, (bytes, bytearray)):
            raise ShimSourceError("http_get must return bytes")
        try:
            digest = _install_archive(
                spec, bytes(data), leg=leg, archive_url=url, pin=pin
            )
        except ShimArchiveMismatch:
            raise
        except OSError as exc:
            # A local wall — a full disk, a permission, a read-only tree — says
            # nothing about the body: the bytes may be perfect and this host
            # cannot write them.  It is not a corruption signal (the class the
            # stale cache must not soften), it is not a transport failure, and
            # asking the other leg would need the same disk, so it is neither
            # retried nor reported as the source's verdict.
            raise ShimSourceError(
                f"cannot install {leg} {url} locally: {type(exc).__name__}: {exc}"
            ) from exc
        except ShimSourceError as exc:
            # The server answered with something that is not the source.  On
            # an unpinned mirror that is a bad replica — the primary is still
            # asked.  Everywhere else it is final: the primary's answer, or a
            # pinned mirror whose bytes are the operator's declaration.
            unusable = True
            all_unreachable = False
            failures.append(f"{leg} {url}: {type(exc).__name__}: {exc}")
            if index + 1 < len(legs) and leg == "mirror" and not pin:
                _log_leg_failure(spec, url, exc, False, legs[index + 1])
                continue
            raise _exhausted_error(
                exc,
                spec=spec,
                leg=leg,
                url=url,
                failures=failures,
                all_unreachable=all_unreachable,
                unusable=unusable,
            ) from exc
        return leg, url, pin, digest
    raise ShimSourceError("remote source has no URL to fetch")  # pragma: no cover


def _log_leg_failure(
    spec: _RemoteSpec,
    url: str,
    exc: BaseException,
    unreachable: bool,
    next_leg: tuple[str, str, str | None],
) -> None:
    """Say which leg failed and which one is about to be tried."""
    logger.warning(
        "Shim source %s: %s %s (%s); trying the %s %s",
        spec.url,
        url,
        "unreachable" if unreachable else "failed",
        exc,
        next_leg[0],
        next_leg[1],
    )


def _exhausted_error(
    exc: BaseException,
    *,
    spec: _RemoteSpec,
    leg: str,
    url: str,
    failures: list[str],
    all_unreachable: bool,
    unusable: bool = False,
) -> ShimSourceError:
    """The error for a leg that failed and has no untried leg behind it.

    A failure of every leg with every one of them a transport failure is
    :class:`ShimSourceUnreachable`; a body that could not be installed is
    :class:`ShimArchiveUnusable`; anything else is a plain source error.  When
    more than one leg failed the message names each.

    The type is the rule: ``_resolve_remote_source`` re-raises the two
    "the bytes are wrong" classes instead of degrading to the cache, and
    degrades everything else.
    """
    if all_unreachable:
        if len(failures) == 1:
            where = url if url == spec.url else f"{url} (source {spec.url})"
            return ShimSourceUnreachable(
                f"cannot reach {where}: {type(exc).__name__}: {exc}"
            )
        return ShimSourceUnreachable(
            f"no leg reachable for source {spec.url}: " + "; ".join(failures)
        )
    error_type: type[ShimSourceError] = (
        ShimArchiveUnusable if unusable else ShimSourceError
    )
    if len(failures) > 1:
        return error_type(
            f"no usable leg for source {spec.url}: " + "; ".join(failures)
        )
    return error_type(f"{leg} {url}: {type(exc).__name__}: {exc}")


def _is_unreachable(exc: BaseException) -> bool:
    """Whether *exc* is a transport failure — no response from the server.

    Only this class of failure falls back to the mirror.  An HTTP status is a
    response from the source itself, and a bad archive is the source's
    verdict; neither is a network symptom to route around.

    The transport subclasses are named rather than taking ``OSError`` whole:
    what is left of ``OSError`` here is local (a permission error, a path of
    the wrong type), which says nothing about whether the host answers.
    """
    from llm_rosetta_gateway._vendor.httpclient import (
        HTTPError,
        HttpConnectionError,
        HttpTimeoutError,
    )

    if isinstance(exc, HTTPError):
        return False
    return isinstance(
        exc, (HttpConnectionError, HttpTimeoutError, ConnectionError, TimeoutError)
    )


def _leg_headers(
    headers: dict[str, str],
    *,
    leg: str,
    url: str,
    primary_url: str,
    spec: _RemoteSpec,
) -> dict[str, str]:
    """Request headers for one leg.

    A bearer token belongs to the origin that declared it.  The primary's
    ``token_env`` therefore rides only on a leg whose origin is the primary's:
    a mirror on another host never sees it, and one that needs a credential
    declares its own ``token_env``.
    """
    token_env = spec.token_env
    if leg == "mirror":
        if spec.mirror_token_env:
            token_env = spec.mirror_token_env
        elif not _same_origin(url, primary_url):
            token_env = None
    leg_headers = {
        name: value
        for name, value in headers.items()
        if name.lower() != "authorization"
    }
    if token_env:
        leg_headers.update(_bearer_headers(token_env))
    return leg_headers


def _same_origin(first: str, second: str) -> bool:
    """Whether two URLs share a scheme, host and port."""
    left, right = _url_origin(first), _url_origin(second)
    return left is not None and left == right


#: The port a URL scheme implies when it names none.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_origin(url: str) -> tuple[str, str, int | None] | None:
    """The ``(scheme, host, port)`` an Authorization header would reach.

    A scheme's default port is filled in, so ``https://h`` and ``https://h:443``
    are one origin: writing the default explicitly must not cost the credential.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return scheme, parts.hostname.lower(), port


def is_supported_archive_url(url: str) -> bool:
    """Whether the loader can turn *url* into a fetchable archive.

    The rule :func:`_archive_url` applies, exposed so a declared mirror is
    validated by the same code that will fetch it: a GitHub repository URL, a
    URL containing ``{ref}``, or a URL ending in one of the archive suffixes.
    """
    return (
        _parse_github(url) is not None
        or "{ref}" in url
        or url.lower().endswith(_ARCHIVE_SUFFIXES)
    )


def _archive_url(url: str, ref: str | None, github: tuple[str, str] | None) -> str:
    """Translate a declared URL into the archive URL to GET."""
    if github is not None:
        owner, repo = github
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref or 'HEAD'}"
    if "{ref}" in url:
        return url.replace("{ref}", ref or "HEAD")
    if is_supported_archive_url(url):
        return url
    raise ShimSourceError(
        "not a GitHub repository URL and not an archive URL; give an explicit "
        "archive URL (optionally containing '{ref}') or a GitHub repo URL"
    )


def _resolve_github_head(
    owner: str,
    repo: str,
    *,
    ref: str | None = None,
    headers: dict[str, str],
    http_get: HttpGet,
    proxy: str | None,
) -> str:
    """Resolve a GitHub ref (default: HEAD) to a concrete commit SHA."""
    target = quote(ref, safe="") if ref else "HEAD"
    api = f"https://api.github.com/repos/{owner}/{repo}/commits/{target}"
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


def check_remote_updates(
    report: ShimLoadReport,
    *,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Compare each remote source's pinned commit with its repository HEAD.

    A remote is pinned to the commit it was fetched at (``resolved_ref``), so
    "is there an update?" is one API call per source: resolve ``HEAD`` and see
    whether it moved.  Only GitHub sources can answer — an archive URL has no
    head to ask for — and a source that failed to load reports the error
    instead of a verdict.

    Args:
        report: The last applied load report.
        http_get: Injectable HTTP getter (tests).
        proxy: Upstream proxy; defaults to the caller's.

    Returns:
        One entry per remote source: its location, the pinned commit, the
        current head (``None`` when it could not be read) and whether the head
        differs.
    """
    get = http_get or _default_http_get
    out: list[dict[str, Any]] = []
    for status in report.sources:
        if status.kind != "remote":
            continue
        entry: dict[str, Any] = {
            "location": status.location,
            "current": status.resolved_ref,
            "head": None,
            "update": False,
            "error": None,
        }
        if not status.ok:
            # A source that never loaded has no pin to compare; the badge must
            # say so rather than claim the (absent) pin is current.
            entry["error"] = status.error or "source failed to load"
            out.append(entry)
            continue
        pin = status.resolved_ref
        if not pin or not _SHA_RE.match(pin):
            entry["error"] = f"pinned at {pin!r}, not a commit — refetch to re-pin"
            out.append(entry)
            continue
        parts = _parse_github(status.location)
        if parts is None:
            entry["error"] = "not a GitHub repository"
            out.append(entry)
            continue
        try:
            head = _resolve_github_head(
                parts[0],
                parts[1],
                ref=status.ref,
                headers=_auth_headers(status.token_env),
                http_get=get,
                proxy=proxy,
            )
        except Exception as exc:  # noqa: BLE001 - a check must not break the page
            entry["error"] = str(exc)
            out.append(entry)
            continue
        entry["head"] = head
        entry["update"] = head != pin
        out.append(entry)
    return out


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
        headers.update(_bearer_headers(token_env))
    return headers


def _bearer_headers(token_env: str) -> dict[str, str]:
    """The ``Authorization`` header for *token_env*, or an error when it is unset."""
    token = os.environ.get(token_env)
    if not token:
        raise ShimSourceError(f"token_env {token_env!r} is not set in the environment")
    return {"Authorization": f"Bearer {token}"}


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
    default getter rather than a post-hoc sanity check.  The abort raises
    :class:`ShimArchiveUnusable`, the same verdict :func:`_install_archive`
    gives an oversized body, so the two layers cannot disagree about whether a
    warm cache may mask it.  (An injected ``http_get`` still only gets the
    backstop check in :func:`_fetch_remote`, unless it raises that class
    itself.)
    """
    from llm_rosetta_gateway._vendor.httpclient import Client, Response

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
                    # The same verdict the post-download check gives, raised
                    # earlier: the source answered with bytes that cannot be
                    # the archive, so the stale cache must not mask them.
                    raise ShimArchiveUnusable(
                        f"archive exceeds {MAX_ARCHIVE_BYTES} bytes; download aborted"
                    )
                chunks.append(chunk)
            return b"".join(chunks)


__all__ = [
    "ShimSourceError",
    "ShimArchiveMismatch",
    "ShimSourceUnreachable",
    "ShimSourceStatus",
    "ShimLoadReport",
    "rebuild_shims",
    "resolve_remote",
    "is_supported_archive_url",
    "normalize_mirror",
    "DEFAULT_CACHE_DIRNAME",
    "REMOTE_TTL_SECONDS",
]

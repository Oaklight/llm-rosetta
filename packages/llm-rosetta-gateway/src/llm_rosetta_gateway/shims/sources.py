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

import hashlib
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_rosetta.shims.provider_shim import (
    _clear_registry_caches,
    list_shims,
    register_shim,
    staging_registry,
    unregister_shim,
)
from llm_rosetta.shims.providers import (  # noqa: F401
    load_providers,
    load_providers_from_dir,
)

# ``MAX_ARCHIVE_BYTES`` is bound into this module and read here (the size
# check in ``_install_archive``), which is what keeps the tests'
# ``monkeypatch.setattr(sources_mod, "MAX_ARCHIVE_BYTES", ...)`` live.  The
# check must stay in the module that reads the name; the same holds for
# ``_atomic_replace`` and ``_extract_archive``.
from ._types import (  # noqa: F401
    DEFAULT_CACHE_DIRNAME,
    HTTP_TIMEOUT,
    HttpGet,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_MEMBERS,
    MAX_EXTRACTED_BYTES,
    REMOTE_TTL_SECONDS,
    RefreshMode,
    ShimArchiveMismatch,
    ShimArchiveUnusable,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
    ShimSourceUnreachable,
    _ARCHIVE_SUFFIXES,
    _DRIVE_RE,
    _GITHUB_RE,
    _REFRESH_MODES,
    _SHA256_RE,
    _SHA_RE,
    logger,
)

# These names keep the import surface this module had before the split: it
# imported them from ``.manifest``, so callers could take them from here.
from .manifest import (  # noqa: F401
    MANIFEST_FILENAME,
    ShimManifestError,
    core_version_problem,
    read_shim_manifest,
)
from ._spec import (  # noqa: F401
    _RemoteSpec,
    _cache_key,
    _derive_name,
    _parse_github,
    _parse_remote_spec,
    _parse_selection,
    _validate_url_scheme,
    is_supported_archive_url,
    normalize_mirror,
    normalize_subpath,
)
from ._cache import (  # noqa: F401
    _atomic_replace,
    _chosen_legs,
    _chosen_legs_lock,
    _file_lock,
    _leg_subpath,
    _needs_fetch,
    _read_stamp,
    _remote_failure,
    _remote_subdir,
    _stamp_leg,
    _write_stamp,
)
from ._scan import (  # noqa: F401
    _load_local_dir,
    _scan_dir,
    _selection_report,
    _unmatched_patterns,
    selection_predicate,
)
from ._legs import (  # noqa: F401
    _DEFAULT_PORTS,
    _archive_url,
    _auth_headers,
    _bearer_headers,
    _exhausted_error,
    _is_unreachable,
    _leg_headers,
    _leg_plan,
    _log_leg_failure,
    _mirror_slot,
    _resolve_github_head,
    _same_origin,
    _url_origin,
)
from ._archive import (  # noqa: F401
    _common_root_prefix,
    _extract_archive,
    _extract_tar,
    _extract_zip,
    _sniff_archive,
    _strip_prefix,
    _validate_member_name,
)


# In-process guard: a rebuild must not be observed half-empty by another thread.
_rebuild_lock = threading.Lock()

#: Shim names installed by the most recent rebuild from ``extra_dirs`` /
#: ``extra_remotes``.  Tracked so a later rebuild can drop exactly those while
#: leaving shims registered in-process via ``register_shim`` alone.
_source_shims: set[str] = set()


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

    # An ``include`` that selected nothing is a config fact rather than a
    # source failure — the rest of the source loaded — so it is reported on the
    # status and only stops a strict load.  An include whose every hit the
    # exclude dropped lands here too: it selected *something*, but the source
    # still ends up empty and the operator believes a shim is running.  An
    # unmatched ``exclude`` stays a warning under ``strict``: nothing downstream
    # depends on an exclude having hit.
    unmatched = [
        (status.location, pattern)
        for status in statuses
        for pattern in status.unmatched
    ]
    emptied = [
        (status.location, pattern)
        for status in statuses
        for pattern in status.emptied_includes
    ]
    if strict and (unmatched or emptied):
        detail = "; ".join(
            [
                f"{location}: include {pattern!r} matched no shim"
                for location, pattern in unmatched
            ]
            + [
                f"{location}: include {pattern!r} selected only shims the exclude drops"
                for location, pattern in emptied
            ]
        )
        raise ShimSourceError(f"shim selection matched nothing: {detail}")

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
    ``refresh`` (``auto`` / ``always`` / ``never``), ``mirror``, and
    ``include`` / ``exclude`` — shim names or ``*`` patterns to keep and to
    drop, both optional, empty meaning "load everything the source ships".

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
            include=parsed.include,
            exclude=parsed.exclude,
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
        include=status.include,
        exclude=status.exclude,
    )
    return replace(scanned, leg=status.leg, archive_url=status.archive_url)


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
    # The mirror is addressed by the ref the operator declared when there is
    # one: a tag-addressed mirror publishes one directory per release
    # (``...@dist/v0.1.0/...``, #1412), and its URL must not depend on
    # api.github.com answering.  With no declared ref the preset tracks the
    # branch, so the mirror is addressed by the commit the forge resolved:
    # that is what makes the pinned commit and the mirror's bytes name the
    # same tree, which the old ``sha256`` pair used to guarantee.  An
    # unreachable API then costs the fallback, never the primary.
    mirror_ref = _mirror_slot(ref, resolved_ref)
    legs = _leg_plan(spec, primary_url, ref=effective_ref, mirror_ref=mirror_ref)

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
    "selection_predicate",
    "resolve_remote",
    "is_supported_archive_url",
    "normalize_mirror",
    "DEFAULT_CACHE_DIRNAME",
    "REMOTE_TTL_SECONDS",
]

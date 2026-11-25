"""External shim sources — load provider shims from manifests and directories.

Built-in shims are registered at import time from the bundled ``providers/``
directory (see :mod:`llm_rosetta.shims._loader`).  This module adds a second,
*declarative* source of shims: vendors installed from a source's versioned
manifest (:mod:`.manifest_sources`) and extra local directories.  The gateway
exposes both through its ``shims`` config section; this module is the
config-agnostic core.

Public entry point: :func:`rebuild_shims` — full registry rebuild (built-ins +
entry points + manifest installs + ``extra_dirs``).  Idempotent.  It is the
programmatic entry point for callers that want to load shims before calling
:func:`llm_rosetta.convert`.

Security
--------
A shim directory may ship a ``transforms.py`` that is executed on load, so a
declared source is **code execution** — the same trust level as ``pip
install``ing a plugin package.  Only add sources you trust.  A downloaded wheel is
installed by core's standard-library installer, which rejects absolute paths,
``..`` traversal, links and device members and caps the extracted size; the
download itself is size-capped here.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from llm_rosetta.api import (
    list_shims,
    load_providers,
    register_shim,
    staging_registry,
    unregister_shim,
)

from ._scan import _load_local_dir
from ._types import (
    HTTP_TIMEOUT,
    MAX_ARCHIVE_BYTES,
    REMOTE_TTL_SECONDS,
    HttpGet,
    ShimArchiveUnusable,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
    logger,
)
from .manifest_sources.install import install_from_manifests
from .manifest_sources.plan import ManifestPlan
from .manifest_sources.status import check_manifest_updates, scan_manifest_status


# In-process guard: a rebuild must not be observed half-empty by another thread.
_rebuild_lock = threading.Lock()

#: Shim names installed by the most recent rebuild from manifest installs and
#: ``extra_dirs``.  Tracked so a later rebuild can drop exactly those while
#: leaving shims registered in-process via ``register_shim`` alone.
_source_shims: set[str] = set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rebuild_shims(
    extra_dirs: Sequence[str | os.PathLike[str]] = (),
    *,
    cache_root: str | os.PathLike[str] | None = None,
    strict: bool = False,
    http_get: HttpGet | None = None,
    now: Callable[[], float] | None = None,
    ttl: int = REMOTE_TTL_SECONDS,
    proxy: str | None = None,
    disabled: Sequence[str] = (),
    manifests: ManifestPlan | None = None,
    force_refresh: bool = False,
) -> ShimLoadReport:
    """Rebuild the shim registry from built-ins, extensions and sources.

    The registry is cleared and repopulated in a fixed order — built-ins, then
    the installed extensions (those a manifest install just placed in the
    extensions directory included), then ``extra_dirs`` in order — so a later
    source overrides an earlier one of the same shim name.  An extension is the
    exception to that rule against core: one whose shim names another owner
    holds is not loaded (core wins, see :mod:`.manifest_sources`).
    Idempotent: calling twice with the same inputs yields the same registry.

    Args:
        extra_dirs: Local directories to scan for ``provider.yaml`` (flat or
            grouped layout, as :func:`load_providers_from_dir` accepts).
        cache_root: Directory for fetched manifests and installs.  Required
            when ``manifests`` is given.
        strict: When ``True``, raise :class:`ShimSourceError` if any declared
            source fails; otherwise failures are reported in the result.
        http_get: Injectable ``(url, *, headers, timeout, proxy) -> bytes``.
            Defaults to the vendored synchronous HTTP client.
        now: Injectable clock (defaults to :func:`time.time`).
        ttl: Seconds a fetched manifest is reused without a refetch.
        proxy: Upstream proxy URL for fetches.
        disabled: Shim names to leave out of the registry once everything has
            loaded.  A name is dropped after its source registers it, so the
            sources stay the one account of what a configuration declares.
        manifests: The ``shims.sources`` / ``shims.install`` plan.
        force_refresh: Refetch every manifest regardless of its age.

    Returns:
        A :class:`ShimLoadReport` describing each source and the final registry
        size.

    Raises:
        ShimSourceError: Only when ``strict`` is set and a source failed.
    """
    now_fn = now or time.time
    statuses: list[ShimSourceStatus] = []

    with _rebuild_lock:
        # Slow, network-bound work first: fetch and install every manifest
        # vendor *before* the registry is touched, so shims are never missing
        # from the visible registry for the duration of a fetch.
        manifest_statuses = _install_manifests(
            manifests,
            cache_root=cache_root,
            http_get=http_get,
            now=now_fn,
            ttl=ttl,
            proxy=proxy,
            force_refresh=force_refresh,
        )

        # The registry is rebuilt off to the side and swapped in with a single
        # reference assignment (see ``staging_registry``), so a reader on any
        # other thread — notably the gateway's event loop, which resolves each
        # request's shim by name against this registry — always sees either the
        # old registry or the complete new one, never a partially-built one.
        # ``staging_registry`` drops the caches derived from it on entry, so the
        # new registry starts from a clean slate.
        out_of_band = {s.name: s for s in list_shims() if s.name not in _source_shims}

        with staging_registry():
            load_providers()
            known = {s.name for s in list_shims()}
            for name, shim in out_of_band.items():
                if name not in known:
                    register_shim(shim)

            # Built-ins and plugins are core: a manifest vendor never
            # replaces one of their names.
            for status in manifest_statuses:
                statuses.append(scan_manifest_status(status, known))
            for directory in extra_dirs:
                statuses.append(_load_local_dir(directory))

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


def _install_manifests(
    plan: ManifestPlan | None,
    *,
    cache_root: str | os.PathLike[str] | None,
    http_get: HttpGet | None,
    now: Callable[[], float],
    ttl: int,
    proxy: str | None,
    force_refresh: bool,
) -> list[ShimSourceStatus]:
    """Run the manifest installs of a rebuild, before the registry is touched.

    The core-name estimate only saves a download: the registry minus what
    the previous rebuild loaded from sources.  The scan step re-checks
    against the live core set.

    Returns:
        One status per vendor install, or one failure status when the plan
        cannot run.
    """
    if not plan:
        return []
    if cache_root is None:
        failure = ShimSourceStatus(
            kind="remote",
            location=", ".join(s.url for s in plan.sources),
            ok=False,
            error="manifest sources require a cache directory",
        )
        return [failure]
    estimate = {s.name for s in list_shims()} - _source_shims
    return install_from_manifests(
        plan,
        cache_root=cache_root,
        http_get=http_get or _default_http_get,
        now=now,
        ttl=ttl,
        proxy=proxy,
        force_refresh=force_refresh,
        core_names=estimate,
    )


def check_source_updates(
    report: ShimLoadReport,
    *,
    http_get: HttpGet | None = None,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Compare each installed vendor with its manifest's newest installable version.

    ``current`` and ``head`` are versions, and ``update`` is true only when a
    newer version is installable on this core.  A vendor whose manifest cannot
    be read carries the error and no verdict.

    Args:
        report: The last applied load report.
        http_get: Injectable HTTP getter (tests).
        proxy: Upstream proxy.

    Returns:
        One entry per installed vendor.
    """
    installed = [s for s in report.sources if s.vendor and s.ok and s.release]
    return check_manifest_updates(
        installed, http_get=http_get or _default_http_get, proxy=proxy
    )


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
    "rebuild_shims",
]

"""Remote-source cache bookkeeping: staleness, stamps, atomic swap and locking."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal


from ._types import (
    ShimSourceStatus,
    ShimSourceUnreachable,
    _SHA_RE,
)
from ._spec import (
    _RemoteSpec,
)


#: The leg that last served each remote source, keyed by its cache directory
#: name, so a working mirror is not re-probed on every fetch.  Process-local:
#: a restart costs one failed connect, which is the documented cost of the
#: lazy (non-pre-probing) selection.
_chosen_legs: dict[str, str] = {}
_chosen_legs_lock = threading.Lock()


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

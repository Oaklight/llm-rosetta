"""Install stamps and file locking for the shim cache."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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

"""Config file IO: JSONC reading, the reader/writer protocol, and locks.

``JsoncConfigIO`` is the default reader/writer: it parses JSONC with env-var
substitution, applies any pending migrations, and writes a migrated document
back.  The locks live here too — both the synchronous ``config_lock`` and the
asyncio ``async_config_lock`` guard the same file the IO reads and writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any, Protocol, runtime_checkable

from llm_rosetta._vendor import jsonx
from llm_rosetta.observability.maintenance import Registry

logger = logging.getLogger("llm-rosetta-gateway")


def _lock_exclusive(f: Any) -> None:
    """Acquire an exclusive lock on an open file (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt  # noqa: F811  # Windows-only

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)


@contextmanager
def config_lock(path: str) -> Generator[None, None, None]:
    """Serialize read-modify-write cycles on the same config file.

    Admin routes must wrap their ``load_raw → modify → save`` sequences
    with this lock.  Without it, concurrent requests that each load the
    same version can silently overwrite each other's changes.

    Uses a ``.lock`` sidecar file with ``fcntl.flock`` (Unix) or
    ``msvcrt.locking`` (Windows) for cross-process mutual exclusion.
    This protects against multiple gateway instances sharing the same
    config file (e.g. dev + prod on the same host).
    """
    lock_path = os.path.realpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as lf:
        _lock_exclusive(lf)
        yield


def _get_async_lock(path: str) -> asyncio.Lock:
    real = os.path.realpath(path)
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    key = (real, loop_id)
    if key not in _async_locks:
        # Evict stale entries from previous loops
        stale = [k for k in _async_locks if k[1] != loop_id]
        for k in stale:
            del _async_locks[k]
        _async_locks[key] = asyncio.Lock()
    return _async_locks[key]


@asynccontextmanager
async def async_config_lock(path: str) -> AsyncGenerator[None, None]:
    """Async version of :func:`config_lock` for use in event-loop contexts.

    Acquires an in-process ``asyncio.Lock`` first (non-blocking to the
    event loop), then delegates the cross-process file lock to a thread
    executor so ``fcntl.flock`` never blocks the loop.
    """
    lock = _get_async_lock(path)
    async with lock:
        lock_path = os.path.realpath(path) + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        lf = open(lock_path, "a+")  # noqa: SIM115
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _lock_exclusive, lf)
            yield
        finally:
            lf.close()


@runtime_checkable
class ConfigIO(Protocol):
    """Protocol for reading/writing gateway config files.

    The default :class:`JsoncConfigIO` handles JSONC files.  Downstream
    projects (e.g. argo-proxy) can supply an alternative implementation
    for other formats (YAML, TOML, etc.) via ``setup_admin(..., config_io=...)``.

    A reader that runs load-time maintenance steps records the registry it ran
    as ``last_load_registry``, and the caller that builds a
    :class:`GatewayConfig` from the document it returned passes that on.  A
    reader without that attribute runs no such steps, which reads the same as a
    config built programmatically: nothing ran.
    """

    def load(self, path: str) -> dict[str, Any]:
        """Read config with env-var substitution (for runtime use)."""
        ...

    def load_raw(self, path: str) -> dict[str, Any]:
        """Read config without env-var substitution (for edit round-trips)."""
        ...

    def save(self, path: str, data: dict[str, Any]) -> None:
        """Write config back to disk."""
        ...


class JsoncConfigIO:
    """Default :class:`ConfigIO` — reads/writes JSONC with env-var substitution.

    On load, applies any pending config migrations and writes back the
    migrated config if changes were made.
    """

    def __init__(self) -> None:
        #: The registry the most recent read ran, for the config object built
        #: from the document that read returned.  A load-time step runs while a
        #: document is read, so the reader is what knows the answer — it is not
        #: left in a process-wide slot, where the *next* :class:`GatewayConfig`
        #: construction would take it whether or not it came from a document that
        #: ran anything.  The caller that reads and then builds the config hands
        #: it over explicitly, so a config built from a literal reports nothing —
        #: the same answer as "no load ran".
        self.last_load_registry: Registry | None = None

    def load(self, path: str) -> dict[str, Any]:
        raw = _parse_config(path)
        return self._apply_migrations(raw, path, is_raw=False)

    def load_raw(self, path: str) -> dict[str, Any]:
        raw = _parse_config_raw(path)
        return self._apply_migrations(raw, path, is_raw=True)

    def save(self, path: str, data: dict[str, Any]) -> None:
        write_config(path, data)

    def _apply_migrations(
        self, raw: dict[str, Any], path: str, *, is_raw: bool = False
    ) -> dict[str, Any]:
        """Bring one config document up to date, through the load-time registry.

        The steps are the versioned structure migrations and the name
        correction, in that order, under
        :attr:`Policy.LOAD_TIME <llm_rosetta.observability.maintenance.Policy>`;
        the registry that ran is kept on this reader, which is what read the
        document, and is handed to the config built from it.
        """
        from ..maintenance.config import apply

        raw, changed, registry = apply(raw, path=path)
        self.last_load_registry = registry
        if changed:
            self._write_back(raw, path, is_raw=is_raw)
        return raw

    def _write_back(self, raw: dict[str, Any], path: str, *, is_raw: bool) -> None:
        """Write the migrated document back, and keep the original beside it.

        Best-effort on purpose.  The steps have already run and their result is
        the document this load returns, so nothing here is needed to serve it —
        and a config that is mounted read-only (a k8s ConfigMap, a container
        layer) would otherwise turn a working deployment into a refused start
        the first time a migration had something to do.  A directory that cannot
        be written is logged, and the document is served from memory instead:
        the process holds the migrated one either way, and the version on disk
        simply does not advance.
        """
        from ..maintenance.config import apply

        if is_raw:
            disk_data = raw
        else:
            disk_data = _parse_config_raw(path)
            # The same steps over the on-disk copy, so a substituted env var
            # is not written back in place of its placeholder.
            disk_data, _changed, _registry = apply(disk_data, path=path)
        try:
            import shutil

            bak = path + ".pre-migration.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
                logger.info("config: backed up original to %s", bak)
            with config_lock(path):
                write_config(path, disk_data)
        except OSError as exc:
            logger.warning(
                "config: %s was brought forward in memory but could not be "
                "written back (%s); serving the migrated document and leaving "
                "the file as it is",
                path,
                exc,
            )
            return
        logger.info(
            "config: migrated %s to version %s",
            path,
            disk_data.get("config_version"),
        )


def _substitute_env_vars(text: str) -> str:
    """Replace ${ENV_VAR} placeholders with environment variable values."""

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        value = os.environ.get(var_name)
        if value is None:
            logger.warning("Environment variable %s is not set", var_name)
            return m.group(0)  # leave placeholder intact
        return value

    return _ENV_VAR_RE.sub(_replace, text)


def _loads(text: str) -> Any:
    """Plain JSON first; the JSONC preprocessor only when the text needs it.

    ``jsonx.loads`` strips comments and trailing commas with two
    character-by-character passes *before* handing the text to ``json.loads`` —
    3.5 ms against 0.3 ms for a 35 KiB document, and it charges that to
    documents that have neither, which is every file :func:`write_config` has
    ever written.  JSON is a subset of JSONC, so a ``json.loads`` that succeeds
    is the same answer; only its failure needs the scanning parser.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return jsonx.loads(text)


def _parse_config(path: str) -> dict[str, Any]:
    """Parse a JSONC config file with env-var substitution, and nothing else."""
    with open(path) as f:
        raw = f.read()
    substituted = _substitute_env_vars(raw)
    return _loads(substituted)


def _parse_config_raw(path: str) -> dict[str, Any]:
    """Parse a JSONC config file without env-var substitution, and nothing else."""
    with open(path) as f:
        return _loads(f.read())


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


_async_locks: dict[tuple[str, int], asyncio.Lock] = {}


def write_config(path: str, data: dict[str, Any]) -> None:
    """Write a config dict as formatted JSON to *path* atomically.

    Creates parent directories if needed.  Writes to a temporary file in
    the same directory, flushes to disk, then atomically replaces the
    target via ``os.replace``.  Readers never see a partially-written file.

    Comments in the original JSONC file (if any) are **not** preserved.
    """
    import tempfile

    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise

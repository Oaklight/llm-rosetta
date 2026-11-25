"""Installing, listing and removing extensions with the standard library only.

Where pip is absent (a compiled binary, a slim image) an extension wheel can
still be installed: a wheel is a zip whose members are
``llm_rosetta_ext/<id>/...`` and ``<name>-<version>.dist-info/...``, and
:func:`install_wheel` unpacks exactly that into the extensions directory.
Every refusal happens before a byte is written to the directory: member names
are checked, the manifest is parsed and its compatibility with this core is
checked, and only then is the wheel extracted (into a temporary directory in
the same directory, then swapped in).

The doctor primitives are pure functions; a command line prints their results.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import ipaddress
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from llm_rosetta._vendor.yaml import load as yaml_load
from llm_rosetta.contracts.extension import (
    MANIFEST_FILENAME,
    ExtensionManifest,
    ExtensionManifestInvalid,
    ExternalProgram,
    compatibility_problems,
    parse_extension_manifest,
)

from ._dirs import ExtensionInstallError, extensions_dir
from ._discovery import NAMESPACE

#: The most bytes a wheel may unpack to, and the largest wheel accepted.
MAX_UNPACKED_BYTES = 64 * 1024 * 1024

#: The most members a wheel may hold.
MAX_MEMBERS = 5000

#: The largest ``plugin.yaml`` read.
MAX_MANIFEST_BYTES = 1024 * 1024

#: Seconds a lock file may stay before it is taken for a dead installer's.
LOCK_STALE_SECONDS = 300.0

#: Seconds an installer waits for the lock before it gives up.
LOCK_WAIT_SECONDS = 60.0

_LOCK_NAME = ".llm-rosetta-install.lock"
_OWNER_MARKER = "LLM_ROSETTA_EXTENSION"
_DIST_INFO_RE = re.compile(r"^[^-/]+-[^-/]+\.dist-info$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class InstalledExtension:
    """One extension in the extensions directory.

    Attributes:
        id: The extension id (its folder name).
        version: The manifest's version; ``None`` when the manifest is unreadable.
        path: The extension's folder.
        manifest: The parsed manifest; ``None`` when it could not be read.
        reason: Why the manifest could not be read; ``None`` otherwise.
    """

    id: str
    version: str | None
    path: Path
    manifest: ExtensionManifest | None
    reason: str | None = None


@dataclass(frozen=True)
class ExternalStatus:
    """The result of checking one external program.

    Attributes:
        name: The program's name.
        state: ``"ok"``, ``"missing"`` (not runnable), ``"too_old"`` (older than
            the minimum) or ``"unknown"`` (it ran, a minimum is required, and
            its output holds no version number).
        found: The version found, when there was one.
        minimum: The version the extension requires, if any.
        detail: One sentence on what happened; ``None`` when ``ok``.
    """

    name: str
    state: str
    found: str | None = None
    minimum: str | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- #
# the wheel: member rules
# --------------------------------------------------------------------------- #


def _member_parts(name: str) -> tuple[str, ...]:
    if "\x00" in name:
        raise ExtensionInstallError("wheel member name contains NUL")
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        raise ExtensionInstallError("wheel has a member with an empty name")
    if normalized.startswith("/") or _DRIVE_RE.match(parts[0]):
        raise ExtensionInstallError(f"wheel contains an absolute path: {name}")
    if ".." in parts:
        raise ExtensionInstallError(f"wheel contains a traversal path: {name}")
    return parts


def _classify(
    info: zipfile.ZipInfo,
) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(kind, key, parts)`` for an allowed member, else raise.

    ``kind`` is ``"ext"`` (``key`` is the extension id) or ``"dist"`` (``key``
    is the ``.dist-info`` directory name).
    """
    name = info.filename
    parts = _member_parts(name)
    mode = info.external_attr >> 16
    if mode:
        kind = mode & 0o170000
        if kind == 0o120000:
            raise ExtensionInstallError(f"wheel contains a symlink member: {name}")
        if kind not in (0, 0o100000, 0o040000):
            raise ExtensionInstallError(f"wheel contains a special member: {name}")
    is_dir = info.is_dir()
    if parts[0] == NAMESPACE:
        if len(parts) == 1:
            if not is_dir:
                raise ExtensionInstallError(f"wheel member outside its roots: {name}")
            return "ext", "", parts
        ext_id = parts[1]
        if not ext_id.isidentifier() or ext_id.startswith("_"):
            raise ExtensionInstallError(
                f"wheel member names an invalid extension id {ext_id!r}: {name}"
            )
        if len(parts) == 2 and not is_dir:
            raise ExtensionInstallError(
                f"wheel member lies in the namespace, not in an extension folder: "
                f"{name}"
            )
        return "ext", ext_id, parts
    if _DIST_INFO_RE.match(parts[0]):
        if len(parts) == 1 and not is_dir:
            raise ExtensionInstallError(f"wheel member outside its roots: {name}")
        return "dist", parts[0], parts
    raise ExtensionInstallError(
        f"wheel member lies outside {NAMESPACE}/<id>/ and <name>-<version>.dist-info/: "
        f"{name}"
    )


def _read_source(source: str | os.PathLike[str] | bytes | bytearray) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        path = Path(source)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExtensionInstallError(f"cannot read wheel {path}: {exc}") from exc
        if size > MAX_UNPACKED_BYTES:
            raise ExtensionInstallError(
                f"wheel {path} is {size} bytes, over the {MAX_UNPACKED_BYTES} cap"
            )
        data = path.read_bytes()
    if len(data) > MAX_UNPACKED_BYTES:
        raise ExtensionInstallError("wheel exceeds the size cap")
    return data


def _manifest_from(archive: zipfile.ZipFile, ext_id: str) -> ExtensionManifest:
    member = f"{NAMESPACE}/{ext_id}/{MANIFEST_FILENAME}"
    try:
        info = archive.getinfo(member)
    except KeyError:
        raise ExtensionInstallError(
            f"wheel has no {MANIFEST_FILENAME} for extension {ext_id!r}"
        ) from None
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ExtensionInstallError(f"{MANIFEST_FILENAME} exceeds the size cap")
    try:
        text = archive.read(info).decode("utf-8")
        document = yaml_load(text)
        return parse_extension_manifest(document)
    except ExtensionManifestInvalid as exc:
        raise ExtensionInstallError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - any parse failure is a refusal
        raise ExtensionInstallError(
            f"cannot read {MANIFEST_FILENAME}: {type(exc).__name__}: {exc}"
        ) from exc


def _inspect(
    archive: zipfile.ZipFile,
) -> tuple[ExtensionManifest, list[tuple[zipfile.ZipInfo, tuple[str, ...]]]]:
    """Check every member and the manifest; return what to extract."""
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ExtensionInstallError("wheel has too many members")
    ids: set[str] = set()
    dists: set[str] = set()
    seen: set[tuple[str, ...]] = set()
    total = 0
    planned: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    for info in infos:
        kind, key, parts = _classify(info)
        if parts in seen:
            raise ExtensionInstallError(
                f"wheel contains a member twice: {info.filename}"
            )
        seen.add(parts)
        if kind == "ext" and key:
            ids.add(key)
        elif kind == "dist":
            dists.add(key)
        total += info.file_size
        if total > MAX_UNPACKED_BYTES:
            raise ExtensionInstallError("wheel unpacks beyond the size cap")
        planned.append((info, parts))
    if len(ids) != 1:
        found = f" (found {', '.join(sorted(ids))})" if ids else ""
        raise ExtensionInstallError(
            f"a wheel must hold exactly one extension id, not {len(ids)}{found}"
        )
    if len(dists) > 1:
        raise ExtensionInstallError(
            f"wheel holds more than one dist-info: {', '.join(sorted(dists))}"
        )
    ext_id = next(iter(ids))
    manifest = _manifest_from(archive, ext_id)
    if manifest.id != ext_id:
        raise ExtensionInstallError(
            f"{MANIFEST_FILENAME} names id {manifest.id!r} but the folder is "
            f"{ext_id!r}; an extension is imported under its id"
        )
    problems = compatibility_problems(manifest)
    if problems:
        raise ExtensionInstallError(
            f"extension {manifest.id} {manifest.version} cannot run here: "
            + "; ".join(problems)
        )
    return manifest, planned


def _extract(
    archive: zipfile.ZipFile,
    planned: list[tuple[zipfile.ZipInfo, tuple[str, ...]]],
    stage: Path,
) -> None:
    written = 0
    for info, parts in planned:
        target = stage.joinpath(*parts)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, open(target, "wb") as out:
            while True:
                chunk = src.read(64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UNPACKED_BYTES:
                    raise ExtensionInstallError("wheel unpacks beyond the size cap")
                out.write(chunk)


# --------------------------------------------------------------------------- #
# the directory: lock, swap
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _locked(dest: Path) -> Iterator[None]:
    """Hold an exclusive lock on *dest* (an ``O_EXCL`` file with a stale limit)."""
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / _LOCK_NAME
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_SECONDS:
                with contextlib.suppress(FileNotFoundError):
                    lock.unlink()
                continue
            if time.monotonic() > deadline:
                raise ExtensionInstallError(
                    f"another install holds {lock}; remove it if no installer runs"
                ) from None
            time.sleep(0.02)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        break
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _owned_dist_infos(dest: Path, ext_id: str) -> list[Path]:
    """The ``.dist-info`` directories in *dest* that belong to *ext_id*."""
    owned: list[Path] = []
    prefix = f"{NAMESPACE}/{ext_id}/"
    for entry in dest.glob("*.dist-info"):
        if not entry.is_dir() or entry.is_symlink():
            continue
        marker = entry / _OWNER_MARKER
        try:
            if marker.is_file() and marker.read_text("utf-8").strip() == ext_id:
                owned.append(entry)
                continue
            record = entry / "RECORD"
            if record.is_file() and prefix in record.read_text("utf-8").replace(
                "\\", "/"
            ):
                owned.append(entry)
        except (OSError, UnicodeDecodeError):
            continue
    return owned


def _discard(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _swap_in(dest: Path, stage: Path, ext_id: str) -> None:
    """Replace the installed *ext_id* by the one unpacked in *stage*."""
    (dest / NAMESPACE).mkdir(exist_ok=True)
    trash = Path(tempfile.mkdtemp(prefix=".trash-", dir=dest))
    undo: list[tuple[Path, Path]] = []  # (moved to, came from)
    placed: list[Path] = []

    def move(src: Path, dst: Path) -> None:
        os.rename(src, dst)

    try:
        old = dest / NAMESPACE / ext_id
        if old.is_symlink() or old.exists():
            holder = trash / "folder"
            move(old, holder)
            undo.append((holder, old))
        for index, dist in enumerate(_owned_dist_infos(dest, ext_id)):
            holder = trash / f"dist{index}"
            move(dist, holder)
            undo.append((holder, dist))
        new_folder = stage / NAMESPACE / ext_id
        target = dest / NAMESPACE / ext_id
        move(new_folder, target)
        placed.append(target)
        for dist in stage.glob("*.dist-info"):
            target = dest / dist.name
            move(dist, target)
            placed.append(target)
    except BaseException:
        for path in placed:
            _discard(path)
        for held, origin in reversed(undo):
            with contextlib.suppress(OSError):
                os.rename(held, origin)
        shutil.rmtree(trash, ignore_errors=True)
        raise
    shutil.rmtree(trash, ignore_errors=True)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #


def _read_manifest_file(folder: Path) -> ExtensionManifest:
    path = folder / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        return parse_extension_manifest(yaml_load(handle.read()))


def install_wheel(
    source: str | os.PathLike[str] | bytes | bytearray,
    *,
    sha256: str | None = None,
    dest: str | os.PathLike[str] | None = None,
) -> InstalledExtension:
    """Install an extension wheel with the standard library.

    The previous version of the same extension, if any, is replaced.  Nothing is
    written to *dest* unless the wheel passes every check.

    Args:
        source: A path to a ``.whl`` file, or its bytes.
        sha256: The expected hex digest of the wheel; verified when given.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        The installed extension.

    Raises:
        ExtensionInstallError: The digest differs, the wheel is not a safe zip,
            holds not exactly one extension, has no valid ``plugin.yaml``, or
            declares a requirement this core does not meet.
    """
    data = _read_source(source)
    if sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256.strip().lower():
            raise ExtensionInstallError(
                f"sha256 mismatch: expected {sha256.strip().lower()}, got {actual}"
            )
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtensionInstallError(f"not a valid wheel (zip) file: {exc}") from exc
    with archive:
        manifest, planned = _inspect(archive)
        target = Path(dest) if dest is not None else extensions_dir()
        with _locked(target):
            stage = Path(tempfile.mkdtemp(prefix=".install-", dir=target))
            try:
                _extract(archive, planned, stage)
                for dist in stage.glob("*.dist-info"):
                    (dist / _OWNER_MARKER).write_text(manifest.id, encoding="utf-8")
                _swap_in(target, stage, manifest.id)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
    return InstalledExtension(
        id=manifest.id,
        version=manifest.version,
        path=target / NAMESPACE / manifest.id,
        manifest=manifest,
    )


def _url_allowed(url: str) -> str | None:
    """Return why *url* may not be fetched, or ``None``."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https"):
        return f"{url!r} has the scheme {parts.scheme!r}; only https is allowed"
    if not host:
        return f"{url!r} has no host"
    if parts.scheme == "https":
        return None
    if parts.scheme == "http":
        loopback = host in _LOOPBACK
        if not loopback:
            with contextlib.suppress(ValueError):
                loopback = ipaddress.ip_address(host).is_loopback
        if loopback:
            return None
        return f"{url!r} is plain http; only https (or a loopback host) is allowed"
    return f"{url!r} has the scheme {parts.scheme!r}; only https is allowed"


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        problem = _url_allowed(newurl)
        if problem is not None:
            raise ExtensionInstallError(f"refused redirect: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def install_wheel_url(
    url: str,
    *,
    sha256: str,
    dest: str | os.PathLike[str] | None = None,
    timeout: float = 30.0,
) -> InstalledExtension:
    """Download an extension wheel and install it.

    Args:
        url: An ``https`` URL (``http`` only for a loopback host).
        sha256: The wheel's hex digest; required, because the transport alone
            does not say whose wheel it is.
        dest: The extensions directory; :func:`extensions_dir` by default.
        timeout: Seconds for each network operation.

    Returns:
        The installed extension.

    Raises:
        ExtensionInstallError: The URL is not allowed, the digest is missing or
            differs, the download is too large or fails, or the wheel is refused.
    """
    if not sha256 or not sha256.strip():
        raise ExtensionInstallError("a sha256 is required to install from a URL")
    problem = _url_allowed(url)
    if problem is not None:
        raise ExtensionInstallError(problem)
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    handlers: list[urllib.request.BaseHandler] = [_CheckedRedirect()]
    if urllib.parse.urlsplit(url).scheme == "http" or host in _LOOPBACK:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    with tempfile.TemporaryDirectory(prefix="llm-rosetta-ext-") as tmp:
        wheel = Path(tmp) / "download.whl"
        total = 0
        try:
            with (
                opener.open(url, timeout=timeout) as response,
                open(wheel, "wb") as out,
            ):
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UNPACKED_BYTES:
                        raise ExtensionInstallError(
                            f"download exceeds the {MAX_UNPACKED_BYTES} byte cap"
                        )
                    out.write(chunk)
        except ExtensionInstallError:
            raise
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ExtensionInstallError(f"cannot download {url}: {exc}") from exc
        return install_wheel(wheel, sha256=sha256, dest=dest)


# --------------------------------------------------------------------------- #
# manage
# --------------------------------------------------------------------------- #


def installed_extensions(
    dest: str | os.PathLike[str] | None = None,
) -> list[InstalledExtension]:
    """List what is installed in the extensions directory.

    An extension whose manifest cannot be read is listed with ``manifest=None``
    and a ``reason``; this function does not raise for one.

    Args:
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        One record per extension folder, sorted by id.
    """
    root = (Path(dest) if dest is not None else extensions_dir()) / NAMESPACE
    if not root.is_dir():
        return []
    found: list[InstalledExtension] = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name):
        name = folder.name
        if name.startswith((".", "_")) or not folder.is_dir():
            continue
        try:
            manifest = _read_manifest_file(folder)
        except ExtensionManifestInvalid as exc:
            found.append(InstalledExtension(name, None, folder, None, str(exc)))
        except Exception as exc:  # noqa: BLE001 - listing never raises for one
            found.append(
                InstalledExtension(
                    name, None, folder, None, f"{type(exc).__name__}: {exc}"
                )
            )
        else:
            found.append(InstalledExtension(name, manifest.version, folder, manifest))
    return found


def remove_extension(ext_id: str, dest: str | os.PathLike[str] | None = None) -> bool:
    """Remove an installed extension and its ``.dist-info``.

    Args:
        ext_id: The extension id.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        ``True`` when it was removed, ``False`` when it was not installed.

    Raises:
        ExtensionInstallError: *ext_id* is not a plain extension id (so it
            cannot name anything outside *dest*).
    """
    if (
        not isinstance(ext_id, str)
        or not ext_id.isidentifier()
        or ext_id.startswith("_")
    ):
        raise ExtensionInstallError(f"{ext_id!r} is not an extension id")
    target = Path(dest) if dest is not None else extensions_dir()
    folder = target / NAMESPACE / ext_id
    if not target.is_dir() or not (folder.is_symlink() or folder.exists()):
        return False
    with _locked(target):
        if not (folder.is_symlink() or folder.exists()):
            return False
        for dist in _owned_dist_infos(target, ext_id):
            _discard(dist)
        _discard(folder)
    return True


# --------------------------------------------------------------------------- #
# doctor primitives
# --------------------------------------------------------------------------- #


def _version_tuple(text: str) -> tuple[int, ...] | None:
    match = _VERSION_RE.search(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def _at_least(found: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(found), len(minimum))
    pad = lambda v: v + (0,) * (width - len(v))  # noqa: E731
    return pad(found) >= pad(minimum)


def check_external(
    program: ExternalProgram, *, timeout: float = 10.0
) -> ExternalStatus:
    """Run an external program's version check and compare to its minimum.

    The command runs without a shell.  The first dotted number in its output
    (standard output, then standard error) is the version.

    Args:
        program: The declaration from the manifest.
        timeout: Seconds the command may run.

    Returns:
        The status: ``ok``, ``missing``, ``too_old`` or ``unknown``.

    Raises:
        ValueError: ``program.minimum`` holds no dotted version number.
    """
    minimum = None
    if program.minimum is not None:
        minimum = _version_tuple(program.minimum)
        if minimum is None:
            raise ValueError(f"minimum {program.minimum!r} is not a dotted version")
    base = {"name": program.name, "minimum": program.minimum}
    if not program.check:
        return ExternalStatus(state="missing", detail="no check command", **base)
    try:
        run = subprocess.run(
            list(program.check),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return ExternalStatus(
            state="missing", detail=f"{program.check[0]!r} is not installed", **base
        )
    except subprocess.TimeoutExpired:
        return ExternalStatus(
            state="missing", detail=f"the check timed out after {timeout:g}s", **base
        )
    except OSError as exc:
        return ExternalStatus(state="missing", detail=f"cannot run it: {exc}", **base)
    output = f"{run.stdout}\n{run.stderr}"
    found = _version_tuple(output)
    if found is None:
        if minimum is None:
            return ExternalStatus(state="ok", **base)
        return ExternalStatus(
            state="unknown",
            detail="its output holds no version number",
            **base,
        )
    shown = ".".join(str(part) for part in found)
    if minimum is not None and not _at_least(found, minimum):
        return ExternalStatus(
            state="too_old",
            found=shown,
            detail=f"version {shown} is older than {program.minimum}",
            **base,
        )
    return ExternalStatus(state="ok", found=shown, **base)


def missing_dependencies(
    manifest: ExtensionManifest,
    installed_manifests: Mapping[str, object] | Iterable[ExtensionManifest | None],
) -> list[str]:
    """List the required extensions that are not installed.

    Args:
        manifest: The extension whose ``requires.extensions`` is checked.
        installed_manifests: The installed manifests, or a mapping keyed by id.

    Returns:
        The missing ids, in the order the manifest lists them.
    """
    if isinstance(installed_manifests, Mapping):
        have = set(installed_manifests)
    else:
        have = {item.id for item in installed_manifests if item is not None}
    return [name for name in manifest.requires.extensions if name not in have]


# --------------------------------------------------------------------------- #
# pip
# --------------------------------------------------------------------------- #


def pip_available() -> bool:
    """Whether pip can be imported in this interpreter."""
    return importlib.util.find_spec("pip") is not None


def install_with_pip(
    spec: str, dest: str | os.PathLike[str] | None = None
) -> InstalledExtension:
    """Build *spec* into a wheel with pip and install that wheel.

    pip is only the wheel builder here, never the installer: ``pip install
    --target`` into the shared extensions directory replaces the whole
    ``llm_rosetta_ext`` folder (the namespace every extension lives in) and
    would delete the code of every other installed extension while leaving
    their metadata behind.  The wheel pip builds goes through
    :func:`install_wheel`, which touches only this extension's own folder and
    makes the same compatibility and safety checks any wheel gets.

    A local directory is copied first and built from the copy: an extension
    folder is itself the package, so a build writes ``build/`` and
    ``*.egg-info`` inside it, and the next build would read them back as
    package content.

    Args:
        spec: A requirement specifier, a path or a URL pip accepts.
        dest: The extensions directory; :func:`extensions_dir` by default.

    Returns:
        The installed extension.

    Raises:
        ExtensionInstallError: pip is not available, pip could not build the
            wheel, it did not produce exactly one wheel, or the wheel is
            refused by :func:`install_wheel`.
    """
    if not pip_available():
        raise ExtensionInstallError(
            "pip is not available in this interpreter; install a wheel with "
            "install_wheel instead"
        )
    with tempfile.TemporaryDirectory(prefix="llm-rosetta-pip-") as tmp:
        work = Path(tmp)
        source = spec
        if Path(spec).is_dir():
            clean = work / "src" / Path(spec).resolve().name
            shutil.copytree(
                spec,
                clean,
                ignore=shutil.ignore_patterns(
                    "build", "*.egg-info", "__pycache__", "*.pyc"
                ),
            )
            source = str(clean)
        out = work / "wheels"
        out.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(out), source],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
            raise ExtensionInstallError(
                f"pip could not build a wheel for {spec!r} "
                f"(exit {result.returncode}): " + " | ".join(tail)
            )
        wheels = sorted(out.glob("*.whl"))
        if len(wheels) != 1:
            raise ExtensionInstallError(
                f"pip produced {len(wheels)} wheels for {spec!r}; an extension "
                "builds exactly one (it has no dependencies)"
            )
        return install_wheel(wheels[0], dest=dest)

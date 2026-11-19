"""Safe extraction of tar/zip archives into a directory."""

from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal


from ._types import (
    MAX_ARCHIVE_MEMBERS,
    MAX_EXTRACTED_BYTES,
    ShimSourceError,
    _DRIVE_RE,
)


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

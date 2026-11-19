"""Parsing and normalising a declared remote source into a ``_RemoteSpec``."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ._types import (
    RefreshMode,
    ShimSourceError,
    _ARCHIVE_SUFFIXES,
    _GITHUB_RE,
    _REFRESH_MODES,
    _SHA256_RE,
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
    mirror_url: str | None = None
    mirror_sha256: str | None = None
    mirror_token_env: str | None = None
    #: Scan directory inside the *mirror's* archive.  ``None`` inherits
    #: ``subpath``: the legs need different paths only when the archives they
    #: serve are laid out differently.
    mirror_subpath: str | None = None
    #: Selection carried from the declaration: exact names and ``*`` globs to
    #: keep, and to drop.  Both empty means "load everything the source ships".
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


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
    include = _parse_selection(spec.get("include"), where="'include'")
    exclude = _parse_selection(spec.get("exclude"), where="'exclude'")
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
        include=include,
        exclude=exclude,
    )


def _parse_selection(value: Any, *, where: str) -> tuple[str, ...]:
    """Normalize an ``include`` / ``exclude`` declaration.

    Args:
        value: The declared value: ``None``, a single name, or a list of them.
        where: The config key, for the error message.

    Returns:
        The declared patterns, in declaration order.  Empty when nothing was
        declared, which means "no restriction" for ``include``.

    Raises:
        ShimSourceError: On a non-string entry, or a blank one.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ShimSourceError(f"remote source {where} must be a list of shim names")
    patterns: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ShimSourceError(
                f"remote source {where} entries must be non-empty strings"
            )
        patterns.append(item)
    return tuple(patterns)


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


def _cache_key(name: str, url: str, ref: str | None) -> str:
    """Return a stable, filesystem-safe cache directory name."""
    digest = hashlib.sha256(f"{url}@{ref or 'HEAD'}".encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.") or "source"
    return f"{slug}-{digest}"


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

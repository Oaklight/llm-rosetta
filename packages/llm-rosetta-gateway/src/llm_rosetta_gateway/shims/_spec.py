"""Validation helpers for declared shim paths and URLs."""

from __future__ import annotations

from typing import Any

from ._types import (
    ShimSourceError,
)


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

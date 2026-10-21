"""Load admin panel resources from the package."""

from __future__ import annotations

import importlib.resources
import os

_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}

_ALLOWED_PREFIXES = ("css/", "js/")


def load_admin_html() -> str:
    """Return the contents of ``admin.html`` bundled with this package."""
    return (
        importlib.resources.files(__package__ or __name__)
        .joinpath("admin.html")
        .read_text("utf-8")
    )


def load_static_file(subpath: str) -> tuple[bytes, str]:
    """Load a static CSS/JS file from the package.

    Returns:
        ``(content_bytes, content_type)`` tuple.

    Raises:
        FileNotFoundError: If the path is invalid or the file doesn't exist.
    """
    if any(seg == ".." for seg in subpath.split("/")) or not subpath.startswith(
        _ALLOWED_PREFIXES
    ):
        raise FileNotFoundError(subpath)

    ext = os.path.splitext(subpath)[1]
    content_type = _MIME_TYPES.get(ext)
    if not content_type:
        raise FileNotFoundError(subpath)

    try:
        data = (
            importlib.resources.files(__package__ or __name__)
            .joinpath(subpath)
            .read_bytes()
        )
    except (FileNotFoundError, TypeError):
        raise FileNotFoundError(subpath) from None

    return data, content_type

"""Load the admin panel HTML from package resources."""

from __future__ import annotations

import importlib.resources


def load_admin_html() -> str:
    """Return the contents of ``admin.html`` bundled with this package."""
    return (
        importlib.resources.files(__package__ or __name__)
        .joinpath("admin.html")
        .read_text("utf-8")
    )

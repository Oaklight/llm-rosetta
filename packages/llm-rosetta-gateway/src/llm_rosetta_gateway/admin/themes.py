"""The admin panel's themes: one stylesheet each, one registry (#1471).

A theme is a file under ``admin/css/themes/`` that declares values only: the
structural tokens shared by its light and dark modes, then each mode's colours.
Components consume those tokens and own where they land (epic #1154).

The registry is the single place a theme is named.  The boot script, the
Settings picker, ``tools/theme_matrix`` and the packaging test all read it from
here, so a new theme is one file plus one entry — no JavaScript list, no second
``<option>``, no matrix edit.

This module is data and pure helpers only: it is imported by tools that never
start a gateway, so importing it must not touch the filesystem or the network.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """One theme: its name, its display label, and the file that declares it."""

    name: str
    """The ``data-scheme`` value and the file stem."""

    label: str
    """English label.  A translation arrives as the i18n key ``scheme.<name>``;
    a theme without one still shows this."""

    file: str
    """Path under the admin package, e.g. ``css/themes/minimal.css``."""

    default: bool = False
    """Whether this is the theme the panel uses when nothing is stored."""


THEMES: tuple[Theme, ...] = (
    Theme("minimal", "Minimal", "css/themes/minimal.css", default=True),
    Theme("emerald", "Emerald", "css/themes/emerald.css"),
)

#: URL prefix the panel serves static files under (``static.py``).
STATIC_URL_PREFIX = "/admin/static/"


def default_theme() -> Theme:
    """Return the theme the panel falls back to."""
    return next(theme for theme in THEMES if theme.default)


def theme_names() -> tuple[str, ...]:
    """Return every registered theme name, in registry order."""
    return tuple(theme.name for theme in THEMES)


def theme_by_name(name: str | None) -> Theme | None:
    """Return the theme called *name*, or ``None``.

    A stored preference can name a theme this build no longer ships; callers
    treat ``None`` as "use the default" rather than as an error.
    """
    for theme in THEMES:
        if theme.name == name:
            return theme
    return None


def theme_url(theme: Theme) -> str:
    """Return the URL the panel loads *theme* from."""
    return STATIC_URL_PREFIX + theme.file


def registry_payload() -> dict[str, object]:
    """Return the registry as the JSON the page's boot script reads."""
    return {
        "default": default_theme().name,
        "themes": [
            {"name": theme.name, "label": theme.label, "url": theme_url(theme)}
            for theme in THEMES
        ],
    }

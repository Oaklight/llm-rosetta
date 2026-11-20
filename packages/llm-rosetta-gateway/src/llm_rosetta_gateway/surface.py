"""The core surface level this gateway needs (#1630).

The gateway reaches ``llm_rosetta`` only through ``llm_rosetta.api`` and four
contract modules.  ``llm_rosetta.api.SURFACE_VERSION`` rises when a name is
removed or changes meaning, never when one is added; the gateway therefore
asserts a minimum at startup instead of failing later on a missing name.
"""

from __future__ import annotations

#: The lowest ``llm_rosetta.api.SURFACE_VERSION`` this gateway runs on.
REQUIRED_SURFACE_VERSION = 1


def check_core_surface(installed: object | None = None) -> None:
    """Fail clearly when the installed core's surface is older than needed.

    Args:
        installed: The installed core's ``SURFACE_VERSION``.  ``None`` reads it
            from ``llm_rosetta.api``; a core that predates the constant counts
            as level 0.

    Raises:
        RuntimeError: The installed level is below ``REQUIRED_SURFACE_VERSION``
            or is not an integer.
    """
    if installed is None:
        from llm_rosetta import api

        installed = getattr(api, "SURFACE_VERSION", 0)
    if (
        not isinstance(installed, int)
        or isinstance(installed, bool)
        or installed < REQUIRED_SURFACE_VERSION
    ):
        raise RuntimeError(
            f"llm-rosetta-gateway needs llm-rosetta core surface level "
            f">= {REQUIRED_SURFACE_VERSION}, but the installed core reports "
            f"{installed!r}. Upgrade llm-rosetta to a release that matches this "
            f"gateway (same minor)."
        )

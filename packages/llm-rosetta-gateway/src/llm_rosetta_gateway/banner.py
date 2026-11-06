"""Startup banner for llm-rosetta-gateway."""

from __future__ import annotations

from collections.abc import Sequence

from llm_rosetta import __version__ as core_version

from . import __version__ as gateway_version

BANNER = r"""
 ██╗     ██╗     ███╗   ███╗      ██████╗  ██████╗ ███████╗███████╗████████╗████████╗ █████╗
 ██║     ██║     ████╗ ████║      ██╔══██╗██╔═══██╗██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██╔══██╗
 ██║     ██║     ██╔████╔██║█████╗██████╔╝██║   ██║███████╗█████╗     ██║      ██║   ███████║
 ██║     ██║     ██║╚██╔╝██║╚════╝██╔══██╗██║   ██║╚════██║██╔══╝     ██║      ██║   ██╔══██║
 ███████╗███████╗██║ ╚═╝ ██║      ██║  ██║╚██████╔╝███████║███████╗   ██║      ██║   ██║  ██║
 ╚══════╝╚══════╝╚═╝     ╚═╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝
"""


def print_banner() -> None:
    """Print the startup banner with version info."""
    print(BANNER)
    print(
        f"  llm-rosetta-gateway  v{gateway_version}  (llm-rosetta core v{core_version})"
    )
    print()


def print_shim_sources(lines: Sequence[str]) -> None:
    """Print the shim-source summary beneath the banner.

    The lines come from
    :func:`llm_rosetta_gateway.shim_sources.describe_shim_sources`; each names
    the leg a remote source loaded from, so a mirror that took over is visible
    at startup without reading the log.
    """
    if not lines:
        return
    print("  shim sources:")
    for line in lines:
        print(f"    {line}")
    print()

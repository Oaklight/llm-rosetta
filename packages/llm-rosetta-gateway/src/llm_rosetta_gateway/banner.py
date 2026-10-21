"""Startup banner for llm-rosetta-gateway."""

from __future__ import annotations

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

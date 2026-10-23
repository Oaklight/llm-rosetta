"""Command line entry point: ``python -m tools.theme_matrix``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .gateway import running_gateway
from .runner import build_cells, declared_modes, declared_schemes, run_matrix


def _csv(value: str) -> list[str]:
    """Split a comma-separated option, dropping blanks."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _widths(value: str) -> list[int]:
    """Parse the width axis and reject a non-positive viewport."""
    widths = [int(part) for part in _csv(value)]
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("widths must be positive integers")
    return widths


def _setup_error(exc: RuntimeError) -> int:
    """Print a setup failure as a sentence, not a traceback, and return 2.

    Missing Playwright, a browser that will not launch and a gateway that
    never came up are expected paths for a first-time user; a stack trace
    buries the one line that says what to install.
    """
    print(f"error: {exc}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.theme_matrix",
        description=(
            "Render the admin panel for every scheme × mode × width, assert "
            "the render-level contract, and save screenshots for review."
        ),
    )
    parser.add_argument(
        "--out",
        default="theme-matrix-out",
        help="directory for screenshots and the report (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "drive an already running gateway instead of spawning a disposable "
            "one; useful for a real database with realistic numbers"
        ),
    )
    parser.add_argument(
        "--admin-password",
        default=os.environ.get("THEME_MATRIX_ADMIN_PASSWORD", ""),
        help=(
            "admin password for --base-url gateways "
            "(default: $THEME_MATRIX_ADMIN_PASSWORD)"
        ),
    )
    parser.add_argument(
        "--schemes",
        default=None,
        help=(
            "comma-separated schemes (default: every scheme base.css "
            "declares, so a new scheme is covered automatically)"
        ),
    )
    parser.add_argument(
        "--modes",
        default=None,
        help=("comma-separated modes (default: every mode base.css declares)"),
    )
    parser.add_argument(
        "--widths",
        type=_widths,
        default=[390, 768, 1280],
        help="comma-separated viewport widths (default: 390,768,1280)",
    )
    parser.add_argument(
        "--keep-gateway",
        action="store_true",
        help="keep the disposable gateway's config, data and log on disk",
    )

    args = parser.parse_args(argv)
    schemes = _csv(args.schemes) if args.schemes else declared_schemes()
    modes = _csv(args.modes) if args.modes else declared_modes()
    cells = build_cells(schemes, modes, args.widths)
    out_dir = Path(args.out)

    if args.base_url:
        try:
            report = run_matrix(
                args.base_url.rstrip("/"),
                out_dir,
                cells,
                password=args.admin_password,
                source="external",
            )
        except RuntimeError as exc:
            return _setup_error(exc)
    else:
        try:
            with running_gateway(keep=args.keep_gateway) as base_url:
                report = run_matrix(
                    base_url,
                    out_dir,
                    cells,
                    password=args.admin_password,
                    source="spawned",
                )
        except RuntimeError as exc:
            return _setup_error(exc)

    passed = sum(1 for cell in report["cells"] if cell["ok"])
    print(
        f"\n{passed}/{len(report['cells'])} cells pass; report: {out_dir / 'report.md'}"
    )
    return 0 if report["ok"] else 1

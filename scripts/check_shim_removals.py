#!/usr/bin/env python3
"""Fail when the repository still references a shim that has been removed.

The gate is self-arming.  While the package version is below a removal's
``removed_in``, the repository is *supposed* to keep mentioning those shims —
they still work — so it passes.  The day the version reaches it, every remaining
mention under the scanned roots, and every one still registered, becomes an
error, so a removal cannot land without the cleanup that belongs with it:

* the built-in shim directory and its ``LEGACY_SHIM_ALIASES`` entry,
* the ``SHIM_REMOVALS`` entry in ``llm_rosetta.shims.removals``,
* the examples under ``examples/`` and the tables in ``docs_en`` / ``docs_zh``.

The manifest is the single source of what is going away, so this script reads it
rather than keeping its own list.

Usage::

    python scripts/check_shim_removals.py            # the real check
    python scripts/check_shim_removals.py --version 0.16.0   # what it would say then

Exits 0 when the repository is consistent with the version, 1 when it is not,
and 2 on a usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `pytest.ini` puts `src` on the path for the test suite; a plain
# `python scripts/...` run does not, so do it here too and let the gate work
# from a checkout — the package it inspects is the one next to it, not whichever
# llm-rosetta happens to be installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_rosetta import __version__  # noqa: E402
from llm_rosetta.shims import (  # noqa: E402
    SHIM_REMOVALS,
    RemovalReference,
    get_shim,
    removals_due,
    scan_removal_references,
)
from llm_rosetta.shims.providers import load_providers  # noqa: E402

#: Where the repository spells shim names.  ``src`` and ``scripts`` are here
#: because the cleanup a removal owes is not only prose: a leftover
#: ``LEGACY_SHIM_ALIASES`` entry, a built-in ``provider.yaml``, or a config
#: *example* in a helper script would otherwise outlive the shim it names — and
#: the registration half cannot stand in for the scan, since a shim that is
#: already gone resolves to ``None`` whether or not its alias remains.
#: ``design`` holds the UI mock-ups, which spell shim names too.
#: ``docs_en`` and ``docs_zh`` are separate worktrees, so CI has to check them
#: out for those two to be scanned — a missing root is skipped, not an error.
DEFAULT_ROOTS = (
    "examples",
    ".github/ci",
    "docker",
    "design",
    "scripts",
    "docs_en/docs",
    "docs_zh/docs",
    "src",
)

#: The changelogs record what happened; their history is not rewritten, and a
#: "removed in 0.16.0" line is exactly the kind of mention that must survive.
#: ``removals.py`` is the manifest itself — it names every shim that is going
#: away, by design, so scanning it would fail each entry on its own name.
DEFAULT_EXEMPT_FILES = (
    "docs_en/docs/changelog.md",
    "docs_zh/docs/changelog.md",
    "src/llm_rosetta/shims/removals.py",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        default=__version__,
        help="Version to check against (default: the installed package version).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Files or directories to scan for stale mentions.",
    )
    parser.add_argument(
        "--exempt-file",
        action="append",
        default=list(DEFAULT_EXEMPT_FILES),
        help="A file to skip entirely (repeatable).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing on success.",
    )
    return parser.parse_args(argv)


def _reference_message(reference: RemovalReference) -> str:
    return (
        f"{reference.path}:{reference.line}: still references removed shim "
        f"{reference.shim!r} — {reference.text}"
    )


def check(version: str, roots: list[str], exempt_files: list[str]) -> list[str]:
    """Return one message per inconsistency between *version* and the repository.

    Args:
        version: The version to check against.
        roots: Files or directories to scan.
        exempt_files: Files to skip entirely.

    Returns:
        The problems found, empty when the repository is consistent.
    """
    # The check compares the manifest against the live registry, so the built-ins
    # have to be in it.  Importing the package normally does that; doing it here
    # as well means the answer does not depend on what the calling process
    # happens to have loaded.
    load_providers()
    due = {entry.shim for entry in removals_due(version)}
    problems: list[str] = []
    for entry in SHIM_REMOVALS:
        registered = get_shim(entry.shim) is not None
        if entry.shim in due:
            if registered:
                problems.append(
                    f"{entry.shim!r} is still registered, but was scheduled for "
                    f"removal in {entry.removed_in} — delete the shim and this "
                    f"SHIM_REMOVALS entry together."
                )
            problems.extend(
                _reference_message(reference)
                for reference in scan_removal_references(
                    roots, removals=[entry], exempt_files=exempt_files
                )
            )
        elif not registered:
            problems.append(
                f"{entry.shim!r} is not registered, but is not scheduled for "
                f"removal until {entry.removed_in} — it was removed early, or the "
                f"manifest entry outlived the rename."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the check and report it.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 when the repository is consistent, 1 when it is not.
    """
    args = _parse_args(argv)
    problems = check(args.version, args.roots, args.exempt_file)
    if problems:
        print(f"shim removals: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            f"shim removals: {len(SHIM_REMOVALS)} scheduled, none due at {args.version}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

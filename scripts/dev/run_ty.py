#!/usr/bin/env python3
"""Run ``ty check`` the way CI does, from any checkout or worktree (stdlib only).

CI installs the project editable from its own checkout, so ``ty`` resolves
``llm_rosetta`` and ``llm_rosetta_gateway`` to the sources under test.  A local
environment may hold an editable install that points at ANOTHER checkout (for
example the main checkout while you work in a git worktree).  ``ty`` then
type-checks this checkout's tests against the other checkout's sources and
reports spurious diagnostics.

This wrapper puts this checkout's package sources first with
``--extra-search-path`` and points ``ty`` at the active environment, so the
result is the same from every worktree.  Extra arguments go to ``ty check``.
Exit status is ``ty``'s: non-zero on the first diagnostic, as in CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SEARCH_PATHS = ("packages/llm-rosetta/src", "packages/llm-rosetta-gateway/src")


def find_repo_root(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` that holds the package sources."""
    for p in (start, *start.parents):
        if all((p / s).is_dir() for s in SEARCH_PATHS):
            return p
    return None


def pick_python() -> str:
    """Return the environment prefix: $VIRTUAL_ENV, $CONDA_PREFIX, sys.prefix."""
    return os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX") or sys.prefix


def find_ty() -> list[str] | None:
    """Locate ``ty`` on PATH, beside this interpreter, or as ``python -m ty``."""
    exe = shutil.which("ty")
    if exe:
        return [exe]
    sibling = Path(sys.executable).with_name("ty")
    if sibling.exists():
        return [str(sibling)]
    probe = subprocess.run(
        [sys.executable, "-m", "ty", "--version"], capture_output=True, text=True
    )
    return [sys.executable, "-m", "ty"] if probe.returncode == 0 else None


def main(argv: list[str]) -> int:
    """Run ``ty check`` with this checkout's sources first on the search path."""
    root = find_repo_root(Path.cwd())
    if root is None:
        print("run_ty: run from inside the llm-rosetta checkout.", file=sys.stderr)
        return 2
    ty = find_ty()
    if ty is None:
        print(
            "run_ty: `ty` is not installed. Install the pinned dev tools with "
            '`pip install -e "packages/llm-rosetta[dev]"` (pins ty==0.0.54).',
            file=sys.stderr,
        )
        return 2
    prefix = pick_python()
    if not (os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")):
        print(
            f"run_ty: no virtualenv or conda env is active; checking against {prefix}. "
            "If that interpreter lacks the project's dependencies, ty reports "
            "unrelated errors: activate the env that has "
            '`pip install -e "packages/llm-rosetta[all]"` and run again.',
            file=sys.stderr,
        )
    cmd = [*ty, "check", "--python", prefix]
    for sp in SEARCH_PATHS:
        cmd += ["--extra-search-path", sp]
    return subprocess.call([*cmd, *argv], cwd=root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

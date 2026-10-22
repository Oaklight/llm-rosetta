"""Run a docs branch's own Read the Docs build, exactly as the branch states it.

The docs live on two orphan branches and each one carries a
``.readthedocs.yaml`` whose ``pre_build`` copies the source out of this
repository before the site is built.  A CI job that *re-describes* those steps
(``cp -r packages/...``) cannot catch a branch whose yaml has rotted: it keeps
passing while Read the Docs fails.  This runs the branch's own steps instead.

The clone is redirected at the local checkout with git's ``url.*.insteadOf``,
so the branch's ``git clone https://github.com/Oaklight/llm-rosetta.git`` reads
this working tree and no network is needed.

Usage:
    python scripts/dev/rtd_build.py docs_en [--repo-root .]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_URL = "https://github.com/Oaklight/llm-rosetta.git"


def _commands(entry: object) -> list[str]:
    """Flatten one ``pre_build``/``build.html`` entry into shell lines."""
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, list):
        return [str(item) for item in entry]
    raise SystemExit(f"Unsupported step in .readthedocs.yaml: {entry!r}")


def _redirect_env(repo_root: Path) -> dict[str, str]:
    """Point the documented clone at *repo_root* instead of GitHub."""
    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = f"url.file://{repo_root}.insteadOf"
    env["GIT_CONFIG_VALUE_0"] = REPO_URL
    # A file:// clone only carries the refs that exist locally.  The CI
    # checkout is a detached commit, so give it a name first.
    subprocess.run(
        ["git", "-C", str(repo_root), "update-ref", "refs/heads/_rtd-checkout", "HEAD"],
        check=True,
    )
    return env


def main(argv: list[str] | None = None) -> int:
    """Run the branch's pre_build and build steps in order.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 when every step succeeded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch_dir", help="The docs branch checkout, e.g. docs_en")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository the branch's clone should read (default: cwd)",
    )
    args = parser.parse_args(argv)

    branch = Path(args.branch_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    spec = yaml.safe_load((branch / ".readthedocs.yaml").read_text())
    jobs = spec["build"]["jobs"]

    steps: list[tuple[str, list[str]]] = []
    if jobs.get("pre_build"):
        steps.append(("pre_build", _commands(jobs["pre_build"])))
    build = jobs["build"]
    html = build["html"] if isinstance(build, dict) else build
    steps.append(("build", _commands(html)))

    env = _redirect_env(repo_root)
    # A previous run — or an interrupted one — leaves the branch's clone
    # behind, and ``git clone`` refuses a destination that exists.  RTD always
    # starts from a clean checkout; locally that has to be arranged.
    shutil.rmtree(branch / "src_repo", ignore_errors=True)
    try:
        for name, commands in steps:
            for command in commands:
                print(f"[{branch.name}] {name}: {command}", flush=True)
                result = subprocess.run(command, shell=True, cwd=branch, env=env)
                if result.returncode != 0:
                    print(
                        f"[{branch.name}] {name} failed (exit {result.returncode})",
                        file=sys.stderr,
                    )
                    return result.returncode
    finally:
        # Never leave the branch's clone behind, on success or failure: it is
        # scratch, and a stale one makes the next run fail on the clone.
        shutil.rmtree(branch / "src_repo", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

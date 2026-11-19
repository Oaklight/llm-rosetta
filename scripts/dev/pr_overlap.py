"""Warn when open PRs touch the same files as the current branch.

Run it before opening a PR (series rule, see CONTRIBUTING.md): it lists the
files the branch changes against its base, asks ``gh`` for every open PR with
its files, and prints each other PR that touches any of the same paths.  Local
branches can be checked as well with ``--local``.

By default it exits 0 and prints a WARNING block on overlap, so it is safe in a
hook.  ``--fail-on-overlap`` is the gate mode, for CI or a pre-push check:

* exit 0: the check ran and found no overlap;
* exit 1: the check ran and found overlap;
* exit 2: the check could not run or could not see every open PR (``gh``
  missing or failing, bad JSON, or the PR list hit its ``--limit``).  A gate
  never passes on a check that did not finish.

The PR the branch belongs to is excluded by head ref name and by head commit,
so ``--head HEAD``, a SHA or ``origin/<branch>`` work as well as a branch name
(``$GITHUB_HEAD_REF`` is also honoured, for a detached merge-commit checkout).

    python scripts/dev/pr_overlap.py
    python scripts/dev/pr_overlap.py --head fix/x --base origin/master --local

Only the standard library and the ``gh`` CLI are used.  The ``run`` argument of
the functions is the one place a subprocess starts, so tests inject a stub.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence

Runner = Callable[[Sequence[str]], str]

PR_LIMIT = 200
EXIT_OVERLAP = 1
EXIT_UNVERIFIED = 2


def run_command(argv: Sequence[str]) -> str:
    """Run a command and return its stdout.

    Args:
        argv: The command and its arguments.

    Returns:
        The decoded standard output.

    Raises:
        RuntimeError: When the command is missing or exits non-zero.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{argv[0]} is not installed") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed: {proc.stderr.strip()}")
    return proc.stdout


def current_branch(run: Runner = run_command) -> str:
    """Return the checked-out branch name."""
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()


def changed_files(head: str, base: str, run: Runner = run_command) -> set[str]:
    """Return the paths ``head`` changes relative to the merge base with ``base``."""
    out = run(["git", "diff", "--name-only", f"{base}...{head}"])
    return {line for line in out.splitlines() if line}


def resolve_sha(head: str, run: Runner = run_command) -> str | None:
    """Return the full commit SHA ``head`` points to, or None if it does not resolve."""
    try:
        out = run(["git", "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}"])
    except RuntimeError:
        return None
    return out.strip() or None


def open_prs(run: Runner = run_command) -> list[dict]:
    """Return open PRs as ``gh`` reports them.

    Each entry has number, title, headRefName, headRefOid and files.  ``gh pr
    list`` has no pagination, so a result of :data:`PR_LIMIT` entries may be
    truncated; the caller checks the length.
    """
    out = run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            str(PR_LIMIT),
            "--json",
            "number,title,headRefName,headRefOid,files",
        ]
    )
    return json.loads(out or "[]")


def overlapping_prs(
    files: set[str],
    prs: list[dict],
    own_branch: str,
    own_sha: str | None = None,
    own_refs: Sequence[str] = (),
) -> list[tuple[dict, list[str]]]:
    """Pair every other PR with the paths it shares with ``files``.

    Args:
        files: Paths the current branch changes.
        prs: Open PRs from :func:`open_prs`.
        own_branch: The branch's own head ref, excluded from the result.
        own_sha: Full SHA of the head; a PR whose ``headRefOid`` equals it is
            the branch's own PR and is excluded too.
        own_refs: Further head ref names to exclude (``$GITHUB_HEAD_REF``).

    Returns:
        ``(pr, sorted shared paths)`` for each overlapping PR, by PR number.
    """
    found = []
    for pr in prs:
        if pr.get("headRefName") in (own_branch, *own_refs):
            continue
        if own_sha and pr.get("headRefOid") == own_sha:
            continue
        theirs = {f.get("path") for f in pr.get("files") or []}
        shared = sorted(files & theirs)
        if shared:
            found.append((pr, shared))
    return sorted(found, key=lambda item: item[0].get("number", 0))


def overlapping_branches(
    files: set[str], head: str, base: str, run: Runner = run_command
) -> list[tuple[str, list[str]]]:
    """Pair every other local branch with the paths it shares with ``files``."""
    names = run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"])
    found = []
    for name in names.split():
        if name == head:
            continue
        try:
            theirs = changed_files(name, base, run)
        except RuntimeError:
            continue
        shared = sorted(files & theirs)
        if shared:
            found.append((name, shared))
    return found


def format_report(
    head: str,
    prs: list[tuple[dict, list[str]]],
    branches: list[tuple[str, list[str]]] | None = None,
) -> str:
    """Render the result; an empty overlap yields a one-line all-clear."""
    if not prs and not branches:
        return f"pr_overlap: no open PR or checked branch touches the files of {head}."
    lines = [
        "=" * 70,
        f"WARNING: {head} overlaps work in flight. Merge a series one at a time",
        "and rebase the rest; decide the order before opening this PR.",
        "=" * 70,
    ]
    for pr, shared in prs:
        lines.append(f"PR #{pr['number']} {pr.get('title', '')} [{pr['headRefName']}]")
        lines.extend(f"    {path}" for path in shared)
    for name, shared in branches or []:
        lines.append(f"local branch {name}")
        lines.extend(f"    {path}" for path in shared)
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, run: Runner = run_command) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--head", help="branch to check (default: current branch)")
    parser.add_argument("--base", default="origin/master", help="base ref")
    parser.add_argument(
        "--local", action="store_true", help="also compare other local branches"
    )
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="gate mode: exit 1 on overlap, 2 when the check could not complete",
    )
    args = parser.parse_args(argv)

    unverified = EXIT_UNVERIFIED if args.fail_on_overlap else 0
    try:
        head = args.head or current_branch(run)
        files = changed_files(head, args.base, run)
        if not files:
            print(f"pr_overlap: {head} changes no files against {args.base}.")
            return 0
        listed = open_prs(run)
        own_refs = (
            [os.environ["GITHUB_HEAD_REF"]] if os.environ.get("GITHUB_HEAD_REF") else []
        )
        prs = overlapping_prs(files, listed, head, resolve_sha(head, run), own_refs)
        branches = (
            overlapping_branches(files, head, args.base, run) if args.local else []
        )
    except (RuntimeError, ValueError) as exc:
        print(f"pr_overlap: could not check ({exc}); overlap is unverified.")
        return unverified
    truncated = len(listed) >= PR_LIMIT
    if truncated:
        print(
            f"pr_overlap: gh returned {len(listed)} open PRs (the limit); "
            "older PRs may be missing, so overlap is not fully verified."
        )
    print(format_report(head, prs, branches))
    if prs or branches:
        return EXIT_OVERLAP if args.fail_on_overlap else 0
    return unverified if truncated else 0


if __name__ == "__main__":
    sys.exit(main())

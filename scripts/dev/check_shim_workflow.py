"""The five published shim packages must carry the same branch-publish workflow.

Each package repository ships `.github/workflows/publish-branch.yml` verbatim:
five copies of a hundred-line job, rather than a reusable workflow that couples
six repositories to one another.  Verbatim means drift, and this drift is
silent — a copy that stops copying the `.sha256` sibling leaves a mirror whose
only symptom is a missing file on a branch nobody reads until a mirror is
wanted.

So the master copy lives here, in `shim-publish-branch.yml`, and this script
compares every package's copy with it.  A copy that differs is a failure; a
repository that has no copy yet is a warning until the rollout finishes, and a
failure once `--require-all` is passed (which the nightly job will do as soon
as all five carry the file).

Usage::

    python scripts/dev/check_shim_workflow.py                  # token from $GITHUB_TOKEN
    python scripts/dev/check_shim_workflow.py --require-all
    python scripts/dev/check_shim_workflow.py --from-dir saved  # offline
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]

#: The master copy every package repository must reproduce byte for byte.
MASTER = Path(__file__).resolve().with_name("shim-publish-branch.yml")

#: Where the copy lives inside a package repository.
WORKFLOW_PATH = ".github/workflows/publish-branch.yml"

#: The default branch the packages publish from.
BRANCH = "main"


@dataclass(frozen=True)
class Copy:
    """One package's copy of the workflow.

    Attributes:
        key: The catalogue key of the package (``argo``, ``alcf``, …).
        repo: The repository URL the catalogue names.
        content: The file's bytes, or ``None`` when the repository has none.
    """

    key: str
    repo: str
    content: bytes | None


def _slug(repo: str) -> str:
    """``https://github.com/Owner/name`` → ``Owner/name``."""
    return repo.rstrip("/").removeprefix("https://github.com/").removesuffix(".git")


def raw_url(repo: str, *, branch: str = BRANCH) -> str:
    """The raw file URL for *repo*'s copy of the workflow."""
    return f"https://raw.githubusercontent.com/{_slug(repo)}/{branch}/{WORKFLOW_PATH}"


def fetch_copy(
    repo: str, *, token: str | None = None, branch: str = BRANCH
) -> bytes | None:
    """Read one repository's copy, or ``None`` when it does not carry one.

    Args:
        repo: The repository URL.
        token: A GitHub token; raw.githubusercontent.com serves public files
            without one, so this is optional.
        branch: The branch the file is read from.

    Returns:
        The file's bytes, or ``None`` on ``404``.

    Raises:
        SystemExit: Any other answer — a 403 or a 5xx must not be read as "the
            file is missing", which is the failure this script exists to catch.
    """
    url = raw_url(repo, branch=branch)
    headers = {"User-Agent": "llm-rosetta"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=20) as response:  # noqa: S310
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise SystemExit(f"GitHub answered {exc.code} for {url}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot reach GitHub: {exc.reason}") from exc


def compare_copies(
    master: bytes, copies: Iterable[Copy]
) -> tuple[list[Copy], list[Copy], list[Copy]]:
    """Sort the copies into agreeing, missing and differing.

    Args:
        master: The bytes of the master copy.
        copies: One entry per published package.

    Returns:
        ``(agreeing, missing, differing)``, each in catalogue order.
    """
    agreeing: list[Copy] = []
    missing: list[Copy] = []
    differing: list[Copy] = []
    for copy in copies:
        if copy.content is None:
            missing.append(copy)
        elif copy.content == master:
            agreeing.append(copy)
        else:
            differing.append(copy)
    return agreeing, missing, differing


def _first_difference(master: bytes, copy: bytes) -> str:
    """Describe where two copies part company, in terms a person can act on."""
    master_lines = master.decode("utf-8", "replace").splitlines()
    copy_lines = copy.decode("utf-8", "replace").splitlines()
    for index, (left, right) in enumerate(zip(master_lines, copy_lines), start=1):
        if left != right:
            return (
                f"line {index} differs\n"
                f"      master: {left.strip()}\n"
                f"      copy:   {right.strip()}"
            )
    if len(master_lines) != len(copy_lines):
        return (
            f"length differs: master has {len(master_lines)} lines, "
            f"the copy {len(copy_lines)}"
        )
    return "the bytes differ without a differing line (line endings or encoding)"


def _load_saved(directory: Path, packages: list) -> list[Copy]:
    """Read copies from a directory of saved files, for an offline run.

    A missing file is a package without a copy; that is the same state the
    online run reports, so the offline mode stays a faithful rehearsal.
    """
    copies: list[Copy] = []
    for package in packages:
        path = directory / f"{package.key}.yml"
        copies.append(
            Copy(
                key=package.key,
                repo=package.repo,
                content=path.read_bytes() if path.is_file() else None,
            )
        )
    return copies


def report(
    master: bytes,
    agreeing: list[Copy],
    missing: list[Copy],
    differing: list[Copy],
    *,
    require_all: bool,
) -> int:
    """Print the verdict and return the exit code.

    Args:
        master: The master copy's bytes.
        agreeing: Packages whose copy matches.
        missing: Packages with no copy yet.
        differing: Packages whose copy departs from the master.
        require_all: Whether a missing copy is a failure.

    Returns:
        ``0`` when nothing is wrong, ``1`` otherwise.
    """
    for copy in agreeing:
        print(f"ok      {copy.key}: {copy.repo} matches the master copy")
    for copy in missing:
        label = "problem" if require_all else "warning"
        print(
            f"{label} {copy.key}: {copy.repo} has no {WORKFLOW_PATH} on "
            f"{BRANCH}; copy {MASTER.name} into it"
        )
    for copy in differing:
        assert copy.content is not None
        print(
            f"problem {copy.key}: {copy.repo} differs from the master copy —\n"
            f"      {_first_difference(master, copy.content)}"
        )
    if differing or (missing and require_all):
        print(
            f"\n{len(differing)} differing, {len(missing)} missing; the master "
            f"copy is {MASTER.relative_to(ROOT).as_posix()}"
        )
        return 1
    if missing:
        print(
            f"\n{len(missing)} package(s) have no copy yet; "
            "pass --require-all once every repository carries one"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Args:
        argv: Command-line arguments; ``--from-dir`` rehearses a run offline.

    Returns:
        The process exit code: ``0`` when the copies agree, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Treat a package with no copy as a failure (use after the rollout)",
    )
    parser.add_argument(
        "--from-dir",
        metavar="DIR",
        help="Read <key>.yml files from DIR instead of the network",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token; defaults to $GITHUB_TOKEN",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "packages/llm-rosetta-gateway/src"))
    sys.path.insert(0, str(ROOT / "packages/llm-rosetta/src"))
    from llm_rosetta_gateway.shims.catalog import SHIM_SOURCE_PACKAGES

    master = MASTER.read_bytes()
    if args.from_dir:
        copies = _load_saved(Path(args.from_dir), list(SHIM_SOURCE_PACKAGES))
    else:
        copies = [
            Copy(
                key=package.key,
                repo=package.repo,
                content=fetch_copy(package.repo, token=args.token),
            )
            for package in SHIM_SOURCE_PACKAGES
        ]
    agreeing, missing, differing = compare_copies(master, copies)
    return report(master, agreeing, missing, differing, require_all=args.require_all)


if __name__ == "__main__":
    raise SystemExit(main())

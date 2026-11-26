"""Check that every shim repository the account publishes is in the catalogue.

The manifest sources the gateway reads (``SHIM_SOURCES``) are a hand-written
record.  A new source repository therefore goes unnoticed until someone reads
about it, and an operator who knows it exists has no key for it.  This check
compares the account's repositories with the record and says which side to
edit:

- a repository named ``llm-rosetta-ext-*`` that the record does not name
  (add a ``ShimSource`` — the output prints a skeleton to paste);
- a source whose repository no longer exists (drop the entry, or fix the URL
  after a rename).

Run it by hand (``python scripts/dev/check_shim_catalog.py``) or through the
``shim-catalog`` CI job, which fails on drift so the reminder arrives as a red
check rather than a review comment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from collections.abc import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The name prefix of an extension repository.
SHIM_REPO_PREFIXES = ("llm-rosetta-ext-",)


def _normalise(url: str) -> str:
    """A repository URL reduced to ``host/owner/name``, for comparison."""
    value = (url or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[: -len(".git")]
    value = value.replace("git@github.com:", "https://github.com/")
    return value.lower()


def shim_repositories(repos: Iterable[dict[str, Any]]) -> dict[str, str]:
    """The account's shim-package repositories, as ``normalised name -> url``.

    Args:
        repos: Repository objects as the GitHub API returns them.

    Returns:
        Every repository whose name carries a shim-package prefix.  Archived
        repositories are included: an archived package still has a name in the
        catalogue, and retiring one is a deliberate edit on both sides.
    """
    found: dict[str, str] = {}
    for repo in repos:
        name = str(repo.get("name") or "")
        if not name.startswith(SHIM_REPO_PREFIXES):
            continue
        html_url = str(repo.get("html_url") or repo.get("url") or "")
        owner = (repo.get("owner") or {}).get("login") or ""
        if not html_url and owner:
            html_url = f"https://github.com/{owner}/{name}"
        found[_normalise(html_url)] = name
    return found


def compare_catalog(
    repos: dict[str, str],
    catalog: Iterable[Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Diff the repositories against the manifest sources.

    Args:
        repos: ``normalised url -> name``, from :func:`shim_repositories`.
        catalog: ``ShimSource`` records (``key`` and ``repo``).

    Returns:
        ``(missing, stale)`` — repositories the record does not name, and
        sources whose repository is not in the listing.  Each item is
        ``(key_or_name, url)``.
    """
    by_url = {pkg.key: _normalise(pkg.repo) for pkg in catalog}
    known = set(by_url.values())
    missing = [(name, url) for url, name in sorted(repos.items()) if url not in known]
    stale = [(key, url) for key, url in sorted(by_url.items()) if url not in repos]
    return missing, stale


def fetch_repositories(owner: str, *, token: str | None = None) -> list[dict[str, Any]]:
    """Read every public repository of *owner* from the GitHub API.

    Args:
        owner: The account or organisation name.
        token: A GitHub token; unauthenticated requests work but are limited to
            60 per hour, which CI shares with everything else.

    Returns:
        The repository objects, across every page.

    Raises:
        SystemExit: The API could not be reached or answered with an error.
    """
    repos: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{owner}/repos?per_page=100&page={page}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "llm-rosetta"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:  # noqa: S310
                batch = json.load(response)
        except HTTPError as exc:  # noqa: BLE001 - the status is the answer
            raise SystemExit(f"GitHub answered {exc.code} for {url}") from exc
        except URLError as exc:
            raise SystemExit(f"cannot reach GitHub: {exc.reason}") from exc
        if not isinstance(batch, list) or not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def _skeleton(name: str, url: str) -> str:
    """A ``ShimSource`` entry to paste, with the fields only a human can fill."""
    key = (
        name.removeprefix("llm-rosetta-ext-")
        .removeprefix("llm-rosetta-shim-")
        .removeprefix("llm-rosetta-shims-")
    )
    return (
        "    ShimSource(\n"
        f'        key="{key}",\n'
        f'        label="{key.title()}",\n'
        f'        repo="{url}",\n'
        "    ),"
    )


def report(
    owner: str, missing: list[tuple[str, str]], stale: list[tuple[str, str]]
) -> int:
    """Print the drift and return the exit code.

    Args:
        owner: The account the listing came from.
        missing: Repositories the record does not name.
        stale: Sources whose repository is not in the listing.

    Returns:
        ``0`` when the two agree, ``1`` otherwise.
    """
    if not missing and not stale:
        print(f"shim catalogue: {owner}'s packages match SHIM_SOURCES")
        return 0
    print(f"shim catalogue drift for {owner}:")
    for name, url in missing:
        print(f"\n  add   {name}\n        {url}\n\n{_skeleton(name, url)}")
    for key, url in stale:
        print(
            f"\n  stale {key}: the catalogue names {url}, which is not in the account"
        )
    print(
        "\nUpdate SHIM_SOURCES in "
        "packages/llm-rosetta-gateway/src/llm_rosetta_gateway/shims/catalog.py "
        "so the gateway offers every source."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the check.

    Args:
        argv: Command-line arguments; ``--from-json`` reads a saved listing
            instead of calling GitHub, which the tests and an offline run use.

    Returns:
        ``0`` when the catalogue matches, ``1`` on drift.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", default="Oaklight", help="GitHub account to list")
    parser.add_argument(
        "--from-json",
        default=None,
        help="Read the repository listing from this file instead of the API",
    )
    args = parser.parse_args(argv)

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as handle:
            repos = json.load(handle)
    else:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        repos = fetch_repositories(args.owner, token=token)

    sys.path.insert(0, str(REPO_ROOT / "packages/llm-rosetta-gateway/src"))
    sys.path.insert(0, str(REPO_ROOT / "packages/llm-rosetta/src"))
    from llm_rosetta_gateway.shims.catalog import SHIM_SOURCES  # noqa: E402

    missing, stale = compare_catalog(shim_repositories(repos), SHIM_SOURCES)
    return report(args.owner, missing, stale)


if __name__ == "__main__":
    raise SystemExit(main())

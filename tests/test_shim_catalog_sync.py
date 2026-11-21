"""The account's shim repositories and the catalogue must name the same set."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The script under test.  Loaded by path rather than by putting ``scripts/dev``
#: on ``sys.path``: a path insert is invisible to the type checker, which then
#: cannot resolve the import at all.
SCRIPT = ROOT / "scripts" / "dev" / "check_shim_catalog.py"

spec = importlib.util.spec_from_file_location("check_shim_catalog", SCRIPT)
assert spec is not None and spec.loader is not None, SCRIPT
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

sys.path.insert(0, str(ROOT / "packages/llm-rosetta-gateway/src"))
sys.path.insert(0, str(ROOT / "packages/llm-rosetta/src"))

from llm_rosetta_gateway.shims.catalog import SHIM_SOURCE_PACKAGES  # noqa: E402


def _repo(name: str, owner: str = "Oaklight", **extra) -> dict:
    return {
        "name": name,
        "html_url": f"https://github.com/{owner}/{name}",
        "owner": {"login": owner},
        **extra,
    }


def test_only_shim_package_repositories_are_considered():
    repos = check.shim_repositories(
        [
            _repo("llm-rosetta-shim-acme"),
            _repo("llm-rosetta-shims-legacy"),
            _repo("llm-rosetta"),
            _repo("llm-rosetta-gateway"),
        ]
    )

    assert sorted(repos.values()) == [
        "llm-rosetta-shim-acme",
        "llm-rosetta-shims-legacy",
    ]


def test_a_catalogue_url_is_compared_without_case_or_suffix_noise():
    repos = check.shim_repositories(
        [
            _repo(
                "llm-rosetta-shim-acme",
                html_url="https://github.com/Oaklight/llm-rosetta-shim-acme.git",
            )
        ]
    )

    class Pkg:
        key = "acme"
        repo = "https://GitHub.com/oaklight/llm-rosetta-shim-acme/"

    missing, stale = check.compare_catalog(repos, [Pkg()])

    assert missing == [] and stale == []


def test_a_repository_the_catalogue_does_not_name_is_missing():
    repos = check.shim_repositories([_repo("llm-rosetta-shim-new")])

    missing, stale = check.compare_catalog(repos, [])

    assert missing == [
        ("llm-rosetta-shim-new", "https://github.com/oaklight/llm-rosetta-shim-new")
    ]
    assert stale == []


def test_a_catalogue_entry_whose_repository_is_gone_is_stale():
    class Pkg:
        key = "gone"
        repo = "https://github.com/Oaklight/llm-rosetta-shim-gone"

    missing, stale = check.compare_catalog({}, [Pkg()])

    assert missing == []
    assert stale == [("gone", "https://github.com/oaklight/llm-rosetta-shim-gone")]


def test_the_current_catalogue_matches_its_own_repositories():
    """The published packages are the baseline: this is what CI runs."""
    repos = check.shim_repositories(
        [_repo(pkg.repo.rsplit("/", 1)[1]) for pkg in SHIM_SOURCE_PACKAGES]
    )

    missing, stale = check.compare_catalog(repos, SHIM_SOURCE_PACKAGES)

    assert (missing, stale) == ([], [])


def test_every_catalogue_entry_points_at_a_shim_repository():
    """A typo in a URL would read as "stale" forever; catch it here instead."""
    for pkg in SHIM_SOURCE_PACKAGES:
        assert pkg.repo.startswith("https://github.com/"), pkg.key
        assert "llm-rosetta-shim" in pkg.repo, pkg.key


def test_the_skeleton_names_the_package_the_repository_holds(capsys):
    code = check.report(
        "Oaklight",
        [
            (
                "llm-rosetta-shim-acme",
                "https://github.com/Oaklight/llm-rosetta-shim-acme",
            )
        ],
        [],
    )

    out = capsys.readouterr().out
    assert code == 1
    assert 'key="acme"' in out
    assert 'package="llm_rosetta_shim_acme"' in out


def test_offline_run_reads_a_saved_listing(tmp_path, capsys):
    listing = [_repo(pkg.repo.rsplit("/", 1)[1]) for pkg in SHIM_SOURCE_PACKAGES]
    source = tmp_path / "repos.json"
    source.write_text(json.dumps(listing), encoding="utf-8")

    assert check.main(["--from-json", str(source)]) == 0
    assert "match SHIM_SOURCE_PACKAGES" in capsys.readouterr().out

    listing.append(_repo("llm-rosetta-shim-extra"))
    source.write_text(json.dumps(listing), encoding="utf-8")

    assert check.main(["--from-json", str(source)]) == 1
    assert "llm-rosetta-shim-extra" in capsys.readouterr().out

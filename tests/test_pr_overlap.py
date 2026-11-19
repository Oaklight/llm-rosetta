"""Tests for scripts/dev/pr_overlap.py with a stubbed command runner."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pr_overlap", Path(__file__).parent.parent / "scripts" / "dev" / "pr_overlap.py"
)
assert _SPEC is not None and _SPEC.loader is not None
pr_overlap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pr_overlap)

FULL = "abcdef0123456789abcdef0123456789abcdef01"

PRS = [
    {
        "number": 7,
        "title": "own",
        "headRefName": "feat/mine",
        "headRefOid": FULL,
        "files": [{"path": "a.py"}],
    },
    {
        "number": 9,
        "title": "other",
        "headRefName": "feat/other",
        "files": [{"path": "a.py"}, {"path": "z.py"}],
    },
    {
        "number": 11,
        "title": "unrelated",
        "headRefName": "feat/un",
        "files": [{"path": "q.py"}],
    },
]


class Runner:
    """A stub for the subprocess call; ``calls`` records every argv."""

    def __init__(self, diffs, prs, branches, fail_gh):
        self.diffs = diffs
        self.prs = prs
        self.branches = branches
        self.fail_gh = fail_gh
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[:2] == ["git", "rev-parse"]:
            if "--abbrev-ref" in argv:
                return "feat/mine\n"
            rev = argv[-1].removesuffix("^{commit}")
            if rev in ("HEAD", "feat/mine", FULL) or FULL.startswith(rev):
                return FULL + "\n"
            raise RuntimeError(f"unknown revision {rev}")
        if argv[:2] == ["git", "diff"]:
            head = argv[-1].split("...")[1]
            return "\n".join(self.diffs.get(head, [])) + "\n"
        if argv[:2] == ["git", "for-each-ref"]:
            return self.branches
        if argv[0] == "gh":
            if self.fail_gh:
                raise RuntimeError("gh is not installed")
            return self.prs if isinstance(self.prs, str) else json.dumps(self.prs)
        raise AssertionError(argv)


def make_runner(diffs=None, prs=PRS, branches="feat/mine\nfeat/loc\n", fail_gh=False):
    return Runner(diffs or {}, prs, branches, fail_gh)


def test_overlap_lists_other_prs_and_excludes_own(capsys):
    run = make_runner({"feat/mine": ["a.py", "b.py"]})
    assert pr_overlap.main([], run) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "PR #9" in out and "a.py" in out
    assert "PR #7" not in out and "PR #11" not in out


def test_fail_on_overlap_exits_one():
    run = make_runner({"feat/mine": ["a.py"]})
    assert pr_overlap.main(["--fail-on-overlap"], run) == 1


def test_no_overlap_exits_zero_even_with_flag(capsys):
    run = make_runner({"feat/mine": ["b.py"]})
    assert pr_overlap.main(["--fail-on-overlap"], run) == 0
    assert "WARNING" not in capsys.readouterr().out


def test_head_and_base_arguments_reach_git():
    run = make_runner({"feat/x": ["a.py"]})
    pr_overlap.main(["--head", "feat/x", "--base", "main"], run)
    assert ["git", "diff", "--name-only", "main...feat/x"] in run.calls
    assert not any("--abbrev-ref" in c for c in run.calls)


def test_local_branches_are_compared_when_asked(capsys):
    run = make_runner({"feat/mine": ["a.py"], "feat/loc": ["a.py", "k.py"]}, prs=[])
    assert pr_overlap.main(["--local"], run) == 0
    out = capsys.readouterr().out
    assert "local branch feat/loc" in out
    assert "local branch feat/mine" not in out


def test_local_branches_skipped_by_default(capsys):
    run = make_runner({"feat/mine": ["a.py"], "feat/loc": ["a.py"]}, prs=[])
    pr_overlap.main([], run)
    assert not any(c[:2] == ["git", "for-each-ref"] for c in run.calls)
    assert "WARNING" not in capsys.readouterr().out


def test_gh_failure_is_reported_not_raised(capsys):
    run = make_runner({"feat/mine": ["a.py"]}, fail_gh=True)
    assert pr_overlap.main([], run) == 0
    assert "unverified" in capsys.readouterr().out


def test_gh_failure_fails_closed_in_gate_mode(capsys):
    run = make_runner({"feat/mine": ["a.py"]}, fail_gh=True)
    assert pr_overlap.main(["--fail-on-overlap"], run) == 2
    assert "unverified" in capsys.readouterr().out


def test_bad_json_fails_closed_in_gate_mode(capsys):
    run = make_runner({"feat/mine": ["a.py"]}, prs="{not json")
    assert pr_overlap.main([], run) == 0
    assert pr_overlap.main(["--fail-on-overlap"], run) == 2
    assert "unverified" in capsys.readouterr().out


@pytest.mark.parametrize("head", ["feat/mine", FULL, "HEAD", FULL[:7]])
def test_own_pr_is_excluded_for_any_head_spelling(head, capsys):
    # Every spelling of the head, including a SHA that never equals headRefName.
    run = make_runner({head: ["a.py"]})
    assert pr_overlap.main(["--head", head, "--fail-on-overlap"], run) == 1
    out = capsys.readouterr().out
    assert "PR #9" in out and "PR #7" not in out


def test_own_pr_alone_is_not_an_overlap():
    prs = [PRS[0]]
    run = make_runner({"HEAD": ["a.py"]}, prs=prs)
    assert pr_overlap.main(["--head", "HEAD", "--fail-on-overlap"], run) == 0


def test_github_head_ref_excludes_own_pr(monkeypatch):
    # A detached merge-commit checkout matches neither name nor SHA.
    monkeypatch.setenv("GITHUB_HEAD_REF", "feat/mine")
    prs = [PRS[0]]
    run = make_runner({"merge-sha": ["a.py"]}, prs=prs)
    assert pr_overlap.main(["--head", "merge-sha", "--fail-on-overlap"], run) == 0


def test_truncated_list_warns_and_gate_fails_closed(capsys, monkeypatch):
    monkeypatch.setattr(pr_overlap, "PR_LIMIT", 3)
    run = make_runner({"feat/mine": ["b.py"]})
    assert pr_overlap.main([], run) == 0
    assert "limit" in capsys.readouterr().out
    assert pr_overlap.main(["--fail-on-overlap"], run) == 2


def test_empty_diff_short_circuits():
    run = make_runner({})
    assert pr_overlap.main([], run) == 0
    assert not any(c[0] == "gh" for c in run.calls)

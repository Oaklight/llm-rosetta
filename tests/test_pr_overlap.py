"""Tests for scripts/dev/pr_overlap.py with a stubbed command runner."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "pr_overlap", Path(__file__).parent.parent / "scripts" / "dev" / "pr_overlap.py"
)
assert _SPEC is not None and _SPEC.loader is not None
pr_overlap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pr_overlap)

PRS = [
    {
        "number": 7,
        "title": "own",
        "headRefName": "feat/mine",
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
            return "feat/mine\n"
        if argv[:2] == ["git", "diff"]:
            head = argv[-1].split("...")[1]
            return "\n".join(self.diffs.get(head, [])) + "\n"
        if argv[:2] == ["git", "for-each-ref"]:
            return self.branches
        if argv[0] == "gh":
            if self.fail_gh:
                raise RuntimeError("gh is not installed")
            return json.dumps(self.prs)
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
    assert not any(c[:2] == ["git", "rev-parse"] for c in run.calls)


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
    assert pr_overlap.main(["--fail-on-overlap"], run) == 0
    assert "unverified" in capsys.readouterr().out


def test_empty_diff_short_circuits():
    run = make_runner({})
    assert pr_overlap.main([], run) == 0
    assert not any(c[0] == "gh" for c in run.calls)

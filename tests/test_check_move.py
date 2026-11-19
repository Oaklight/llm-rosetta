"""Tests for scripts/dev/check_move.py, run against tiny temporary git repos."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "check_move.py"

_spec = importlib.util.spec_from_file_location("check_move", SCRIPT)
assert _spec is not None and _spec.loader is not None
check_move = importlib.util.module_from_spec(_spec)
sys.modules["check_move"] = check_move
_spec.loader.exec_module(check_move)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))


@pytest.fixture
def make_repo(tmp_path):
    """Return ``build(before, after) -> repo`` committing two snapshots."""

    def build(before: dict[str, str], after: dict[str, str], delete=()):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        _git(repo, "config", "commit.gpgsign", "false")
        _write(repo, before)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        _git(repo, "branch", "base")
        for name in delete:
            (repo / name).unlink()
        _write(repo, after)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "head")
        return repo

    return build


def _run(repo: Path, *extra: str) -> int:
    return check_move.main(["base", "HEAD", "--repo", str(repo), *extra])


def _report(repo: Path, allow=None):
    return check_move.run_check(str(repo), "base", "HEAD", allow or {})


BASE = textwrap.dedent(
    """\
    import functools


    class Foo:
        @classmethod
        def make(cls, x: int) -> int:
            \"\"\"Doc.\"\"\"
            return x + 1  # comment

        def other(self):
            return 2


    def helper(a, b=1):
        return a + b
"""
)


def test_identical_move_passes(make_repo, capsys):
    repo = make_repo(
        {"a.py": BASE},
        {
            "a.py": "",
            "b.py": BASE.replace("x + 1  # comment", "x + 1").replace(
                "def helper(a, b=1):", "def helper(a,\n           b=1):"
            ),
        },
    )
    assert _run(repo) == 0
    report = _report(repo)
    assert len(report.moved_identical) >= 3
    assert not report.moved_with_diff and not report.missing


def test_dropped_decorator_fails(make_repo, capsys):
    repo = make_repo(
        {"a.py": BASE},
        {"a.py": "", "b.py": BASE.replace("    @classmethod\n", "")},
    )
    assert _run(repo) == 1
    out = capsys.readouterr().out
    assert "MOVED-WITH-DIFF" in out and "Foo.make" in out
    assert "-@classmethod" in out


def test_changed_body_fails(make_repo):
    repo = make_repo(
        {"a.py": BASE},
        {"a.py": "", "b.py": BASE.replace("return a + b", "return a - b")},
    )
    assert _run(repo) == 1
    assert [b.qual for b, _ in _report(repo).moved_with_diff] == ["helper"]


def test_deleted_function_fails(make_repo):
    repo = make_repo(
        {"a.py": BASE},
        {"a.py": BASE.replace("def helper(a, b=1):\n    return a + b\n", "")},
    )
    assert _run(repo) == 1
    assert [m.qual for m in _report(repo).missing] == ["helper"]


def test_type_checking_stub_ignored(make_repo):
    after = BASE + textwrap.dedent(
        """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:

        def stub(self) -> int: ...
    """
    )
    repo = make_repo({"a.py": BASE}, {"a.py": after})
    assert _run(repo) == 0
    report = _report(repo)
    assert [h.qual for h in report.type_checking] == ["stub"]
    assert not report.added


def test_hoisted_local_import_ignored(make_repo):
    before = """\
    def f():
        import warnings

        warnings.warn("x")
    """
    after = """\
    import warnings


    def f():
        warnings.warn("x")
    """
    repo = make_repo({"a.py": before}, {"a.py": after})
    assert _run(repo) == 0
    assert len(_report(repo).import_hoisted) == 1


def test_removed_local_import_not_hoisted_fails(make_repo):
    before = """\
    def f():
        import warnings

        warnings.warn("x")
    """
    after = """\
    def f():
        warnings.warn("x")
    """
    repo = make_repo({"a.py": before}, {"a.py": after})
    assert _run(repo) == 1


def test_allow_file_honoured(make_repo, tmp_path, capsys):
    repo = make_repo(
        {"a.py": BASE},
        {"a.py": "", "b.py": BASE.replace("return a + b", "return a - b")},
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("# accepted\nhelper  intentional sign change\nnothing.here  x\n")
    assert _run(repo, "--allow", str(allow)) == 0
    out = capsys.readouterr().out
    assert "allowed: intentional sign change" in out
    assert "nothing.here" in out  # reported as an unused entry


def test_class_split_into_mixins_matched_by_bare_name(make_repo):
    before = """\
    class Big:
        def one(self):
            return 1

        def two(self):
            return 2
    """
    after = """\
    class _OneMixin:
        def one(self):
            return 1


    class _TwoMixin:
        def two(self):
            return 2


    class Big(_OneMixin, _TwoMixin):
        pass
    """
    repo = make_repo({"a.py": before}, {"a.py": after})
    assert _run(repo) == 0
    report = _report(repo)
    assert {(b.qual, h.qual) for b, h in report.moved_identical} == {
        ("Big.one", "_OneMixin.one"),
        ("Big.two", "_TwoMixin.two"),
    }
    assert not report.missing and not report.moved_with_diff


def test_vendor_files_skipped(make_repo):
    repo = make_repo(
        {"pkg/_vendor/v.py": "def f():\n    return 1\n"},
        {"pkg/_vendor/v.py": "def f():\n    return 2\n"},
    )
    assert _run(repo) == 0


def test_markdown_output(make_repo, capsys):
    repo = make_repo(
        {"a.py": BASE},
        {"a.py": "", "b.py": BASE.replace("return a + b", "return a - b")},
    )
    assert _run(repo, "--markdown") == 1
    out = capsys.readouterr().out
    assert "### Pure-move check: FAIL" in out
    assert "| moved-with-diff | 1 |" in out
    assert "```diff" in out


def test_bad_ref_is_usage_error(make_repo):
    repo = make_repo({"a.py": BASE}, {"a.py": BASE + "\nx = 1\n"})
    assert check_move.main(["nope", "HEAD", "--repo", str(repo)]) == 2

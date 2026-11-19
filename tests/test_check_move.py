"""Tests for scripts/dev/check_move.py, run against tiny temporary git repos."""

from __future__ import annotations

import importlib.util
import re
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
    allow.write_text(
        "# accepted\nb.py::helper  intentional sign change\nb.py::nothing.here  x\n"
    )
    assert _run(repo, "--allow", str(allow)) == 0
    out = capsys.readouterr().out
    assert "allowed by b.py::helper: intentional sign change" in out
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


GLUE = "\n\nclass Glue:\n    @staticmethod\n    def value():\n        return 1\n"


def test_split_adding_decorated_helper_passes(make_repo):
    # Moving one function and adding a `@staticmethod` helper: added never fails.
    repo = make_repo(
        {"a.py": "def f():\n    return 1\n"},
        {"a.py": "", "b.py": "def f():\n    return 1\n" + GLUE},
    )
    report = _report(repo)
    assert report.census["staticmethod"] == (0, 0)
    assert [h.qual for h in report.added] == ["Glue", "Glue.value"]
    assert _run(repo) == 0


def test_dropped_decorator_on_moved_def_fails_census(make_repo, capsys):
    after = BASE.replace("    @classmethod\n", "")
    repo = make_repo({"a.py": BASE}, {"a.py": "", "b.py": after})
    report = _report(repo)
    assert report.census["classmethod"] == (1, 0)
    assert _run(repo) == 1
    assert "DECORATOR-CENSUS" in capsys.readouterr().out


def test_census_still_fails_when_the_diff_is_allowed(make_repo, tmp_path, capsys):
    after = BASE.replace("    @classmethod\n", "")
    repo = make_repo({"a.py": BASE}, {"a.py": "", "b.py": after})
    allow = tmp_path / "allow.txt"
    allow.write_text("a.py::Foo.make  dropped classmethod\n")
    assert _run(repo, "--allow", str(allow)) == 1
    out = capsys.readouterr().out
    assert "DECORATOR-CENSUS:" in out and "allowed by a.py::Foo.make" in out
    allow.write_text("a.py::Foo.make  x\ndecorator-census:classmethod  y\n")
    assert _run(repo, "--allow", str(allow)) == 0


def test_decorator_census_mismatch_can_be_allowed(make_repo, tmp_path):
    after = BASE.replace("    @classmethod\n", "    @property\n")
    repo = make_repo({"a.py": BASE}, {"a.py": "", "b.py": after})
    allow = tmp_path / "allow.txt"
    allow.write_text(
        "b.py::Foo.make  became a property\n"
        "decorator-census:classmethod  r\ndecorator-census:property  r\n"
    )
    assert _run(repo, "--allow", str(allow)) == 0


TWO_MODULES = "class C:\n    def m(self):\n        return {n}\n"


def test_allow_is_keyed_on_the_file(make_repo, tmp_path, capsys):
    repo = make_repo(
        {"a.py": TWO_MODULES.format(n=1), "b.py": TWO_MODULES.format(n=1)},
        {"a.py": TWO_MODULES.format(n=2), "b.py": TWO_MODULES.format(n=3)},
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("a.py::C.m  only a changes\n")
    assert _run(repo, "--allow", str(allow)) == 1
    out = capsys.readouterr().out
    assert "MOVED-WITH-DIFF (allowed by a.py::C.m: only a changes): a.py:C.m" in out
    assert "MOVED-WITH-DIFF: b.py:C.m" in out
    allow.write_text("a.py::C.m  x\nb.py::C.m  y\n")
    assert _run(repo, "--allow", str(allow)) == 0


def test_allow_accepts_base_path_of_a_moved_def(make_repo, tmp_path):
    repo = make_repo(
        {"a.py": TWO_MODULES.format(n=1)},
        {"a.py": "", "b.py": TWO_MODULES.format(n=2)},
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("a.py::C.m  base path\n")
    assert _run(repo, "--allow", str(allow)) == 0
    allow.write_text("b.py::C.m  head path\n")
    assert _run(repo, "--allow", str(allow)) == 0
    allow.write_text("c.py::C.m  wrong file\n")
    assert _run(repo, "--allow", str(allow)) == 1


def test_bare_allow_entry_is_rejected(make_repo, tmp_path, capsys):
    repo = make_repo(
        {"a.py": TWO_MODULES.format(n=1)}, {"a.py": TWO_MODULES.format(n=2)}
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("C.m  legacy\n")
    assert _run(repo, "--allow", str(allow)) == 2
    err = capsys.readouterr().err
    assert "no file path" in err and "<path>::C.m" in err
    allow.write_text("undefined-global:C.m  legacy\n")
    assert _run(repo, "--allow", str(allow)) == 2


def test_markdown_shows_allow_key_with_path(make_repo, tmp_path, capsys):
    repo = make_repo(
        {"a.py": TWO_MODULES.format(n=1)}, {"a.py": TWO_MODULES.format(n=2)}
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("a.py::C.m  why\nb.py::C.m  stale\n")
    assert _run(repo, "--allow", str(allow), "--markdown") == 0
    out = capsys.readouterr().out
    assert "(allowed by a.py::C.m: why)" in out
    assert "Unused allow entries: `b.py::C.m`" in out


def test_workflow_does_not_claim_a_default_shell_behaviour():
    text = (SCRIPT.parents[2] / ".github" / "workflows" / "pure-move.yml").read_text()
    assert "shell: bash" in text
    assert "without it" not in text and "turns on `pipefail`" not in text


def test_decorator_census_balanced_move_passes(make_repo):
    repo = make_repo({"a.py": BASE}, {"a.py": "", "b.py": BASE})
    assert _report(repo).census["classmethod"] == (1, 1)
    assert _run(repo) == 0


LIMIT_MODULE = """\
LIMIT = 3


def f():
    return LIMIT
"""


def test_undefined_global_after_move_fails(make_repo, capsys):
    repo = make_repo(
        {"a.py": LIMIT_MODULE},
        {"a.py": "LIMIT = 3\n", "b.py": "def f():\n    return LIMIT\n"},
    )
    assert _run(repo) == 1
    report = _report(repo)
    assert [(h.qual, names) for _, h, names in report.undefined_global] == [
        ("f", ["LIMIT"])
    ]
    assert "UNDEFINED-GLOBAL" in capsys.readouterr().out


def test_undefined_global_resolved_by_import_passes(make_repo):
    repo = make_repo(
        {"a.py": LIMIT_MODULE},
        {
            "a.py": "LIMIT = 3\n",
            "b.py": "from a import LIMIT\n\n\ndef f():\n    return LIMIT\n",
        },
    )
    assert _run(repo) == 0
    assert not _report(repo).undefined_global


def test_undefined_global_allowed(make_repo, tmp_path):
    repo = make_repo(
        {"a.py": LIMIT_MODULE},
        {"a.py": "LIMIT = 3\n", "b.py": "def f():\n    return LIMIT\n"},
    )
    allow = tmp_path / "allow.txt"
    allow.write_text("undefined-global:b.py::f  b.py gets LIMIT injected\n")
    assert _run(repo, "--allow", str(allow)) == 0


def test_lost_logger_in_method_fails(make_repo):
    before = (
        "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
        "class C:\n    def m(self):\n        logger.info('x')\n"
    )
    after = "class C:\n    def m(self):\n        logger.info('x')\n"
    repo = make_repo({"a.py": before}, {"a.py": after})
    assert _run(repo) == 1
    assert [(h.qual, n) for _, h, n in _report(repo).undefined_global] == [
        ("C.m", ["logger"])
    ]


def test_logger_name_literal_ignored(make_repo, capsys):
    before = "import logging\n\nlogger = logging.getLogger(__name__)\n"
    after = 'import logging\n\nlogger = logging.getLogger("pkg.a")\n'
    repo = make_repo({"pkg/a.py": before}, {"pkg/a.py": "", "pkg/b.py": after})
    assert _run(repo) == 0
    report = _report(repo)
    assert len(report.logger_name) == 1 and not report.moved_with_diff
    assert "logger-name (ignored)" in capsys.readouterr().out


def test_logger_name_wrong_literal_fails(make_repo):
    before = "import logging\n\nlogger = logging.getLogger(__name__)\n"
    after = 'import logging\n\nlogger = logging.getLogger("pkg.other")\n'
    repo = make_repo({"pkg/a.py": before}, {"pkg/a.py": "", "pkg/b.py": after})
    assert _run(repo) == 1


def test_census_ignores_type_checking_stub_decorators(make_repo):
    stub = (
        "\n\nfrom typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n\n"
        "    class S:\n"
        "        @staticmethod\n"
        "        def s() -> int: ...\n"
    )
    repo = make_repo({"a.py": BASE}, {"a.py": BASE + stub})
    assert _report(repo).census["staticmethod"] == (0, 0)
    assert _run(repo) == 0


def _property_drop(make_repo):
    base = "class Foo:\n    @property\n    def p(self):\n        return 1\n"
    return make_repo(
        {"a.py": base}, {"a.py": "", "b.py": base.replace("    @property\n", "")}
    )


def test_printed_census_key_round_trips_through_the_allow_file(
    make_repo, tmp_path, capsys
):
    repo = _property_drop(make_repo)
    assert _run(repo) == 1
    out = capsys.readouterr().out
    match = re.search(r"allow entry: (decorator-census:\S+)\)", out)
    assert match
    allow = tmp_path / "allow.txt"
    allow.write_text(f"b.py::Foo.p  r\n{match.group(1)}  r\n")
    assert _run(repo, "--allow", str(allow)) == 0
    assert "matches nothing" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "entry", ["decorator-census:property", "decorator-census:@property"]
)
def test_census_allow_accepts_both_spellings(make_repo, tmp_path, capsys, entry):
    repo = _property_drop(make_repo)
    allow = tmp_path / "allow.txt"
    allow.write_text(f"b.py::Foo.p  r\n{entry}  r\n")
    assert check_move.load_allow(str(allow))["decorator-census:property"] == "r"
    assert _run(repo, "--allow", str(allow)) == 0
    assert "matches nothing" not in capsys.readouterr().out


def test_unknown_census_decorator_is_reported_unused(make_repo, tmp_path, capsys):
    repo = _property_drop(make_repo)
    allow = tmp_path / "allow.txt"
    allow.write_text(
        "b.py::Foo.p  r\ndecorator-census:@property  r\ndecorator-census:@nosuch  r\n"
    )
    assert _run(repo, "--allow", str(allow)) == 0
    assert (
        "allow entry matches nothing failing: decorator-census:nosuch"
        in capsys.readouterr().out
    )

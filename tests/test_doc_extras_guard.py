"""Unit tests for the guard that keeps documented extras declared (#1407).

The docs told users to install a ``[profiling]`` extra no distribution declared
for a long time, and nothing noticed because the docs are orphan branches.  The
guard runs in the ``docs-build`` CI job, where both trees are checked out; these
tests pin its rules against synthetic trees.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_doc_extras.py"


@pytest.fixture(scope="module")
def guard():
    """The script loaded as a module, since ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("check_doc_extras", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator resolves the class's module through sys.modules,
    # so a module loaded by path has to be registered before it runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path: Path) -> Path:
    """Write the two pyprojects and return the repository root."""
    for name, declared in {
        "llm-rosetta": ["all", "conformance", "dev", "gateway"],
        "llm-rosetta-gateway": [],
    }.items():
        path = tmp_path / "packages" / name / "pyproject.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(f'{key} = ["x"]\n' for key in declared)
        path.write_text(
            f'[project]\nname = "{name}"\n\n[project.optional-dependencies]\n{body}'
        )
    return tmp_path


def _docs(repo: Path, name: str, body: str) -> Path:
    """Write one docs tree with one page and return the tree's root."""
    root = repo / name
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "page.md").write_text(textwrap.dedent(body))
    return root


class TestDeclaration:
    """What counts as declared is what the pyproject says."""

    def test_a_declared_extra_passes(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta[dev]\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.checked == 1

    def test_an_undeclared_extra_fails_with_the_declared_set(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta[profiling]\n")
        report = guard.check(repo, [root])
        assert len(report.failures) == 1
        failure = report.failures[0]
        assert failure.distribution == "llm-rosetta"
        assert failure.extra == "profiling"
        assert failure.declared == ("all", "conformance", "dev", "gateway")
        message = str(failure)
        assert "profiling" in message
        assert "llm-rosetta" in message
        assert "['all', 'conformance', 'dev', 'gateway']" in message

    def test_a_distribution_with_no_extras_fails_any_extra(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", 'pip install "llm-rosetta-gateway[profiling]"\n')
        report = guard.check(repo, [root])
        assert len(report.failures) == 1
        assert report.failures[0].declared == ()

    def test_several_extras_on_one_line_are_each_checked(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta[dev,profiling]\n")
        report = guard.check(repo, [root])
        assert [failure.extra for failure in report.failures] == ["profiling"]


class TestDistributionSpelling:
    """A name is compared the way pip compares it (PEP 503)."""

    @pytest.mark.parametrize(
        "spelling", ["llm_rosetta", "LLM-Rosetta", "llm.rosetta", "Llm__Rosetta"]
    )
    def test_an_undeclared_extra_fails_under_any_spelling(
        self, guard, tmp_path, spelling
    ):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", f"pip install {spelling}[bogus]\n")
        report = guard.check(repo, [root])
        assert len(report.failures) == 1
        assert report.failures[0].distribution == "llm-rosetta"
        assert report.failures[0].extra == "bogus"

    def test_a_declared_extra_passes_under_another_spelling(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm_rosetta[dev]\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.unattributed == []


class TestLineShapes:
    """Every shape an install line takes in these docs is read the same way."""

    def test_a_path_form_resolves_to_its_distribution(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", 'pip install -e "packages/llm-rosetta[dev]"\n')
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.checked == 1

    def test_a_path_form_is_still_checked(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", 'pip install -e "packages/llm-rosetta[nope]"\n')
        report = guard.check(repo, [root])
        assert [failure.extra for failure in report.failures] == ["nope"]

    def test_uv_pip_install_is_scanned(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "uv pip install llm-rosetta[nope]\n")
        report = guard.check(repo, [root])
        assert [failure.extra for failure in report.failures] == ["nope"]

    def test_pip3_is_scanned_like_pip(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip3 install llm-rosetta[nope]\n")
        report = guard.check(repo, [root])
        assert [failure.extra for failure in report.failures] == ["nope"]

    def test_a_shell_escaped_extra_is_reported_not_skipped(self, guard, tmp_path):
        """The scanner cannot read ``\\[bogus\\]``; silence is what it must not do."""
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta\\[bogus\\]\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert len(report.unattributed) == 1
        assert "cannot read an extra" in report.unattributed[0]

    def test_an_escaped_extra_is_reported_beside_a_readable_one(self, guard, tmp_path):
        """A passing sibling must not hide it: the two are classified apart."""
        repo = _tree(tmp_path)
        root = _docs(
            repo,
            "docs_en",
            'pip install llm-rosetta[dev] "llm-rosetta\\[bogus\\]"\n',
        )
        report = guard.check(repo, [root])
        assert report.failures == []
        assert len(report.unattributed) == 1
        assert "cannot read an extra" in report.unattributed[0]

    def test_an_escaped_bracket_of_prose_is_not_reported(self, guard, tmp_path):
        """``see\\[1\\]`` names no distribution of ours, so it is not a finding."""
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta[dev] # see\\[1\\]\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.unattributed == []

    def test_pip_with_a_version_suffix_is_scanned(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip3.11 install llm-rosetta[nope]\n")
        report = guard.check(repo, [root])
        assert [failure.extra for failure in report.failures] == ["nope"]

    def test_only_what_follows_the_command_is_read(self, guard, tmp_path):
        """``foo[bar]`` before the command is prose, not what gets installed."""
        repo = _tree(tmp_path)
        root = _docs(
            repo, "docs_en", "see foo[bar]; then pip install llm-rosetta[dev]\n"
        )
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.unattributed == []

    def test_an_install_line_without_an_extra_is_not_a_finding(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-rosetta\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert report.checked == 1

    def test_a_third_party_distribution_is_reported_not_failed(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "pip install llm-comply[rich]\n")
        report = guard.check(repo, [root])
        assert report.failures == []
        assert len(report.unattributed) == 1
        assert "llm-comply" in report.unattributed[0]

    def test_the_line_number_is_reported(self, guard, tmp_path):
        repo = _tree(tmp_path)
        root = _docs(repo, "docs_en", "# title\n\npip install llm-rosetta[nope]\n")
        report = guard.check(repo, [root])
        assert report.failures[0].line == 3


class TestMain:
    """The command line turns a finding into a failing job."""

    def test_exit_zero_when_everything_is_declared(self, guard, tmp_path, capsys):
        repo = _tree(tmp_path)
        _docs(repo, "docs_en", "pip install llm-rosetta[dev]\n")
        assert guard.main(["--repo-root", str(repo), str(repo / "docs_en")]) == 0

    def test_exit_one_and_a_named_finding(self, guard, tmp_path, capsys):
        repo = _tree(tmp_path)
        _docs(repo, "docs_en", "pip install llm-rosetta[profiling]\n")
        assert guard.main(["--repo-root", str(repo), str(repo / "docs_en")]) == 1
        err = capsys.readouterr().err
        assert "docs name extra 'profiling' of distribution 'llm-rosetta'" in err

    def test_a_missing_docs_root_fails_instead_of_passing_vacuously(
        self, guard, tmp_path, capsys
    ):
        repo = _tree(tmp_path)
        assert guard.main(["--repo-root", str(repo), str(repo / "docs_en")]) == 1
        assert "not found" in capsys.readouterr().err

    def test_the_default_roots_are_the_two_language_trees(
        self, guard, tmp_path, monkeypatch, capsys
    ):
        """CI calls the script with no positional arguments."""
        repo = _tree(tmp_path)
        _docs(repo, "docs_en", "pip install llm-rosetta[nope-en]\n")
        _docs(repo, "docs_zh", "pip install llm-rosetta[nope-zh]\n")
        monkeypatch.chdir(repo)
        assert guard.main(["--repo-root", str(repo)]) == 1
        err = capsys.readouterr().err
        assert "nope-en" in err
        assert "nope-zh" in err

"""Fail when the docs name an extra no distribution declares.

The docs are where a user learns an optional dependency exists, and they live
on two orphan branches that nothing type-checks against the packaging.  That is
how ``pip install llm-rosetta[profiling]`` survived long after there was no
``profiling`` extra to install (#1407): the line reads like an instruction and
silently installs nothing.

This reads every ``pip install`` line under the docs roots, takes the
``<distribution>[<extra>]`` forms in it, and compares them with the extras the
two pyprojects declare.  A line naming an undeclared extra fails the run.  A
line the scanner cannot attribute to a distribution we publish — a third-party
distribution, a shell-escaped ``\\[extra\\]`` it cannot read — is printed
rather than skipped: a typo of one of ours would otherwise hide among them.

The asymmetry is deliberate, not an omission.  A third-party distribution
written readably is listed as unattributed — visible, but nothing this check
can verify — while the shell-escaped spelling of the same thing is not reported
at all: the escaping makes it impossible to tell a distribution from literal
brackets in prose, so warning there would trade a quiet noise for no finding.
This check passes judgement only on tokens it can attribute to a distribution
this repository publishes.

The scope is pip: ``uv pip install`` and ``python -m pip install`` are the same
command line and are read, while ``uv add``, ``pipx install`` and
``poetry add`` are not.  Distribution names are compared after PEP 503
normalization, so ``llm_rosetta[extra]`` and ``LLM-Rosetta[extra]`` are checked
against ``llm-rosetta`` instead of sliding past as unknown names.

It has to run where both trees are checked out — the docs branches and this
repository.  The ``docs-build`` job is that place, which is also why this is
not a test under ``tests/``: with no ``docs_en``/``docs_zh`` in the tree, such
a test could only skip.

Usage:
    python scripts/dev/check_doc_extras.py [--repo-root .] [docs_root ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

#: Our own distributions and where each declares its extras.  A ``pip install``
#: line naming one of these is checked; naming any other is reported as
#: unattributable.
DISTRIBUTIONS: dict[str, Path] = {
    "llm-rosetta": Path("packages/llm-rosetta/pyproject.toml"),
    "llm-rosetta-gateway": Path("packages/llm-rosetta-gateway/pyproject.toml"),
}

#: A shell line that installs something.  ``uv pip install``, a
#: ``python -m pip install`` and a ``pip3``/``pip3.11`` all match, because the
#: command word is the same.
_INSTALL = re.compile(r"\bpip(?:3(?:\.\d+)?)?\s+install\b")

#: A distribution-shaped token followed by a *shell-escaped* bracket.  The
#: check cannot read what is inside, so one naming a distribution of ours is
#: reported, while a bracket of prose (``see\\[1\\]``) or one naming a project we
#: do not publish (``.[all]``, a third party) is not an install of ours.
_ESCAPED_EXTRA = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\\\[")

#: ``<distribution>[<extra>, <extra>]``.  The name pattern starts at a word
#: character, so a path form — ``packages/llm-rosetta[dev]`` — yields the
#: distribution ``llm-rosetta``, which is exactly the project it names.
_DIST_EXTRAS = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\[([^\[\]]+)\]")


@dataclass(frozen=True)
class UndeclaredExtra:
    """One documented extra a distribution does not declare.

    Attributes:
        path: The docs file the line is in.
        line: 1-based line number.
        distribution: The distribution the line installs from.
        extra: The extra it names.
        declared: Every extra that distribution does declare.
    """

    path: Path
    line: int
    distribution: str
    extra: str
    declared: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}: docs name extra {self.extra!r} of "
            f"distribution {self.distribution!r}, which declares "
            f"{list(self.declared)}"
        )


@dataclass
class Report:
    """What one run found.

    Attributes:
        failures: Documented extras no distribution declares.
        unattributed: Install lines that name an extra but not a distribution
            this script knows, printed so the third-party ones stay visible.
        checked: Install lines examined.
    """

    failures: list[UndeclaredExtra] = field(default_factory=list)
    unattributed: list[str] = field(default_factory=list)
    checked: int = 0


def canonical_name(name: str) -> str:
    """Return *name* under PEP 503 normalization.

    A distribution is ``llm-rosetta`` on PyPI and ``llm_rosetta`` in an import;
    ``pip`` treats the two spellings as the same project, and so must this
    check, or an undeclared extra hides behind the spelling.

    Args:
        name: A distribution name as written.

    Returns:
        The normalized name: lower case, ``_`` and ``.`` folded into ``-``, and
        runs of ``-`` collapsed.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_extras(pyproject: Path) -> tuple[str, ...]:
    """Return the extras *pyproject* declares, sorted.

    Args:
        pyproject: Path to a distribution's ``pyproject.toml``.

    Returns:
        The extra names under ``[project.optional-dependencies]``, or an empty
        tuple when the distribution declares none.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    return tuple(sorted(extras))


def extras_in(line: str) -> list[tuple[str, list[str]]]:
    """Return every ``(distribution, extras)`` pair named in *line*.

    Args:
        line: One line of documentation.

    Returns:
        The pairs, in the order they appear; each extra is stripped of
        whitespace and quotes.
    """
    pairs: list[tuple[str, list[str]]] = []
    for name, body in _DIST_EXTRAS.findall(line):
        extras = [part.strip().strip("\"'") for part in body.split(",")]
        pairs.append((name, [extra for extra in extras if extra]))
    return pairs


def check(repo_root: Path, docs_roots: list[Path]) -> Report:
    """Compare every documented extra with the packaging declares.

    Args:
        repo_root: Repository root, where the pyprojects live.
        docs_roots: The docs trees to scan, each searched recursively for
            ``*.md``.

    Returns:
        The report; ``failures`` is what makes the run fail.
    """
    declared = {
        canonical_name(name): declared_extras(repo_root / pyproject)
        for name, pyproject in DISTRIBUTIONS.items()
    }
    report = Report()
    for root in docs_roots:
        for path in sorted(root.rglob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                install = _INSTALL.search(line)
                if install is None:
                    continue
                report.checked += 1
                # Everything after the command word: what the line installs,
                # rather than whatever else shares the line with it.
                tail = line[install.end() :]
                for distribution, extras in extras_in(tail):
                    name = canonical_name(distribution)
                    known = declared.get(name)
                    if known is None:
                        report.unattributed.append(
                            f"{path}:{number}: cannot attribute {distribution!r} "
                            "to a distribution this repository publishes"
                        )
                        continue
                    for extra in extras:
                        if extra not in known:
                            report.failures.append(
                                UndeclaredExtra(
                                    path=path,
                                    line=number,
                                    distribution=name,
                                    extra=extra,
                                    declared=known,
                                )
                            )
                # Classified separately from the readable pairs: a line may
                # carry both, and the escaped one must not vanish behind a
                # readable sibling that happened to pass.
                for distribution in _ESCAPED_EXTRA.findall(tail):
                    if canonical_name(distribution) in declared:
                        report.unattributed.append(
                            f"{path}:{number}: cannot read an extra from "
                            f"{line.strip()!r}"
                        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the check and print its findings.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when every documented extra is declared, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root holding the pyprojects (default: cwd)",
    )
    parser.add_argument(
        "docs_roots",
        nargs="*",
        default=["docs_en", "docs_zh"],
        help="Docs trees to scan (default: docs_en docs_zh)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    docs_roots = [Path(root).resolve() for root in args.docs_roots]

    # A root with no docs in it means the caller is not where the docs are, and
    # every finding below would be vacuously absent.
    missing = [str(root) for root in docs_roots if not root.is_dir()]
    if missing:
        print(
            "check_doc_extras: docs root(s) not found: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "  run this where docs_en/ and docs_zh/ are checked out "
            "(the docs-build job does)",
            file=sys.stderr,
        )
        return 1

    report = check(repo_root, docs_roots)

    for note in report.unattributed:
        print(f"warning: {note}", file=sys.stderr)

    if report.failures:
        for failure in report.failures:
            print(failure, file=sys.stderr)
        print(
            f"check_doc_extras: {len(report.failures)} documented extra(s) "
            "are not declared; declare them in the distribution's pyproject or "
            "correct the docs",
            file=sys.stderr,
        )
        return 1

    print(
        f"check_doc_extras: {report.checked} install line(s) checked; "
        f"{len(report.unattributed)} unattributed",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

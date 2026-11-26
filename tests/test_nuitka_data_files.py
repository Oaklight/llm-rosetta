"""The Nuitka binary ships every data file the code reads (#1703).

A compiled binary contains the Python modules it follows, and nothing else:
a JSON schema, a YAML shim or an admin asset that the Makefile does not list is
simply absent at run time.  The first binary built from the extension work
loaded no extension at all because two schema files were missing.  This test
reads the Makefile's two flag lists (``build-binary`` and ``build-binary-musl``)
and fails when a runtime data file in the source tree is covered by neither an
``--include-data-files`` entry nor an ``--include-data-dir`` entry above it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
SOURCES = (
    ROOT / "packages" / "llm-rosetta" / "src",
    ROOT / "packages" / "llm-rosetta-gateway" / "src",
)

#: Suffixes of files the code opens at run time.
RUNTIME_SUFFIXES = {".json", ".yaml", ".html", ".css", ".js", ".tmpl"}

#: Data files that no code path of the binary reads, with the reason.
NOT_SHIPPED = {
    # ``scripts/dev/new_shim.py`` renders the built-in shim template; the binary
    # has no command that does (``ext new`` uses ``_ext_template``).
    "llm_rosetta/devtools/_shim_template",
}


def _flag_blocks() -> dict[str, str]:
    lines = MAKEFILE.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("NUITKA_FLAGS"))
    end = start
    while lines[end].rstrip().endswith("\\"):
        end += 1
    native = "\n".join(lines[start : end + 1])
    musl_start = next(
        i for i, line in enumerate(lines) if line.startswith("build-binary-musl:")
    )
    musl_end = next(
        i
        for i in range(musl_start + 1, len(lines))
        if re.match(r"^[a-z][a-z-]*:", lines[i])
    )
    return {
        "build-binary": native,
        "build-binary-musl": "\n".join(lines[musl_start:musl_end]),
    }


def _listed(block: str) -> tuple[set[str], set[str]]:
    files = set(re.findall(r"--include-data-files=([^=\s]+)=", block))
    dirs = set(re.findall(r"--include-data-dir=([^=\s]+)=", block))
    return files, dirs


def _runtime_data_files() -> list[Path]:
    found: list[Path] = []
    for src in SOURCES:
        for path in src.rglob("*"):
            if (
                path.is_file()
                and path.suffix in RUNTIME_SUFFIXES
                and "__pycache__" not in path.parts
                and "_vendor" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
                and "build" not in path.relative_to(src).parts
            ):
                found.append(path)
    return found


def _relative(path: Path) -> str:
    for src in SOURCES:
        if src in path.parents:
            return path.relative_to(src).as_posix()
    raise AssertionError(path)


@pytest.mark.parametrize("target", ["build-binary", "build-binary-musl"])
def test_every_runtime_data_file_is_listed(target):
    files, dirs = _listed(_flag_blocks()[target])
    uncovered = []
    for path in _runtime_data_files():
        relative = _relative(path)
        if any(relative.startswith(skip + "/") for skip in NOT_SHIPPED):
            continue
        as_listed = path.relative_to(ROOT).as_posix()
        in_dir = any(as_listed.startswith(d.rstrip("/") + "/") for d in dirs)
        if as_listed not in files and not in_dir:
            uncovered.append(as_listed)
    assert not uncovered, (
        f"`make {target}` does not ship these data files; add an "
        "--include-data-files or --include-data-dir line:\n  "
        + "\n  ".join(sorted(uncovered))
    )


def test_the_two_flag_lists_ship_the_same_data():
    blocks = _flag_blocks()
    assert _listed(blocks["build-binary"]) == _listed(blocks["build-binary-musl"])


def _editable_installs(text: str) -> list[tuple[str, ...]]:
    """The ``-e <path>`` arguments of every ``pip install`` line in *text*."""
    found = []
    for line in text.splitlines():
        if "pip install" in line and " -e " in f" {line} ":
            found.append(tuple(re.findall(r"-e\s+(\"[^\"]*\"|\S+)", line)))
    return found


def test_the_musl_recipe_installs_the_packages_the_workflow_installs():
    """The repository root is not a package since the split (#1158).

    ``pip install -e .`` at the root fails with "Multiple top-level packages
    discovered", so the Alpine container installs the two distributions, as the
    native build in ``.github/workflows/nuitka.yml`` does.
    """
    path = ROOT / ".github" / "workflows" / "nuitka.yml"
    if not path.is_file():
        pytest.skip("the workflow files are not part of this checkout")
    workflow = path.read_text(encoding="utf-8")
    wanted = ("packages/llm-rosetta", "packages/llm-rosetta-gateway")
    assert wanted in _editable_installs(workflow)
    assert _editable_installs(_flag_blocks()["build-binary-musl"]) == [wanted]


def test_the_musl_recipe_pins_nuitka():
    """The Alpine recipe installs one Nuitka release, not whatever is newest.

    Nuitka 4.4 fails on the musl Python (``'ldd' failed with exit code 127`` for
    ``_asyncio.cpython-312-x86_64-linux-musl.so``), and the unpinned install
    picked it up the day it came out.  4.2.1 built the musl binary, and the
    binary ran every check, on 2026-11-26 (it is also what the last green
    release run used, on 2026-09-05).
    """
    block = _flag_blocks()["build-binary-musl"]
    pins = re.findall(r"nuitka\[onefile\]([^\s\"']*)", block)
    assert pins and all(re.fullmatch(r"==\d+(\.\d+)+", pin) for pin in pins), pins

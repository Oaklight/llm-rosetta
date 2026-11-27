"""The image carries no Node.js: ``ext install-cli`` puts it in the volume (#1820).

``llm-rosetta ext install-cli`` installs Node.js and the agent CLIs into
``/config/tools``, so they follow the ``/config`` volume and not the image.
What the image keeps is the one library the musl build of Node.js is linked
against.  These checks read the build files; no Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (".github/workflows/docker.yml", ".github/workflows/test-release.yml")


def _read(path: str) -> str:
    file = ROOT / path
    if not file.is_file():
        pytest.skip(f"{path} is not part of this checkout")
    return file.read_text(encoding="utf-8")


def _apk_packages(dockerfile: str) -> set[str]:
    """Every package a ``RUN apk add`` line of the file names."""
    packages: set[str] = set()
    for line in dockerfile.replace("\\\n", " ").splitlines():
        if "apk add" in line:
            packages |= {w for w in line.split("apk add", 1)[1].split() if w[0] != "-"}
    return packages


def test_the_dockerfile_adds_no_node_or_npm():
    text = _read("docker/Dockerfile")
    assert not {"nodejs", "npm"} & _apk_packages(text)
    assert "WITH_AGENTS" not in text


def test_the_dockerfile_carries_the_library_the_musl_node_needs():
    # Measured: node of unofficial-builds exits 127 on this base without it
    # ("Error loading shared library libstdc++.so.6"); the package brings libgcc.
    assert "libstdc++" in _apk_packages(_read("docker/Dockerfile"))


@pytest.mark.parametrize("path", WORKFLOWS)
def test_no_workflow_publishes_an_agents_tag_or_asks_for_node(path):
    text = _read(path)
    assert "agents" not in text
    assert "WITH_AGENTS" not in text


def test_make_has_no_agents_target():
    text = _read("Makefile")
    assert "agents" not in text
    assert "WITH_AGENTS" not in text
    assert not re.search(r"^build-docker-agents:", text, re.M)

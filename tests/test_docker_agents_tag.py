"""The ``-agents`` image tag: the Python image plus Node.js and npm (#1820).

``llm-rosetta ext install-cli`` installs an agent CLI that is distributed through
npm, so the image that runs it needs ``node`` and ``npm``.  The default image
does not carry them.  These checks read the build files; no Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    file = ROOT / path
    if not file.is_file():
        pytest.skip(f"{path} is not part of this checkout")
    return file.read_text(encoding="utf-8")


def test_the_dockerfile_adds_node_only_when_asked():
    text = _read("docker/Dockerfile")
    assert re.search(r"^ARG WITH_AGENTS=0$", text, re.M), "the default must be off"
    guarded = re.search(
        r'if \[ "\$WITH_AGENTS" = "1" \]; then apk add --no-cache nodejs npm; fi', text
    )
    assert guarded, "nodejs and npm are installed only behind WITH_AGENTS"
    assert text.count("nodejs") == 1, "nothing else puts Node.js in the image"


def test_the_release_workflow_publishes_the_agents_tags():
    text = _read(".github/workflows/docker.yml")
    assert "WITH_AGENTS=1" in text
    assert "-agents" in text and "latest-agents" in text


def test_the_test_release_workflow_publishes_the_dev_agents_tags():
    text = _read(".github/workflows/test-release.yml")
    assert "WITH_AGENTS=1" in text
    assert "-dev-agents" in text and "dev-agents" in text


def test_only_the_agents_build_asks_for_node():
    """A build with ``WITH_AGENTS`` pushes agents tags; no other build asks."""
    for path in (".github/workflows/docker.yml", ".github/workflows/test-release.yml"):
        builds = _read(path).split("docker/build-push-action@v6")[1:]
        asking = [b for b in builds if "WITH_AGENTS=1" in b]
        assert len(asking) == 1, path
        tags = re.search(r"tags: \|\n(.*?)\n\s+cache-from", asking[0], re.S)
        assert tags is not None, path
        assert all("agents" in t for t in tags.group(1).splitlines()), path
        for build in builds:
            if "WITH_AGENTS=1" not in build and "docker/Dockerfile\n" in build:
                assert "agents" not in build.split("cache-from")[0], path


def test_make_builds_the_agents_image():
    text = _read("Makefile")
    assert re.search(r"^build-docker-agents:", text, re.M)
    assert "WITH_AGENTS=1" in text

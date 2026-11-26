"""The image entrypoints write a config template only when the server starts.

``docker/entrypoint.sh`` (the Python image) and ``docker/entrypoint-binary.sh``
(the binary image) used to generate ``config.jsonc`` and print "Edit ... restart
the container" before *any* command, including ``llm-rosetta ext install`` and
``llm-rosetta gateway --help``.  The template belongs to the default start only:
the default CMD, an argument list of options, and an explicit ``serve``.  The
tests run the scripts under ``sh`` with stub executables on ``PATH``; no Docker.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1] / "docker"
SCRIPTS = ("entrypoint.sh", "entrypoint-binary.sh")

_STUB = """#!/bin/sh
echo "$0 $*" >> "$STUB_LOG"
# The real ``init`` writes the template; a stub does the same.
case " $* " in
  *" init "*) mkdir -p "$CONFIG_DIR"; : > "$CONFIG_DIR/config.jsonc" ;;
esac
"""

TEMPLATE_MESSAGE = "No config.jsonc found"

#: argv, where the server starts (and a missing config gets a template).
SERVES = [
    ["llm-rosetta-gateway", "--config", "/config/config.jsonc", "--host", "0.0.0.0"],
    ["--port", "9000"],
    ["--config", "init.jsonc", "--host", "0.0.0.0"],
    ["llm-rosetta-gateway"],
    ["llm-rosetta-gateway", "serve"],
    ["llm-rosetta-gateway", "--port", "9000", "serve"],
    ["llm-rosetta-gateway", "gateway", "serve"],
    ["llm-rosetta", "serve"],
    ["llm-rosetta", "serve", "--config", "x.jsonc"],
    ["llm-rosetta", "gateway", "serve"],
]

#: argv, a one-off command that must not write a template.
ONE_OFFS = [
    ["llm-rosetta", "gateway", "--help"],
    ["llm-rosetta", "gateway", "init"],
    ["llm-rosetta", "gateway", "usage", "--db", "/data"],
    ["llm-rosetta", "ext", "list"],
    ["llm-rosetta", "ext", "install", "acme"],
    ["llm-rosetta"],
    ["llm-rosetta", "gateway"],
    ["llm-rosetta-gateway", "init"],
    ["llm-rosetta-gateway", "--config", "serve.jsonc", "init"],
    ["llm-rosetta-gateway", "ext", "list"],
    ["llm-rosetta-gateway", "gateway", "init"],
    ["sh", "-c", "true"],
]


def _run(script: str, argv: list[str], tmp_path: Path, *, config: bool = False):
    if not (DOCKER / script).is_file():
        pytest.skip("the docker/ directory is not part of this checkout")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("llm-rosetta-gateway", "llm-rosetta"):
        stub = bin_dir / name
        stub.write_text(_STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    if config:
        (config_dir / "config.jsonc").write_text("{}")
    log = tmp_path / "stub.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CONFIG_DIR": str(config_dir),
        "HOME": str(tmp_path / "home"),
        "STUB_LOG": str(log),
    }
    result = subprocess.run(
        ["sh", str(DOCKER / script), *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    return result, (log.read_text() if log.exists() else "")


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("argv", SERVES, ids=" ".join)
def test_the_server_start_writes_a_template_when_there_is_no_config(
    script, argv, tmp_path
):
    result, _log = _run(script, argv, tmp_path)

    assert TEMPLATE_MESSAGE in result.stdout, result.stderr
    assert "restart the container" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("argv", SERVES, ids=" ".join)
def test_the_server_start_leaves_an_existing_config_alone(script, argv, tmp_path):
    result, _log = _run(script, argv, tmp_path, config=True)

    assert TEMPLATE_MESSAGE not in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("argv", ONE_OFFS, ids=" ".join)
def test_a_one_off_command_writes_no_template_and_prints_no_restart_note(
    script, argv, tmp_path
):
    result, log = _run(script, argv, tmp_path)

    assert TEMPLATE_MESSAGE not in result.stdout
    assert "restart the container" not in result.stdout
    if argv[0] != "sh":
        # The command is the only thing that ran: no ``init`` before it.  (A
        # command that is itself ``init`` writes its own config, as it should.)
        assert log.count("\n") == 1, log


def test_the_python_entrypoint_runs_the_given_command_after_the_template(tmp_path):
    result, log = _run("entrypoint.sh", ["--port", "9000"], tmp_path)

    lines = log.splitlines()
    assert lines[0].endswith(f"--config {tmp_path / 'config' / 'config.jsonc'} init")
    assert lines[1].endswith("llm-rosetta-gateway --port 9000")
    assert result.returncode == 0

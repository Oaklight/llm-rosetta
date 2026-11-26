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
echo "EXT=${LLM_ROSETTA_EXTENSIONS_DIR:-} HOME=$HOME" >> "$STUB_ENV"
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


def _run(
    script: str,
    argv: list[str],
    tmp_path: Path,
    *,
    config: bool = False,
    env: dict[str, str | None] | None = None,
    uid: int | None = None,
):
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
    if uid is not None:
        # A uid with no passwd entry, as Compose's ``user:`` selects.
        fake_id = bin_dir / "id"
        fake_id.write_text(f"#!/bin/sh\necho {uid}\n")
        fake_id.chmod(fake_id.stat().st_mode | stat.S_IXUSR)
    merged = {
        **{k: v for k, v in os.environ.items() if k != "LLM_ROSETTA_EXTENSIONS_DIR"},
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CONFIG_DIR": str(config_dir),
        "HOME": str(tmp_path / "home"),
        "STUB_LOG": str(log),
        "STUB_ENV": str(tmp_path / "stub.env"),
        **(env or {}),
    }
    # ``None`` in *env* removes a variable the test runner may have set.
    merged = {key: value for key, value in merged.items() if value is not None}
    result = subprocess.run(
        ["sh", str(DOCKER / script), *argv],
        capture_output=True,
        text=True,
        env=merged,
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


def _stub_env(tmp_path: Path) -> dict[str, str]:
    """What the command saw: ``EXT=<dir> HOME=<dir>`` of its first line."""
    line = (tmp_path / "stub.env").read_text().splitlines()[-1]
    return dict(part.split("=", 1) for part in line.split(" "))


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize(
    "argv", [SERVES[0], ["llm-rosetta", "ext", "list"]], ids=["serve", "one-off"]
)
def test_the_extensions_directory_defaults_to_the_config_volume(script, argv, tmp_path):
    result, _log = _run(
        script, argv, tmp_path, env={"LLM_ROSETTA_EXTENSIONS_DIR": None}
    )

    assert result.returncode == 0, result.stderr
    wanted = tmp_path / "config" / "extensions"
    assert _stub_env(tmp_path)["EXT"] == str(wanted)
    assert wanted.is_dir()


@pytest.mark.parametrize("script", SCRIPTS)
def test_an_extensions_directory_the_operator_set_is_kept(script, tmp_path):
    mine = tmp_path / "mine"

    result, _log = _run(
        script,
        ["llm-rosetta", "ext", "list"],
        tmp_path,
        env={"LLM_ROSETTA_EXTENSIONS_DIR": str(mine)},
    )

    assert result.returncode == 0, result.stderr
    assert _stub_env(tmp_path)["EXT"] == str(mine)
    assert not (tmp_path / "config" / "extensions").exists()


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_config_volume_that_cannot_hold_the_directory_stops_with_a_clear_error(
    script, tmp_path
):
    """The volume is a file here; a read-only volume fails the same check."""
    blocker = tmp_path / "volume"
    blocker.write_text("x")

    result, log = _run(
        script,
        ["llm-rosetta", "ext", "list"],
        tmp_path,
        env={"CONFIG_DIR": str(blocker), "LLM_ROSETTA_EXTENSIONS_DIR": None},
    )

    assert result.returncode == 1
    assert f"cannot use {blocker}/extensions" in result.stderr
    assert "LLM_ROSETTA_EXTENSIONS_DIR" in result.stderr
    assert log == "", "the command did not run"


def test_a_uid_with_no_passwd_entry_keeps_home_and_extensions_in_the_volume(tmp_path):
    result, _log = _run(
        "entrypoint.sh",
        ["llm-rosetta", "ext", "list"],
        tmp_path,
        env={"HOME": "/", "LLM_ROSETTA_EXTENSIONS_DIR": None},
        uid=4242,
    )

    assert result.returncode == 0, result.stderr
    seen = _stub_env(tmp_path)
    assert seen["HOME"] == str(tmp_path / "config" / "home")
    assert seen["EXT"] == str(tmp_path / "config" / "extensions")


def test_the_binary_image_with_an_unusable_home_still_gets_a_directory(tmp_path):
    """It has no HOME handling: with HOME=/ the core would refuse to choose one."""
    result, _log = _run(
        "entrypoint-binary.sh",
        ["llm-rosetta", "ext", "install", "acme"],
        tmp_path,
        env={"HOME": "/", "LLM_ROSETTA_EXTENSIONS_DIR": None},
    )

    assert result.returncode == 0, result.stderr
    assert _stub_env(tmp_path)["EXT"] == str(tmp_path / "config" / "extensions")

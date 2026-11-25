"""The shared CLI sign-in driver used by the vendor-managed exchanges.

Every test runs a real child process: the point of the helper is that it
survives a CLI that prints a URL, waits on stdin, and then either accepts or
rejects the pasted value.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from llm_rosetta.contracts.cli_login import (
    CliLoginError,
    begin_cli_login,
    cancel_cli_login,
    complete_cli_login,
)

FAKE_CLI = """
import json, os, sys

args = sys.argv[1:]
state = os.environ["FAKE_STATE"]

if "login" in args:
    print("Opening browser to sign in…")
    print("\\x1b]8;;https://accounts.example/oauth2/auth?state=abc\\x1b\\\\"
          "https://accounts.example/oauth2/auth?state=abc\\x1b]8;;\\x1b\\\\")
    print("Paste code here if prompted > ", end="", flush=True)
    line = sys.stdin.readline().strip()
    if line != "good-code":
        print("the code was refused: " + line)
        sys.exit(3)
    with open(state, "w", encoding="utf-8") as handle:
        handle.write('{"loggedIn": true, "authMethod": "claude.ai"}')
    sys.exit(0)

if "logout" in args:
    try:
        os.unlink(state)
    except FileNotFoundError:
        pass
    sys.exit(0)
"""


@pytest.fixture()
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "signed-in.json"))
    return [sys.executable, str(script)]


def test_begin_returns_the_url_and_not_the_escape_wrapper(fake_cli):
    key = "test-claude"
    try:
        started = begin_cli_login(key, fake_cli + ["login"])
    finally:
        cancel_cli_login(key)

    assert (
        started["authorization_url"] == "https://accounts.example/oauth2/auth?state=abc"
    )
    assert started["expires_at"] > 0
    assert "Paste" in started["paste_label"]


def test_complete_writes_the_code_and_the_cli_exits_zero(fake_cli):
    key = "test-claude-complete"
    begin_cli_login(key, fake_cli + ["login"])

    complete_cli_login(key, "good-code")

    import os

    assert Path(os.environ["FAKE_STATE"]).exists()


def test_a_refused_code_is_reported_and_redacted(fake_cli):
    key = "test-claude-refused"
    begin_cli_login(key, fake_cli + ["login"])

    with pytest.raises(CliLoginError) as raised:
        complete_cli_login(key, "bad-code")

    assert "exited with 3" in str(raised.value)
    assert "bad-code" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


def test_token_shaped_cli_output_is_redacted():
    from llm_rosetta.contracts.cli_login import _redact

    line = (
        "access ya29.a0AfH6SMBexample refresh 1//0gExample "
        "key sk-ant-example12345 google AIzaSyExample0123456789abcdef "
        "header Bearer eyJhbGciOiJIUzI1NiJ9.payload"
    )

    redacted = _redact(line, "")

    for secret in ("ya29.", "1//0g", "sk-ant", "AIzaSy", "eyJhbGci"):
        assert secret not in redacted
    assert redacted.count("<redacted>") == 5


def test_completing_without_a_pending_sign_in_is_an_error():
    with pytest.raises(CliLoginError, match="no sign-in is pending"):
        complete_cli_login("never-started", "code")


def test_an_empty_code_is_refused_before_the_process_is_touched(fake_cli):
    key = "test-claude-empty"
    begin_cli_login(key, fake_cli + ["login"])
    try:
        with pytest.raises(CliLoginError, match="code or callback URL"):
            complete_cli_login(key, "   \n")
    finally:
        cancel_cli_login(key)


def test_the_error_says_what_a_cli_that_exits_early_printed(tmp_path: Path):
    script = tmp_path / "dies.py"
    script.write_text(
        "import sys\nprint('no credentials file at ~/.tool/auth.json')\nsys.exit(2)\n",
        encoding="utf-8",
    )

    with pytest.raises(CliLoginError) as raised:
        begin_cli_login("dies-early", [sys.executable, str(script)])

    message = str(raised.value)
    assert "no authorization URL" in message
    assert "no credentials file" in message


def test_starting_again_replaces_the_previous_process(fake_cli):
    key = "test-claude-supersede"
    first = begin_cli_login(key, fake_cli + ["login"])
    second = begin_cli_login(key, fake_cli + ["login"])
    try:
        assert first["authorization_url"] == second["authorization_url"]
        # The first process is gone, so its value can no longer complete.
        complete_cli_login(key, "good-code")
    finally:
        cancel_cli_login(key)


def test_a_command_that_prints_no_url_fails_fast(tmp_path, monkeypatch):
    script = tmp_path / "silent.py"
    script.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
    with pytest.raises(CliLoginError, match="printed no authorization URL"):
        begin_cli_login("silent", [sys.executable, str(script)], url_timeout=5)


def test_a_missing_command_is_an_error_not_a_crash(tmp_path):
    with pytest.raises(CliLoginError, match="could not start"):
        begin_cli_login("missing", [str(tmp_path / "no-such-binary")])


BROWSER_CLI = """
import os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

seen = sys.argv[1]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(seen, "w", encoding="utf-8") as handle:
            handle.write(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
port = server.server_port
callback = "http://127.0.0.1:%d/callback?code=good&state=s" % port
os.spawnv(os.P_WAIT, os.environ["BROWSER"], [os.environ["BROWSER"], callback])
print("If the browser didn't open, visit: "
      "https://claude.com/cai/oauth/authorize?code=true")
print("Paste code here if prompted > ", end="", flush=True)
server.handle_request()
sys.exit(0)
"""


def _browser_cli(tmp_path: Path) -> tuple[list[str], Path]:
    script = tmp_path / "browser_cli.py"
    script.write_text(BROWSER_CLI, encoding="utf-8")
    seen = tmp_path / "callback-path.txt"
    return [sys.executable, str(script), str(seen)], seen


def test_the_browser_recorder_reports_the_page_the_cli_would_open(tmp_path):
    argv, _ = _browser_cli(tmp_path)
    key = "test-browser-record"
    try:
        started = begin_cli_login(key, argv, browser=True)
    finally:
        cancel_cli_login(key)

    assert started["authorization_url"] == (
        "https://claude.com/cai/oauth/authorize?code=true"
    )
    assert started["callback_origin"].startswith("http://127.0.0.1:")
    assert "?" not in started["callback_origin"]


def test_a_pasted_callback_url_is_replayed_on_this_host(tmp_path):
    argv, seen = _browser_cli(tmp_path)
    key = "test-browser-replay"
    started = begin_cli_login(key, argv, browser=True)

    complete_cli_login(key, started["callback_origin"] + "/callback?code=good&state=s")

    assert seen.read_text(encoding="utf-8") == "/callback?code=good&state=s"


def test_a_flow_that_finished_on_its_own_is_not_an_error(tmp_path):
    script = tmp_path / "done.py"
    script.write_text(
        "import sys\n"
        "print('visit https://accounts.example/oauth2/auth?state=x')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    key = "test-finished"
    started = begin_cli_login(key, [sys.executable, str(script)])
    assert started["authorization_url"]

    time.sleep(0.3)
    complete_cli_login(key, "good-code")


AUTHORIZE_CLI = """
import os, sys

os.spawnv(
    os.P_WAIT,
    os.environ["BROWSER"],
    [
        os.environ["BROWSER"],
        "https://claude.com/cai/oauth/authorize?code=true"
        "&redirect_uri=http%3A%2F%2Flocalhost%3A45109%2Fcallback&state=s",
    ],
)
print("visit https://claude.com/cai/oauth/authorize?code=true")
sys.stdin.readline()
sys.exit(0)
"""


def test_the_port_comes_from_the_redirect_of_the_page_the_cli_opens(tmp_path):
    script = tmp_path / "authorize_cli.py"
    script.write_text(AUTHORIZE_CLI, encoding="utf-8")
    key = "test-browser-authorize"
    started = begin_cli_login(key, [sys.executable, str(script)], browser=True)
    try:
        assert started["callback_origin"] == "http://127.0.0.1:45109"
        with pytest.raises(CliLoginError, match="isn't from this sign-in"):
            complete_cli_login(key, "http://127.0.0.1:1/callback?code=other")
        with pytest.raises(CliLoginError, match="isn't from this sign-in"):
            complete_cli_login(key, "http://127.0.0.1:45109/other?code=other")
        # The refused addresses leave the flow usable for its own code.
        complete_cli_login(key, "good-code")
    finally:
        cancel_cli_login(key)


BARE_LOOPBACK_CLI = """
import os, sys

os.spawnv(
    os.P_WAIT,
    os.environ["BROWSER"],
    [os.environ["BROWSER"], "http://127.0.0.1:45110/callback"],
)
print("visit https://claude.com/cai/oauth/authorize?code=true")
sys.stdin.readline()
sys.exit(0)
"""


def test_a_bare_loopback_page_still_teaches_the_port(tmp_path):
    script = tmp_path / "bare_cli.py"
    script.write_text(BARE_LOOPBACK_CLI, encoding="utf-8")
    key = "test-browser-bare"
    started = begin_cli_login(key, [sys.executable, str(script)], browser=True)
    try:
        assert started["callback_origin"] == "http://127.0.0.1:45110"
        with pytest.raises(CliLoginError, match="isn't from this sign-in"):
            complete_cli_login(key, "http://127.0.0.1:1/callback?code=other")
    finally:
        cancel_cli_login(key)


def test_a_callback_from_another_port_is_refused(tmp_path):
    argv, _ = _browser_cli(tmp_path)
    key = "test-browser-foreign"
    begin_cli_login(key, argv, browser=True)
    try:
        with pytest.raises(CliLoginError, match="isn't from this sign-in"):
            complete_cli_login(key, "http://127.0.0.1:1/callback?code=other")
    finally:
        cancel_cli_login(key)


def test_import_loads_neither_gateway_nor_api():
    """A sidecar imports the module before ``llm_rosetta.api`` can load."""
    import subprocess

    code = (
        "import sys, llm_rosetta.contracts.cli_login;"
        "bad = [m for m in sys.modules if m == 'llm_rosetta_gateway'"
        " or m.startswith('llm_rosetta_gateway.') or m == 'llm_rosetta.api'];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_module_imports_only_stdlib():
    """The module's own imports are standard library, nothing from llm_rosetta."""
    import ast

    from llm_rosetta.contracts import cli_login

    tree = ast.parse(Path(cli_login.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            roots.add((node.module or "").split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}, roots

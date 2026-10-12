"""Tests for the standalone AskSage token helper."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "asksage-token.py"


@pytest.fixture
def asksage_token():
    """Load the hyphenated helper as a test module."""
    spec = importlib.util.spec_from_file_location("asksage_token", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jwt(payload: object) -> str:
    """Build an unsigned JWT-shaped string carrying *payload*."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS512"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.signature"


# ---------------------------------------------------------------------------
# _token_expiry
# ---------------------------------------------------------------------------


def test_token_expiry_reads_the_exp_claim(asksage_token):
    assert asksage_token._token_expiry(_jwt({"exp": 1791888322})) == 1791888322.0


def test_token_expiry_returns_none_without_an_exp_claim(asksage_token):
    assert asksage_token._token_expiry(_jwt({"sub": "someone"})) is None


@pytest.mark.parametrize("payload", [123, "abc", [1, 2], None])
def test_token_expiry_ignores_json_that_is_not_an_object(asksage_token, payload):
    """A non-object claims segment must not raise AttributeError.

    The middle segment can decode to any JSON value; ``.get`` on a list or an
    int used to escape as an uncaught AttributeError and bury the diagnostic
    ``--status`` is meant to print.
    """
    assert asksage_token._token_expiry(_jwt(payload)) is None


@pytest.mark.parametrize("token", ["", "onlyonesegment", "two.segments"])
def test_token_expiry_ignores_things_that_are_not_jwts(asksage_token, token):
    assert asksage_token._token_expiry(token) is None


def test_token_expiry_ignores_undecodable_payloads(asksage_token):
    assert asksage_token._token_expiry("aaa.!!!not-base64!!!.ccc") is None


# ---------------------------------------------------------------------------
# _exchange
# ---------------------------------------------------------------------------


def _exchange_with(asksage_token, monkeypatch, envelope):
    """Run _exchange against a stubbed urllib that returns *envelope*."""
    body = json.dumps(envelope).encode()

    class _Response:
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        asksage_token.urllib.request, "urlopen", lambda *a, **kw: _Response()
    )
    return asksage_token._exchange("https://host", "user@example.gov", "key")


def test_exchange_returns_the_token(asksage_token, monkeypatch):
    assert (
        _exchange_with(asksage_token, monkeypatch, {"response": {"access_token": "t"}})
        == "t"
    )


@pytest.mark.parametrize(
    "envelope",
    [
        {"response": {"access_token": None}},
        {"response": {"access_token": 123}},
        {"response": {"access_token": ""}},
        {"response": {"access_token": []}},
        {"response": {}},
        {"response": "Token is invalid [2]", "status": 400},
    ],
)
def test_exchange_exits_on_a_token_that_is_not_a_usable_string(
    asksage_token, monkeypatch, envelope
):
    """token_command seeds whatever stdout carries.

    A literal ``None`` or ``123`` reaching stdout would be installed as the
    provider's credential and every later request would fail with 401.
    """
    with pytest.raises(SystemExit) as excinfo:
        _exchange_with(asksage_token, monkeypatch, envelope)
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def test_api_key_prefers_names_the_shims_do_not_claim(asksage_token, monkeypatch):
    """``ASKSAGE_API_KEY`` is the shims' api_key_env, so it must lose.

    The gateway turns that variable into ``api_key`` before it reads
    ``token_command`` and then rejects the pair as mutually exclusive, so the
    dedicated name has to win when both are present.
    """
    monkeypatch.setenv("ASKSAGE_API_KEY", "shim-owned")
    monkeypatch.setenv("ASKSAGE_TOKEN_API_KEY", "dedicated")
    assert asksage_token._resolve_api_key() == "dedicated"


def test_api_key_falls_back_to_the_anl_sample_name(asksage_token, monkeypatch):
    monkeypatch.delenv("ASKSAGE_TOKEN_API_KEY", raising=False)
    monkeypatch.setenv("ASK_SAGE_API", "anl")
    monkeypatch.setenv("ASKSAGE_API_KEY", "shim-owned")
    assert asksage_token._resolve_api_key() == "anl"


def test_api_key_still_accepts_the_shim_variable(asksage_token, monkeypatch, capsys):
    monkeypatch.delenv("ASKSAGE_TOKEN_API_KEY", raising=False)
    monkeypatch.delenv("ASK_SAGE_API", raising=False)
    monkeypatch.setenv("ASKSAGE_API_KEY", "shim-owned")

    assert asksage_token._resolve_api_key() == "shim-owned"
    assert "mutually exclusive" in capsys.readouterr().err


def test_api_key_exits_without_any_credential(asksage_token, monkeypatch):
    for name in ("ASKSAGE_TOKEN_API_KEY", "ASK_SAGE_API", "ASKSAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        asksage_token._resolve_api_key()
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_main(asksage_token, monkeypatch, argv, token="tok"):
    monkeypatch.setattr(asksage_token, "_exchange", lambda *a, **kw: token)
    monkeypatch.setenv("ASKSAGE_TOKEN_EMAIL", "user@example.gov")
    monkeypatch.setenv("ASKSAGE_TOKEN_API_KEY", "key")
    monkeypatch.setattr(sys, "argv", ["asksage-token.py", *argv])
    asksage_token.main()


def test_probe_with_an_empty_model_fails_instead_of_skipping(
    asksage_token, monkeypatch, capsys
):
    """``--probe ""`` used to read as "no probe" and print the token anyway."""
    monkeypatch.setattr(asksage_token, "_probe", pytest.fail)

    with pytest.raises(SystemExit) as excinfo:
        _run_main(asksage_token, monkeypatch, ["--probe", ""])

    assert excinfo.value.code == 1
    assert "empty model" in capsys.readouterr().err


def test_probe_without_a_value_uses_the_default_model(asksage_token, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        asksage_token,
        "_probe",
        lambda host, token, model: seen.append(model) or True,
    )

    _run_main(asksage_token, monkeypatch, ["--probe"])

    assert seen == [asksage_token.DEFAULT_PROBE_MODEL]


def test_a_failed_probe_exits(asksage_token, monkeypatch):
    monkeypatch.setattr(asksage_token, "_probe", lambda *a, **kw: False)

    with pytest.raises(SystemExit) as excinfo:
        _run_main(asksage_token, monkeypatch, ["--probe", "gpt-4.1"])

    assert excinfo.value.code == 1


def test_status_prints_no_token(asksage_token, monkeypatch, capsys):
    _run_main(asksage_token, monkeypatch, ["--status"], token=_jwt({"exp": 9999999999}))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Token valid for" in captured.err


def test_plain_run_prints_only_the_token(asksage_token, monkeypatch, capsys):
    _run_main(asksage_token, monkeypatch, [], token="the-token")

    captured = capsys.readouterr()
    assert captured.out == "the-token\n"


def test_help_names_the_variables_the_code_actually_reads(
    asksage_token, monkeypatch, capsys
):
    """``--help`` is the one place a renamed env var silently goes stale.

    It is also the place someone looks when the script says "No account
    email", so a help string naming the wrong variable sends them to the
    wrong export.
    """
    monkeypatch.setattr(sys, "argv", ["asksage-token.py", "--help"])

    with pytest.raises(SystemExit):
        asksage_token.main()

    out = capsys.readouterr().out
    assert "ASKSAGE_TOKEN_EMAIL" in out
    assert "ASKSAGE_HOST" in out
    assert "ASKSAGE_EMAIL" not in out


def test_timeout_leaves_room_under_the_gateway_budget(asksage_token):
    """The gateway kills a ``token_command`` at COMMAND_TIMEOUT.

    This script has to give up first, or a mint that would have succeeded is
    discarded and the operator sees the gateway's timeout instead of the
    script's diagnostic.
    """
    from llm_rosetta.gateway.transport.token_refresh import COMMAND_TIMEOUT

    assert asksage_token.TIMEOUT < COMMAND_TIMEOUT

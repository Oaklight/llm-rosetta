"""A provider save must not drop the credential *source* (#946).

The provider editor has no field for ``token_command`` — the gateway mints the
credential from a command instead of storing one — so an ordinary save never
mentions it.  Rebuilding the entry from the request body alone dropped it,
which turned a command-driven provider (ALCF, AskSage) into a static-key one
on the next edit; the same entry is what the panel then prefilled with the
literal string ``token_command`` as the key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes._shared import _build_provider_entry

COMMAND = ["python3", "token.py"]


def _save(body: dict, existing: dict | None = None, name: str = "alcf") -> dict:
    return _build_provider_entry(
        body,
        api_key=body.get("api_key", ""),
        base_url=body.get("base_url", ""),
        existing_providers={} if existing is None else existing,
        resolve_name=name,
    )


def test_an_ordinary_save_keeps_the_stored_command():
    stored = {
        "type": "alcf--sophia",
        "base_url": "https://api.example/v1",
        "token_command": COMMAND,
        "token_refresh_interval": 7200,
    }

    entry = _save(
        {"type": "alcf--sophia", "base_url": "https://api.example/v1"}, {"alcf": stored}
    )

    assert entry["token_command"] == COMMAND
    assert entry["token_refresh_interval"] == 7200


def test_sending_the_command_stores_it():
    entry = _save({"base_url": "https://api.example/v1", "token_command": COMMAND})

    assert entry["token_command"] == COMMAND


def test_an_explicitly_empty_command_clears_it():
    stored = {"token_command": COMMAND, "token_refresh_interval": 7200}

    entry = _save(
        {"base_url": "https://api.example/v1", "token_command": []}, {"alcf": stored}
    )

    assert "token_command" not in entry


def test_a_new_provider_gains_no_phantom_keys():
    entry = _save({"base_url": "https://api.example/v1", "type": "chat_completions"})

    assert "token_command" not in entry
    assert "token_refresh_interval" not in entry


def test_an_empty_key_is_not_stored_beside_a_command():
    """Two credential sources in one entry is what the loader refuses."""
    entry = _save({"base_url": "https://api.example/v1", "token_command": COMMAND})

    assert "api_key" not in entry


def test_an_explicitly_empty_command_stores_no_empty_key():
    """``token_command: []`` means "no command", not "no key either".

    The editor sends it for a source that stores no command (a shim command or
    a sign-in).  Writing ``api_key: ""`` as well would stop the loader from
    filling the shim's ``api_key_env``, which is how AskSage receives the
    account key its token command reads.
    """
    entry = _save({"base_url": "https://api.example/v1", "token_command": []})

    assert "api_key" not in entry
    assert "token_command" not in entry


def test_an_empty_key_is_still_stored_for_an_ordinary_provider():
    """The rule is about the command, not about empty keys in general."""
    entry = _save({"base_url": "https://api.example/v1", "type": "chat_completions"})

    assert entry["api_key"] == ""


def test_clear_api_key_removes_a_stored_key_the_body_never_mentions():
    """Switching static -> command/exchange deletes the key.

    An empty ``api_key`` has always meant "keep the stored one", so the editor
    needs a word of its own for the other instruction.
    """
    stored = {
        "type": "chat_completions",
        "base_url": "https://api.example/v1",
        "api_key": "sk-old",
    }

    entry = _save(
        {
            "base_url": "https://api.example/v1",
            "type": "chat_completions",
            "clear_api_key": True,
            "token_command": COMMAND,
        },
        {"p": stored},
        name="p",
    )

    assert "api_key" not in entry
    assert entry["token_command"] == COMMAND
    # The flag is an instruction, not a setting.
    assert "clear_api_key" not in entry


def test_clear_api_key_without_another_source_stores_an_empty_key():
    """The key is gone; the entry says so rather than keeping the old one."""
    stored = {"base_url": "https://api.example/v1", "api_key": "sk-old"}

    entry = _save(
        {"base_url": "https://api.example/v1", "clear_api_key": True},
        {"p": stored},
        name="p",
    )

    assert entry["api_key"] == ""


def test_an_empty_command_env_removes_the_entry_override():
    """The editor clears a command env by sending ``""``.

    The loader treats a *present* ``token_command_env`` as the override, even
    when it is empty, so this pop is what lets a shim's own env var apply
    again.  The command itself is untouched when the body omits it.
    """
    stored = {
        "base_url": "https://api.example/v1",
        "token_command": COMMAND,
        "token_command_env": "OLD_ACCOUNT_KEY",
    }

    entry = _save(
        {"base_url": "https://api.example/v1", "token_command_env": ""},
        {"p": stored},
        name="p",
    )

    assert "token_command_env" not in entry
    assert entry["token_command"] == COMMAND


def test_an_omitted_command_env_keeps_the_stored_one():
    stored = {
        "base_url": "https://api.example/v1",
        "token_command": COMMAND,
        "token_command_env": "ACCOUNT_KEY",
    }

    entry = _save({"base_url": "https://api.example/v1"}, {"p": stored}, name="p")

    assert entry["token_command_env"] == "ACCOUNT_KEY"


def test_an_ordinary_save_keeps_the_key_the_caller_resolved():
    """Without the flag the resolved key is stored as given.

    ``put_provider`` reads the stored key when the body omits one; the builder
    must not invent a second rule for the same field.
    """
    stored = {"base_url": "https://api.example/v1", "api_key": "sk-old"}

    entry = _build_provider_entry(
        {"base_url": "https://api.example/v1"},
        api_key="sk-old",
        base_url="https://api.example/v1",
        existing_providers={"p": stored},
        resolve_name="p",
    )

    assert entry["api_key"] == "sk-old"


@pytest.mark.asyncio
async def test_clearing_the_key_is_judged_on_the_entry_it_leaves(tmp_path):
    """A source switch that leaves no credential is refused, and the old key
    with it: the save must not pass validation on a credential it deletes."""
    from llm_rosetta_gateway.admin.routes.config.providers import put_provider

    raw = {
        "providers": {
            "p": {
                "type": "chat_completions",
                "base_url": "https://api.example/v1",
                "api_key": "sk-old",
            }
        },
        "models": {},
    }
    request = _save_request(
        tmp_path,
        raw,
        {
            "type": "chat_completions",
            "base_url": "https://api.example/v1",
            "clear_api_key": True,
        },
    )

    response = await put_provider(request)

    assert response.status_code == 400
    assert b"api_key" in response.body
    assert _saved_file(request)["providers"]["p"]["api_key"] == "sk-old"


def test_a_shim_that_ships_an_exchange_needs_no_key():
    """``auth.py`` mints the credential, so the editor has nothing to demand."""
    from types import SimpleNamespace

    from llm_rosetta_gateway.admin.routes._shared import (
        _shim_ships_credential_exchange,
    )

    assert _shim_ships_credential_exchange(SimpleNamespace(auth=object())) is True
    assert _shim_ships_credential_exchange(SimpleNamespace(auth=None)) is False
    assert _shim_ships_credential_exchange(None) is False


def test_the_old_display_marker_is_never_a_key():
    """The literal ``token_command`` was a card label, not a credential.

    It reached the key slot through the old editor's "keep the stored key"
    path.  Beside a real command the loader refuses the entry; the save path
    must drop it instead of carrying it forward.
    """
    entry = _build_provider_entry(
        {"base_url": "https://api.example/v1", "token_command": COMMAND},
        api_key="token_command",
        base_url="https://api.example/v1",
        existing_providers={},
        resolve_name="alcf",
    )

    assert "api_key" not in entry
    assert entry["token_command"] == COMMAND


def test_the_old_display_marker_is_dropped_without_a_command():
    entry = _build_provider_entry(
        {"base_url": "https://api.example/v1"},
        api_key="token_command",
        base_url="https://api.example/v1",
        existing_providers={},
        resolve_name="plain",
    )

    assert entry["api_key"] == ""


def test_a_config_the_loader_rejects_is_named_before_it_is_written():
    from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error

    error = _config_preflight_error(
        {
            "providers": {
                "alcf": {
                    "type": "openai",
                    "base_url": "https://api.example/v1",
                    "api_key": "sk-x",
                    "token_command": COMMAND,
                }
            }
        }
    )

    assert error is not None
    assert "mutually exclusive" in error


def _capture_provider_logs():
    """Collect ``(level, message)`` from the providers logger."""
    import logging

    from llm_rosetta_gateway import providers as prov

    records: list[tuple[int, str]] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append((record.levelno, record.getMessage()))

    handler = _Capture()
    prov.logger.addHandler(handler)
    previous = prov.logger.level
    prov.logger.setLevel(logging.DEBUG)
    return prov, handler, records, previous


def _stop_capture(prov, handler, previous) -> None:
    prov.logger.removeHandler(handler)
    prov.logger.setLevel(previous)


def test_preflight_keeps_the_provider_startup_notices_quiet():
    """Every enabled provider is rebuilt on every save.

    The "executable not found" warning describes the provider, not the save;
    repeating it per save turns a real warning into background noise.
    """
    import logging

    from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error

    prov, handler, records, previous = _capture_provider_logs()
    try:
        error = _config_preflight_error(
            {
                "providers": {
                    "p": {
                        "type": "openai",
                        "base_url": "https://api.example/v1",
                        "token_command": ["/nonexistent/binary-xyz"],
                    }
                }
            }
        )
    finally:
        _stop_capture(prov, handler, previous)

    assert error is None
    binary = [level for level, msg in records if "not found on PATH" in msg]
    assert binary == [logging.DEBUG]


def test_a_normal_build_still_warns_about_a_missing_binary():
    """The quiet switch is the validator's, not the builder's default."""
    import logging

    prov, handler, records, previous = _capture_provider_logs()
    try:
        prov.build_provider_info(
            "openai",
            {
                "base_url": "https://api.example/v1",
                "token_command": ["/nonexistent/binary-xyz"],
            },
        )
    finally:
        _stop_capture(prov, handler, previous)

    binary = [level for level, msg in records if "not found on PATH" in msg]
    assert binary == [logging.WARNING]


def test_a_config_the_loader_accepts_passes_preflight():
    from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error

    assert (
        _config_preflight_error(
            {
                "providers": {
                    "alcf": {
                        "type": "openai",
                        "base_url": "https://api.example/v1",
                        "token_command": COMMAND,
                    }
                }
            }
        )
        is None
    )


def test_a_switched_off_provider_does_not_block_another_save():
    """Only enabled entries are built.

    A provider the operator already switched off may still be broken; it must
    not turn every other save into a 400 naming it.
    """
    from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error

    data = {
        "providers": {
            "off": {"type": "open_responses", "enabled": False},
            "on": {
                "type": "openai",
                "base_url": "https://api.example/v1",
                "token_command": COMMAND,
            },
        }
    }

    assert _config_preflight_error(data) is None


def _save_request(tmp_path, raw: dict, body: dict, name: str = "p") -> SimpleNamespace:
    import json

    from llm_rosetta_gateway.config import JsoncConfigIO

    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    app = SimpleNamespace(
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=None,
        provider_validation_warnings=[],
        shim_source_signature=None,
        rate_limit_state=None,
    )
    return SimpleNamespace(
        app=app,
        path_params={"name": name},
        method="PUT",
        headers={},
        json=lambda: dict(body),
    )


def _saved_file(request: SimpleNamespace) -> dict:
    import json

    with open(request.app.config_path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.asyncio
async def test_a_rejected_save_returns_400_and_writes_nothing(tmp_path):
    """The preflight runs before the write, not after the damage.

    ``config_mutate`` saves first and reloads second, so without this the bad
    document is on disk when the 500 comes back, and the next process start
    fails on it.
    """
    from llm_rosetta_gateway.admin.routes.config.providers import put_provider

    raw = {"providers": {}, "models": {}}
    request = _save_request(
        tmp_path,
        raw,
        {
            "type": "openai",
            "base_url": "https://api.example/v1",
            "api_key": "sk-x",
            "token_command": COMMAND,
        },
    )

    response = await put_provider(request)

    assert response.status_code == 400
    assert b"mutually exclusive" in response.body
    # The reader may migrate the document (``config_version``); the save must
    # add nothing to it.
    assert _saved_file(request)["providers"] == {}


@pytest.mark.asyncio
async def test_clearing_the_only_credential_names_what_is_missing(tmp_path):
    """A command provider has no key box, so "missing api_key" is not a field
    the operator can see.  The refusal says what the save would leave behind.
    """
    from llm_rosetta_gateway.admin.routes.config.providers import put_provider

    raw = {
        "providers": {
            "p": {
                "type": "openai",
                "base_url": "https://api.example/v1",
                "token_command": COMMAND,
            }
        },
        "models": {},
    }
    request = _save_request(
        tmp_path,
        raw,
        {"type": "openai", "base_url": "https://api.example/v1", "token_command": []},
    )

    response = await put_provider(request)

    assert response.status_code == 400
    assert b"no credential" in response.body
    assert b"api_key" not in response.body
    assert _saved_file(request)["providers"]["p"]["token_command"] == COMMAND


def test_clearing_a_command_on_a_shim_command_provider_keeps_the_right_message(
    monkeypatch,
):
    """The shim's own command is the fallback, so the credential is not gone.

    The shim here supplies the command but not the URL, so the save is refused
    for ``base_url`` — and the sentence must say that.  A bare ``if clearing:``
    would claim the provider has no credential instead.
    """
    import json

    from llm_rosetta_gateway.admin.routes.config import providers as mod

    shim = SimpleNamespace(
        connection=SimpleNamespace(
            base_url=None,
            keyless=False,
            token_command=["python3", "vendor-token.py"],
            token_command_env=None,
        ),
        auth=None,
    )
    monkeypatch.setattr(mod, "get_shim", lambda name: shim)

    error = mod._missing_provider_fields_error(
        {"type": "alcf--sophia", "token_command": []},
        api_key="",
        base_url="",
        resolve_name="p",
        existing_providers={},
    )

    assert error is not None
    message = json.loads(error.body)["error"]
    assert "base_url" in message
    assert "no credential" not in message


@pytest.mark.asyncio
async def test_clearing_the_command_still_names_other_missing_fields(tmp_path):
    """Clearing the command and the base_url together reports both.

    The specific sentence is about the credential; it must not hide a second
    problem behind the one it explains.
    """
    from llm_rosetta_gateway.admin.routes.config.providers import put_provider

    raw = {
        "providers": {
            "p": {
                "type": "openai",
                "base_url": "https://api.example/v1",
                "token_command": COMMAND,
            }
        },
        "models": {},
    }
    # A bare format has no shim base_url to fall back on, so clearing the
    # command without one leaves two things missing at once.
    request = _save_request(
        tmp_path,
        raw,
        {"type": "chat_completions", "token_command": []},
    )

    response = await put_provider(request)

    assert response.status_code == 400
    assert b"no credential" in response.body
    assert b"base_url" in response.body


def test_preflight_resolves_env_placeholders_without_touching_the_document(
    monkeypatch,
):
    """The admin edits the raw document; the loader substitutes env first."""
    from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error

    monkeypatch.setenv("UPSTREAM_TIMEOUT", "42")
    data = {
        "providers": {
            "p": {
                "type": "openai",
                "base_url": "https://api.example/v1",
                "timeout": "${UPSTREAM_TIMEOUT}",
            }
        }
    }

    assert _config_preflight_error(data) is None
    # The saved file keeps the placeholder.
    assert data["providers"]["p"]["timeout"] == "${UPSTREAM_TIMEOUT}"

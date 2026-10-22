"""A provider save must not drop the credential *source* (#946).

The provider editor has no field for ``token_command`` — the gateway mints the
credential from a command instead of storing one — so an ordinary save never
mentions it.  Rebuilding the entry from the request body alone dropped it,
which turned a command-driven provider (ALCF, AskSage) into a static-key one
on the next edit; the same entry is what the panel then prefilled with the
literal string ``token_command`` as the key.
"""

from __future__ import annotations

from llm_rosetta_gateway.admin.routes._shared import _build_provider_entry

COMMAND = ["python3", "scripts/alcf-token.py"]


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


def test_an_empty_key_is_still_stored_for_an_ordinary_provider():
    """The rule is about the command, not about empty keys in general."""
    entry = _save({"base_url": "https://api.example/v1", "type": "chat_completions"})

    assert entry["api_key"] == ""

"""A provider save carries the per-endpoint ``<type>_enabled`` flags.

``False`` is a value, not an absence: the save path's set-or-clear rule treats
every other falsy value as "remove the key", so a ``False`` sent by the editor
must be stored as sent.  ``""`` is how the editor says "enabled", and it must
not leave a key behind that only restates the default.
"""

from __future__ import annotations

from llm_rosetta_gateway.admin.routes._shared import _build_provider_entry


def _save(body: dict, existing: dict | None = None, name: str = "multi") -> dict:
    return _build_provider_entry(
        body,
        api_key=body.get("api_key", "k"),
        base_url=body.get("base_url", "https://x"),
        existing_providers={} if existing is None else existing,
        resolve_name=name,
    )


def test_false_is_stored():
    entry = _save({"base_url": "https://x", "embeddings_enabled": False})

    assert entry["embeddings_enabled"] is False


def test_true_is_stored_as_sent():
    entry = _save({"base_url": "https://x", "embeddings_enabled": True})

    assert entry["embeddings_enabled"] is True


def test_an_empty_value_clears_the_key():
    stored = {"base_url": "https://x", "embeddings_enabled": False}

    entry = _save(
        {"base_url": "https://x", "embeddings_enabled": ""}, {"multi": stored}
    )

    assert "embeddings_enabled" not in entry


def test_the_flag_is_typed_not_truthy():
    """A string that happens to be truthy is not a boolean switch."""
    entry = _save({"base_url": "https://x", "embeddings_enabled": "false"})

    assert "embeddings_enabled" not in entry


class TestProviderEnabled:
    """The LLM section's switch is the provider's own `enabled` flag.

    Chat is the provider rather than one of its endpoints, so there is no
    `llm_enabled`: the editor's LLM switch writes `enabled`, the same key the
    card's toggle writes.
    """

    def test_false_is_stored(self):
        entry = _save({"base_url": "https://x", "enabled": False})

        assert entry["enabled"] is False

    def test_true_clears_the_key(self):
        """`True` is the default, so it must not leave a key behind."""
        stored = {"base_url": "https://x", "enabled": False}

        entry = _save({"base_url": "https://x", "enabled": True}, {"multi": stored})

        assert "enabled" not in entry

    def test_an_absent_flag_keeps_what_is_stored(self):
        """A card toggle must survive an edit that does not mention it."""
        stored = {"base_url": "https://x", "enabled": False}

        entry = _save({"base_url": "https://y"}, {"multi": stored})

        assert entry["enabled"] is False

    def test_saving_restores_an_automatically_disabled_provider(self):
        """The editor opens it ticked, so saving drops `enabled` and the marker."""
        stored = {
            "base_url": "https://x",
            "enabled": False,
            "_auto_disabled": {"reason": "no base_url", "category": "missing_base_url"},
        }

        entry = _save({"base_url": "https://x", "enabled": True}, {"multi": stored})

        assert "enabled" not in entry
        assert "_auto_disabled" not in entry

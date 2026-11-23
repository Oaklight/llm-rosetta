"""A provider save must not drop the ACP lane's hand-written settings.

The provider editor has no ACP pane.  ``transport``, ``acp`` (the persistent
home and the agent command), and the older flat ``acp_argv``/``acp_env`` live
in ``config.jsonc`` only.  Rebuilding the entry from the editor's fields alone
dropped them, so the next request took the HTTP path — or the agent started
with the default profile, losing the vendor login the home exists for.
"""

from __future__ import annotations

from llm_rosetta_gateway.admin.routes._shared import _build_provider_entry


def _save(body: dict, existing: dict | None = None, name: str = "antigravity") -> dict:
    return _build_provider_entry(
        body,
        api_key=body.get("api_key", ""),
        base_url=body.get("base_url", ""),
        existing_providers={} if existing is None else existing,
        resolve_name=name,
    )


def test_an_ordinary_save_keeps_the_lane_and_its_profile():
    stored = {
        "type": "antigravity--acp",
        "transport": "acp",
        "acp": {
            "home": "saved/antigravity-work",
            "argv": ["/opt/agy-1.7/bin/agy", "acp"],
            "harden": True,
        },
        "acp_env": {"EXTRA": "1"},
    }

    entry = _save({"type": "antigravity--acp"}, {"antigravity": stored})

    assert entry["transport"] == "acp"
    assert entry["acp"] == stored["acp"]
    assert entry["acp_env"] == {"EXTRA": "1"}


def test_the_flat_spellings_survive_a_save_too():
    stored = {
        "type": "antigravity--acp",
        "acp_argv": ["/usr/local/bin/agy", "acp"],
        "acp_env": {"CLAUDE_CONFIG_DIR": "/srv/profile"},
    }

    entry = _save({"type": "antigravity--acp"}, {"antigravity": stored})

    assert entry["acp_argv"] == ["/usr/local/bin/agy", "acp"]
    assert entry["acp_env"] == {"CLAUDE_CONFIG_DIR": "/srv/profile"}


def test_a_body_that_names_the_block_stores_it():
    entry = _save(
        {
            "type": "antigravity--acp",
            "transport": "acp",
            "acp": {"home": "/srv/agy", "argv": ["agy", "acp"]},
        }
    )

    assert entry["transport"] == "acp"
    assert entry["acp"] == {"home": "/srv/agy", "argv": ["agy", "acp"]}


def test_an_empty_block_keeps_the_stored_one():
    """``{}`` says "nothing to change"; it must not drop the install pin."""
    stored = {
        "type": "antigravity--acp",
        "transport": "acp",
        "acp": {"home": "/x", "argv": ["/opt/agy", "acp"]},
    }

    entry = _save(
        {"type": "antigravity--acp", "acp": {}},
        {"antigravity": stored},
    )

    assert entry["acp"] == stored["acp"]


def test_a_null_block_clears_it():
    stored = {"type": "antigravity--acp", "transport": "acp", "acp": {"home": "/x"}}

    entry = _save(
        {"type": "antigravity--acp", "transport": "", "acp": None},
        {"antigravity": stored},
    )

    assert "transport" not in entry
    assert "acp" not in entry

"""The admin config payload's credential facts.

The provider editor decides between a key box and a credential section from
``mints_credential`` — the editor must not guess from the shim's name — and a
plan's usage page is a vendor fact, so it travels with the shim rather than
living in the panel's JavaScript.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from llm_rosetta.shims.providers import load_providers

PLAN_SHIM = "openai_chatgpt--open_responses"


def _shims() -> dict[str, Any]:
    return {s.name: s for s in load_providers()}


class TestTheShimDeclaresIt:
    def test_a_shim_with_an_exchange_says_what_it_mints(self):
        shim = _shims()[PLAN_SHIM]

        assert shim.auth is not None
        assert shim.auth.kind == "oauth"

    def test_a_shim_without_one_says_nothing(self):
        assert _shims()["openai--open_responses"].auth is None

    def test_the_plan_route_names_its_usage_page(self):
        """A per-app cap lives in the vendor's settings, not behind an API."""
        shim = _shims()[PLAN_SHIM]

        assert shim.connection.manage_usage_url == "https://chatgpt.com/settings/usage"

    def test_the_field_travels_through_both_yaml_shapes(self, tmp_path):
        from llm_rosetta.shims.providers import _load_single_provider

        grouped = tmp_path / "grouped"
        grouped.mkdir()
        (grouped / "provider.yaml").write_text(
            "name: g--open_responses\n"
            "model_type: llm\n"
            "format: open_responses\n"
            "connection:\n"
            "  base_url: https://example.test/v1\n"
            "  manage_usage_url: https://example.test/usage\n"
        )
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "provider.yaml").write_text(
            "name: l--open_responses\n"
            "model_type: llm\n"
            "format: open_responses\n"
            "default_base_url: https://example.test/v1\n"
            "manage_usage_url: https://example.test/usage2\n"
        )

        grouped_shim = _load_single_provider(grouped)
        legacy_shim = _load_single_provider(legacy)
        assert grouped_shim is not None and legacy_shim is not None

        assert grouped_shim.connection.manage_usage_url == "https://example.test/usage"
        assert legacy_shim.connection.manage_usage_url == "https://example.test/usage2"


class TestTheConfigPayload:
    def _payload(self, monkeypatch, config: Any) -> dict[str, Any]:
        import asyncio

        from llm_rosetta_gateway.admin.routes.config import payload

        monkeypatch.setattr(payload, "_get_config_path", lambda request: "/tmp/c.jsonc")
        monkeypatch.setattr(
            payload,
            "_get_config_io",
            lambda request: SimpleNamespace(
                load_raw=lambda path: {"providers": {"chatgpt": {}}}
            ),
        )
        monkeypatch.setattr(payload, "_shim_status_payload", lambda request: {})
        monkeypatch.setattr(payload, "_shim_removal_advisories", lambda r, raw: [])
        monkeypatch.setattr(payload, "_shim_source_presets_payload", lambda: [])
        config.credential_visible = False
        config.api_keys_db = None
        config.admin_session_idle_minutes = 30
        config.admin_session_remember_days = 30
        request = SimpleNamespace(
            app=SimpleNamespace(
                gateway_config=config,
                auth_state=SimpleNamespace(admin_password=None),
            )
        )
        return json.loads(asyncio.run(payload.get_config(request)).body)

    def test_registered_shims_carry_the_credential_facts(self, monkeypatch):
        payload = self._payload(monkeypatch, SimpleNamespace(providers={}))

        by_name = {s["name"]: s for s in payload["registered_shims"]}
        plan = by_name[PLAN_SHIM]
        assert plan["mints_credential"] is True
        assert plan["credential_kind"] == "oauth"
        assert plan["manage_usage_url"] == "https://chatgpt.com/settings/usage"

        plain = by_name["openai--open_responses"]
        assert plain["mints_credential"] is False
        assert plain["credential_kind"] is None
        assert plain["manage_usage_url"] is None

    def test_registered_shims_name_the_vendor_manifest(self, monkeypatch):
        """The credential lives in the vendor file now, so the payload names it.

        The Basic/Auth drawer reads both fields: the manifest path is what an
        operator is pointed at, and ``consumes_key`` keeps the key field for an
        exchange that takes the operator's key as its input.
        """
        payload = self._payload(monkeypatch, SimpleNamespace(providers={}))

        by_name = {s["name"]: s for s in payload["registered_shims"]}
        assert by_name[PLAN_SHIM]["manifest_path"].endswith(
            "openai_chatgpt/provider.yaml"
        )
        assert by_name["openai--open_responses"]["manifest_path"].endswith(
            "openai/provider.yaml"
        )
        assert all(s["manifest_path"] for s in payload["registered_shims"])
        assert all(s["consumes_key"] is False for s in payload["registered_shims"])

    def test_registered_shims_carry_the_language_mapped_hint(self, monkeypatch):
        """A shim's panel note travels as one entry per language."""
        payload = self._payload(monkeypatch, SimpleNamespace(providers={}))

        by_name = {s["name"]: s for s in payload["registered_shims"]}
        hint = by_name["opencode--chat_completions"]["hint_msg"]
        assert set(hint) == {"en", "zh"}
        assert "zen/v1" in hint["en"]
        assert by_name["openai--open_responses"]["hint_msg"] is None

    def test_the_refresh_log_is_bounded_here_too(self, monkeypatch):
        """The panel loads this payload on open.

        The credential endpoints clip the error text; this surface must clip
        it as well, or the bounded copy is not the one the browser sees first.
        """
        config = SimpleNamespace(
            providers={
                "chatgpt": SimpleNamespace(
                    mints_credential=True,
                    keyless=False,
                    token_status={
                        "consecutive_failures": 1,
                        "last_error": "x" * 900,
                    },
                    credentials_pending=False,
                    credential_source="config_command",
                )
            }
        )

        payload = self._payload(monkeypatch, config)

        status = payload["providers"]["chatgpt"]["token_status"]
        assert len(status["last_error"]) == 300
        assert status["last_error"].endswith("…")
        # The rest of the log is untouched.
        assert status["consecutive_failures"] == 1

    def test_a_provider_entry_says_whether_it_mints_its_own(self, monkeypatch):
        config = SimpleNamespace(
            providers={
                "chatgpt": SimpleNamespace(
                    mints_credential=True,
                    keyless=False,
                    token_status=None,
                    credentials_pending=True,
                    credential_source="auth_exchange",
                )
            }
        )

        payload = self._payload(monkeypatch, config)

        entry = payload["providers"]["chatgpt"]
        assert entry["mints_credential"] is True
        assert entry["credentials_pending"] is True
        assert entry["credential_source"] == "auth_exchange"


class TestTheLegacyMarker:
    """The old panel stored the literal word ``token_command`` in the key slot.

    Masking hides even that, so the drawer needs a flag recorded before the
    value is masked — otherwise the repair it offers can never appear.
    """

    def test_a_marker_becomes_a_flag_not_a_key(self):
        from llm_rosetta_gateway.admin.routes.config.payload import (
            _mask_provider_entry,
        )

        masked = _mask_provider_entry(
            "p", {"api_key": "token_command", "base_url": "https://x.test/v1"}
        )

        assert masked["legacy_credential_marker"] is True
        assert masked["api_key"] != "token_command"

    def test_a_real_key_has_no_marker(self):
        from llm_rosetta_gateway.admin.routes.config.payload import (
            _mask_provider_entry,
        )

        assert "legacy_credential_marker" not in _mask_provider_entry(
            "p", {"api_key": "sk-abcdefgh", "base_url": "https://x.test/v1"}
        )
        assert "legacy_credential_marker" not in _mask_provider_entry("p", {})

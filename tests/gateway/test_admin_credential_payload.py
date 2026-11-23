"""The admin config payload's credential facts.

The provider editor decides between a key box and a credential section from
``mints_credential`` — the editor must not guess from the shim's name — and a
plan's usage page is a vendor fact, so it travels with the shim rather than
living in the panel's JavaScript.
"""

from __future__ import annotations

import json

import pytest
from types import SimpleNamespace
from typing import Any

from _moved_vendors import load_providers_with_moved
from llm_rosetta.shims import load_providers_from_dir

# The vendors these tests read moved to contrib (#1660); their frozen copies
# are registered for every test here.
pytestmark = pytest.mark.usefixtures("moved_vendors")

PLAN_SHIM = "openai_chatgpt--open_responses"


def _shims() -> dict[str, Any]:
    return {s.name: s for s in load_providers_with_moved()}


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

    def test_the_field_travels_through_the_connection_block(self, tmp_path):
        loaded = _load_dir(tmp_path, grouped=GROUPED)

        shim = loaded["g--open_responses"]
        assert shim.connection.manage_usage_url == "https://example.test/usage"

    def test_the_flat_yaml_shape_is_refused(self, tmp_path):
        """The flat connection spelling was removed in 0.14.0."""
        with pytest.raises(ValueError, match="connection.manage_usage_url"):
            _load_dir(tmp_path, legacy=LEGACY)


GROUPED = (
    "name: g--open_responses\n"
    "model_type: llm\n"
    "format: open_responses\n"
    "connection:\n"
    "  base_url: https://example.test/v1\n"
    "  manage_usage_url: https://example.test/usage\n"
)
LEGACY = (
    "name: l--open_responses\n"
    "model_type: llm\n"
    "format: open_responses\n"
    "default_base_url: https://example.test/v1\n"
    "manage_usage_url: https://example.test/usage2\n"
)


def _load_dir(tmp_path, grouped=None, legacy=None):
    """Load shim definitions from a directory through the public loader."""
    from llm_rosetta.api import (
        load_providers_from_dir,
        restore_registry,
        snapshot_registry,
    )

    for name, text in (("grouped", grouped), ("legacy", legacy)):
        if text is not None:
            leaf = tmp_path / name
            leaf.mkdir()
            (leaf / "provider.yaml").write_text(text)
    snapshot = snapshot_registry()
    try:
        return dict((s.name, s) for s in load_providers_from_dir(tmp_path))
    finally:
        restore_registry(snapshot)


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
        config.credential_visible = False
        config.api_keys_db = None
        # The payload now sends the resolved switch; a real GatewayConfig
        # always has it (#1502 §2.2).
        config.open_on_no_keys = False
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
        assert plain["vendor_managed"] is False
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

    def test_a_vendor_managed_shim_says_so_and_mints_nothing(self, monkeypatch):
        """The editor hides the credential ring from such a lane."""
        from llm_rosetta.contracts.shim import ConnectionConfig, ProviderShim
        from llm_rosetta.shims.provider_shim import register_shim, unregister_shim

        class Exchange:
            kind = "vendor-cli"
            consumes_key = False
            vendor_managed = True

        name = "__test_vendor_managed_payload__"
        register_shim(
            ProviderShim(
                name=name,
                format="open_responses",
                connection=ConnectionConfig(
                    base_url="https://vendor.example/v1", keyless=True
                ),
                auth=Exchange(),
            )
        )
        try:
            payload = self._payload(monkeypatch, SimpleNamespace(providers={}))
        finally:
            unregister_shim(name)

        entry = {s["name"]: s for s in payload["registered_shims"]}[name]
        assert entry["vendor_managed"] is True
        assert entry["keyless"] is True
        assert entry["mints_credential"] is False

    def test_a_configured_vendor_managed_provider_says_so(self, monkeypatch):
        """The saved entry carries the same sign-in fact as the shim list."""
        pinfo = SimpleNamespace(
            keyless=True,
            mints_credential=False,
            credentials_pending=False,
            credential_source="none",
            token_status=None,
            auth_exchange=SimpleNamespace(vendor_managed=True),
        )
        payload = self._payload(
            monkeypatch, SimpleNamespace(providers={"chatgpt": pinfo})
        )

        entry = payload["providers"]["chatgpt"]
        assert entry["vendor_managed"] is True
        assert entry["mints_credential"] is False

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


class TestTheDeclaredSetupFields:
    """The credential form's inputs travel from the shim, not from the panel.

    A shim states them in ``provider.yaml``; the payload reports them in the
    shape the form renders, so changing the form for every vendor is a change
    to the YAML rather than a gateway release.
    """

    def test_a_shim_declaration_reaches_the_payload(self, monkeypatch, tmp_path):

        vendor = tmp_path / "acme"
        (vendor / "llm").mkdir(parents=True)
        (vendor / "provider.yaml").write_text(
            "fields:\n  - key: email\n    label: Account email\n    required: true\n"
        )
        (vendor / "llm" / "provider.yaml").write_text(
            "name: acme--chat_completions\nmodel_type: llm\nformat: chat_completions\n"
        )
        shim = load_providers_from_dir(tmp_path)[0]

        from llm_rosetta_gateway.admin.routes.config import payload

        monkeypatch.setattr(payload, "list_shims", lambda: [shim])
        body = TestTheConfigPayload()._payload(
            monkeypatch, SimpleNamespace(providers={})
        )

        assert body["registered_shims"][0]["fields"] == [
            {"key": "email", "label": "Account email", "required": True}
        ]

    def test_the_code_declaration_is_a_fallback_beside_the_yaml(self):
        """One list, two sources: the data wins per key, the code adds the rest."""
        from types import SimpleNamespace as Shim

        from llm_rosetta_gateway.admin.routes._shared import declared_fields

        exchange = SimpleNamespace(
            fields=lambda: (
                {"key": "email", "label": "From the exchange"},
                {"key": "tenant", "label": "Tenant"},
            )
        )
        shim = Shim(
            fields=({"key": "email", "label": "Account email"},),
            auth=exchange,
        )

        assert declared_fields(shim) == [
            {"key": "email", "label": "Account email"},
            {"key": "tenant", "label": "Tenant"},
        ]

    def test_an_entry_the_panel_cannot_render_is_dropped(self):
        from types import SimpleNamespace as Shim

        from llm_rosetta_gateway.admin.routes._shared import declared_fields

        shim = Shim(
            fields=(
                {"key": "../etc", "label": "Escape"},
                {"key": "ok", "label": "Fine", "help": 7},
                "not a mapping",
            ),
            auth=None,
        )

        # A non-identifier key and a non-mapping entry cannot be rendered; a
        # non-string hint is dropped while the row it belongs to stays.
        assert declared_fields(shim) == [{"key": "ok", "label": "Fine"}]


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

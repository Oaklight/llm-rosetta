"""Runtime coverage for the provider toggle response (#1106).

The toggle endpoint reports the *effective* post-revalidation state: turning a
provider on clears the automatic-disable marker, but the reload validates the
config again, so a provider that is still invalid stays disabled and the
response says so instead of echoing the request.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_rosetta.gateway.admin.routes.config.providers import toggle_provider
from llm_rosetta.gateway.config import JsoncConfigIO


def _request(tmp_path, raw: dict) -> SimpleNamespace:
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
        app=app, path_params={"name": "p"}, method="POST", headers={}
    )


def _on_disk(request) -> dict:
    return json.loads(
        (request.app.config_path and open(request.app.config_path)).read()
    )


@pytest.mark.asyncio
async def test_toggle_off_then_on_returns_the_effective_state(tmp_path):
    raw = {
        "providers": {"p": {"type": "openai", "base_url": "https://a.example/v1"}},
        "models": {"m": {"providers": [{"name": "p"}]}},
    }
    request = _request(tmp_path, raw)

    off = json.loads((await toggle_provider(request)).body)
    assert off == {"ok": True, "provider": "p", "enabled": False}
    assert _on_disk(request)["providers"]["p"]["enabled"] is False

    on = json.loads((await toggle_provider(request)).body)
    assert on == {"ok": True, "provider": "p", "enabled": True}
    assert "enabled" not in _on_disk(request)["providers"]["p"]


@pytest.mark.asyncio
async def test_turning_on_a_still_invalid_provider_reports_it_disabled(tmp_path):
    raw = {
        "providers": {"p": {"type": "open_responses", "enabled": False}},
        "models": {},
    }
    request = _request(tmp_path, raw)

    resp = json.loads((await toggle_provider(request)).body)

    assert resp["ok"] is True
    assert resp["enabled"] is False
    assert resp["provider_warnings"][0]["provider"] == "p"
    assert resp["provider_warnings"][0]["category"] == "missing_base_url"

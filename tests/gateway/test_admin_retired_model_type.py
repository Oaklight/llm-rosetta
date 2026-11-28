"""A retired model type in the admin config payload names its location (#919).

The loader refuses a retired spelling at startup with the file and the JSON
path.  The admin panel reads the file again on every open, so an entry edited
by hand after startup reached ``declared_model_type`` without a path and the
request failed with no location.  These tests pin the location on both the
helper and the route.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.config import payload
from llm_rosetta_gateway.admin.routes.config.payload import _normalize_model_entry
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.config.errors import RetiredConfigName
from _admin_app import fake_app

_CONFIG_PATH = "/srv/gateway/config.jsonc"


def test_the_helper_names_the_json_path_of_a_retired_type():
    with pytest.raises(RetiredConfigName) as caught:
        _normalize_model_entry({"provider": "p", "type": "embedding"}, "emb")

    assert caught.value.where == "models.emb.type"
    assert "embeddings" in caught.value.detail


def test_the_helper_names_the_index_of_a_retired_capability():
    with pytest.raises(RetiredConfigName) as caught:
        _normalize_model_entry({"provider": "p", "capabilities": ["decision"]}, "d")

    assert caught.value.where == "models.d.capabilities[0]"


def _get_config(monkeypatch, models: dict) -> tuple[int, dict]:
    raw = {"providers": {}, "models": models, "server": {}}
    monkeypatch.setattr(payload, "_get_config_path", lambda request: _CONFIG_PATH)
    monkeypatch.setattr(
        payload,
        "_get_config_io",
        lambda request: SimpleNamespace(load_raw=lambda path: raw),
    )
    monkeypatch.setattr(payload, "_shim_status_payload", lambda request: {})
    cfg = GatewayConfig({"providers": {}, "models": {}, "server": {}})
    cfg.credential_visible = False
    cfg.api_keys_db = None
    request = SimpleNamespace(
        app=fake_app(
            gateway_config=cfg,
            auth_state=SimpleNamespace(admin_password=None),
        )
    )
    response = asyncio.run(payload.get_config(request))
    return response.status_code, json.loads(response.body)


def test_the_route_reports_the_file_and_the_json_path(monkeypatch):
    status, body = _get_config(
        monkeypatch, {"emb": {"provider": "p", "type": "embedding"}}
    )

    assert status == 500
    assert body["path"] == _CONFIG_PATH
    assert body["where"] == "models.emb.type"
    assert body["error"].startswith(
        "Failed to read config: " + _CONFIG_PATH + ": models.emb.type: "
    )
    assert "llm-rosetta gateway migrate" in body["error"]


def test_a_current_spelling_still_loads(monkeypatch):
    status, body = _get_config(
        monkeypatch, {"emb": {"provider": "p", "type": "embeddings"}}
    )

    assert status == 200
    assert body["models"]["emb"]["type"] == "embeddings"

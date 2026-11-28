"""Deleting a provider takes its lanes, not the models that others serve (#1593).

A model entry may route through several providers.  Deleting one provider used
to delete every model entry that named it on any lane, so an entry that five
other providers were still serving vanished with a single click — this is how
``deepseek-v4.1-flash`` lost its other six lanes.

The cascade now removes the lanes that name the provider and keeps the rest,
deleting an entry only when no lane is left.  The panel's confirmation and its
toast read the same distinction from the response.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
from llm_rosetta_gateway.admin.routes.config.providers import delete_provider
from llm_rosetta_gateway.config import JsoncConfigIO


def _request(tmp_path, raw: dict, *, cascade: bool = True) -> SimpleNamespace:
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
    app.admin = GatewayAdminContext(app)
    query = "?cascade=true" if cascade else ""
    return SimpleNamespace(
        app=app,
        path_params={"name": "gone"},
        method="DELETE",
        headers={},
        query_params={"cascade": ["true"]} if cascade else {},
        query_string=query,
        raw_path=f"/admin/api/config/providers/gone{query}",
    )


def _on_disk(request) -> dict:
    return json.loads(open(request.app.config_path).read())


def _providers(*names: str) -> dict:
    return {
        name: {"base_url": f"https://{name}.example/v1", "type": "chat_completions"}
        for name in names
    }


class TestTheCascadeKeepsTheOtherLanes:
    @pytest.mark.asyncio
    async def test_multiple_surviving_lanes_stay_a_list(self, tmp_path):
        raw = {
            "providers": _providers("gone", "a", "b"),
            "models": {
                "m": {
                    "providers": [
                        {"name": "a", "weight": 3},
                        {"name": "gone", "weight": 2},
                        {"name": "b", "weight": 4},
                    ],
                    "capabilities": ["text"],
                }
            },
        }
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["ok"] is True
        assert body["cascade_trimmed_models"] == ["m"]
        assert "cascade_deleted_models" not in body
        entry = _on_disk(request)["models"]["m"]
        assert [p["name"] for p in entry["providers"]] == ["a", "b"]
        assert entry["providers"][0]["weight"] == 3

    @pytest.mark.asyncio
    async def test_one_lane_left_collapses_to_a_single_provider(self, tmp_path):
        raw = {
            "providers": _providers("gone", "a"),
            "models": {
                "m": {
                    "providers": [
                        {"name": "gone", "upstream_model": "x"},
                        {"name": "a", "upstream_model": "y"},
                    ],
                    "capabilities": ["text", "tools"],
                    "flatten_system": False,
                }
            },
        }
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["cascade_trimmed_models"] == ["m"]
        entry = _on_disk(request)["models"]["m"]
        assert entry["provider"] == "a"
        assert entry["upstream_model"] == "y"
        # The entry's own fields survive the collapse; they describe the model.
        assert entry["capabilities"] == ["text", "tools"]
        assert entry["flatten_system"] is False

    @pytest.mark.asyncio
    async def test_every_lane_removed_deletes_the_entry(self, tmp_path):
        raw = {
            "providers": _providers("gone"),
            "models": {
                "m": {
                    "providers": [{"name": "gone", "weight": 2}],
                    "capabilities": ["text"],
                }
            },
        }
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["cascade_deleted_models"] == ["m"]
        assert "m" not in _on_disk(request)["models"]


class TestTheSingleProviderEntry:
    @pytest.mark.asyncio
    async def test_a_string_entry_is_deleted(self, tmp_path):
        raw = {"providers": _providers("gone"), "models": {"m": "gone"}}
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["cascade_deleted_models"] == ["m"]
        assert "m" not in _on_disk(request)["models"]

    @pytest.mark.asyncio
    async def test_a_single_provider_dict_is_deleted(self, tmp_path):
        raw = {
            "providers": _providers("gone"),
            "models": {"m": {"provider": "gone", "upstream_model": "u"}},
        }
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["cascade_deleted_models"] == ["m"]
        assert "m" not in _on_disk(request)["models"]


class TestTheRefusal:
    @pytest.mark.asyncio
    async def test_without_cascade_nothing_is_written(self, tmp_path):
        raw = {
            "providers": _providers("gone", "a"),
            "models": {"m": {"providers": [{"name": "gone"}, {"name": "a"}]}},
        }
        request = _request(tmp_path, raw, cascade=False)

        response = await delete_provider(request)

        assert response.status_code == 409
        on_disk = _on_disk(request)
        assert "gone" in on_disk["providers"]
        assert len(on_disk["models"]["m"]["providers"]) == 2

    @pytest.mark.asyncio
    async def test_an_unknown_provider_is_a_404(self, tmp_path):
        request = _request(tmp_path, {"providers": _providers("a"), "models": {}})

        assert (await delete_provider(request)).status_code == 404


class TestAnEntryWithTwoLanesOfIt:
    @pytest.mark.asyncio
    async def test_both_of_its_lanes_go_and_the_rest_stay(self, tmp_path):
        """The same provider twice, plus one other: none of the two survives."""
        raw = {
            "providers": _providers("gone", "a"),
            "models": {
                "m": {
                    "providers": [
                        {"name": "gone", "upstream_model": "one"},
                        {"name": "a"},
                        {"name": "gone", "upstream_model": "two"},
                    ],
                    "capabilities": ["text"],
                }
            },
        }
        request = _request(tmp_path, raw)

        body = json.loads((await delete_provider(request)).body)

        assert body["cascade_trimmed_models"] == ["m"]
        entry = _on_disk(request)["models"]["m"]
        assert entry["provider"] == "a"


class TestTheDeleteRouteGivesBackTheProfile:
    @pytest.mark.asyncio
    async def test_a_deleted_entry_removes_its_managed_profile(self, tmp_path):
        data_dir = tmp_path / "data"
        home = data_dir / "acp" / "gone"
        home.mkdir(parents=True)
        request = _request(tmp_path, {"providers": _providers("gone"), "models": {}})
        request.app.gateway_config = SimpleNamespace(
            data_dir=str(data_dir),
            providers={
                "gone": SimpleNamespace(
                    managed_acp_state_dir=str(home),
                    managed_auth_state_dir=None,
                )
            },
        )

        body = json.loads((await delete_provider(request)).body)

        assert body["ok"] is True
        assert not home.exists()
        assert body["profile"]["removed"] == [str(home)]

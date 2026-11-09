"""The admin can store and read back two lanes of one provider (#1430, step 2).

A model may route one provider to several upstream models.  The rename-merge
path and the model editor both have to judge duplicates by the lane
``(provider, upstream_model)`` — otherwise the second lane is either refused
or silently replaces the first.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes.config.model_providers import (
    remove_model_provider,
    update_model_provider,
)
from llm_rosetta_gateway.admin.routes.config.models import put_model
from llm_rosetta_gateway.admin.routes.config.payload import _normalize_model_entry
from llm_rosetta_gateway.config import GatewayConfig, JsoncConfigIO

PROVIDER = {"type": "openai", "base_url": "https://p.test"}
MODEL = "deepseek-v4.1-flash"


def _raw(*, lanes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "providers": {"p": dict(PROVIDER)},
        "models": {
            MODEL: {
                "providers": lanes
                or [
                    {"name": "p", "upstream_model": "fast"},
                    {"name": "p", "upstream_model": "eco"},
                ],
                "capabilities": ["text"],
            }
        },
    }


def _single(upstream: str) -> dict[str, Any]:
    """The same model, written as a single-provider entry."""
    return {
        "providers": {"p": dict(PROVIDER)},
        "models": {MODEL: {"provider": "p", "upstream_model": upstream}},
    }


def _request(
    tmp_path: Path, raw: dict[str, Any], *, name: str, body: dict[str, Any]
) -> SimpleNamespace:
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


def _on_disk(request: SimpleNamespace) -> dict[str, Any]:
    return json.loads(Path(request.app.config_path).read_text(encoding="utf-8"))


class TestThePayloadKeepsBothLanes:
    def test_a_hand_written_entry_comes_back_whole(self):
        lanes = [
            {"name": "p", "upstream_model": "fast", "weight": 2},
            {"name": "p", "upstream_model": "eco", "capabilities": ["text"]},
        ]

        entry = _normalize_model_entry({"providers": lanes, "capabilities": ["text"]})

        assert entry["providers"] == lanes
        assert entry["provider"] == "p"

    def test_the_config_loads_two_lanes(self):
        config = GatewayConfig(_raw())

        route = config.models[MODEL]
        assert [entry.key for entry in route.providers] == [
            ("p", "fast"),
            ("p", "eco"),
        ]


class TestThePanelNoLongerCreatesAFallback:
    """A multi-lane entry keeps every upstream on its own lane.

    The entry-level ``upstream_model`` is a fallback the resolver still honours
    for hand-written configs, but the panel must stop feeding it: the editor
    stores the upstream it was given on the lane it named, the payload reports
    the fallback read-only under its own key, and deleting it takes an explicit
    request.
    """

    @pytest.mark.asyncio
    async def test_the_editor_stores_the_upstream_on_the_new_lane(self, tmp_path: Path):
        request = _request(
            tmp_path,
            _raw(),
            name=MODEL,
            body={"provider": "p", "upstream_model": "turbo", "merge": True},
        )

        await put_model(request)

        on_disk = _on_disk(request)["models"][MODEL]
        assert "upstream_model" not in on_disk
        assert [lane.get("upstream_model") for lane in on_disk["providers"]] == [
            "fast",
            "eco",
            "turbo",
        ]

    @pytest.mark.asyncio
    async def test_a_hand_written_top_level_upstream_survives_an_edit(
        self, tmp_path: Path
    ):
        """Dropping it would change what the lanes without one resolve to."""
        raw = _raw()
        raw["models"][MODEL]["upstream_model"] = "legacy-top-level"
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "capabilities": ["text", "vision"]},
        )

        await put_model(request)

        on_disk = _on_disk(request)["models"][MODEL]
        assert on_disk["upstream_model"] == "legacy-top-level"
        assert on_disk["capabilities"] == ["text", "vision"]

    @pytest.mark.asyncio
    async def test_the_panel_deletes_it_only_when_asked(self, tmp_path: Path):
        raw = _raw()
        raw["models"][MODEL]["upstream_model"] = "legacy-top-level"
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "clear_upstream_fallback": True},
        )

        await put_model(request)

        on_disk = _on_disk(request)["models"][MODEL]
        assert "upstream_model" not in on_disk
        assert [lane["upstream_model"] for lane in on_disk["providers"]] == [
            "fast",
            "eco",
        ]

    def test_the_payload_reports_it_under_its_own_key(self):
        entry = _normalize_model_entry(
            {
                "providers": [{"name": "p", "upstream_model": "fast"}],
                "upstream_model": "legacy-top-level",
                "capabilities": ["text"],
            }
        )

        assert entry["upstream_model_fallback"] == "legacy-top-level"
        assert "upstream_model" not in entry, "the editor must not re-post it"

    def test_the_payload_keeps_the_lane_key_for_a_single_provider(self):
        entry = _normalize_model_entry(
            {"provider": "p", "upstream_model": "gpt-4o-2024-08-06"}
        )

        assert entry["upstream_model"] == "gpt-4o-2024-08-06"
        assert "upstream_model_fallback" not in entry


def _post_request(
    tmp_path: Path, raw: dict[str, Any], *, name: str, body: dict[str, Any]
) -> SimpleNamespace:
    request = _request(tmp_path, raw, name=name, body=body)
    request.method = "POST"
    return request


class TestLaneAdminEndpoints:
    """remove/update-provider address one lane; a bare name must be enough."""

    def _two_lane(self) -> dict[str, Any]:
        raw = _raw()
        raw["models"][MODEL]["providers"] = [
            {"name": "p", "upstream_model": "fast", "weight": 2},
            {"name": "p", "upstream_model": "eco", "weight": 1},
        ]
        return raw

    def test_declared_lanes_are_one_row_each(self):
        config = GatewayConfig(self._two_lane())

        assert config.declared_model_lanes[MODEL] == [
            {"name": "p", "upstream_model": "fast", "weight": 2, "enabled": True},
            {"name": "p", "upstream_model": "eco", "weight": 1, "enabled": True},
        ]

    @pytest.mark.asyncio
    async def test_removing_a_lane_keeps_its_sibling(self, tmp_path: Path):
        request = _post_request(
            tmp_path,
            self._two_lane(),
            name=MODEL,
            body={"provider": "p", "upstream_model": "eco"},
        )

        payload = json.loads((await remove_model_provider(request)).body)

        assert payload["ok"] is True
        assert payload["removed_upstream_model"] == "eco"
        # One lane left collapses back to the single-provider shape, carrying
        # the upstream it kept.
        entry = _on_disk(request)["models"][MODEL]
        assert entry["provider"] == "p"
        assert entry["upstream_model"] == "fast"

    @pytest.mark.asyncio
    async def test_removing_by_name_alone_is_ambiguous(self, tmp_path: Path):
        raw = self._two_lane()
        request = _post_request(tmp_path, raw, name=MODEL, body={"provider": "p"})

        response = await remove_model_provider(request)

        assert response.status_code == 409
        assert "'fast'" in json.loads(response.body)["error"]
        assert _on_disk(request)["models"] == raw["models"]

    @pytest.mark.asyncio
    async def test_updating_a_lane_changes_only_it(self, tmp_path: Path):
        request = _post_request(
            tmp_path,
            self._two_lane(),
            name=MODEL,
            body={"provider": "p", "upstream_model": "eco", "weight": 7},
        )

        payload = json.loads((await update_model_provider(request)).body)

        assert payload["ok"] is True
        lanes = _on_disk(request)["models"][MODEL]["providers"]
        assert [lane.get("weight") for lane in lanes] == [2, 7]

    @pytest.mark.asyncio
    async def test_updating_by_name_alone_is_ambiguous(self, tmp_path: Path):
        raw = self._two_lane()
        request = _post_request(
            tmp_path, raw, name=MODEL, body={"provider": "p", "weight": 7}
        )

        response = await update_model_provider(request)

        assert response.status_code == 409
        assert _on_disk(request)["models"] == raw["models"]

    @pytest.mark.asyncio
    async def test_one_lane_still_takes_a_bare_name(self, tmp_path: Path):
        raw = _raw(lanes=[{"name": "p", "upstream_model": "fast", "weight": 2}])
        request = _post_request(
            tmp_path, raw, name=MODEL, body={"provider": "p", "weight": 5}
        )

        payload = json.loads((await update_model_provider(request)).body)

        assert payload["ok"] is True
        lanes = _on_disk(request)["models"][MODEL]["providers"]
        assert [lane.get("weight") for lane in lanes] == [5]

    @pytest.mark.asyncio
    async def test_collapsing_keeps_the_fallback_the_survivor_uses(
        self, tmp_path: Path
    ):
        """The surviving lane declared none, so the entry value is its upstream.

        Dropping it would make that lane start asking for the model's own name
        — the silent edit the read-only fallback row exists to prevent.
        """
        raw = _raw()
        raw["models"][MODEL]["providers"] = [
            {"name": "p", "upstream_model": "fast"},
            {"name": "p"},
        ]
        raw["models"][MODEL]["upstream_model"] = "legacy-top-level"
        request = _post_request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "fast"},
        )

        await remove_model_provider(request)

        entry = _on_disk(request)["models"][MODEL]
        assert entry["provider"] == "p"
        assert entry["upstream_model"] == "legacy-top-level"

    @pytest.mark.asyncio
    async def test_the_undeclared_lane_is_addressed_as_empty(self, tmp_path: Path):
        """``""`` names the lane without an upstream; absent means "not said"."""
        raw = _raw()
        raw["models"][MODEL]["providers"] = [
            {"name": "p", "upstream_model": "fast"},
            {"name": "p"},
        ]
        request = _post_request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": ""},
        )

        payload = json.loads((await remove_model_provider(request)).body)

        assert payload["ok"] is True
        entry = _on_disk(request)["models"][MODEL]
        assert entry["provider"] == "p"
        assert entry["upstream_model"] == "fast"


class TestATopLevelUpstreamIsTheLaneFallback:
    """An entry with ``providers[]`` and a top-level ``upstream_model`` is known.

    The editor posts the upstream at the entry level, so a multi-provider entry
    can carry both.  The top-level value is not ignored: it is the **fallback**
    for a lane that declares none — ``config/resolve.py`` reads
    ``entry.upstream_model or config.model_upstream_names.get(model)``.  Only a
    shape with one lane declaring an upstream and one without can tell the two
    readings apart; when every lane declares one, both readings give the same
    answer.
    """

    def _raw_with_legacy_top_level(self) -> dict[str, Any]:
        raw = _raw()
        raw["models"][MODEL]["upstream_model"] = "legacy-top-level"
        return raw

    def test_the_payload_keeps_the_lanes_authoritative(self):
        entry = _normalize_model_entry(
            self._raw_with_legacy_top_level()["models"][MODEL]
        )

        assert [lane["upstream_model"] for lane in entry["providers"]] == [
            "fast",
            "eco",
        ]

    def test_the_route_has_only_the_lane_upstreams(self):
        """Every lane declares one, so the top-level value changes nothing."""
        config = GatewayConfig(self._raw_with_legacy_top_level())

        route = config.models[MODEL]
        assert [entry.key for entry in route.providers] == [
            ("p", "fast"),
            ("p", "eco"),
        ]
        picked = {route.select().upstream_model for _ in range(8)}
        assert picked == {"fast", "eco"}

    def test_a_lane_without_an_upstream_gets_the_top_level_one(self):
        """The mixed shape is the one that separates fallback from "ignored"."""
        raw = _raw()
        raw["models"][MODEL]["providers"] = [
            {"name": "p", "upstream_model": "fast"},
            {"name": "p"},
        ]
        raw["models"][MODEL]["upstream_model"] = "legacy-top-level"
        config = GatewayConfig(raw)

        picked: set[str | None] = set()
        # The strategy is deterministic (weighted round-robin alternates), so
        # two calls already see both lanes; the loop is margin, not sampling.
        for _ in range(32):
            route, _ = config.resolve(config.selection_context(MODEL, "openai_chat"))
            picked.add(route.upstream_model)

        assert picked == {"fast", "legacy-top-level"}


class TestTheEditorAddsASecondLane:
    @pytest.mark.asyncio
    async def test_a_second_upstream_of_the_same_provider_offers_the_merge(
        self, tmp_path: Path
    ):
        raw = _single("fast")
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "eco"},
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["merge_possible"] is True
        assert payload["existing_provider"] == "p"
        assert _on_disk(request)["models"] == raw["models"]

    @pytest.mark.asyncio
    async def test_the_merge_adds_the_lane_and_keeps_the_first(self, tmp_path: Path):
        raw = _single("fast")
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "eco", "merge": True},
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["ok"] is True
        assert payload["merged"] is True
        lanes = _on_disk(request)["models"][MODEL]["providers"]
        assert [lane.get("upstream_model") for lane in lanes] == ["fast", "eco"]

    @pytest.mark.asyncio
    async def test_editing_the_same_lane_stays_one_lane(self, tmp_path: Path):
        raw = _single("fast")
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={
                "provider": "p",
                "upstream_model": "fast",
                "capabilities": ["text", "vision"],
            },
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["ok"] is True
        on_disk = _on_disk(request)["models"][MODEL]
        assert "providers" not in on_disk
        assert on_disk["upstream_model"] == "fast"
        assert on_disk["capabilities"] == ["text", "vision"]

    @pytest.mark.asyncio
    async def test_a_new_lane_of_a_multi_provider_model_asks_first(
        self, tmp_path: Path
    ):
        """Without the flag the editor must not silently drop the new lane."""
        raw = _raw()
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "turbo"},
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["merge_possible"] is True
        assert _on_disk(request)["models"] == raw["models"]
        # The refusal names the lanes that exist, not the model name: on a
        # providers[] entry the upstream lives on the lane.
        assert "'fast'" in payload["error"] and "'eco'" in payload["error"]
        assert f"to '{MODEL}'" not in payload["error"]

    @pytest.mark.asyncio
    async def test_a_second_lane_of_a_multi_provider_model_is_added(
        self, tmp_path: Path
    ):
        raw = _raw()
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "turbo", "merge": True},
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["ok"] is True
        lanes = _on_disk(request)["models"][MODEL]["providers"]
        assert [lane.get("upstream_model") for lane in lanes] == [
            "fast",
            "eco",
            "turbo",
        ]

    @pytest.mark.asyncio
    async def test_the_same_lane_twice_is_not_added_again(self, tmp_path: Path):
        raw = _raw()
        request = _request(
            tmp_path,
            raw,
            name=MODEL,
            body={"provider": "p", "upstream_model": "eco", "merge": True},
        )

        payload = json.loads((await put_model(request)).body)

        assert payload["ok"] is True
        lanes = _on_disk(request)["models"][MODEL]["providers"]
        assert [lane.get("upstream_model") for lane in lanes] == ["fast", "eco"]

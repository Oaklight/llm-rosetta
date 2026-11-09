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

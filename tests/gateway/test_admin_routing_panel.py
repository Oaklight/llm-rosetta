"""The Routing panel's endpoint: what the selector decided, and why.

``GET /admin/api/routing/<model>`` exists so the panel never re-derives a
routing number in the browser.  These tests pin the projection — the share a
candidate serves at, the state word, the block end — and the refusal paths of
``PUT``, which writes the model's ``routing`` block through the ordinary
config rebuild (so the parser, not the panel, is the validator).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes.routing import (
    _decision,
    _expected_shares,
    _iso,
    _state,
    get_model_routing,
    list_routable_models,
    put_model_routing,
)
from llm_rosetta_gateway.config import JsoncConfigIO
from llm_rosetta_gateway.routing import (
    Eligibility,
    ModelRoute,
    ProviderEntry,
    parse_model_routing,
)


def _provider(name: str) -> SimpleNamespace:
    """A stand-in transport configuration the eligibility predicate accepts."""
    return SimpleNamespace(
        name=name,
        config_name=name,
        keyless=True,
        credentials_pending=False,
        key_ring=None,
        quota_probe=None,
    )


def _route(*entries: ProviderEntry) -> ModelRoute:
    return ModelRoute(
        providers=list(entries) or [ProviderEntry(name="a")],
        route_id="m",
    )


def _request(
    *,
    model: str = "m",
    models: dict[str, Any] | None = None,
    providers: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    tmp_path: Any = None,
    request_log: Any = None,
    query: dict[str, list[str]] | None = None,
) -> SimpleNamespace:
    app: SimpleNamespace = SimpleNamespace(
        gateway_config=SimpleNamespace(
            models=models if models is not None else {model: _route()},
            providers=providers if providers is not None else {"a": _provider("a")},
        ),
        request_log=request_log,
    )
    if raw is not None and tmp_path is not None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        app.config_path = str(path)
        app.config_io = JsoncConfigIO()
        app.provider_validation_warnings = []
        app.shim_source_signature = None
    return SimpleNamespace(
        app=app,
        path_params={"model": model},
        method="PUT" if body is not None else "GET",
        headers={},
        query_params=query or {},
        json=lambda: body or {},
    )


class TestStateWord:
    """One provider, one word, so the panel never invents a state."""

    def test_serving(self):
        assert _state(SimpleNamespace(eligible=True, sunk=False)) == "serving"

    def test_sunk_but_still_eligible(self):
        assert _state(SimpleNamespace(eligible=True, sunk=True)) == "sunk"

    @pytest.mark.parametrize(
        "reason",
        ["blocked", "rate_limited", "spent", "no_credential", "initializing"],
    )
    def test_reason_passes_through(self, reason):
        answer = SimpleNamespace(eligible=False, sunk=False, reason=reason)
        assert _state(answer) == reason

    def test_an_unknown_provider_is_not_serving(self):
        assert _state(None) == "unknown"


class TestIso:
    def test_utc_gets_a_z_suffix(self):
        moment = datetime(2026, 11, 2, 12, 30, tzinfo=timezone.utc)
        assert _iso(moment) == "2026-11-02T12:30:00Z"

    def test_no_time_is_none(self):
        assert _iso(None) is None

    def test_a_future_block_still_serializes(self):
        soon = datetime.now(timezone.utc) + timedelta(minutes=1)
        stamp = _iso(soon)
        assert stamp is not None
        assert stamp.endswith("Z")


class TestExpectedShare:
    """The Share column is the strategy's arithmetic, not the raw weight."""

    def _lanes(self, *weights, eligible=None, sunk=None):
        entries = [
            ProviderEntry(name=chr(ord("a") + index), weight=weight)
            for index, weight in enumerate(weights)
        ]
        flags = eligible or [True] * len(entries)
        sunk_flags = sunk or [False] * len(entries)
        answers = [
            Eligibility(name=entry.name, eligible=True, sunk=is_sunk)
            for entry, is_sunk in zip(entries, sunk_flags)
        ]
        return [
            (entry, answer, 1.0, flag)
            for entry, answer, flag in zip(entries, answers, flags)
        ]

    def _route(self, strategy: str, count: int) -> ModelRoute:
        return ModelRoute(
            providers=[ProviderEntry(name=chr(ord("a") + i)) for i in range(count)],
            route_id="m",
            routing=parse_model_routing({"strategy": strategy}, label="t"),
        )

    def test_ordered_hands_everything_to_the_first_eligible_lane(self):
        lanes = self._lanes(1, 5)

        shares = _expected_shares(self._route("ordered", 2), lanes)

        assert shares == [1.0, 0.0]

    def test_ordered_skips_a_lane_that_cannot_serve(self):
        lanes = self._lanes(1, 1, eligible=[False, True])

        shares = _expected_shares(self._route("ordered", 2), lanes)

        assert shares == [0.0, 1.0]

    def test_rotate_ignores_the_configured_weights(self):
        lanes = self._lanes(1, 5)

        shares = _expected_shares(self._route("rotate", 2), lanes)

        assert shares == [1.0, 1.0]

    def test_pace_follows_the_allowance_not_the_weight(self):
        lanes = self._lanes(1, 5)

        shares = _expected_shares(self._route("pace", 2), lanes)

        assert shares == [1.0, 1.0]

    def test_weighted_keeps_the_weight(self):
        lanes = self._lanes(1, 5)

        shares = _expected_shares(self._route("weighted", 2), lanes)

        assert shares == [1.0, 5.0]

    def test_a_sunk_lane_holds_no_share_while_another_can_serve(self):
        # Selection only offers a sunk lane when nothing un-sunk is left, so
        # the column must not advertise traffic the lane cannot get.
        lanes = self._lanes(1, 5, sunk=[True, False])

        shares = _expected_shares(self._route("weighted", 2), lanes)

        assert shares == [0.0, 5.0]

    def test_a_sunk_lane_still_serves_when_nothing_else_can(self):
        lanes = self._lanes(1, 5, sunk=[True, True])

        shares = _expected_shares(self._route("weighted", 2), lanes)

        assert shares == [1.0, 5.0]

    def test_ordered_falls_back_to_a_sunk_lane_only(self):
        lanes = self._lanes(1, 1, eligible=[True, True], sunk=[True, False])

        shares = _expected_shares(self._route("ordered", 2), lanes)

        assert shares == [0.0, 1.0]

    def test_a_sunk_lane_keeps_its_share_when_the_model_turns_sink_off(self):
        # ``sink: false`` is the model saying it does not want refusals to
        # demote a lane; the selector keeps both in rotation, so the column
        # must too.
        route = ModelRoute(
            providers=[ProviderEntry(name="a"), ProviderEntry(name="b", weight=5)],
            route_id="m",
            routing=parse_model_routing(
                {"strategy": "weighted", "sink": False}, label="t"
            ),
        )
        lanes = self._lanes(1, 5, sunk=[True, False])

        shares = _expected_shares(route, lanes)

        assert shares == [1.0, 5.0]

    def test_no_eligible_lane_holds_a_share(self):
        lanes = self._lanes(1, 5, eligible=[False, False])

        shares = _expected_shares(self._route("weighted", 2), lanes)

        assert shares == [0.0, 0.0]


class TestGetModelRouting:
    """The endpoint's answer is the selector's own numbers."""

    @pytest.mark.asyncio
    async def test_weights_shares_and_state(self):
        route = _route(ProviderEntry(name="a"), ProviderEntry(name="b", weight=3))
        request = _request(
            models={"m": route},
            providers={"a": _provider("a"), "b": _provider("b")},
        )

        response = await get_model_routing(request)
        payload = json.loads(response.body)

        assert response.status_code == 200
        assert payload["model"] == "m"
        assert [entry["name"] for entry in payload["providers"]] == ["a", "b"]
        # Nothing has measured these lanes, so the factor is 1.0 and the share
        # is the configured weight.
        assert [entry["share"] for entry in payload["providers"]] == [1.0, 3.0]
        assert [entry["state"] for entry in payload["providers"]] == [
            "serving",
            "serving",
        ]
        assert all(entry["headroom"] is None for entry in payload["providers"])
        assert payload["decisions"] == []

    @pytest.mark.asyncio
    async def test_a_lane_the_route_dropped_is_listed_after_the_serving_ones(self):
        """A switched-off provider keeps its row on the board.

        The route drops it at parse time, so it is not a candidate; the
        document still declares it, and a lane missing from the page is a lane
        nobody notices is down.
        """
        route = _route(ProviderEntry(name="a"))
        request = _request(models={"m": route}, providers={"a": _provider("a")})
        request.app.gateway_config.declared_model_lanes = {
            "m": [{"name": "a", "weight": 1}, {"name": "off", "weight": 2}]
        }
        request.app.gateway_config.raw_providers = {
            "off": {"type": "chat_completions", "enabled": False}
        }

        response = await get_model_routing(request)
        payload = json.loads(response.body)

        assert [entry["name"] for entry in payload["providers"]] == ["a", "off"]
        off = payload["providers"][1]
        assert off["state"] == "off"
        assert off["excluded"] is True
        assert off["eligible"] is False
        assert off["share"] == 0
        assert off["weight"] == 2
        assert off["format"] == "chat_completions"
        assert payload["providers"][0]["excluded"] is False

        assert off["reason_kind"] == "provider"

    @pytest.mark.asyncio
    async def test_each_way_a_lane_leaves_the_route_is_named(self):
        """The row says which switch is off: the provider's, the gateway's, or the model's."""
        route = _route(ProviderEntry(name="a"))
        request = _request(models={"m": route}, providers={"a": _provider("a")})
        request.app.gateway_config.declared_model_lanes = {
            "m": [
                {"name": "a", "weight": 1, "enabled": True},
                {"name": "auto", "weight": 1, "enabled": True},
                {"name": "entry", "weight": 2, "enabled": False},
            ]
        }
        request.app.gateway_config.raw_providers = {
            "auto": {
                "type": "argo--chat_completions",
                "enabled": False,
                "_auto_disabled": {"reason": "required external shim missing"},
            },
            "entry": {"type": "chat_completions", "enabled": True},
        }

        payload = json.loads((await get_model_routing(request)).body)
        rows = {row["name"]: row for row in payload["providers"]}

        assert rows["auto"]["reason_kind"] == "auto"
        assert rows["auto"]["reason"] == "required external shim missing"
        # The lane's own switch is not the provider's, so the row must not
        # blame the providers section for it.
        assert rows["entry"]["reason_kind"] == "entry"
        assert rows["entry"]["reason"] == "switched off on this model"
        assert rows["entry"]["weight"] == 2

    @pytest.mark.asyncio
    async def test_a_model_with_every_lane_off_still_has_a_board(self):
        """An empty route is a model that stopped serving, not an unknown name.

        404 here would take away the only page where the lane can be switched
        back on — the model is still in the document.
        """
        request = _request(models={}, providers={"a": _provider("a")})
        request.app.gateway_config.declared_model_lanes = {
            "m": [{"name": "a", "weight": 1, "enabled": False}]
        }
        request.app.gateway_config.raw_providers = {
            "a": {"type": "chat_completions", "enabled": False}
        }

        response = await get_model_routing(request)
        payload = json.loads(response.body)

        assert response.status_code == 200
        assert payload["routing"] is None
        assert [row["name"] for row in payload["providers"]] == ["a"]
        assert payload["providers"][0]["excluded"] is True
        assert payload["providers"][0]["reason_kind"] == "entry"

    @pytest.mark.asyncio
    async def test_an_unknown_model_is_still_a_404(self):
        request = _request(models={}, providers={})
        request.app.gateway_config.declared_model_lanes = {}

        response = await get_model_routing(request)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_a_row_without_a_conversation_folds_by_client_identity(self):
        """Rows written before the log kept an id read the way they always did."""
        rows = [
            {
                "timestamp": "2026-11-02T12:00:00Z",
                "model": "m",
                "api_key_label": "codex",
                "user_agent": "codex-cli/1.0",
                "target_provider_name": "b",
            },
            {
                "timestamp": "2026-11-02T11:59:00Z",
                "model": "m",
                "api_key_label": "codex",
                "user_agent": "codex-cli/1.0",
                "target_provider_name": "a",
            },
            {
                "timestamp": "2026-11-02T11:58:00Z",
                "model": "m",
                "client_ip": "10.0.0.7",
                "target_provider_name": "a",
            },
        ]

        class _Log:
            async def get_entries(self, **_kwargs: Any):
                return rows, len(rows)

        request = _request(request_log=_Log())

        payload = json.loads((await get_model_routing(request)).body)

        assert payload["sessions"] == [
            {
                "client": "codex",
                "user_agent": "codex-cli/1.0",
                "provider": "b",
                "requests": 2,
                "last": "2026-11-02T12:00:00Z",
                "derived": False,
            },
            {
                "client": "10.0.0.7",
                "user_agent": None,
                "provider": "a",
                "requests": 1,
                "last": "2026-11-02T11:58:00Z",
                "derived": False,
            },
        ]
        # Every row is a fallback group rather than one conversation.
        assert payload["session_count"] == 2
        assert payload["session_sample"] == 3

    @pytest.mark.asyncio
    async def test_two_conversations_from_one_key_get_two_rows(self):
        """The case the old grouping could not show: one key, two sessions."""
        rows = [
            {
                "timestamp": "2026-11-02T12:00:00Z",
                "model": "m",
                "api_key_label": "claude-code",
                "user_agent": "claude-cli/2.0",
                "conversation_id": "b" * 32,
                "target_provider_name": "b",
            },
            {
                "timestamp": "2026-11-02T11:59:00Z",
                "model": "m",
                "api_key_label": "claude-code",
                "user_agent": "claude-cli/2.0",
                "conversation_id": "a" * 32,
                "target_provider_name": "a",
            },
            {
                "timestamp": "2026-11-02T11:58:00Z",
                "model": "m",
                "api_key_label": "claude-code",
                "user_agent": "claude-cli/2.0",
                "conversation_id": "b" * 32,
                "target_provider_name": "b",
            },
        ]

        class _Log:
            async def get_entries(self, **_kwargs: Any):
                return rows, len(rows)

        request = _request(request_log=_Log())

        payload = json.loads((await get_model_routing(request)).body)

        assert [(row["provider"], row["requests"]) for row in payload["sessions"]] == [
            ("b", 2),
            ("a", 1),
        ]
        assert all(row["derived"] for row in payload["sessions"])
        assert payload["session_count"] == 2
        assert payload["session_sample"] == 3

    @pytest.mark.asyncio
    async def test_a_fallback_row_never_merges_with_a_conversation(self):
        """A row with an id and a row without stay on separate lines."""
        rows = [
            {
                "timestamp": "2026-11-02T12:00:00Z",
                "model": "m",
                "api_key_label": "codex",
                "conversation_id": "c" * 32,
                "target_provider_name": "b",
            },
            {
                "timestamp": "2026-11-02T11:59:00Z",
                "model": "m",
                "api_key_label": "codex",
                "target_provider_name": "a",
            },
        ]

        class _Log:
            async def get_entries(self, **_kwargs: Any):
                return rows, len(rows)

        payload = json.loads(
            (await get_model_routing(_request(request_log=_Log()))).body
        )

        assert [row["derived"] for row in payload["sessions"]] == [True, False]
        assert payload["session_count"] == 2

    @pytest.mark.asyncio
    async def test_a_lane_reports_where_it_goes(self):
        route = _route(ProviderEntry(name="a", upstream_model="vendor/model-1"))
        request = _request(models={"m": route})
        request.app.gateway_config.provider_types = {"a": "messages"}
        request.app.gateway_config.provider_shim_names = {"a": "argo"}

        payload = json.loads((await get_model_routing(request)).body)

        lane = payload["providers"][0]
        assert lane["format"] == "messages"
        assert lane["shim"] == "argo"
        assert lane["upstream_model"] == "vendor/model-1"
        # This stub provider carries no quota probe, so the panel can say
        # "no probe" rather than a bare dash.
        assert lane["probe"] is False

    @pytest.mark.asyncio
    async def test_an_ineligible_lane_holds_no_share(self):
        route = _route(ProviderEntry(name="a"), ProviderEntry(name="b", weight=5))
        pending = _provider("b")
        pending.credentials_pending = True
        request = _request(
            models={"m": route}, providers={"a": _provider("a"), "b": pending}
        )

        payload = json.loads((await get_model_routing(request)).body)

        # The strategy never sees an ineligible lane, so it holds no share —
        # the panel must not show a lane at "no credential" serving 56%.
        assert payload["providers"][1]["eligible"] is False
        assert payload["providers"][1]["share"] == 0.0
        assert payload["providers"][0]["share"] > 0.0

    @pytest.mark.asyncio
    async def test_a_lane_reports_its_quota_windows(self):
        request = _request(models={"m": _route(ProviderEntry(name="a"))})

        payload = json.loads((await get_model_routing(request)).body)

        # This stub carries no probe, so there are no windows to list; the
        # field is always present so the panel never guesses.
        assert payload["providers"][0]["windows"] == []

    @pytest.mark.asyncio
    async def test_the_block_is_reported_not_re_derived(self):
        request = _request(models={"m": _route(ProviderEntry(name="a"))})

        payload = json.loads((await get_model_routing(request)).body)

        # A route with no stored block reports the default vocabulary.
        assert payload["routing"]["strategy"] == "weighted"
        assert payload["routing"]["affinity"] == "off"
        assert payload["routing"]["sink"] is True
        assert isinstance(payload["strategy"], str)

    @pytest.mark.asyncio
    async def test_an_unknown_model_names_the_configured_ones(self):
        request = _request(model="nope", models={"m": _route(ProviderEntry(name="a"))})

        response = await get_model_routing(request)
        payload = json.loads(response.body)

        assert response.status_code == 404
        assert "nope" in payload["error"]
        assert "m" in payload["error"]

    @pytest.mark.asyncio
    async def test_recent_decisions_come_from_the_request_log(self):
        rows = [
            {
                "timestamp": "2026-11-02T12:00:00Z",
                "model": "m",
                "request_id": "req_0123456789abcdef",
                "api_key_label": "codex",
                # The log keeps the client's User-Agent, so the panel shows it
                # under the key label: a decision names the client that made it.
                "user_agent": "claude-cli/1.2.3 (external, cli)",
                "target_provider_name": "a",
                "status_code": 200,
                "duration_ms": 12,
                # The proxy records a failover inside the row's profile, so the
                # panel must read it there.
                "profile": {"failover": {"from": "a", "to": "b"}},
            }
        ]

        class _Log:
            async def get_entries(self, **_kwargs: Any):
                return rows, 1

        request = _request(request_log=_Log())

        payload = json.loads((await get_model_routing(request)).body)

        assert payload["decisions"] == [
            {
                "time": "2026-11-02T12:00:00Z",
                "model": "m",
                "request_id": "req_0123456789abcdef",
                "client": "codex",
                "user_agent": "claude-cli/1.2.3 (external, cli)",
                "provider": "a",
                "status": 200,
                "duration_ms": 12,
                "failover": {"from": "a", "to": "b"},
            }
        ]

    @pytest.mark.asyncio
    async def test_a_broken_log_does_not_break_the_board(self):
        class _Log:
            async def get_entries(self, **_kwargs: Any):
                raise RuntimeError("db closed")

        request = _request(request_log=_Log())

        response = await get_model_routing(request)

        assert response.status_code == 200
        assert json.loads(response.body)["decisions"] == []


class TestRoutableModels:
    """The picker lists what the gateway can route, not every config key."""

    @pytest.mark.asyncio
    async def test_models_the_runtime_dropped_are_not_offered(self):
        # The config document may name a model whose provider was skipped at
        # load; the runtime's route table is the answer.
        request = _request(models={"b": _route(ProviderEntry(name="a")), "a": _route()})

        payload = json.loads((await list_routable_models(request)).body)

        assert payload["models"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_each_model_carries_its_lanes_and_traffic(self):
        class _Log:
            async def activity_by_model(self, since: str):
                return {"m": {"requests": 4, "last_seen": "2026-11-02T12:00:00Z"}}

        request = _request(
            models={"m": _route(ProviderEntry(name="a"), ProviderEntry(name="b"))},
            providers={"a": _provider("a"), "b": _provider("b")},
            request_log=_Log(),
        )

        payload = json.loads((await list_routable_models(request)).body)

        assert payload["window"] == "5h"
        assert payload["models"] == ["m"]
        assert payload["summary"]["m"] == {
            "providers": 2,
            "serving": 2,
            "blocked": 0,
            "shares": [1.0, 1.0],
            # The list's bar is a share stack, so the summary names the lane
            # behind each slice.
            "lanes": [
                {"name": "a", "share": 1.0, "eligible": True, "state": "serving"},
                {"name": "b", "share": 1.0, "eligible": True, "state": "serving"},
            ],
            "requests": 4,
            "last_seen": "2026-11-02T12:00:00Z",
        }

    @pytest.mark.asyncio
    async def test_a_failing_log_leaves_the_counts_unknown(self):
        # A locked or half-migrated log must not read as an idle gateway:
        # the roster still lists every model, and says the counts are not
        # the log's answer.
        class _Log:
            async def activity_by_model(self, since: str):
                raise RuntimeError("database is locked")

        request = _request(models={"m": _route()}, request_log=_Log())

        payload = json.loads((await list_routable_models(request)).body)

        assert payload["activity_known"] is False
        assert payload["models"] == ["m"]
        assert payload["summary"]["m"]["requests"] == 0

    @pytest.mark.asyncio
    async def test_a_window_it_does_not_know_is_refused(self):
        request = _request(query={"window": ["forever"]})

        response = await list_routable_models(request)

        assert response.status_code == 400
        assert "forever" in json.loads(response.body)["error"]

    @pytest.mark.asyncio
    async def test_a_model_with_no_traffic_still_appears(self):
        request = _request(models={"quiet": _route(), "busy": _route()})

        payload = json.loads((await list_routable_models(request)).body)

        assert payload["models"] == ["busy", "quiet"]
        assert payload["summary"]["quiet"]["requests"] == 0

    @pytest.mark.asyncio
    async def test_the_most_recently_requested_model_comes_first(self):
        class _Log:
            async def activity_by_model(self, since: str):
                return {
                    "old": {"requests": 9, "last_seen": "2026-11-02T10:00:00Z"},
                    "fresh": {"requests": 1, "last_seen": "2026-11-02T12:00:00Z"},
                }

        request = _request(
            models={"old": _route(), "fresh": _route(), "never": _route()},
            request_log=_Log(),
        )

        payload = json.loads((await list_routable_models(request)).body)

        assert payload["models"] == ["fresh", "old", "never"]

    @pytest.mark.asyncio
    async def test_no_config_is_an_error_not_an_empty_roster(self, monkeypatch):
        # The reader falls back to the process-wide config when the app has
        # none, so the missing case is patched rather than left to global
        # state another test may have left behind.
        from llm_rosetta_gateway.admin.routes import routing as routing_routes

        monkeypatch.setattr(routing_routes, "_get_gateway_config", lambda request: None)
        request = _request()

        response = await list_routable_models(request)

        assert response.status_code == 500


class TestDecisionRow:
    """One log row reduced to what the stream shows."""

    def test_provider_falls_back_to_the_target_column(self):
        entry = {"target_provider": "b", "status_code": 429}
        assert _decision(entry)["provider"] == "b"

    def test_missing_fields_are_empty_not_invented(self):
        decision = _decision({})
        assert decision["time"] is None
        assert decision["provider"] is None
        assert decision["status"] is None

    def test_a_failover_is_read_from_the_row_profile(self):
        entry = {"profile": {"failover": {"from": "a", "to": "b"}}}
        assert _decision(entry)["failover"] == {"from": "a", "to": "b"}

    def test_the_client_falls_back_to_the_address(self):
        assert _decision({"api_key_label": "codex"})["client"] == "codex"
        assert _decision({"client_ip": "10.0.0.4"})["client"] == "10.0.0.4"


class TestPutModelRouting:
    """The write is validated by the config parser, so the panel shows its words."""

    @pytest.mark.asyncio
    async def test_a_body_without_routing_is_refused(self):
        request = _request(body={"other": 1})

        response = await put_model_routing(request)

        assert response.status_code == 400
        assert "routing" in json.loads(response.body)["error"]

    @pytest.mark.asyncio
    async def test_an_unknown_model_is_refused(self, tmp_path):
        raw = {"providers": {}, "models": {"other": {"provider": "p"}}}
        request = _request(model="m", body={"routing": {}}, raw=raw, tmp_path=tmp_path)

        response = await put_model_routing(request)

        assert response.status_code == 404
        assert "m" in json.loads(response.body)["error"]

    @pytest.mark.asyncio
    async def test_an_unknown_key_is_refused_before_it_is_written(self, tmp_path):
        # A raw write of a block the loader will not read stops the next
        # start; the refusal has to come before the document changes.
        raw = {"providers": {}, "models": {"m": {"provider": "a"}}}
        request = _request(
            body={"routing": {"providers": ["b", "a"]}}, raw=raw, tmp_path=tmp_path
        )

        response = await put_model_routing(request)

        assert response.status_code == 400
        assert "unknown key" in json.loads(response.body)["error"]
        on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert on_disk == raw

    @pytest.mark.asyncio
    async def test_a_non_object_entry_is_refused(self, tmp_path):
        raw = {"providers": {}, "models": {"m": 5}}
        request = _request(body={"routing": {}}, raw=raw, tmp_path=tmp_path)

        response = await put_model_routing(request)

        assert response.status_code == 400
        assert "object" in json.loads(response.body)["error"]


class TestPinnedBlockIsWhatTheParserReads:
    """A block the panel could write is one ``parse_model_routing`` accepts."""

    def test_the_default_vocabulary_round_trips(self):
        block = {"strategy": "weighted", "affinity": "off", "sink": True}
        routing = parse_model_routing(block, label="models.m")

        assert routing.strategy == "weighted"
        assert routing.affinity == "off"
        assert routing.sink is True

    def test_an_unimplemented_affinity_is_refused_with_a_reason(self):
        with pytest.raises(ValueError, match="affinity"):
            parse_model_routing({"affinity": "turn"}, label="models.m")

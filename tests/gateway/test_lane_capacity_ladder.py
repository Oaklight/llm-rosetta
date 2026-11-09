"""The capacity ladder belongs to the upstream lane, not the provider (#1452).

A ``503`` says one upstream model cannot serve.  The ladder that counts those
refusals used to be keyed by provider, and a success on *any* lane cleared it —
so a healthy sibling kept erasing the failed lane's evidence, and every request
paid one doomed upstream call.  These tests pin the lane key, the clearing
rule, and the evidence a reader can see.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import ProviderMetadataStore, handle_non_streaming
from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy, RateLimitTracker
from llm_rosetta_gateway.quota.service import provider_cooldown
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing.eligibility import (
    lane_eligibility,
    provider_eligibility,
)
from llm_rosetta_gateway.transport._base import UpstreamResponse

_OK_BODY: dict[str, Any] = {
    "id": "resp-1",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0}],
}

_REQUEST: dict[str, Any] = {
    "model": "m",
    "messages": [{"role": "user", "content": "hello"}],
}

_REFUSAL_BODY = b'{"error": {"message": "not now", "type": "server_error"}}'


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The ladder is process-wide: no test may inherit another's counts."""
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


@pytest.fixture()
def gateway_warnings():
    """Collect this package's records: its logger does not propagate.

    Yields:
        The list the handler appends to, in order.
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("llm-rosetta-gateway")
    handler = _Collect()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


class _LaneTransport:
    """Answers by the model the attempt asks for, recording each attempt."""

    def __init__(self, *, failing: dict[str, int] | None = None) -> None:
        self._failing = failing or {}
        self.models: list[Any] = []

    async def send(
        self, provider_info: Any, url: str, body: Any, **kwargs: Any
    ) -> UpstreamResponse:
        model = body.get("model")
        self.models.append(model)
        status = self._failing.get(model) if isinstance(model, str) else None
        if status is not None:
            return UpstreamResponse(
                status_code=status, body=None, raw_content=_REFUSAL_BODY
            )
        return UpstreamResponse(
            status_code=200,
            body=_OK_BODY,
            raw_content=json.dumps(_OK_BODY).encode(),
        )

    async def send_streaming(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("these tests never stream")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""


def _lane_config(
    *,
    providers: tuple[str, ...] = ("a",),
    lanes: list[dict[str, Any]] | None = None,
) -> GatewayConfig:
    """A model whose route is the given lanes, with failover opted in.

    The default route is provider ``a``'s two lanes, plus one entry for every
    further provider named in *providers*.
    """
    if lanes is None:
        lanes = [
            {"name": "a", "upstream_model": "fast", "weight": 100},
            {"name": "a", "upstream_model": "eco"},
        ]
        lanes += [{"name": name} for name in providers if name != "a"]
    config = GatewayConfig(
        {
            "providers": {
                name: {
                    "type": "chat_completions",
                    "api_key": f"sk-{name}",
                    "base_url": f"https://{name}.test",
                }
                for name in providers
            },
            "models": {"m": {"providers": lanes}},
        }
    )
    config.model_failover = {"m": True}
    return config


def _serve(
    config: GatewayConfig, transport: _LaneTransport, model: str = "m"
) -> tuple[Any, list[Any]]:
    """Run one request; return its response and the models it asked for."""
    before = len(transport.models)
    selection = config.selection_context(model, "chat_completions")
    route, pinfo = config.resolve(selection)
    body = dict(_REQUEST, model=model)
    # The route handler rewrites the model to the lane's upstream alias before
    # the proxy runs; a handler called directly must do the same.
    if route.upstream_model:
        body["model"] = route.upstream_model

    async def _go() -> Any:
        return await handle_non_streaming(
            route,
            pinfo,
            body,
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    response, _profile = asyncio.run(_go())
    return response, transport.models[before:]


class TestTheBadLaneIsPaidForOnce:
    def test_four_requests_pay_one_failed_attempt_in_total(self):
        """The issue's scenario: ``fast`` always 503, ``eco`` always serves.

        The first request has to discover the failure; from the second on the
        failed lane is sunk behind its sibling and is not tried at all.
        """
        config = _lane_config()
        transport = _LaneTransport(failing={"fast": 503})

        first, models = _serve(config, transport)
        assert first.status_code == 200
        assert models == ["fast", "eco"]

        for _ in range(3):
            response, models = _serve(config, transport)
            assert response.status_code == 200
            assert models == ["eco"]

    def test_a_siblings_success_does_not_clear_the_failed_lanes_evidence(self):
        """The bug itself: ``eco`` serving said nothing about ``fast``."""
        config = _lane_config()
        transport = _LaneTransport(failing={"fast": 503})

        _serve(config, transport)

        record = QUOTA_STATE.rate_limits.upstream_record("a", "fast")
        assert record is not None
        assert record.refusals == 1
        assert QUOTA_STATE.rate_limits.upstream_sunk("a", "fast")
        # The sibling's own lane has no evidence against it.
        assert QUOTA_STATE.rate_limits.upstream_record("a", "eco") is None


class TestTheLaneOwnsItsCount:
    def test_a_lane_that_recovers_clears_its_own_count(self):
        """The other half of the key: only this lane's success clears it.

        With ``sink`` off nothing reorders the lanes, so the count is what
        disables one; a recovery that left the count behind would spend the
        three-strike rung on failures that already healed.
        """
        config = _lane_config()
        config.routing_rate_limit = RateLimitPolicy(sink=False, after=3)
        transport = _LaneTransport(failing={"fast": 503})

        for _ in range(2):
            _serve(config, transport)
        record = QUOTA_STATE.rate_limits.upstream_record("a", "fast")
        assert record is not None and record.refusals == 2

        transport._failing = {}
        _serve(config, transport)  # ``fast`` serves again

        assert QUOTA_STATE.rate_limits.upstream_record("a", "fast") is None


class TestTwoModelsWithoutAnAlias:
    def test_each_model_keeps_its_own_evidence(self):
        """The lane key falls back to the model, not to ``None``.

        Two models routed through one provider, neither naming an upstream
        model, used to share one key: the healthy model's success cleared the
        failing model's count, so the count never reached the threshold.  The
        fallback key is the model the client asked for, so each model's
        failures accumulate on their own.
        """
        config = _two_model_config()
        transport = _LaneTransport(failing={"bad": 503})

        for attempt in (1, 2):
            bad, models = _serve(config, transport, model="bad")
            assert bad.status_code == 503
            assert models == ["bad"]
            assert _refusals("bad") == attempt

            good, models = _serve(config, transport, model="good")
            assert good.status_code == 200
            assert models == ["good"]
            # The healthy model's success is not evidence about the failing
            # one: its count is still where it was.
            assert _refusals("bad") == attempt

        # The third refusal reaches ``after`` and takes the lane out; the count
        # starts over with it.
        bad, models = _serve(config, transport, model="bad")
        assert bad.status_code == 503
        assert QUOTA_STATE.rate_limits.upstream_disabled("a", "bad") is not None
        assert QUOTA_STATE.rate_limits.upstream_record("a", "good") is None


def _two_model_config() -> GatewayConfig:
    """One provider, two models, neither naming an upstream model."""
    config = GatewayConfig(
        {
            "providers": {
                "a": {
                    "type": "chat_completions",
                    "api_key": "sk-a",
                    "base_url": "https://a.test",
                }
            },
            "models": {
                "bad": {"provider": "a"},
                "good": {"provider": "a"},
            },
        }
    )
    config.model_failover = {"bad": True, "good": True}
    return config


def _refusals(lane: str) -> int:
    """The refusal count the ladder holds for one lane."""
    record = QUOTA_STATE.rate_limits.upstream_record("a", lane)
    return 0 if record is None else record.refusals


class TestTheWindow:
    def test_the_lane_is_probed_again_after_its_window(self):
        """A sunk lane is a fallback, not a tombstone: time brings it back."""
        tracker = RateLimitTracker()
        policy = RateLimitPolicy(window_s=300)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        tracker.note_upstream("a", "fast", policy, now=start)

        inside = start + timedelta(seconds=10)
        assert tracker.upstream_sunk("a", "fast", now=inside)
        assert tracker.upstream_record("a", "fast", now=inside) is not None

        after = start + timedelta(seconds=301)
        assert not tracker.upstream_sunk("a", "fast", now=after)
        assert tracker.upstream_record("a", "fast", now=after) is None


class TestStaleRecordsAreSwept:
    def test_a_write_removes_records_no_read_will_visit(self):
        """A renamed or removed lane must not sit in the tracker for ever.

        Expiry on read only helps while the same key keeps being asked about.
        A write sweeps the store, so the records of lanes nothing refers to any
        more leave on the next refusal.
        """
        tracker = RateLimitTracker()
        policy = RateLimitPolicy(window_s=60)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for index in range(50):
            tracker.note_upstream("p", f"old-{index}", policy, now=start)
        assert tracker.tracked() == 50

        later = start + timedelta(seconds=61)
        tracker.note_upstream("p", "fresh", policy, now=later)

        assert tracker.tracked() == 1


class TestADisabledRecordSurvivesTheSweep:
    def test_the_cooldown_is_not_cut_short_by_a_write(self):
        """The sweep may not collect a record that is still disabling a lane.

        A cooldown can outlast the window that counted the refusals — the
        default pair does not show it, a custom one does — and a record
        dropped early would let the lane back in while it is still out.
        """
        tracker = RateLimitTracker()
        policy = RateLimitPolicy(window_s=60, cooldown_s=600)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for _ in range(3):
            tracker.note_upstream("p", "sick", policy, now=start)
        assert tracker.upstream_disabled("p", "sick", now=start) is not None

        # Past the window, still inside the cooldown: a write to another lane
        # must leave the disabled record alone.
        inside = start + timedelta(seconds=120)
        tracker.note_upstream("p", "other", policy, now=inside)
        assert tracker.tracked() == 2
        assert tracker.upstream_disabled("p", "sick", now=inside) is not None

        # Past the cooldown, the next write collects it.
        expired = start + timedelta(seconds=601)
        tracker.note_upstream("p", "other", policy, now=expired)
        assert tracker.tracked() == 1
        assert tracker.upstream_record("p", "sick", now=expired) is None


class TestTheCredentialAxisIsShared:
    def test_a_429_still_takes_every_lane_of_the_provider_out(self):
        """A 429 is the key's state: the sibling shares it and leaves too."""
        config = _lane_config(providers=("a", "b"))
        transport = _LaneTransport(failing={"fast": 429})

        first, models = _serve(config, transport)
        assert first.status_code == 200
        # ``a`` refused with a 429, so the request moves to another provider —
        # not to ``a``'s sibling.
        assert models == ["fast", "m"]

        _response, models = _serve(config, transport)
        assert models == ["m"]
        # The credential ladder counted it; the lane ladder did not.
        assert QUOTA_STATE.rate_limits.upstream_record("a", "fast") is None


class TestEveryLaneBad:
    def test_the_route_moves_to_another_provider(self):
        """Both lanes refuse: the model's other provider serves next time."""
        config = _lane_config(providers=("a", "b"))
        transport = _LaneTransport(failing={"fast": 503, "eco": 503})

        first, models = _serve(config, transport)
        assert first.status_code == 200
        assert models == ["fast", "eco", "m"]

        _response, models = _serve(config, transport)
        assert models == ["m"]


class TestTheSingleLaneProvider:
    """A one-lane route selects and disables exactly as it did before.

    Two things are deliberately *not* unchanged, and this test states them
    rather than leaving them to be discovered: the credential-level accessors
    (``disabled``, ``latest_disable``, and through them the provider ``block``)
    no longer see a capacity refusal — those belong to the lane now, which the
    admin's per-lane row reports.  Summarising lane disables at the provider is
    a separate issue (#1463).
    """

    def test_routing_is_equivalent_and_visibility_moves_to_the_lane(
        self, gateway_warnings
    ):
        config = _lane_config(
            providers=("a",), lanes=[{"name": "a", "upstream_model": "solo"}]
        )
        transport = _LaneTransport(failing={"solo": 503})

        # Three refusals reach ``policy.after``, and with nothing else to try
        # every request still goes to the lane.
        for _ in range(3):
            response, models = _serve(config, transport)
            assert response.status_code == 503
            assert models == ["solo"]

        # The ladder did what it always did.
        assert QUOTA_STATE.rate_limits.upstream_disabled("a", "solo") is not None
        # The credential ladder is not involved in a capacity refusal.
        assert QUOTA_STATE.rate_limits.disabled("a") is None
        assert QUOTA_STATE.rate_limits.latest_disable("a") is None
        # The provider-level block stays a 429 story: a lane disable no longer
        # reaches it.  Reporting lane disables there is a separate issue.
        assert provider_cooldown("a", config.providers["a"]) is None
        # The admin's lane row reads the lane's own answer.
        answer = lane_eligibility(config.providers["a"], "a", lane="solo")
        assert answer.eligible is False
        assert answer.reason == "rate_limited"
        assert answer.until is not None
        # A caller with no lane in mind keeps the provider-wide answer.
        assert provider_eligibility(config.providers["a"], "a").eligible is True
        # The warning names the lane the ladder actually took out.
        assert any("a/solo" in record.getMessage() for record in gateway_warnings)

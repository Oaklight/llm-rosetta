"""Why a route is down: four states, one answer, one place.

``resolve`` used to work this out inline: a quota block, a spent allowance with
or without a reset, a failed startup and a plain "still initializing".  The
order is the rule — a client that just received a plan refusal must not be told
the gateway is booting — so it is pinned here rather than through the exception
it ends up in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.maintenance.startup import ProviderInitState
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing import SelectionContext
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway.routing.unavailable import (
    Unavailable,
    blocked_until,
    classify,
)

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def _ctx(names: list[str], excluded: frozenset[str] = frozenset()) -> SelectionContext:
    ctx = SelectionContext("m", "chat_completions")
    ctx.model_route = ModelRoute(
        providers=[ProviderEntry(name=n) for n in names],
        route_id="r-unavailable",
    )
    ctx.quota_excluded = excluded
    return ctx


def _config(providers: dict[str, object], deferred: object | None = None):
    return SimpleNamespace(providers=providers, _deferred_startup=deferred)


def _unready(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, ready=False)


def _blocked(name: str, identity: str, seconds: int) -> SimpleNamespace:
    QUOTA_STATE.block_provider(identity, seconds, "usage limit")
    pinfo = SimpleNamespace(
        name=name,
        ready=False,
        quota_identity=identity,
        quota_provider_blocked=lambda: QUOTA_STATE.provider_block(identity) is not None,
    )
    return pinfo


class TestTheBlock:
    def test_every_provider_blocked_is_a_block_with_the_latest_end(self):
        a = _blocked("a", "@unavail-block-a", 30)
        b = _blocked("b", "@unavail-block-b", 90)
        ctx = _ctx(["a", "b"])

        state = classify(_config({"a": a, "b": b}), ctx)

        assert state.reason == "blocked"
        assert state.until is not None
        ends = {
            QUOTA_STATE.provider_block("@unavail-block-a").until,
            QUOTA_STATE.provider_block("@unavail-block-b").until,
        }
        assert state.until == max(ends)

    def test_one_provider_unblocked_is_not_a_block(self):
        """The route's story is not (only) a quota story, so it falls through."""
        blocked = _blocked("a", "@unavail-block-mixed", 30)
        other = SimpleNamespace(
            name="b", ready=False, quota_provider_blocked=lambda: False
        )

        assert blocked_until([blocked, other]) is None

    def test_a_block_that_cannot_be_dated_is_unknown(self):
        pinfo = SimpleNamespace(
            name="a", ready=False, quota_provider_blocked=lambda: True
        )
        assert blocked_until([pinfo]) is None


class TestTheSpentAllowance:
    def test_a_known_reset_is_reported(self, monkeypatch):
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(
            service,
            "provider_exhausted_until",
            lambda pinfo: NOW + timedelta(minutes=5),
        )
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo}), ctx)

        assert state.reason == "exhausted"
        assert state.until == NOW + timedelta(minutes=5)
        assert state.as_kwargs() == {"exhausted_until": state.until, "exhausted": False}

    def test_no_named_reset_is_still_exhausted(self, monkeypatch):
        """The gateway refuses to invent a reset it was not told."""
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(service, "provider_exhausted_until", lambda pinfo: None)
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo}), ctx)

        assert state.reason == "exhausted"
        assert state.until is None
        assert state.as_kwargs() == {"exhausted_until": None, "exhausted": True}

    def test_one_candidate_left_out_is_not_exhausted(self, monkeypatch):
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(service, "provider_exhausted_until", lambda pinfo: NOW)
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a", "b"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo, "b": pinfo}), ctx)

        assert state.reason == "initializing"

    def test_a_route_with_no_candidates_is_not_exhausted(self):
        ctx = _ctx([])

        state = classify(_config({}), ctx)

        assert state.reason == "initializing"


class TestTheStartupStates:
    def test_every_unready_provider_failed(self):
        deferred = SimpleNamespace(provider_state=lambda name: ProviderInitState.FAILED)
        ctx = _ctx(["a", "b"])

        state = classify(
            _config({"a": _unready("a"), "b": _unready("b")}, deferred), ctx
        )

        assert state.reason == "failed"
        assert state.as_kwargs() == {"failed": True}

    def test_one_provider_still_initializing(self):
        states = {"a": ProviderInitState.FAILED, "b": ProviderInitState.READY}
        deferred = SimpleNamespace(provider_state=lambda name: states[name])
        ctx = _ctx(["a", "b"])

        state = classify(
            _config({"a": _unready("a"), "b": _unready("b")}, deferred), ctx
        )

        assert state.reason == "initializing"
        assert state.as_kwargs() == {"failed": False}

    def test_no_deferred_startup_is_initializing(self):
        ctx = _ctx(["a"])

        assert classify(_config({"a": _unready("a")}), ctx).reason == "initializing"


class TestTheAnswerShape:
    @pytest.mark.parametrize(
        "state, expected",
        [
            (Unavailable("blocked", NOW), {"blocked_until": NOW}),
            (
                Unavailable("exhausted", NOW),
                {"exhausted_until": NOW, "exhausted": False},
            ),
            (Unavailable("exhausted"), {"exhausted_until": None, "exhausted": True}),
            (Unavailable("failed"), {"failed": True}),
            (Unavailable("initializing"), {"failed": False}),
        ],
    )
    def test_as_kwargs_speaks_provider_not_ready(self, state, expected):
        assert state.as_kwargs() == expected

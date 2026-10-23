"""Quota-informed weighting: a nearly spent provider serves less, not nothing.

The quota layer knows how much allowance each credential has left.  Routing
used to ignore it until a provider was entirely blocked, so two "ready"
providers split the traffic evenly even when one was at 96%.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway.quota.models import QuotaReading, QuotaWindow
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing import (
    AnchorStage,
    QuotaStage,
    SelectionContext,
    create_strategy,
)
from llm_rosetta_gateway.routing.select import quota_weight, select_ready_entry
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway.transport.provider_info import ProviderInfo


def _pinfo(name: str, *, keys: str = "k1", probe: str | None = None) -> Any:
    pinfo = ProviderInfo(
        name,
        api_key=keys,
        base_url="https://example.test",
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template="{base_url}/x",
    )
    pinfo.quota_probe = probe
    return pinfo


def _reading(probe: str, key: str, used: float) -> QuotaReading:
    return QuotaReading(
        probe=probe,
        credential="fingerprint",
        windows=[QuotaWindow(name="5h", used_percent=used, span_seconds=18000)],
    )


class TestTheWeight:
    def test_an_unmeasured_provider_keeps_its_weight(self):
        assert quota_weight(None) == 1.0

    def test_an_empty_window_keeps_it_too(self):
        assert quota_weight(1.0) == 1.0

    def test_a_spent_window_still_serves_some(self):
        """Zero would starve a live provider; the state removes it, not this."""
        assert quota_weight(0.0) == 0.25

    def test_it_is_monotonic_in_between(self):
        assert quota_weight(0.75) > quota_weight(0.5) > quota_weight(0.25)


class TestTheStage:
    def _ctx(self, providers: dict[str, Any], names: list[str]) -> SelectionContext:
        ctx = SelectionContext("m", "chat_completions")
        ctx.providers = providers
        ctx.model_route = ModelRoute(
            providers=[ProviderEntry(name=n) for n in names],
            route_id="r",
            strategy=create_strategy("weighted_round_robin"),
        )
        return ctx

    def test_a_measured_provider_reports_its_smallest_share(self):
        probe = "p-test-stage"
        pinfo = _pinfo("a", keys="k1,k2", probe=probe)
        QUOTA_STATE.put_reading(_reading(probe, "k1", 10))
        # The reading is keyed by fingerprint, so put it where the lookup looks.
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=__import__(
                    "llm_rosetta_gateway.quota.models", fromlist=["x"]
                ).credential_fingerprint("k2"),
                windows=[QuotaWindow(name="5h", used_percent=90, span_seconds=18000)],
            )
        )
        ctx = self._ctx({"a": pinfo}, ["a"])

        QuotaStage().run(ctx)

        assert ctx.quota_headroom["a"] == 0.1

    def test_an_unmeasured_provider_is_one_not_zero(self):
        ctx = self._ctx({"a": _pinfo("a"), "b": _pinfo("b")}, ["a", "b"])

        QuotaStage().run(ctx)

        assert ctx.quota_headroom == {"a": 1.0, "b": 1.0}


class TestTheAnchor:
    def test_the_nearly_spent_provider_gets_materially_less(self):
        route = ModelRoute(
            providers=[ProviderEntry(name="fresh"), ProviderEntry(name="spent")],
            route_id="r-quota-split",
            strategy=create_strategy("weighted_round_robin"),
        )
        picks = {"fresh": 0, "spent": 0}
        for _ in range(40):
            entry = select_ready_entry(
                route,
                is_ready=lambda name: True,
                headroom={"fresh": 1.0, "spent": 0.0},
            )
            assert entry is not None
            picks[entry.name] += 1

        assert picks["fresh"] > picks["spent"] * 2
        assert picks["spent"] > 0  # it is still a candidate, not excluded

    def test_without_headroom_the_split_is_even(self):
        route = ModelRoute(
            providers=[ProviderEntry(name="a"), ProviderEntry(name="b")],
            route_id="r-quota-even",
            strategy=create_strategy("weighted_round_robin"),
        )
        picks = {"a": 0, "b": 0}
        for _ in range(20):
            entry = select_ready_entry(route, is_ready=lambda name: True)
            assert entry is not None
            picks[entry.name] += 1

        assert picks == {"a": 10, "b": 10}


class TestTheChainOrder:
    def test_the_anchor_refuses_to_run_before_the_quota_stage(self):
        """A reversed chain would weight nothing and say nothing."""
        from llm_rosetta_gateway.routing import RoutingChain
        from llm_rosetta_gateway.routing.stage import StageOrderError

        try:
            RoutingChain([AnchorStage(), QuotaStage()])
        except StageOrderError as exc:
            assert "quota_headroom" in str(exc)
        else:  # pragma: no cover - the failure this test exists for
            raise AssertionError("the reversed chain was accepted")

    def test_the_real_order_builds(self):
        from llm_rosetta_gateway.routing import RoutingChain

        RoutingChain([QuotaStage(), AnchorStage()])


class TestExclusionByReading:
    """A fresh reading that says "spent" removes the candidate, not just weight."""

    def _spent_provider(self, probe: str, *, age: float = 0.0) -> Any:
        from datetime import datetime, timedelta, timezone

        from llm_rosetta_gateway.quota.models import credential_fingerprint

        pinfo = _pinfo("spent", keys="k1", probe=probe)
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100, span_seconds=18000)],
                read_at=datetime.now(timezone.utc) - timedelta(seconds=age),
            )
        )
        return pinfo

    def test_a_fresh_exhausted_reading_excludes_the_provider(self):
        from llm_rosetta_gateway.quota.service import provider_spent

        probe = "p-exclude-fresh"
        assert provider_spent(self._spent_provider(probe)) is True

    def test_a_stale_reading_is_not_a_statement_about_now(self):
        from llm_rosetta_gateway.quota.service import (
            READING_MAX_AGE_SECONDS,
            provider_spent,
        )

        pinfo = self._spent_provider(
            "p-exclude-stale", age=READING_MAX_AGE_SECONDS + 60
        )

        assert provider_spent(pinfo) is False

    def test_an_unmeasured_provider_is_not_spent(self):
        from llm_rosetta_gateway.quota.service import provider_spent

        assert provider_spent(_pinfo("unmeasured", probe="p-exclude-none")) is False

    def test_one_credential_with_room_keeps_the_provider(self):
        from llm_rosetta_gateway.quota.models import credential_fingerprint
        from llm_rosetta_gateway.quota.service import provider_spent

        probe = "p-exclude-mixed"
        pinfo = _pinfo("mixed", keys="k1,k2", probe=probe)
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
            )
        )
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k2"),
                windows=[QuotaWindow(name="5h", used_percent=20)],
            )
        )

        assert provider_spent(pinfo) is False

    def test_a_spent_provider_is_not_chosen_even_with_an_even_split(self):
        route = ModelRoute(
            providers=[ProviderEntry(name="spent"), ProviderEntry(name="fresh")],
            route_id="r-exclude-split",
            strategy=create_strategy("weighted_round_robin"),
        )

        picks = set()
        for _ in range(20):
            entry = select_ready_entry(
                route,
                is_ready=lambda name: True,
                excluded=frozenset({"spent"}),
            )
            assert entry is not None
            picks.add(entry.name)

        assert picks == {"fresh"}


class TestExclusionNeedsCompleteEvidence:
    """Every credential read, and every one spent.

    An exclusion cannot correct itself: nothing routes to a provider that is
    out, so an unread credential is never read, and the provider would stay
    out until its reading aged past the freshness window.  Down-weighting
    self-heals; excluding must therefore demand complete evidence.
    """

    def _provider(self, probe: str, keys: str) -> Any:
        return _pinfo("p", keys=keys, probe=probe)

    def test_a_spent_key_beside_an_unread_one_is_not_spent(self):
        from llm_rosetta_gateway.quota.service import provider_spent

        probe = "p-evidence-unread"
        pinfo = self._provider(probe, "k1,k2")
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=__import__(
                    "llm_rosetta_gateway.quota.models", fromlist=["x"]
                ).credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
            )
        )

        assert provider_spent(pinfo) is False

    def test_a_key_that_decoded_no_window_is_not_agreement(self):
        from llm_rosetta_gateway.quota.models import credential_fingerprint
        from llm_rosetta_gateway.quota.service import provider_spent

        probe = "p-evidence-undecoded"
        pinfo = self._provider(probe, "k1,k2")
        for key, windows in (
            ("k1", [QuotaWindow(name="5h", used_percent=100)]),
            ("k2", []),
        ):
            QUOTA_STATE.put_reading(
                QuotaReading(
                    probe=probe,
                    credential=credential_fingerprint(key),
                    windows=windows,
                )
            )

        assert provider_spent(pinfo) is False

    def test_every_key_spent_is_spent(self):
        from llm_rosetta_gateway.quota.models import credential_fingerprint
        from llm_rosetta_gateway.quota.service import provider_spent

        probe = "p-evidence-all"
        pinfo = self._provider(probe, "k1,k2")
        for key in ("k1", "k2"):
            QUOTA_STATE.put_reading(
                QuotaReading(
                    probe=probe,
                    credential=credential_fingerprint(key),
                    windows=[QuotaWindow(name="5h", used_percent=100)],
                )
            )

        assert provider_spent(pinfo) is True

    def test_a_poll_slower_than_the_baseline_still_counts(self):
        """A reading cannot be stale on arrival because the operator polls slowly."""
        from datetime import datetime, timedelta, timezone

        from llm_rosetta_gateway.quota.models import credential_fingerprint
        from llm_rosetta_gateway.quota.service import (
            READING_MAX_AGE_SECONDS,
            provider_spent,
        )

        probe = "p-evidence-interval"
        pinfo = self._provider(probe, "k1")
        pinfo.quota_interval = READING_MAX_AGE_SECONDS * 2
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
                read_at=datetime.now(timezone.utc)
                - timedelta(seconds=READING_MAX_AGE_SECONDS + 300),
            )
        )

        assert provider_spent(pinfo) is True


class TestTheChainEndToEnd:
    """The stage's output must be what selection actually consumes.

    Testing ``provider_spent`` and ``select_ready_entry`` separately leaves the
    wire between them untested: a stage that never adds to ``excluded``, or an
    anchor that drops the set before selecting, passes both.
    """

    def _chain_ctx(
        self, providers: dict[str, Any], names: list[str]
    ) -> SelectionContext:
        from llm_rosetta_gateway.routing import RoutingChain

        ctx = SelectionContext("m", "chat_completions")
        ctx.providers = providers
        ctx.model_route = ModelRoute(
            providers=[ProviderEntry(name=n) for n in names],
            route_id="r-chain",
            strategy=create_strategy("weighted_round_robin"),
        )
        self._chain = RoutingChain([QuotaStage(), AnchorStage()])
        return ctx

    def test_a_spent_provider_never_serves_beside_an_unmeasured_one(self):
        from llm_rosetta_gateway.quota.models import credential_fingerprint

        probe = "p-chain-exclude"
        spent = _pinfo("spent", keys="k1", probe=probe)
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
            )
        )
        ctx = self._chain_ctx(
            {"spent": spent, "unmeasured": _pinfo("unmeasured")},
            ["spent", "unmeasured"],
        )

        picks = {"spent": 0, "unmeasured": 0}
        for _ in range(40):
            ctx.chosen = None
            self._chain.run(ctx)
            assert ctx.chosen is not None
            picks[ctx.chosen.name] += 1

        assert picks == {"spent": 0, "unmeasured": 40}
        assert ctx.quota_excluded == frozenset({"spent"})

    def test_the_survivors_split_evenly(self):
        """An excluded provider must not consume turns of the round-robin.

        Taking its pick and falling through would hand extra turns to
        whichever survivor comes first in route order.
        """
        from llm_rosetta_gateway.quota.models import credential_fingerprint

        probe = "p-chain-skew"
        spent = _pinfo("spent", keys="k1", probe=probe)
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
            )
        )
        ctx = self._chain_ctx(
            {"spent": spent, "a": _pinfo("a"), "b": _pinfo("b")},
            ["spent", "a", "b"],
        )

        picks = {"a": 0, "b": 0}
        for _ in range(60):
            ctx.chosen = None
            self._chain.run(ctx)
            assert ctx.chosen is not None
            picks[ctx.chosen.name] += 1

        assert picks == {"a": 30, "b": 30}


class TestEveryProviderSpent:
    """The client is told "out of allowance", not "initializing".

    A reading knows when its window resets — the same fact #1188 made
    ``Retry-After`` carry for provider blocks.  Saying "initializing" sends a
    client looking for a startup problem that does not exist.
    """

    def _config(self, *, reset_in: int | None) -> Any:
        from datetime import datetime, timedelta, timezone

        from llm_rosetta_gateway.config import GatewayConfig
        from llm_rosetta_gateway.quota.models import credential_fingerprint

        config = GatewayConfig(
            {
                "providers": {
                    "a": {
                        "type": "chat_completions",
                        "api_key": "k",
                        "base_url": "https://example.test",
                    }
                },
                "models": {"m": "a"},
            }
        )
        pinfo = config.providers["a"]
        pinfo.quota_probe = "p-all-spent"
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe="p-all-spent",
                credential=credential_fingerprint("k"),
                windows=[
                    QuotaWindow(
                        name="5h",
                        used_percent=100,
                        resets_at=(
                            None
                            if reset_in is None
                            else datetime.now(timezone.utc)
                            + timedelta(seconds=reset_in)
                        ),
                    )
                ],
            )
        )
        return config

    def test_the_answered_reset_is_the_retry_after(self):
        from llm_rosetta_gateway._vendor.httpserver import Response
        from llm_rosetta_gateway.app.routes_llm import _resolve_or_error

        config = self._config(reset_in=600)
        selection = config.selection_context("m", "chat_completions")

        resp = _resolve_or_error(config, selection, "req-1")

        assert isinstance(resp, Response)
        assert resp.status_code == 503
        assert 540 <= int(resp.headers["Retry-After"]) <= 600
        assert b"spent their allowance" in resp.body
        assert b"initializing" not in resp.body

    def test_an_unknown_reset_is_not_invented(self):
        from llm_rosetta_gateway._vendor.httpserver import Response
        from llm_rosetta_gateway.app.routes_llm import _resolve_or_error

        config = self._config(reset_in=None)
        selection = config.selection_context("m", "chat_completions")

        resp = _resolve_or_error(config, selection, "req-2")

        assert isinstance(resp, Response)
        assert resp.status_code == 503
        assert b"no reset time is known" in resp.body
        # The startup-shaped default still applies, but the reason does not.
        assert resp.headers["Retry-After"] == "5"


class TestTheRealignment:
    """`prefer_same_format` must not undo what the quota stage decided.

    The tiers pick among the providers that already speak the client's format,
    which is a second selection — and it read neither headroom nor exclusions
    until #1194/#1196.  A spent provider taking the realigned request is a
    silent failure: the status is still 200 as long as the other one serves.
    """

    @staticmethod
    def _config(*, readings: dict[str, float]) -> Any:
        from llm_rosetta_gateway.config import GatewayConfig
        from llm_rosetta_gateway.quota.models import credential_fingerprint

        config = GatewayConfig(
            {
                "providers": {
                    "a": {
                        "type": "messages",
                        "api_key": "sk-a",
                        "base_url": "https://a.test",
                    },
                    "b": {
                        "type": "chat_completions",
                        "api_key": "sk-b",
                        "base_url": "https://b.test",
                    },
                    "c": {
                        "type": "chat_completions",
                        "api_key": "sk-c",
                        "base_url": "https://c.test",
                    },
                },
                "models": {"m": {"providers": ["a", "b", "c"]}},
                "server": {"prefer_same_format": True},
            }
        )
        for name, used in readings.items():
            pinfo = config.providers[name]
            pinfo.quota_probe = "p-real"
            QUOTA_STATE.put_reading(
                QuotaReading(
                    probe="p-real",
                    credential=credential_fingerprint(f"sk-{name}"),
                    windows=[QuotaWindow(name="5h", used_percent=used)],
                )
            )
        return config

    @staticmethod
    def _served(config: Any, rounds: int = 24) -> dict[str, int]:
        served: dict[str, int] = {}
        for _ in range(rounds):
            selection = config.selection_context("m", "chat_completions")
            config.resolve(selection)
            assert selection.chosen is not None
            served[selection.chosen.name] = served.get(selection.chosen.name, 0) + 1
        return served

    def test_a_spent_same_format_provider_is_never_realigned_onto(self):
        config = self._config(readings={"b": 100})

        served = self._served(config)

        assert "b" not in served, served
        assert served == {"c": 24}

    def test_the_nearly_spent_sibling_gets_less_of_the_realigned_traffic(self):
        config = self._config(readings={"b": 96, "c": 10})

        served = self._served(config, rounds=40)

        assert served.get("c", 0) > served.get("b", 0) * 2, served

    def test_a_failover_excluded_provider_is_not_realigned_onto(self):
        """The provider that just refused must not come back through a tier.

        Only ``b`` is excluded, so the anchor still lands on ``a`` half the
        time — and from there the realignment is the only thing that could
        hand the request back to ``b``.
        """
        config = self._config(readings={})

        served: dict[str, int] = {}
        for _ in range(24):
            selection = config.selection_context("m", "chat_completions")
            selection.with_failover_excluded({"b"})
            config.resolve(selection)
            assert selection.chosen is not None
            served[selection.chosen.name] = served.get(selection.chosen.name, 0) + 1

        assert "b" not in served, served
        assert served.get("c", 0) > 0, served

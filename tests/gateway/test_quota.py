"""Tests for the quota probe layer, its state, and key-ring demotion."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.proxy import merge_chat_completion_events
from llm_rosetta_gateway.quota.models import (
    QuotaReading,
    QuotaWindow,
    credential_fingerprint,
)
from llm_rosetta_gateway.quota.probes import resolve_probe
from llm_rosetta_gateway.quota.state import QuotaState
from llm_rosetta_gateway.transport.provider_info import KeyRing
from llm_rosetta.shims.providers.openai.quota import OpenAICodexProbe
from llm_rosetta.shims.providers.opencode.quota import OpenCodeGoProbe

# The probes ship with the shims; a test resolves against them directly.
_PROBES = (OpenCodeGoProbe(), OpenAICodexProbe())


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The state is process-wide; no test may inherit another's demotions."""
    from llm_rosetta_gateway.quota.state import QUOTA_STATE

    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_fingerprint_is_stable_and_not_the_key(self):
        assert credential_fingerprint("sk-secret") == credential_fingerprint(
            "sk-secret"
        )
        assert credential_fingerprint("sk-secret") != credential_fingerprint("sk-other")
        assert "sk-secret" not in credential_fingerprint("sk-secret")
        assert len(credential_fingerprint("sk-secret")) == 16

    def test_window_exhaustion_from_status(self):
        window = QuotaWindow(name="5 hours", used_percent=12, status="rate-limited")
        assert window.exhausted

    def test_window_exhaustion_from_percent(self):
        assert QuotaWindow(name="Week", used_percent=100).exhausted
        assert not QuotaWindow(name="Week", used_percent=99.9).exhausted

    def test_window_reset_in_the_past_clears_exhaustion(self):
        window = QuotaWindow(
            name="Week",
            used_percent=100,
            resets_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        assert not window.exhausted

    def test_reading_reports_the_last_blocking_reset(self):
        reading = QuotaReading(
            probe="p",
            credential="c",
            windows=[
                QuotaWindow(name="5h", used_percent=100, resets_at=_utc(2099, 1, 1)),
                QuotaWindow(name="week", used_percent=100, resets_at=_utc(2099, 1, 8)),
                QuotaWindow(name="month", used_percent=10, resets_at=_utc(2099, 2, 1)),
            ],
        )
        assert reading.exhausted_until == _utc(2099, 1, 8)

    def test_demote_at_lowers_the_threshold(self):
        reading = QuotaReading(
            probe="p",
            credential="c",
            demote_at=90.0,
            windows=[
                QuotaWindow(
                    name="week",
                    used_percent=95,
                    resets_at=datetime.now(timezone.utc) + timedelta(days=1),
                )
            ],
        )
        assert reading.exhausted_until is not None

    def test_credits_override_demote_at(self):
        reading = QuotaReading(
            probe="p",
            credential="c",
            demote_at=90.0,
            windows=[
                QuotaWindow(
                    name="week",
                    used_percent=100,
                    status="credits",
                    resets_at=datetime.now(timezone.utc) + timedelta(days=1),
                )
            ],
        )
        assert reading.exhausted_until is None

    def test_unknown_reset_suppresses_the_demotion_until(self):
        reading = QuotaReading(
            probe="p",
            credential="c",
            windows=[QuotaWindow(name="week", used_percent=100)],
        )
        assert reading.exhausted_until is None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class TestOpenCodeGoProbe:
    probe = OpenCodeGoProbe()

    def test_matches_the_go_lane(self):
        assert (
            self.probe.match("https://opencode.ai/zen/go/v1")
            == "https://opencode.ai/zen/go/v1/usage"
        )

    def test_reports_the_zen_lane_as_unsupported(self):
        assert self.probe.match("https://opencode.ai/zen/v1") == ""
        match = resolve_probe("https://opencode.ai/zen/v1", probes=_PROBES)
        assert match is not None and match.url is None
        assert "no keyed usage endpoint" in (match.unsupported or "")

    def test_ignores_other_vendors(self):
        assert self.probe.match("https://api.example.com/v1") is None

    def test_parses_percent_as_used(self):
        plan, windows = self.probe.parse(
            {
                "usage": {
                    "rolling": {
                        "status": "ok",
                        "percent": 12,
                        "resetsAt": "2026-10-19T22:22:28.331Z",
                    },
                    "weekly": {"status": "ok", "percent": 94},
                    "monthly": {
                        "status": "rate-limited",
                        "percent": 100,
                        "resetsAt": "2026-11-11T22:26:38.000Z",
                    },
                }
            }
        )
        assert plan is None
        assert [w.name for w in windows] == ["5 hours", "7 days", "Month"]
        assert windows[0].used_percent == 12
        assert windows[0].span_seconds == 5 * 3600
        assert windows[2].exhausted

    def test_malformed_answer_is_an_error(self):
        import pytest

        from llm_rosetta_gateway.quota.probes import QuotaProbeError

        with pytest.raises(QuotaProbeError):
            self.probe.parse({"usage": {}})

    def test_error_wording(self):
        assert "does not authenticate" in self.probe.describe_error(401, {})
        assert "no Go subscription" in self.probe.describe_error(403, {})

    def test_inband_429_names_the_window(self):
        signal = self.probe.inband(
            429, {"error": {"message": "limit"}, "metadata": {"limitName": "weekly"}}
        )
        assert signal is not None
        assert signal.window == "weekly"
        assert signal.cooldown_seconds == 7 * 24 * 3600

    def test_inband_ignores_other_statuses(self):
        assert self.probe.inband(500, {}) is None


class TestOpenAICodexProbe:
    probe = OpenAICodexProbe()

    @staticmethod
    def _token(account_id: str) -> str:
        claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
        return f"header.{payload.rstrip('=')}.signature"

    def test_reads_the_account_id_from_the_token(self):
        headers = self.probe.headers(self._token("acct-1"))
        assert headers == {"chatgpt-account-id": "acct-1"}

    def test_missing_claim_yields_no_header(self):
        assert self.probe.headers("not-a-jwt") == {}

    def test_matches_the_codex_backend(self):
        assert (
            self.probe.match("https://chatgpt.com/backend-api/codex")
            == "https://chatgpt.com/backend-api/wham/usage"
        )

    def test_parses_windows(self):
        plan, windows = self.probe.parse(
            {
                "plan_type": "plus",
                "rate_limit": {
                    "allowed": True,
                    "primary_window": {
                        "used_percent": 37,
                        "limit_window_seconds": 18000,
                        "reset_after_seconds": 600,
                    },
                    "secondary_window": {
                        "used_percent": 80,
                        "limit_window_seconds": 604800,
                        "reset_at": 1791072000,
                    },
                },
            }
        )
        assert plan == "plus"
        assert [w.name for w in windows] == ["5 hours", "7 days"]
        assert windows[0].used_percent == 37
        assert windows[1].resets_at is not None

    def test_allowed_false_blocks_even_below_100(self):
        _, windows = self.probe.parse(
            {
                "rate_limit": {
                    "allowed": False,
                    "primary_window": {
                        "used_percent": 96,
                        "limit_window_seconds": 18000,
                    },
                }
            }
        )
        assert windows[0].exhausted

    def test_credits_keep_the_account_usable(self):
        _, windows = self.probe.parse(
            {
                "credits": {"has_credits": True, "unlimited": False, "balance": "12"},
                "rate_limit": {
                    "allowed": False,
                    "primary_window": {
                        "used_percent": 100,
                        "limit_window_seconds": 18000,
                    },
                },
            }
        )
        assert not windows[0].exhausted


class TestProbeResolution:
    def test_explicit_probe_wins(self):
        match = resolve_probe(
            "https://proxy.example.com/v1",
            probes=_PROBES,
            probe_id="openai_codex",
            url="https://proxy.example.com/usage",
        )
        assert match is not None
        assert match.probe.id == "openai_codex"
        assert match.url == "https://proxy.example.com/usage"

    def test_explicit_codex_probe_falls_back_to_its_endpoint(self):
        match = resolve_probe(
            "https://api.openai.com/v1", probes=_PROBES, probe_id="openai_codex"
        )
        assert match is not None
        assert match.url == "https://chatgpt.com/backend-api/wham/usage"

    def test_unknown_probe_is_an_error(self):
        import pytest

        with pytest.raises(ValueError):
            resolve_probe("https://api.example.com", probes=_PROBES, probe_id="nope")

    def test_shims_attach_their_probes(self):
        from llm_rosetta.shims import _load_providers, get_shim

        if get_shim("opencode--chat_completions") is None:
            _load_providers()
        opencode = get_shim("opencode--chat_completions")
        openai = get_shim("openai--open_responses")
        assert opencode is not None and openai is not None
        assert [p.id for p in opencode.quota_probes] == ["opencode_go"]
        assert [p.id for p in openai.quota_probes] == ["openai_codex"]

    def test_the_pivot_holds_no_vendor_probe(self):
        import llm_rosetta_gateway.quota.probes as pivot

        assert not hasattr(pivot, "OpenCodeGoProbe")
        assert not hasattr(pivot, "OpenAICodexProbe")


# ---------------------------------------------------------------------------
# State and rotation
# ---------------------------------------------------------------------------


class TestQuotaState:
    def test_demotion_expires(self):
        state = QuotaState()
        state.demote("p", "c", datetime.now(timezone.utc) - timedelta(seconds=1), "old")
        assert state.demotion("p", "c") is None

    def test_is_usable_follows_demotion(self):
        state = QuotaState()
        key = "sk-one"
        fp = credential_fingerprint(key)
        assert state.is_usable("p", key)
        state.demote_for("p", fp, 60, "spent")
        assert not state.is_usable("p", key)

    def test_is_usable_follows_an_exhausted_reading(self):
        state = QuotaState()
        key = "sk-two"
        reading = QuotaReading(
            probe="p",
            credential=credential_fingerprint(key),
            windows=[
                QuotaWindow(
                    name="week",
                    used_percent=100,
                    resets_at=datetime.now(timezone.utc) + timedelta(days=1),
                )
            ],
        )
        state.put_reading(reading)
        assert not state.is_usable("p", key)

    def test_a_failed_reading_is_not_a_demotion(self):
        state = QuotaState()
        key = "sk-three"
        state.put_reading(
            QuotaReading(
                probe="p",
                credential=credential_fingerprint(key),
                status="error",
                error="timeout",
            )
        )
        assert state.is_usable("p", key)

    def test_snapshot_never_contains_the_key(self):
        state = QuotaState()
        key = "sk-very-secret"
        state.demote_for("p", credential_fingerprint(key), 60, "spent")
        snapshot = state.snapshot("p", [key])
        assert key not in json.dumps(snapshot)
        assert snapshot[0]["credential"] == credential_fingerprint(key)


class TestQuotaIntervals:
    def test_reads_the_shim_declaration(self):
        from types import SimpleNamespace

        from llm_rosetta_gateway.quota.service import quota_intervals
        from llm_rosetta.shims import _load_providers, get_shim

        # Other test modules reset the shim registry; load the built-ins so
        # this test reads the declaration it is about.
        if get_shim("opencode--chat_completions") is None:
            _load_providers()

        config = SimpleNamespace(
            providers={"oc": object(), "plain": object()},
            _raw_providers={
                "oc": {"type": "opencode--chat_completions"},
                "plain": {"type": "openai--chat_completions"},
            },
            provider_shim_names={
                "oc": "opencode--chat_completions",
                "plain": "openai--chat_completions",
            },
        )
        assert quota_intervals(config) == {"oc": 300}

    def test_provider_config_overrides_the_interval(self):
        from types import SimpleNamespace

        from llm_rosetta_gateway.quota.service import quota_intervals
        from llm_rosetta.shims import _load_providers, get_shim

        if get_shim("opencode--chat_completions") is None:
            _load_providers()

        config = SimpleNamespace(
            providers={"oc": object()},
            _raw_providers={
                "oc": {"type": "opencode--chat_completions", "quota": {"interval": 60}}
            },
            provider_shim_names={"oc": "opencode--chat_completions"},
        )
        assert quota_intervals(config) == {"oc": 60}


class TestInBandDemotion:
    def test_429_demotes_the_serving_key(self):
        from types import SimpleNamespace

        from llm_rosetta_gateway.quota.service import note_upstream_quota_error
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        key = "sk-inband-detour-1"
        pinfo = SimpleNamespace(
            name="p",
            keyless=False,
            quota_probe="opencode_go",
            quota_probe_impl=OpenCodeGoProbe(),
            quota_identity=None,
            token_command=None,
            mints_credential=False,
            active_key=lambda: key,
        )
        signal = note_upstream_quota_error(
            pinfo,
            429,
            json.dumps(
                {"error": {"message": "limit"}, "metadata": {"limitName": "weekly"}}
            ),
        )
        assert signal is not None and signal.window == "weekly"
        assert not QUOTA_STATE.is_usable("opencode_go", key)

    def test_rotating_token_demotes_the_provider(self):
        from types import SimpleNamespace

        from llm_rosetta_gateway.quota.service import note_upstream_quota_error
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        identity = "cmd:inband-rotating-token"
        pinfo = SimpleNamespace(
            name="p2",
            keyless=False,
            quota_probe="opencode_go",
            quota_probe_impl=OpenCodeGoProbe(),
            quota_identity=identity,
            token_command=["true"],
            mints_credential=True,
            active_key=lambda: "sk-rotating",
        )
        signal = note_upstream_quota_error(pinfo, 429, "{}")
        assert signal is not None
        assert QUOTA_STATE.provider_block(identity) is not None

    def test_keyless_429_cools_the_provider(self):
        from types import SimpleNamespace

        from llm_rosetta_gateway.quota.service import note_upstream_quota_error
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        identity = "@free-lane-test"
        pinfo = SimpleNamespace(
            name="free",
            keyless=True,
            quota_probe=None,
            quota_identity=identity,
            active_key=lambda: None,
        )
        note_upstream_quota_error(pinfo, 429, "{}")
        assert QUOTA_STATE.provider_block(identity) is not None


class TestKeyRingDemotion:
    def test_skips_a_demoted_key(self):
        ring = KeyRing("a,b,c", usable=lambda key: key != "b")
        assert [ring.next() for _ in range(4)] == ["a", "c", "a", "c"]

    def test_affinity_selection_skips_a_demoted_key(self):
        ring = KeyRing("a,b", usable=lambda key: key != "b")
        assert ring.select("identity") == "a"

    def test_all_demoted_falls_back_to_rotation(self):
        ring = KeyRing("a,b", usable=lambda key: False)
        assert [ring.next() for _ in range(3)] == ["a", "b", "a"]

    def test_keys_are_exposed_for_probing(self):
        assert KeyRing("a, b").keys() == ("a", "b")


# ---------------------------------------------------------------------------
# Stream reassembly
# ---------------------------------------------------------------------------


class TestMergeChatCompletionEvents:
    def test_merges_content_and_usage(self):
        merged = merge_chat_completion_events(
            [
                {
                    "id": "chatcmpl-1",
                    "created": 123,
                    "model": "m",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "he"}}
                    ],
                },
                {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "llo"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            ]
        )
        assert merged["object"] == "chat.completion"
        assert merged["choices"][0]["message"]["content"] == "hello"
        assert merged["choices"][0]["finish_reason"] == "stop"
        assert merged["usage"]["completion_tokens"] == 2

    def test_merges_tool_calls_by_index(self):
        merged = merge_chat_completion_events(
            [
                {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "read", "arguments": "{"},
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "id": "x",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"p":1}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ]
        )
        call = merged["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "read"
        assert call["function"]["arguments"] == '{"p":1}'


class TestDeferredQuotaPolling:
    """The poller is wired through the maintenance package, not the old path."""

    async def test_start_launches_the_poller_when_an_interval_is_declared(
        self, monkeypatch
    ):
        from llm_rosetta_gateway.maintenance.startup import DeferredStartup

        config = SimpleNamespace(providers={})
        app = SimpleNamespace(
            persistence=None, _counter_rebuild_needed=False, gateway_config=None
        )
        monkeypatch.setattr(
            "llm_rosetta_gateway.quota.service.quota_intervals",
            lambda _config: {"p": 60.0},
        )

        startup = DeferredStartup(config, app)
        await startup.start()

        assert "quota-poller" in [task.get_name() for task in startup._tasks]
        await startup.shutdown()

    async def test_start_skips_the_poller_without_an_interval(self, monkeypatch):
        from llm_rosetta_gateway.maintenance.startup import DeferredStartup

        config = SimpleNamespace(providers={})
        app = SimpleNamespace(
            persistence=None, _counter_rebuild_needed=False, gateway_config=None
        )
        monkeypatch.setattr(
            "llm_rosetta_gateway.quota.service.quota_intervals",
            lambda _config: {},
        )

        startup = DeferredStartup(config, app)
        await startup.start()

        assert "quota-poller" not in [task.get_name() for task in startup._tasks]
        await startup.shutdown()


class TestInBandOnlyLane:
    """A lane with no usage endpoint: the probe exists only for its classifier.

    The ChatGPT plan route is the case — the app can be told "this window is
    spent" and can never ask "how much is left".
    """

    def _config(self, monkeypatch):
        from llm_rosetta_gateway.quota.service import resolve_provider_probe_for_info
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        pinfo = SimpleNamespace(
            name="openai_chatgpt--open_responses",
            keyless=False,
            base_url="https://api.openai.com/v1",
            quota_probes=(OpenAICodexProbe(),),
            quota_probe_id="openai_codex",
            quota_probe="openai_codex",
            quota_url=None,
            quota_usage_url=None,
            quota_in_band=True,
            quota_unsupported=None,
            key_ring=KeyRing("plan-token"),
        )
        return resolve_provider_probe_for_info("chatgpt", pinfo), pinfo

    def test_the_probe_resolves_through_the_real_loader(self, tmp_path):
        """The hand-built provider in the other tests supplies the probe the
        loader might not — this one goes through `load_gateway_config`."""
        from llm_rosetta_gateway.config import load_gateway_config
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        config_path = tmp_path / "config.jsonc"
        config_path.write_text(
            json.dumps(
                {
                    "providers": {
                        "chatgpt": {
                            "type": "openai_chatgpt--open_responses",
                            "base_url": "https://api.openai.com/v1",
                        }
                    },
                    "models": {},
                }
            )
        )

        config = load_gateway_config(str(config_path))
        pinfo = config.providers["chatgpt"]

        assert pinfo.quota_probe_impl is not None
        assert pinfo.quota_in_band is True
        assert pinfo.quota_usage_url is None
        assert pinfo.quota_unsupported and "in-band" in pinfo.quota_unsupported

    def test_the_plan_shim_declares_it_in_band(self):
        from llm_rosetta.shims.providers import load_providers

        shim = {s.name: s for s in load_providers()}["openai_chatgpt--open_responses"]

        assert shim.quota is not None
        assert shim.quota.probe == "openai_codex"
        assert shim.quota.in_band is True
        assert shim.quota.url is None

    def test_resolution_keeps_the_probe_but_offers_no_url(self):
        (probe, url, reason), _ = self._config(None)

        assert url is None
        assert probe is not None
        assert "in-band" in (reason or "")

    def test_the_probe_stays_bound_for_the_classifier(self):
        """Without the binding, `note_upstream_quota_error` returns early and
        the lane never learns its window is spent."""
        _, pinfo = self._config(None)

        assert pinfo.quota_probe_impl is not None
        assert pinfo.quota_usage_url is None

    def test_a_429_demotes_the_plan_credential(self):
        from llm_rosetta_gateway.quota.service import note_upstream_quota_error
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        _, pinfo = self._config(None)
        identity = "acct:inband-only-plan"
        pinfo.quota_identity = identity
        pinfo.mints_credential = True
        pinfo.active_key = lambda: "plan-token"

        signal = note_upstream_quota_error(
            pinfo,
            429,
            json.dumps(
                {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}
            ),
        )

        assert signal is not None
        block = QUOTA_STATE.provider_block(identity)
        assert block is not None
        # Bounded and short: the vendor says nothing about a reset, so the
        # gateway must not invent one.
        left = (block.until - datetime.now(timezone.utc)).total_seconds()
        assert 0 < left <= 61

    def test_a_blocked_plan_provider_is_not_ready(self):
        """The block is the only remedy this lane has — nothing polls it.

        A minting provider cannot carry a per-key demotion (a rotating token
        is a new fingerprint every refresh), so if ``ready`` ignored the
        provider block the refusal would be recorded and the next request
        would go back to the same exhausted account.
        """
        from llm_rosetta_gateway.quota.service import bind_provider_quota
        from llm_rosetta_gateway.quota.state import QUOTA_STATE
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo

        pinfo = ProviderInfo(
            "openai_chatgpt--open_responses",
            api_key="plan-token",
            base_url="https://api.openai.com/v1",
            auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
            url_template="{base_url}/responses",
            # A minting provider: the credential comes from the gateway, so a
            # per-key demotion cannot survive a refresh.
            token_command=["true"],
        )
        pinfo.quota_probes = (OpenAICodexProbe(),)
        pinfo.quota_probe_id = "openai_codex"
        pinfo.quota_in_band = True
        pinfo.credential_token = "acct:inband-ready-test"
        bind_provider_quota("chatgpt", pinfo)

        assert pinfo.ready is True

        QUOTA_STATE.block_provider(
            "acct:inband-ready-test", 60, "ChatGPT plan usage limit exceeded"
        )

        assert pinfo.ready is False
        # And the admin surface can say why.
        from llm_rosetta_gateway.quota.service import provider_cooldown

        cooldown = provider_cooldown("chatgpt", pinfo)
        assert cooldown is not None
        assert "plan usage" in cooldown["reason"]

    def test_the_panel_reports_no_endpoint_rather_than_an_error(self):
        import asyncio

        from llm_rosetta_gateway.quota.service import read_provider_quota

        _, pinfo = self._config(None)
        config = SimpleNamespace(providers={"chatgpt": pinfo})

        payload = asyncio.run(read_provider_quota(config, "chatgpt"))

        assert payload["supported"] is False
        assert "in-band" in payload["reason"]
        # The router's own verdict travels with the reading, so the card
        # describes the decision instead of re-deriving it.
        assert payload["routing"]["excluded"] is False
        assert payload["routing"]["headroom"] is None


class TestTheRoutingFacts:
    """What the panel shows is what selection read."""

    @staticmethod
    def _pinfo(keys: str = "k1") -> object:
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo

        pinfo = ProviderInfo(
            "openai--chat_completions",
            api_key=keys,
            base_url="https://api.example.test/v1",
            auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
            url_template="{base_url}/chat/completions",
        )
        pinfo.quota_probe = "p-panel-facts"
        return pinfo

    def test_a_spent_provider_is_reported_with_its_reason(self):
        from llm_rosetta_gateway.quota.service import routing_facts
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        pinfo = self._pinfo()
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe="p-panel-facts",
                credential=credential_fingerprint("k1"),
                windows=[QuotaWindow(name="5h", used_percent=100)],
            )
        )

        facts = routing_facts(pinfo)

        assert facts["excluded"] is True
        assert facts["headroom"] == 0.0
        assert "spent" in facts["reason"]

    def test_an_unmeasured_provider_claims_nothing(self):
        from llm_rosetta_gateway.quota.service import routing_facts

        facts = routing_facts(self._pinfo())

        assert facts == {
            "headroom": None,
            "weight": 1.0,
            "excluded": False,
            "blocked": False,
            "reason": None,
        }

    def test_a_mixed_ring_reads_zero_but_serves_a_quarter(self):
        """One spent key beside a healthy one is not an exclusion.

        The card used to print "0% left" for a provider the router was still
        serving: the headroom is the tightest credential, the weight is what
        the router actually applies.
        """
        from llm_rosetta_gateway.quota.service import routing_facts
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        pinfo = self._pinfo(keys="k1,k2")
        for key, used in (("k1", 100), ("k2", 15)):
            QUOTA_STATE.put_reading(
                QuotaReading(
                    probe="p-panel-facts",
                    credential=credential_fingerprint(key),
                    windows=[QuotaWindow(name="5h", used_percent=used)],
                )
            )

        facts = routing_facts(pinfo)

        assert facts["headroom"] == 0.0
        assert facts["excluded"] is False
        assert facts["weight"] == 0.25

    def test_a_live_block_wins_over_a_healthy_reading(self):
        """A lane cooling down is skipped, whatever its windows say."""
        from llm_rosetta_gateway.quota.service import (
            bind_provider_quota,
            routing_facts,
        )
        from llm_rosetta_gateway.quota.state import QUOTA_STATE
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo

        pinfo = ProviderInfo(
            "openai_chatgpt--open_responses",
            api_key="plan-token",
            base_url="https://api.openai.com/v1",
            auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
            url_template="{base_url}/responses",
            token_command=["true"],
        )
        pinfo.quota_probes = (OpenAICodexProbe(),)
        pinfo.quota_probe_id = "openai_codex"
        pinfo.quota_in_band = True
        pinfo.credential_token = "acct:blocked-panel-test"
        bind_provider_quota("chatgpt", pinfo)
        assert pinfo.quota_probe is not None
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=pinfo.quota_probe,
                credential=credential_fingerprint("plan-token"),
                windows=[QuotaWindow(name="5h", used_percent=10)],
            )
        )
        assert routing_facts(pinfo)["blocked"] is False

        QUOTA_STATE.block_provider(
            "acct:blocked-panel-test", 60, "keyless lane rate limited"
        )

        facts = routing_facts(pinfo)
        assert facts["blocked"] is True
        assert facts["excluded"] is False
        assert facts["headroom"] == 0.9

    def test_a_blocked_lane_shows_its_cooldown_even_with_nothing_to_read(self):
        import asyncio

        from llm_rosetta_gateway.quota.service import (
            bind_provider_quota,
            read_provider_quota,
        )
        from llm_rosetta_gateway.quota.state import QUOTA_STATE
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo

        pinfo = ProviderInfo(
            "openai_chatgpt--open_responses",
            api_key="plan-token",
            base_url="https://api.openai.com/v1",
            auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
            url_template="{base_url}/responses",
            token_command=["true"],
        )
        pinfo.quota_probes = (OpenAICodexProbe(),)
        pinfo.quota_probe_id = "openai_codex"
        pinfo.quota_in_band = True
        pinfo.credential_token = "acct:inband-panel-test"
        bind_provider_quota("chatgpt", pinfo)
        QUOTA_STATE.block_provider("acct:inband-panel-test", 60, "usage limit")

        payload = asyncio.run(
            read_provider_quota(
                SimpleNamespace(providers={"chatgpt": pinfo}), "chatgpt"
            )
        )

        assert payload["supported"] is False
        assert payload["provider_block"]["reason"] == "usage limit"


class TestTheSupportedPayload:
    """The payload a probe-capable provider's card reads must carry the verdict.

    The in-band-only branch is covered above; this is the other shape, and a
    regression here would silently drop the row.
    """

    def test_the_verdict_travels_with_a_readable_quota(self, monkeypatch):
        import asyncio

        from llm_rosetta_gateway.quota import service as quota_service
        from llm_rosetta_gateway.quota.state import QUOTA_STATE
        from llm_rosetta_gateway.transport.provider_info import ProviderInfo

        pinfo = ProviderInfo(
            "openai--chat_completions",
            api_key="sk-live",
            base_url="https://api.example.test/v1",
            auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
            url_template="{base_url}/chat/completions",
        )
        pinfo.quota_probe = "p-supported"
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe="p-supported",
                credential=credential_fingerprint("sk-live"),
                windows=[QuotaWindow(name="5h", used_percent=62)],
            )
        )

        class _Probe:
            id = "p-supported"

        monkeypatch.setattr(
            quota_service,
            "resolve_provider_probe",
            lambda config, name: (_Probe(), "https://api.example.test/usage", None),
        )

        payload = asyncio.run(
            quota_service.read_provider_quota(
                SimpleNamespace(providers={"p": pinfo}), "p"
            )
        )

        assert payload["supported"] is True
        assert payload["routing"]["headroom"] == pytest.approx(0.38)
        assert payload["routing"]["weight"] == pytest.approx(0.25 + 0.75 * 0.38)
        assert payload["routing"]["excluded"] is False
        assert payload["routing"]["blocked"] is False
        assert "provider_block" not in payload

        # A readable lane can still be cooling down, and then the card must
        # show the block rather than the healthy reading.
        QUOTA_STATE.block_provider("@p", 60, "keyless lane rate limited")

        blocked = asyncio.run(
            quota_service.read_provider_quota(
                SimpleNamespace(providers={"p": pinfo}), "p"
            )
        )

        assert blocked["provider_block"]["reason"] == "keyless lane rate limited"
        assert blocked["routing"]["blocked"] is True

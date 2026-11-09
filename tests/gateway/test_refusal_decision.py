"""One decision for a refused upstream attempt.

Both request handlers send differently — one a JSON request, one an SSE
stream — but a refusal means the same thing to both: answer the client, try
the same upstream again, or hand the request to another provider.  These
tests pin that decision on its own, without either transport.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_rosetta_gateway.proxy import errors
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.proxy.errors import (
    QuotaRefusal,
    RefusalAction,
    decide_on_refusal,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Every refusal test starts from an empty rate-limit state."""
    QUOTA_STATE.rate_limits.clear()
    yield
    QUOTA_STATE.rate_limits.clear()


class _Stub:
    """Whatever the patched helpers hand back; the decision passes it through."""


def _decide(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 429,
    auth: tuple[bool, Any] = (False, None),
    quota: tuple[QuotaRefusal | None, Any] = (None, None),
    lane: str = "lane",
) -> tuple[Any, list[bool]]:
    """Run the decision with both underlying helpers replaced.

    Returns:
        The decision, and a one-element list recording whether the quota
        classifier was consulted.
    """
    consulted: list[bool] = []

    async def fake_401(provider_info, status_code, error_text, source_provider):
        return auth

    def fake_quota(provider_info, status_code, error_text):
        consulted.append(True)
        return quota

    monkeypatch.setattr(errors, "_handle_401", fake_401)
    monkeypatch.setattr(errors, "_quota_refusal", fake_quota)
    # A keyless provider keeps the continuous-429 counter out of the test:
    # the refusals are counted against the provider, not a credential.
    provider = SimpleNamespace(name="p", config_name="p", keyless=True)
    decision = asyncio.run(
        decide_on_refusal(
            cast(Any, provider),
            status,
            "upstream said no",
            "chat_completions",
            lane=lane,
        )
    )
    return decision, consulted


def test_a_session_policy_answer_is_returned(monkeypatch):
    """A 401 the gateway cannot fix answers the client, before any retry."""
    response = _Stub()

    decision, consulted = _decide(monkeypatch, auth=(False, response))

    assert decision.action is RefusalAction.RETURN
    assert decision.response is response
    assert consulted == []


def test_a_refreshed_credential_is_retried_without_asking_quota(monkeypatch):
    """A 401 a refresh fixed is a retry, not a quota refusal.

    The quota classifier would read the same status and could call it
    exhausted; the order in ``decide_on_refusal`` is what stops that.
    """
    decision, consulted = _decide(
        monkeypatch, auth=(True, None), quota=(QuotaRefusal.EXHAUSTED, _Stub())
    )

    assert decision.action is RefusalAction.RETRY
    assert consulted == []


def test_a_quota_refusal_with_a_usable_key_is_retried(monkeypatch):
    decision, _ = _decide(monkeypatch, status=429, quota=(QuotaRefusal.RETRY, _Stub()))

    assert decision.action is RefusalAction.RETRY


def test_an_exhausted_provider_fails_over(monkeypatch):
    signal = _Stub()

    decision, _ = _decide(
        monkeypatch, status=429, quota=(QuotaRefusal.EXHAUSTED, signal)
    )

    assert decision.action is RefusalAction.FAILOVER
    assert decision.signal is signal


@pytest.mark.parametrize(
    "status", (408, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)
)
def test_a_transient_upstream_failure_fails_over(monkeypatch, status):
    """A 5xx or timeout says "not now": try another provider.

    The quota classifier is not consulted — demoting a key would blame it for
    the provider's outage.
    """
    decision, consulted = _decide(monkeypatch, status=status)

    assert decision.action is RefusalAction.FAILOVER
    assert decision.signal.status_code == status
    assert consulted == []


def test_an_unexplained_402_fails_over(monkeypatch):
    """No probe, no signal: the status alone is still worth another provider.

    A bare 429 is the next test: the continuous-429 rule counts it and hands
    back an in-band signal rather than the ``UpstreamRefusal`` a 402 gets.
    """
    decision, _ = _decide(monkeypatch, status=402)

    assert decision.action is RefusalAction.FAILOVER
    assert decision.signal.status_code == 402


def _record_ladder(monkeypatch) -> list[tuple[str, int, str | None]]:
    """Replace both cross-request ladders with recorders.

    ``("lane", status, upstream)`` is the capacity ladder; ``("credential",
    status, None)`` is the provider-wide one the ``402``/``429`` fallback uses.
    """
    calls: list[tuple[str, int, str | None]] = []
    monkeypatch.setattr(
        "llm_rosetta_gateway.quota.service.note_lane_refusal",
        lambda pinfo, status, policy=None, *, lane: calls.append(
            ("lane", status, lane)
        ),
    )
    monkeypatch.setattr(
        "llm_rosetta_gateway.quota.service.note_transient_refusal",
        lambda pinfo, status, policy=None: calls.append(("credential", status, None)),
    )
    return calls


@pytest.mark.parametrize("status", (503, 529, 520, 524))
def test_a_capacity_refusal_counts_toward_the_ladder(monkeypatch, status):
    """The per-request failover and the cross-request ladder both run."""
    calls = _record_ladder(monkeypatch)

    decision, _ = _decide(monkeypatch, status=status)

    assert decision.action is RefusalAction.FAILOVER
    assert calls == [("lane", status, "lane")]


@pytest.mark.parametrize("status", (503, 529, 520, 524))
def test_a_capacity_refusal_is_counted_against_its_lane(monkeypatch, status):
    """The count is the upstream model's, not the provider's (#1452)."""
    calls = _record_ladder(monkeypatch)

    _decide(monkeypatch, status=status, lane="fast")

    assert calls == [("lane", status, "fast")]


def test_an_unexplained_402_counts_against_the_credential(monkeypatch):
    """A ``402`` no probe explained is the credential's state, not a lane's."""
    calls = _record_ladder(monkeypatch)

    decision, _ = _decide(monkeypatch, status=402, lane="fast")

    assert decision.action is RefusalAction.FAILOVER
    assert calls == [("credential", 402, None)]


@pytest.mark.parametrize("status", (408, 500, 502, 504))
def test_a_request_specific_failure_does_not_count(monkeypatch, status):
    """These fail over, but one must not take a provider out of rotation."""
    calls = _record_ladder(monkeypatch)

    decision, _ = _decide(monkeypatch, status=status)

    assert decision.action is RefusalAction.FAILOVER
    assert calls == []


def test_an_unexplained_429_is_not_counted_twice(monkeypatch):
    """The continuous-429 rule counts it above; the capacity ladder stays out."""
    calls = _record_ladder(monkeypatch)

    decision, _ = _decide(monkeypatch, status=429)

    assert decision.action is RefusalAction.FAILOVER
    assert calls == []


def test_a_probe_explained_429_uses_the_demotion_path(monkeypatch):
    """A quota signal demotes the credential; the ladder is not consulted."""
    calls = _record_ladder(monkeypatch)

    decision, _ = _decide(
        monkeypatch, status=429, quota=(QuotaRefusal.EXHAUSTED, _Stub())
    )

    assert decision.action is RefusalAction.FAILOVER
    assert calls == []


@pytest.mark.parametrize("status", (400, 403, 404, 413, 422))
def test_a_client_error_breaks(monkeypatch, status):
    """The request itself is wrong; another provider would refuse it too."""
    decision, consulted = _decide(monkeypatch, status=status)

    assert decision.action is RefusalAction.BREAK
    assert consulted == [True]


def test_an_unlisted_status_breaks(monkeypatch):
    """Nothing to retry and nothing to fail over to: the upstream's answer."""
    decision, consulted = _decide(monkeypatch, status=418)

    assert decision.action is RefusalAction.BREAK
    assert consulted == [True]


def test_a_429_no_probe_can_classify_still_fails_over(monkeypatch):
    """A lane with no probe answered 429; it is still rate limited.

    Before this, an unclassifiable 429 fell through to BREAK: the request was
    answered with the upstream's refusal even when another provider was ready,
    and the next request was retried on the same lane as if nothing happened.
    """
    decision, consulted = _decide(monkeypatch)

    assert decision.action is RefusalAction.FAILOVER
    assert decision.signal is not None
    assert "429" in decision.signal.reason
    assert consulted == [True]


def test_an_unexplained_429_counts_on_the_credential_lane(monkeypatch):
    """One refused key must sink, not the whole provider.

    The provider-wide lane alone would take every key of the provider out
    when one of several is rate limited; the per-credential count is what
    lets the key ring skip just that one.
    """
    from llm_rosetta_gateway.quota.models import credential_fingerprint

    async def fake_401(provider_info, status_code, error_text, source_provider):
        return False, None

    def fake_quota(provider_info, status_code, error_text):
        return None, None

    monkeypatch.setattr(errors, "_handle_401", fake_401)
    monkeypatch.setattr(errors, "_quota_refusal", fake_quota)
    provider = SimpleNamespace(
        name="p", config_name="p", keyless=False, active_key=lambda: "sk-1"
    )

    decision = asyncio.run(
        decide_on_refusal(
            cast(Any, provider),
            429,
            "upstream said no",
            "chat_completions",
            lane="m",
        )
    )

    assert decision.action is RefusalAction.FAILOVER
    lane = credential_fingerprint("sk-1")
    assert QUOTA_STATE.rate_limits.record("p", lane) is not None

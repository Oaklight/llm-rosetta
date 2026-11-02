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
            cast(Any, provider), status, "upstream said no", "chat_completions"
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
    decision, _ = _decide(monkeypatch, quota=(QuotaRefusal.RETRY, _Stub()))

    assert decision.action is RefusalAction.RETRY


def test_an_exhausted_provider_fails_over(monkeypatch):
    signal = _Stub()

    decision, _ = _decide(monkeypatch, quota=(QuotaRefusal.EXHAUSTED, signal))

    assert decision.action is RefusalAction.FAILOVER
    assert decision.signal is signal


def test_an_ordinary_error_breaks(monkeypatch):
    """Nothing to retry and nothing to fail over to: the upstream's answer."""
    decision, consulted = _decide(monkeypatch, status=500)

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

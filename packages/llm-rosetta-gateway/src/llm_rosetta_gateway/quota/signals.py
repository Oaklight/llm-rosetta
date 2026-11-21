"""In-band quota signals: what an upstream's own answers write into the quota
state.

A probe reads a vendor's usage endpoint on a schedule (:mod:`.service`); the
functions here record what a live request learned instead — a quota error, a
429 and its rate-limit window, a lane or transient refusal, and the success
that clears them — and answer when a provider blocked that way is back
(:func:`provider_offline_until`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from typing import Any

from ._common import (
    _iso,
    provider_config_key,
)
from .models import (
    InBandSignal,
    credential_fingerprint,
)
from .rate_limit import RateLimitPolicy
from .state import QUOTA_STATE

logger = logging.getLogger("llm-rosetta-gateway")


def note_upstream_quota_error(
    pinfo: Any,
    status_code: int,
    error_text: str,
) -> InBandSignal | None:
    """Demote the serving credential after an in-band quota refusal.

    Called from the proxy's upstream error path.  The signal parser belongs to
    the probe, so a vendor that names the exhausted window can be demoted
    without waiting for the next poll.

    Args:
        pinfo: The provider that served the request.
        status_code: Upstream HTTP status.
        error_text: Upstream error body.

    Returns:
        The signal that was applied, or ``None`` when the error was not a
        quota refusal (or there is no probe for this provider).
    """
    body: Any = error_text
    try:
        body = json.loads(error_text)
    except (TypeError, ValueError):
        pass

    # A keyless lane has no credential to demote.  Its refusal is counted by
    # the continuous-429 rule, exactly as a keyed lane's is, and the signal is
    # returned so the proxy still treats it as a quota-class refusal: there is
    # no key to demote, but there may be another provider to try.
    if getattr(pinfo, "keyless", False):
        if status_code == 429:
            from llm_rosetta.contracts.quota import InBandSignal

            return InBandSignal(cooldown_seconds=60, reason="keyless lane rate limited")
        return None

    probe = getattr(pinfo, "quota_probe_impl", None)
    probe_id = getattr(pinfo, "quota_probe", None)
    if probe is None or not probe_id:
        return None

    signal = probe.inband(status_code, body)
    if signal is None:
        return None
    if signal.permanent:
        # A balance the vendor said is spent is not a rate limit: parking it
        # on the 429 ladder is what revived an unpaid account every 20
        # minutes, at one full request upload per try.  The park has no
        # expiry; `read_provider_quota` lifts it when a reading reports the
        # account available, and so does a restart (the state is in memory by
        # decision — a judgement from one vendor's wording must not outlive
        # the process).
        identity = getattr(pinfo, "quota_identity", None) or (
            f"@{provider_config_key(pinfo)}"
        )
        QUOTA_STATE.block_provider(identity, None, signal.reason)
        logger.warning("Quota: parked %s until cleared (%s)", pinfo.name, signal.reason)
        return signal
    seconds = signal.cooldown_seconds
    if seconds is None:
        # No cooldown and no reset time: do not guess a window length, but do
        # stop sending at this credential for a short period.
        seconds = 60
    if pinfo.mints_credential:
        # A rotating token is a new key every refresh, so demoting today's
        # fingerprint would forget the refusal at the first rotation.  The
        # upstream account is the stable identity here.
        identity = getattr(pinfo, "quota_identity", None) or (
            f"@{provider_config_key(pinfo)}"
        )
        QUOTA_STATE.block_provider(identity, seconds, signal.reason)
    else:
        key = pinfo.active_key()
        if not key:
            return None
        QUOTA_STATE.demote_for(
            probe_id,
            credential_fingerprint(key),
            seconds,
            signal.reason,
            window=signal.window,
        )
    logger.warning(
        "Quota: %s %s for %ss (%s)",
        "blocked the provider" if pinfo.mints_credential else "demoted a credential",
        probe_id,
        seconds,
        signal.reason,
    )
    return signal


def note_upstream_rate_limit(
    pinfo: Any,
    policy: RateLimitPolicy | None = None,
) -> str:
    """Count one 429 against the lane that answered it.

    Every lane is counted, whatever its quota declaration: a provider with no
    probe used to answer 429 and be retried on the next request as if nothing
    had happened.  The first refusal sinks the provider, ``policy.after``
    refusals disable it for ``policy.cooldown_s``, and each repeat doubles that
    up to ``policy.cooldown_max_s``.

    Args:
        pinfo: The provider that answered.
        policy: The rule in force, or the default when the caller has none.

    Returns:
        ``"disable"``, ``"sink"`` or ``"off"``, as :meth:`RateLimitTracker.note`
        reports it.
    """
    name = provider_config_key(pinfo)
    key = None if getattr(pinfo, "keyless", False) else pinfo.active_key()
    credential = credential_fingerprint(key) if key else None
    outcome = QUOTA_STATE.rate_limits.note(
        name, credential, policy or RateLimitPolicy()
    )
    if outcome == "disable":
        logger.warning(
            "Rate limit: %s disabled until %s after repeated 429s",
            name,
            _iso(QUOTA_STATE.rate_limits.disabled(name, credential)),
        )
    return outcome


def clear_upstream_rate_limit(pinfo: Any) -> None:
    """A success clears its lane's count, strike ladder and sink.

    Counted per provider *and* credential, so one credential succeeding does
    not absolve another — but it does stop one transient refusal from
    accumulating across hours on the lane that keeps working.
    """
    name = provider_config_key(pinfo)
    key = None if getattr(pinfo, "keyless", False) else pinfo.active_key()
    credential = credential_fingerprint(key) if key else None
    QUOTA_STATE.rate_limits.clear_lane(name, credential)


def provider_offline_until(pinfo: Any) -> datetime | None:
    """When a provider-wide block ends, if the state can date one.

    Two states take a whole provider out: a quota block (an in-band refusal, a
    keyless lane, a rotating credential whose account is out) and a rate-limit
    disable that covers every lane.  ``blocked_until`` asks this so the client
    is told when the route comes back rather than "initializing".
    """
    name = provider_config_key(pinfo)
    identity = getattr(pinfo, "quota_identity", None) or f"@{name}"
    ends: list[datetime] = []
    block = QUOTA_STATE.provider_block(identity)
    if block is not None:
        if block.until is None:
            # A permanent park names no end; the caller reports the route as
            # unavailable without a date rather than inventing one.
            return None
        ends.append(block.until)
    disable = QUOTA_STATE.rate_limits.latest_disable(name)
    if disable is not None:
        ends.append(disable)
    return max(ends) if ends else None


def note_lane_refusal(
    pinfo: Any,
    status_code: int,
    policy: RateLimitPolicy | None = None,
    *,
    lane: str,
) -> str:
    """Count one capacity refusal against the upstream lane that answered.

    The per-request failover already serves this request from a sibling lane
    (``#1448``).  This is the cross-request half: the first refusal sinks the
    lane behind its siblings, and ``policy.after`` of them inside
    ``policy.window_s`` disable it for ``policy.cooldown_s``, doubling on
    repeats, so later requests start on a sibling instead of paying the failed
    attempt again.  A lane that is sunk and skipped while a sibling answers
    collects no second refusal, so there the sink is the whole effect and the
    disable belongs to a lane that keeps being tried.

    The key is ``(provider, lane)`` — the lane, not the credential — where a
    lane is the upstream model the attempt asked for, or the model the client
    asked for when the candidate names none.  It is never empty: two models
    routing through one provider are two lanes.
    That is what makes the count work: a capacity status is evidence about one
    upstream model, and the sibling that served the request says nothing about
    it.  Keyed by provider, a healthy sibling's success cleared the failed
    lane's evidence every time, so the count never reached ``policy.after``
    (``#1452``).

    What stays shared on purpose is the *credential* axis: a ``429`` is still
    counted by :func:`note_upstream_rate_limit` against the whole credential,
    because every lane of the provider shares the key.  The two ladders never
    see each other's records.

    The policy's ``sink`` applies as written.  Sinking a lane only reorders it
    behind its siblings, and a provider with no sibling left still serves
    through it — which is why a capacity refusal may sink a lane but must not
    sink a provider the way the old provider-wide count did.

    Args:
        pinfo: The provider that refused.
        status_code: The upstream status, for the log line.
        policy: The rule in force, or the default when the caller has none.
        lane: The lane's key, as :func:`proxy.failover.attempted_lane` builds
            it.

    Returns:
        ``"disable"``, ``"sink"`` or ``"off"``, as
        :meth:`RateLimitTracker.note_upstream` reports it.
    """
    name = provider_config_key(pinfo)
    outcome = QUOTA_STATE.rate_limits.note_upstream(
        name, lane, policy or RateLimitPolicy()
    )
    if outcome == "disable":
        logger.warning(
            "Upstream: %s/%s disabled until %s after repeated capacity refusals "
            "(last status %s)",
            name,
            lane,
            _iso(QUOTA_STATE.rate_limits.upstream_disabled(name, lane)),
            status_code,
        )
    return outcome


def note_transient_refusal(
    pinfo: Any,
    status_code: int,
    policy: RateLimitPolicy | None = None,
) -> str:
    """Count a credential-level refusal the quota probes could not explain.

    A ``402``/``429`` that no probe classified is still evidence that the
    *credential* cannot serve, so this count is provider-wide (credential
    ``None``): every lane of the provider shares the key, and key rotation must
    not spread one outage across several lanes.  Capacity refusals do not come
    here — they are counted per lane by :func:`note_lane_refusal`.

    It does not sink the provider on the first refusal.  Sinking keeps it out
    of the next request for the whole window, so the count could never reach
    ``policy.after`` and the three-strike rung would be unreachable; the
    continuous-429 ladder is what demotes a credential immediately.

    Args:
        pinfo: The provider that refused.
        status_code: The upstream status, for the log line.
        policy: The rule in force, or the default when the caller has none.

    Returns:
        ``"disable"`` or ``"off"``, as :meth:`RateLimitTracker.note` reports it.
    """
    name = provider_config_key(pinfo)
    counting = replace(policy or RateLimitPolicy(), sink=False)
    outcome = QUOTA_STATE.rate_limits.note(name, None, counting)
    if outcome == "disable":
        logger.warning(
            "Upstream: %s disabled until %s after repeated failures (last status %s)",
            name,
            _iso(QUOTA_STATE.rate_limits.disabled(name, None)),
            status_code,
        )
    return outcome


def note_upstream_success(pinfo: Any, *, lane: str) -> None:
    """A served request clears the evidence about the lane that served it.

    Two records are cleared, on purpose:

    - the provider-wide credential-level record (``None``), as before: a
      served request says the credential can serve;
    - the capacity record of *this* lane only.  A sibling's success is not
      evidence about the lane that refused, so it must not clear its count
      (``#1452``).
    """
    name = provider_config_key(pinfo)
    QUOTA_STATE.rate_limits.clear_lane(name, None)
    QUOTA_STATE.rate_limits.clear_upstream(name, lane)

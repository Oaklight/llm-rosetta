"""Probe execution, caching and the bridge to routing.

This module is the only place that talks to a vendor's usage endpoint.  It
resolves the probe from the shim and provider config, probes each credential
in the provider's key ring, and keeps the last reading in
:data:`~.state.QUOTA_STATE` so the router and the admin panel read one truth.

The raw key never leaves this module: readings are stored and returned
against a fingerprint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from llm_rosetta_gateway._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta_gateway.observability.sanitize import scrub_credential_patterns

from .models import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    InBandSignal,
    QuotaReading,
    credential_fingerprint,
)
from .probes import QuotaProbe, QuotaProbeError, resolve_probe
from .rate_limit import RateLimitPolicy
from .state import QUOTA_STATE

logger = logging.getLogger("llm-rosetta-gateway")

# A page refresh should not hit the vendor every time, but a reading older than
# a minute is cheap to refresh and the windows move in minutes, not seconds.
DEFAULT_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 10.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def provider_config_key(pinfo: Any) -> str:
    """The provider's key in the config, which its routing state is keyed on.

    ``ProviderInfo.name`` is the base format, so two configured providers of
    the same format share it; the config key tells them apart.
    """
    return getattr(pinfo, "config_name", None) or pinfo.name


def _quota_declaration(config: Any, name: str) -> dict[str, Any]:
    """Return the merged quota declaration for a provider.

    The shim supplies vendor defaults; a ``quota`` object in the provider
    config overrides them field by field.
    """
    from llm_rosetta.shims import get_shim

    cfg = getattr(config, "_raw_providers", {}).get(name, {}) or {}
    declared: dict[str, Any] = {}
    shim_name = getattr(config, "provider_shim_names", {}).get(name)
    shim = get_shim(shim_name) if shim_name else None
    shim_quota = getattr(shim, "quota", None)
    if shim_quota is not None:
        declared.update(shim_quota.as_dict())
    override = cfg.get("quota")
    if isinstance(override, dict):
        declared.update(override)
    return declared


def resolve_provider_probe(
    config: Any, name: str
) -> tuple[QuotaProbe | None, str | None, str | None]:
    """Return ``(probe, url, unsupported_reason)`` for a configured provider.

    ``(None, None, reason)`` covers every "nothing to read here" case: a
    keyless lane, an embedding provider, or a vendor no probe knows.
    """
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is None:
        return None, None, "provider is not loaded"
    if pinfo.keyless:
        return None, None, "the lane is keyless, so it has no per-key allowance"
    if not getattr(pinfo, "key_ring", None) or len(pinfo.key_ring) == 0:
        return None, None, "the provider holds no credential to probe"

    # Resolution happened when the config was built (the shim attaches its
    # probe implementations); this only reports what was bound.
    probe = getattr(pinfo, "quota_probe_impl", None)
    if probe is None:
        return (
            None,
            None,
            getattr(pinfo, "quota_unsupported", None)
            or "no quota probe is declared by this shim",
        )
    if pinfo.quota_usage_url is None:
        # The probe is bound for its in-band classifier only: say so, rather
        # than reporting a readable quota with no endpoint.
        return probe, None, pinfo.quota_unsupported or "no quota endpoint"
    return probe, pinfo.quota_usage_url, None


def _probe_headers(pinfo: Any, probe: QuotaProbe, key: str) -> dict[str, str]:
    """Build the probe request headers for one credential."""
    headers = {"Accept": "application/json", **pinfo.auth_headers_for(key)}
    headers.update(probe.headers(key))
    return headers


async def _probe_one(
    pinfo: Any,
    probe: QuotaProbe,
    url: str,
    key: str,
    *,
    demote_at: float = 100.0,
) -> QuotaReading:
    """Read one credential and normalize the answer."""
    credential = credential_fingerprint(key)
    proxy = getattr(pinfo, "proxy_url", None)
    try:
        async with AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS, proxy=proxy) as client:
            resp = await client.get(url, headers=_probe_headers(pinfo, probe, key))
            assert isinstance(resp, HttpResponse)
            body: Any
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            if resp.status_code >= 400:
                return QuotaReading(
                    probe=probe.id,
                    credential=credential,
                    status=STATUS_ERROR,
                    error=probe.describe_error(resp.status_code, body),
                    demote_at=demote_at,
                )
            try:
                plan, windows = probe.parse(body)
            except QuotaProbeError as exc:
                return QuotaReading(
                    probe=probe.id,
                    credential=credential,
                    status=exc.status,
                    error=str(exc),
                    demote_at=demote_at,
                )
            return QuotaReading(
                probe=probe.id,
                credential=credential,
                windows=windows,
                plan=plan,
                status=STATUS_OK,
                demote_at=demote_at,
            )
    except Exception as exc:  # noqa: BLE001 - report, never raise into admin
        logger.warning(
            "Quota probe %s failed for %s: %s",
            probe.id,
            pinfo.name,
            scrub_credential_patterns(str(exc)),
        )
        return QuotaReading(
            probe=probe.id,
            credential=credential,
            status=STATUS_ERROR,
            error=f"probe request failed: {scrub_credential_patterns(str(exc))}",
            demote_at=demote_at,
        )


async def read_provider_quota(
    config: Any,
    name: str,
    *,
    force: bool = False,
    ttl: float = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    """Return the normalized quota of every credential of a provider.

    Args:
        config: The live :class:`~gateway.config.GatewayConfig`.
        name: Provider config key.
        force: Ignore cached readings.
        ttl: Maximum age of a cached reading, in seconds.

    Returns:
        A JSON-ready payload.  ``supported`` is ``False`` with a ``reason``
        when there is nothing to probe; the credential list never contains a
        key, only its fingerprint.
    """
    probe, url, unsupported = resolve_provider_probe(config, name)
    if probe is None or url is None:
        payload: dict[str, Any] = {
            "provider": name,
            "supported": False,
            "status": STATUS_UNSUPPORTED,
            "reason": unsupported,
            "credentials": [],
        }
        # A blocked lane has something to say even with nothing to read: an
        # in-band provider is out of rotation for a bounded time, and this is
        # the only place the operator can see it.
        pinfo = getattr(config, "providers", {}).get(name)
        if pinfo is not None:
            block = provider_cooldown(name, pinfo)
            if block is not None:
                payload["provider_block"] = block
            payload["routing"] = routing_facts(pinfo, name)
        return payload

    pinfo = config.providers[name]
    declaration = _quota_declaration(config, name)
    demote_at = float(declaration.get("demote_at", 100.0))
    keys = pinfo.key_ring.keys()
    refreshed = False
    for key in keys:
        credential = credential_fingerprint(key)
        cached = QUOTA_STATE.get_reading(probe.id, credential)
        if not force and cached is not None and QUOTA_STATE.reading_age(cached) < ttl:
            continue
        reading = await _probe_one(pinfo, probe, url, key, demote_at=demote_at)
        QUOTA_STATE.put_reading(reading)
        refreshed = True
        if reading.status == STATUS_OK and reading.exhausted_until is None:
            # A fresh reading that says the credential can serve clears an
            # in-band demotion taken before the window reset.
            QUOTA_STATE.clear_demotion(probe.id, credential)

    payload: dict[str, Any] = {
        "provider": name,
        "supported": True,
        "probe": probe.id,
        "url": scrub_credential_patterns(url),
        "status": STATUS_OK,
        "refreshed": refreshed,
        "ttl_seconds": ttl,
        "credentials": QUOTA_STATE.snapshot(probe.id, keys),
        # The same numbers selection read, so the panel describes the router's
        # decision rather than re-deriving it.
        "routing": routing_facts(pinfo, name),
    }
    # A lane can be readable *and* cooling down; the panel needs both to avoid
    # painting a live reading on a provider the router is skipping.
    block = provider_cooldown(name, pinfo)
    if block is not None:
        payload["provider_block"] = block
    if payload["credentials"]:
        payload["read_at"] = (
            payload["credentials"][0]["reading"]["read_at"]
            if payload["credentials"][0]["reading"]
            else None
        )
    return payload


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
            from llm_rosetta.shims.quota_probe import InBandSignal

            return InBandSignal(cooldown_seconds=60, reason="keyless lane rate limited")
        return None

    probe = getattr(pinfo, "quota_probe_impl", None)
    probe_id = getattr(pinfo, "quota_probe", None)
    if probe is None or not probe_id:
        return None

    signal = probe.inband(status_code, body)
    if signal is None:
        return None
    seconds = signal.cooldown_seconds
    if seconds is None:
        # No cooldown and no reset time: do not guess a window length, but do
        # stop sending at this credential for a short period.
        seconds = 60
    if pinfo.mints_credential:
        # A rotating token is a new key every refresh, so demoting today's
        # fingerprint would forget the refusal at the first rotation.  The
        # upstream account is the stable identity here.
        identity = getattr(pinfo, "quota_identity", None) or f"@{pinfo.name}"
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
        ends.append(block.until)
    disable = QUOTA_STATE.rate_limits.latest_disable(name)
    if disable is not None:
        ends.append(disable)
    return max(ends) if ends else None


def quota_intervals(config: Any) -> dict[str, int]:
    """Return ``{provider: seconds}`` for providers that want a background poll.

    Only providers whose declaration sets ``quota.interval`` above zero are
    polled; the default is on demand (the admin card) plus in-band demotion.
    """
    out: dict[str, int] = {}
    for name in getattr(config, "providers", {}):
        declaration = _quota_declaration(config, name)
        try:
            interval = int(declaration.get("interval", 0) or 0)
        except (TypeError, ValueError):
            continue
        if interval > 0:
            out[name] = interval
    return out


def has_usable_alternative(pinfo: Any) -> bool:
    """Whether another credential of *pinfo* is currently usable.

    Used to decide whether a quota refusal is worth retrying immediately:
    with one credential, or with every credential demoted, a retry would go
    out on the same key and only spend another request.
    """
    probe_id = getattr(pinfo, "quota_probe", None)
    if not probe_id:
        return False
    keys = pinfo.key_ring.keys()
    if len(keys) < 2:
        return False
    return any(QUOTA_STATE.is_usable(probe_id, key) for key in keys)


def bind_provider_quota(name: str, pinfo: Any) -> None:
    """Attach the process-wide quota state to a freshly built provider.

    Called by the config loader after every :class:`ProviderInfo` is built.
    The provider keeps only predicates and an identifier: it never holds the
    state object, so a config reload cannot create a second source of truth.
    """
    # A provider that mints its credential has no ``credential_token`` until a
    # record exists, so the identity falls back to ``@<name>`` and becomes
    # ``acct:<digest>`` on the reload after sign-in.  Both are stable per
    # provider — which is what a block needs — but neither follows the account
    # to another provider entry.
    identity = getattr(pinfo, "credential_token", None) or f"@{name}"
    pinfo.quota_identity = identity
    # A keyless lane has no per-key allowance but can still be cooling down
    # from a 429, so the provider-level predicate is bound for it too.
    pinfo.quota_provider_blocked = lambda: (
        QUOTA_STATE.provider_block(identity) is not None
    )
    probe, _url, _unsupported = resolve_provider_probe_for_info(name, pinfo)
    probe_id = probe.id if probe is not None else None
    if probe_id is not None:
        pinfo.quota_probe = probe_id

    def _usable(key: str) -> bool:
        """One credential's availability: quota demotion *and* 429 disable.

        Bound for every provider, not only one with a probe: a lane that
        answers 429 has a credential that earned it whether or not anything
        can read its allowance, and the ring must skip that credential.
        """
        if probe_id is not None and not QUOTA_STATE.is_usable(probe_id, key):
            return False
        return (
            QUOTA_STATE.rate_limits.disabled(name, credential_fingerprint(key)) is None
        )

    pinfo.key_ring.set_usable(_usable)


def resolve_provider_probe_for_info(
    name: str, pinfo: Any
) -> tuple[QuotaProbe | None, str | None, str | None]:
    """Resolve a probe from the shim's implementations and the declaration.

    Config reloads build the provider before the config object exists, so
    everything needed is carried on the ``ProviderInfo``: the shim attached
    ``quota_probes``, the declaration names one (``quota_probe_id``), and the
    provider may override the URL.
    """
    if pinfo.keyless:
        pinfo.quota_unsupported = "the lane is keyless"
        return None, None, pinfo.quota_unsupported
    try:
        match = resolve_probe(
            pinfo.base_url,
            probes=pinfo.quota_probes,
            probe_id=pinfo.quota_probe_id,
            url=pinfo.quota_url,
            in_band=bool(getattr(pinfo, "quota_in_band", False)),
        )
    except ValueError as exc:
        pinfo.quota_unsupported = str(exc)
        return None, None, pinfo.quota_unsupported
    if match is None:
        pinfo.quota_unsupported = "no quota probe is declared by this shim"
        return None, None, pinfo.quota_unsupported
    if match.url is None:
        # Nothing to poll, but the probe stays bound: its ``inband()`` is the
        # only signal this lane gives, and demoting on it is the point.
        pinfo.quota_probe_impl = match.probe
        pinfo.quota_unsupported = match.unsupported or "no quota endpoint"
        return match.probe, None, pinfo.quota_unsupported
    pinfo.quota_probe_impl = match.probe
    pinfo.quota_usage_url = match.url
    return match.probe, match.url, None


@dataclass(frozen=True, slots=True)
class Allowance:
    """Whether a provider may still spend, and how much room it has.

    The rule is the tightest window: a provider that reports several windows
    (five hours, a week, a month) can spend only as much as the emptiest one
    allows, so the smallest remaining share is the number every caller compares
    on.  :attr:`windows` keeps the names, so a panel can say *which* window is
    binding instead of showing a bare 0.00.

    Attributes:
        has_allowance: ``False`` only when every measured window is empty.
            A provider nothing has measured has allowance — the absence of a
            reading is not evidence of a spent budget.
        value: The tightest remaining share, ``0.0`` to ``1.0``; ``None`` when
            no reading is known.  This is the number to compare providers by.
        windows: Every window the provider reports, tightest first.  Empty
            when it reports none, or when nothing has been read yet.
    """

    has_allowance: bool
    value: float | None
    windows: tuple[dict[str, Any], ...] = ()


def provider_allowance(pinfo: Any) -> Allowance:
    """Read one provider's allowance as a yes/no and a comparable number.

    The read is a lookup in the probe cache, not a call to the vendor.  A
    window that reports no share (a vendor that only says "rate-limited")
    contributes nothing; that case is what the demotion state is for.

    Args:
        pinfo: The provider's transport configuration.

    Returns:
        The allowance answer.  An unmeasured provider answers ``True`` with
        ``value=None``, so a caller that scales by it keeps its weight.
    """
    windows = tuple(provider_windows(pinfo))
    if not windows:
        return Allowance(has_allowance=True, value=None)
    value = min(float(entry["remaining"]) for entry in windows)
    return Allowance(has_allowance=value > 0.0, value=value, windows=windows)


def provider_headroom(pinfo: Any) -> float | None:
    """The smallest remaining share across this provider's credentials.

    ``1.0`` is untouched, ``0.0`` is spent, ``None`` means no reading is known
    — which is not the same as "plenty": a lane nothing has probed must keep
    its configured weight rather than be punished for being unmeasured.

    This is the number :func:`provider_allowance` compares on; it stays a
    separate function because the weighting code asks only for the number.

    Args:
        pinfo: The provider's transport configuration.

    Returns:
        The tightest remaining share, or ``None`` when nothing measured it.
    """
    return provider_allowance(pinfo).value


def provider_windows(pinfo: Any) -> list[dict[str, Any]]:
    """Every allowance window this provider reports, tightest first.

    :func:`provider_headroom` answers "how much is left" with one number — the
    tightest window across the key ring — which says nothing about *which*
    window is binding.  The panel needs that name to explain a lane at 0.00:
    a five-hour window that just reset is a different situation from a monthly
    one that is spent.

    A window name keeps the worst reading across the ring, because the
    tightest credential is what the router acts on.
    """
    probe = getattr(pinfo, "quota_probe", None)
    ring = getattr(pinfo, "key_ring", None)
    if not probe or ring is None:
        return []
    worst: dict[str, dict[str, Any]] = {}
    for key in ring.keys():
        reading = QUOTA_STATE.get_reading(probe, credential_fingerprint(key))
        if reading is None:
            continue
        for window in reading.windows:
            if window.used_percent is None:
                continue
            remaining = max(0.0, 100.0 - float(window.used_percent)) / 100.0
            current = worst.get(window.name)
            if current is None or remaining < current["remaining"]:
                worst[window.name] = {
                    "name": window.name,
                    "remaining": remaining,
                    "resets_at": _iso(window.resets_at),
                    "status": window.status,
                }
    return sorted(worst.values(), key=lambda entry: entry["remaining"])


#: A reading older than this is not evidence about right now.  A vendor that
#: refuses for five minutes should not have a provider removed from rotation
#: for an hour on the strength of a reading taken before that.
READING_MAX_AGE_SECONDS = 900


def reading_max_age(pinfo: Any) -> float:
    """How old a reading may be and still be evidence about this provider.

    The baseline is :data:`READING_MAX_AGE_SECONDS`, raised to the provider's
    declared poll interval when that is longer: a lane polled every half hour
    would otherwise have every reading stale on arrival, and the exclusion
    that depends on freshness would silently never fire.
    """
    interval = getattr(pinfo, "quota_interval", 0) or 0
    return max(float(READING_MAX_AGE_SECONDS), float(interval))


def provider_spent(pinfo: Any) -> bool:
    """Whether *every* credential of this provider is measured and spent.

    The rule is all-or-nothing per provider: each credential in the ring must
    have a fresh reading that decoded a window, and every one of those windows
    must be exhausted.  Silence is not agreement —

    - a credential that has never been read may well have room, and the
      gateway does not know that it does not;
    - a reading too old to speak about now, or one that decoded no window
      at all (the probe reports ``unsupported`` for lanes it cannot read), is
      equally not evidence;
    - and a provider nothing has measured is not empty.

    Exclusion cannot correct itself: nothing routes to a provider that is out,
    so an unread credential would never be read and the provider would stay
    out until a reading aged past the freshness window.  Down-weighting
    (:func:`provider_headroom`) self-heals — traffic keeps flowing, the
    credential gets read, the weight recovers.  That asymmetry is why this
    predicate demands complete evidence.

    This is the exclusion counterpart of :func:`provider_headroom`: the
    weighting reduces a share, this removes the candidate.  A live provider
    block is the gateway's own decision and is already part of ``ready``.
    """
    probe = getattr(pinfo, "quota_probe", None)
    ring = getattr(pinfo, "key_ring", None)
    if not probe or ring is None:
        return False
    keys = list(ring.keys())
    if not keys:
        return False
    max_age = reading_max_age(pinfo)
    for key in keys:
        reading = QUOTA_STATE.get_reading(probe, credential_fingerprint(key))
        if reading is None or QUOTA_STATE.reading_age(reading) > max_age:
            return False
        if not reading.windows:
            return False
        if any(not window.exhausted for window in reading.windows):
            return False
    return True


def provider_exhausted_until(pinfo: Any) -> datetime | None:
    """When this provider could serve again, from its spent readings.

    The earliest reset among its credentials — one credential coming back is
    enough — or ``None`` when the vendor gave no reset, in which case the
    gateway will not invent one.  Only meaningful for a provider that
    :func:`provider_spent` reports as spent.
    """
    probe = getattr(pinfo, "quota_probe", None)
    ring = getattr(pinfo, "key_ring", None)
    if not probe or ring is None:
        return None
    resets: list[datetime] = []
    for key in ring.keys():
        reading = QUOTA_STATE.get_reading(probe, credential_fingerprint(key))
        if reading is None:
            return None
        moment = reading.exhausted_until
        if moment is None:
            return None
        resets.append(moment)
    return min(resets) if resets else None


def routing_facts(pinfo: Any, name: str | None = None) -> dict[str, Any]:
    """What the router concluded about this provider, for the admin panel.

    The panel must not re-derive the rule: the numbers here are the ones
    selection read, so a card that says "skipped" is describing the same
    decision that sent the request elsewhere.

    Three facts, because they answer different questions and the panel once
    conflated them:

    - ``blocked`` — the gateway is skipping this provider right now (an in-band
      refusal or a spent plan lane put it on a cooldown).  It overrides
      everything else: a provider can have a healthy reading and still be out.
    - ``excluded`` — every credential's fresh reading says spent; selection
      removes the candidate.  ``reason`` is a code the panel translates.
    - ``headroom`` — the tightest credential's remaining share, and ``weight``
      — the multiplier the router applies to this provider's configured weight
      (``0.25 + 0.75 × headroom``, never zero).  They differ exactly when a key
      ring is mixed: one spent key and one healthy one reads as ``0.0``
      headroom but still serves at a quarter weight.
    - ``sunk`` — a recent refusal orders this provider last without taking it
      out; the panel shows it as deprioritised rather than blocked.
    """
    from ..routing.eligibility import quota_weight

    headroom = provider_headroom(pinfo)
    excluded = provider_spent(pinfo)
    # Derived from the same call the payload's ``provider_block`` comes from,
    # so the two keys cannot disagree when a provider was built without
    # ``bind_provider_quota``.
    blocked = provider_cooldown(name or pinfo.name, pinfo) is not None
    return {
        "headroom": headroom,
        "weight": quota_weight(headroom),
        "excluded": excluded,
        "blocked": blocked,
        "reason": "spent" if excluded else None,
        "sunk": QUOTA_STATE.rate_limits.sunk(provider_config_key(pinfo)),
    }


def provider_cooldown(name: str, pinfo: Any) -> dict[str, Any] | None:
    """Return a provider-wide block, for the admin surface.

    Three lanes land here: a keyless one, whose 429 has no key to demote; a
    provider that mints its credential, whose rotating token cannot carry a
    per-key demotion across a refresh; and any lane whose continuous refusals
    escalated to a disable.  In every case the block is what keeps the
    provider out of rotation, so it is also what the operator needs to see.
    """
    identity = getattr(pinfo, "quota_identity", None) or f"@{name}"
    block = QUOTA_STATE.provider_block(identity)
    if block is not None:
        return block.describe()
    until = QUOTA_STATE.rate_limits.latest_disable(provider_config_key(pinfo))
    if until is None:
        return None
    return {"until": _iso(until), "reason": "repeated refusals", "window": None}


def note_transient_refusal(
    pinfo: Any,
    status_code: int,
    policy: RateLimitPolicy | None = None,
) -> str:
    """Count one capacity refusal against the provider's own lane.

    The per-request failover already serves this request from another
    provider.  This is the cross-request half: ``policy.after`` refusals
    inside ``policy.window_s`` disable the provider for ``policy.cooldown_s``,
    doubling on repeats, so later requests are routed away instead of paying
    the failed attempt again.

    Unlike a ``429``, a capacity refusal does **not** sink the provider on the
    first one.  Sinking keeps the provider out of the next request for the
    whole window, so the count could never reach ``policy.after`` — the
    three-strike rung would be unreachable with the sink on.

    The count is provider-wide (credential ``None``) rather than per key: a
    capacity status says the provider cannot serve, not that the key that
    happened to answer is spent, and key rotation must not spread one outage
    across several lanes.

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


def note_upstream_success(pinfo: Any) -> None:
    """A served request clears the provider's capacity-refusal count."""
    QUOTA_STATE.rate_limits.clear_lane(provider_config_key(pinfo), None)

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
from .state import QUOTA_STATE

logger = logging.getLogger("llm-rosetta-gateway")

# A page refresh should not hit the vendor every time, but a reading older than
# a minute is cheap to refresh and the windows move in minutes, not seconds.
DEFAULT_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 10.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


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

    # A keyless lane has no credential to demote; its refusal is a cooldown of
    # the provider itself.  Any 429 is enough evidence: there is no key whose
    # quota this could be, so retrying immediately only spends requests.
    if getattr(pinfo, "keyless", False):
        if status_code == 429:
            identity = getattr(pinfo, "quota_identity", None) or f"@{pinfo.name}"
            QUOTA_STATE.block_provider(identity, 60, "keyless lane rate limited")
            # Returned so the proxy can treat this as a quota-class refusal:
            # there is no key to demote, but there is another provider to try.
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
    if probe is None:
        return
    pinfo.quota_probe = probe.id
    pinfo.key_ring.set_usable(lambda key: QUOTA_STATE.is_usable(probe.id, key))


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


def provider_headroom(pinfo: Any) -> float | None:
    """The smallest remaining share across this provider's credentials.

    ``1.0`` is untouched, ``0.0`` is spent, ``None`` means no reading is known
    — which is not the same as "plenty": a lane nothing has probed must keep
    its configured weight rather than be punished for being unmeasured.

    The reading is per credential and already cached by the probe layer, so
    this is a lookup, not a call.  A window that reports no share (a vendor
    that only says "rate-limited") contributes nothing here; that case is what
    the demotion state is for.
    """
    probe = getattr(pinfo, "quota_probe", None)
    ring = getattr(pinfo, "key_ring", None)
    if not probe or ring is None:
        return None
    shares: list[float] = []
    for key in ring.keys():
        reading = QUOTA_STATE.get_reading(probe, credential_fingerprint(key))
        if reading is None:
            continue
        for window in reading.windows:
            if window.used_percent is None:
                continue
            shares.append(max(0.0, 100.0 - float(window.used_percent)) / 100.0)
    return min(shares) if shares else None


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
    """
    from ..routing.select import quota_weight

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
    }


def provider_cooldown(name: str, pinfo: Any) -> dict[str, Any] | None:
    """Return a provider-wide quota block, for the admin surface.

    Two lanes land here: a keyless one, whose 429 has no key to demote, and a
    provider that mints its credential, whose rotating token cannot carry a
    per-key demotion across a refresh.  In both cases the block is what keeps
    the provider out of rotation, so it is also what the operator needs to see.
    """
    identity = getattr(pinfo, "quota_identity", None) or f"@{name}"
    block = QUOTA_STATE.provider_block(identity)
    return block.describe() if block else None

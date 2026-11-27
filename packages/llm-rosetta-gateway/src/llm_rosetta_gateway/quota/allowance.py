"""What routing and the admin panel read from the quota state.

Every function here only reads :data:`~.state.QUOTA_STATE`: the allowance a
provider has left (:class:`Allowance`), its headroom and windows, whether it
is spent and until when, how old a reading may be before it stops counting,
the routing facts the eligibility pass weighs, and the cooldown a refusal
left behind.  The probe that fills the state lives in :mod:`.service`, the
in-band signals that also write it in :mod:`.signals`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ._common import (
    _iso,
    provider_config_key,
)
from .models import (
    credential_fingerprint,
)
from .state import QUOTA_STATE


#: How much of a provider's configured weight survives a spent window.  The
#: floor keeps a live provider in rotation — a headroom of 0.0 is "nearly
#: spent", and the quota state, not this factor, is what removes it.
HEADROOM_FLOOR = 0.25


def quota_weight(headroom: float | None) -> float:
    """Scale a weight by the provider's remaining allowance.

    Args:
        headroom: Smallest remaining share, 0..1, or ``None`` when nothing has
            measured this provider.

    Returns:
        The multiplier: ``1.0`` for an unmeasured or untouched provider, the
        floor for a spent one.
    """
    if headroom is None:
        return 1.0
    return HEADROOM_FLOOR + (1.0 - HEADROOM_FLOOR) * max(0.0, min(1.0, headroom))


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
        if reading is None or reading.advisory:
            # Silence is not agreement, and neither is a reading taken with a
            # different credential: neither says whether this key can serve.
            return False
        if QUOTA_STATE.reading_age(reading) > max_age:
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
        if reading is None or reading.advisory:
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

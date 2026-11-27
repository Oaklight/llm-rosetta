"""Whether a provider may serve, and how much of the traffic it should get.

One predicate answers "can this provider serve this request" for every caller:
the selection chain, the "why is nothing serving" answer, the admin panel and
the proxy.  It used to be three: ``ProviderInfo.ready`` for the provider's own
blocks, ``quota.service.provider_spent`` for its readings, and the block-versus-
exhausted split in ``config/resolve``.  Each answered a different slice of the
same question, so a lane could be "ready" to one and "spent" to another.

The answer is a value, not a boolean, because the caller needs more than yes or
no:

- *why not*, so the refusal can say "blocked until 12:05" rather than
  "initializing";
- *how much*, the weight multiplier a nearly spent provider serves at;
- *where in the order*, since a lane that just answered 429 is still a
  candidate, just the last one tried.

Everything here is a read.  The state itself lives in ``quota/`` — readings,
demotions, provider blocks and the continuous-429 counters — and this module
composes those facts; it never writes them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..quota.allowance import quota_weight
from ..quota.state import QUOTA_STATE
from .stage import RoutingStage
from .strategy import EntryKey

if TYPE_CHECKING:
    from .context import SelectionContext


#: Why a provider is not a candidate.  ``None`` means it is one.
BLOCKED = "blocked"
RATE_LIMITED = "rate_limited"
SPENT = "spent"
NO_CREDENTIAL = "no_credential"
INITIALIZING = "initializing"
#: The entry serves this model but not this request: its declared
#: capabilities lack one the request needs.
CAPABILITY = "capability"


def required_capabilities(facts: Any) -> frozenset[str]:
    """What a request needs a provider to be able to do, from its facts.

    Three facts map to a capability: an image needs ``vision``, a tool list
    needs ``tools``, and an effort needs ``reasoning``.  Everything else — a
    plain text request — needs nothing, so a route whose entries declare
    capabilities is never narrowed for it.

    Args:
        facts: A :class:`~.rules.RuleFacts`, or ``None`` when the edge built
            none.

    Returns:
        The capability names the request needs.
    """
    if facts is None:
        return frozenset()
    needed: set[str] = set()
    if getattr(facts, "images", False):
        needed.add("vision")
    if getattr(facts, "tools", False):
        needed.add("tools")
    if getattr(facts, "effort", None) or getattr(facts, "thinking", False):
        needed.add("reasoning")
    return frozenset(needed)


@dataclass(frozen=True, slots=True)
class Eligibility:
    """One provider's answer to "may this request be served by you?".

    Attributes:
        name: The provider's config key.
        eligible: Whether it is a candidate at all.
        reason: Why it is not, from the module's reason constants.
        until: When it comes back, for the reasons that have a known end.
        headroom: Smallest remaining allowance share, or ``None`` when nothing
            has measured it.
        sunk: Whether a recent refusal orders it last.  A sunk provider is
            still eligible: a route with nothing else must still serve.
    """

    name: str
    eligible: bool
    reason: str | None = None
    until: datetime | None = None
    headroom: float | None = None
    sunk: bool = False
    #: The upstream model the entry asks for, when it names one.  The answer
    #: itself is provider-level — credentials, quota and rate limits are —
    #: but it is stored per lane, so two lanes of one provider keep separate
    #: answers.
    upstream: str | None = None

    @property
    def key(self) -> EntryKey:
        """What this answer is filed under: the candidate's key."""
        return (self.name, self.upstream or None)

    @property
    def weight_factor(self) -> float:
        """The multiplier this provider's configured weight is served at."""
        return quota_weight(self.headroom)

    def describe(self) -> dict[str, Any]:
        """The answer as a JSON-ready dictionary."""
        return {
            "provider": self.name,
            "eligible": self.eligible,
            "reason": self.reason,
            "until": self.until.isoformat() if self.until else None,
            "headroom": self.headroom,
            "weight": self.weight_factor,
            "sunk": self.sunk,
        }


def provider_eligibility(pinfo: Any, name: str | None = None) -> Eligibility:
    """Decide whether *pinfo* may serve, as a provider, and say why not.

    The provider-wide answer: it reads the credential ladder, which is what a
    caller with no lane in mind wants (``ProviderInfo.can_serve``, a typed
    route).  A caller that judges one lane of the provider calls
    :func:`lane_eligibility`, which adds that lane's own capacity record.
    """
    return _eligibility(pinfo, name, per_lane=False, lane="")


def lane_eligibility(pinfo: Any, name: str | None = None, *, lane: str) -> Eligibility:
    """Decide whether one lane of *pinfo* may serve, and say why not.

    Args:
        pinfo: The provider's transport configuration.
        name: The provider's config key.  Defaults to the provider's own.
        lane: The lane's key, as :func:`proxy.failover.attempted_lane` builds
            it — the upstream model the candidate asks for, or the model the
            client asked for when it names none.
    """
    return _eligibility(pinfo, name, per_lane=True, lane=lane)


def _eligibility(
    pinfo: Any, name: str | None, *, per_lane: bool, lane: str
) -> Eligibility:
    """Decide whether *pinfo* may serve, and say why not when it may not.

    The checks are ordered by what the client should be told, and by what can
    change on its own:

    1. a provider-wide quota block (an in-band refusal, a keyless cooldown, a
       rotating credential whose account is out) — it has a known end;
    2. a rate-limit disable from continuous 429s — it also has a known end;
    3. a credential that has not been seeded yet — it is still starting;
    4. a key ring with no usable credential left — a 429 disable or a quota
       demotion covers every key, and the answer names which (a disable is
       ``rate_limited``; a demotion is ``spent``, with the vendor's own end);
    5. readings that say every credential's allowance is spent.

    The provider-wide checks read the credential ladder: a ``429`` takes every
    lane of the provider out, because they share the key.  A lane-level answer
    reads that lane's own capacity record too — its sink makes it a fallback,
    and its disable leaves only that lane out (``#1452``).

    Args:
        pinfo: The provider's transport configuration.  Duck-typed on purpose:
            the admin panel and tests pass stand-ins.
        name: The provider's config key.  Defaults to the provider's own.
        per_lane: Whether to read one lane's capacity record as well.
        lane: The lane's key, read only when *per_lane* is true.

    Returns:
        The answer.  ``eligible`` is ``False`` exactly when ``reason`` is set.
    """
    from ..quota.service import (
        provider_config_key,
        provider_exhausted_until,
        provider_headroom,
        provider_spent,
    )

    key = name or provider_config_key(pinfo)
    headroom = provider_headroom(pinfo)
    sunk = QUOTA_STATE.rate_limits.sunk(key)
    if per_lane:
        sunk = sunk or QUOTA_STATE.rate_limits.upstream_sunk(key, lane)

    def answer(
        reason: str | None,
        *,
        until: datetime | None = None,
        eligible: bool = False,
    ) -> Eligibility:
        return Eligibility(
            name=key,
            eligible=eligible,
            reason=reason,
            until=until,
            headroom=headroom,
            sunk=sunk,
        )

    blocked = getattr(pinfo, "quota_provider_blocked", None)
    if callable(blocked) and blocked():
        return answer(BLOCKED, until=_block_end(pinfo, key))

    # The provider-level lane only: a disable of one credential is not a
    # disable of the provider — the key ring below answers that question.
    disabled = QUOTA_STATE.rate_limits.disabled(key)
    if disabled is not None:
        return answer(RATE_LIMITED, until=disabled)
    if per_lane:
        # A capacity disable is the upstream model's own: its siblings keep
        # serving, so only this lane is ruled out.
        lane_disabled = QUOTA_STATE.rate_limits.upstream_disabled(key, lane)
        if lane_disabled is not None:
            return answer(RATE_LIMITED, until=lane_disabled)

    if getattr(pinfo, "credentials_pending", False):
        return answer(INITIALIZING)

    ring = getattr(pinfo, "key_ring", None)
    if (
        ring is not None
        and not getattr(pinfo, "keyless", False)
        and len(ring)
        and not ring.any_usable()
    ):
        # Every key is out, and the two ways it happens are different facts
        # that end at different times: a 429 ladder disable has its own
        # cooldown, while a quota demotion is the allowance the vendor
        # measured running out.  Reporting the second one as "no credential"
        # told an operator their key was missing when the vendor had merely
        # said "not until then" — and it shadowed the allowance answer below,
        # because this branch returns first.
        #
        # The time reported is when *this provider* can serve again, which is
        # the first credential to come back — not the longest of anything.  The
        # two kinds are therefore resolved per key first (one key held by both
        # is out until the later one), and the keys are then compared by their
        # earliest.
        recovery = _ring_recovery(pinfo, key)
        if recovery is not None:
            until, reason = recovery
            return answer(reason, until=until)
        return answer(NO_CREDENTIAL)

    if provider_spent(pinfo):
        return answer(SPENT, until=provider_exhausted_until(pinfo))

    return answer(None, eligible=True)


def _ring_recovery(pinfo: Any, name: str) -> tuple[datetime, str] | None:
    """When the first credential of *pinfo* comes back, and what holds it.

    The ring is unusable only while **every** key is out, so the provider is
    back as soon as one key is — that key's own binding end, whichever kind
    it is.  A key held by both a ladder disable and a demotion leaves when the
    later of its two ends passes; the provider leaves when the earliest key
    does.

    A key can be out on its allowance alone: a reading that says the window
    is spent, with a reset ahead, keeps the ring unusable without any
    demotion having been written.  That reset is an end like the other two,
    so "spent" carries a time here too.

    Args:
        pinfo: The provider whose key ring is out.
        name: The provider's config key, for the ladder's records.

    Returns:
        ``(until, reason)`` — the moment the provider can serve again and the
        reason constant for what is holding it then — or ``None`` when no key
        can be dated, which is a ring that is out for some other reason.
    """
    from ..quota.models import credential_fingerprint

    ring = getattr(pinfo, "key_ring", None)
    if ring is None:
        return None
    probe = getattr(pinfo, "quota_probe", None)
    # A disable covering the whole provider is one key's end for every key.
    provider_end = QUOTA_STATE.rate_limits.disabled(name)
    best: tuple[datetime, str] | None = None
    for raw in ring.keys():
        fingerprint = credential_fingerprint(raw)
        ends: list[tuple[datetime, str]] = []
        if provider_end is not None:
            ends.append((provider_end, RATE_LIMITED))
        lane_end = QUOTA_STATE.rate_limits.disabled(name, fingerprint)
        if lane_end is not None:
            ends.append((lane_end, RATE_LIMITED))
        if probe:
            demotion = QUOTA_STATE.demotion(probe, fingerprint)
            if demotion is not None and demotion.until is not None:
                # A permanent park names no end; it cannot date the ring, and
                # `ends` staying empty is how the caller learns that.
                ends.append((demotion.until, SPENT))
            # A reading can say the allowance is spent without the gateway
            # having demoted anything: the probe was polled, or the key was
            # marked before this process started.  Its reset is the same kind
            # of end, so a ring held by readings alone is ``spent`` with a
            # time, not "no credential".  An advisory reading is excluded:
            # it was made with another credential, so it cannot date this one.
            reading = QUOTA_STATE.get_reading(probe, fingerprint)
            if (
                reading is not None
                and not reading.advisory
                and reading.exhausted_until is not None
            ):
                ends.append((reading.exhausted_until, SPENT))
        if not ends:
            # This key is out for a reason that carries no end, so the
            # provider's recovery cannot be dated from this ring either.
            continue
        key_end, key_reason = max(ends, key=lambda entry: entry[0])
        if best is None or key_end < best[0]:
            best = (key_end, key_reason)
    return best


def _block_end(pinfo: Any, name: str) -> datetime | None:
    """When a provider-wide quota block ends, if the state can date it."""
    identity = getattr(pinfo, "quota_identity", None) or f"@{name}"
    block = QUOTA_STATE.provider_block(identity)
    return block.until if block is not None else None


class EligibilityStage(RoutingStage):
    """Ask the predicate once, for every candidate, and record the answers.

    The facts this reads are the quota layer's: readings, demotions,
    provider blocks and the continuous-429 counters.  It reads them; it never
    writes them, and it holds no state of its own — which is why the stage is
    still the right shape even though the answer is a value now.
    """

    __slots__ = ()

    name = "eligibility"
    requires = frozenset({"model_route", "providers"})
    # The two derived views are kept because the stages after this one — and
    # the "why is nothing serving" answer — read them by name.
    provides = frozenset({"eligibility", "quota_headroom", "quota_excluded"})

    def run(self, ctx: SelectionContext) -> None:
        """Fill ``ctx.eligibility`` and the two quota views from it."""
        answers: dict[EntryKey, Eligibility] = {}
        headroom: dict[EntryKey, float] = {}
        excluded: set[str] = set()
        for entry in ctx.candidates:
            pinfo = ctx.providers.get(entry.name)
            if pinfo is None:
                # A candidate with no transport config cannot be asked; it
                # stays a candidate, as it did before this stage existed.
                answers[entry.key] = Eligibility(
                    name=entry.name, eligible=True, upstream=entry.upstream_model
                )
                headroom[entry.key] = 1.0
                continue
            answer = lane_eligibility(
                pinfo, entry.name, lane=entry.upstream_model or ctx.model
            )
            # A provider entry may declare what it serves for this model.  The
            # comparison is per request: the same entry stays a candidate for
            # every request whose needs it covers.
            declared = entry.capabilities
            if declared is not None and not ctx.required_capabilities <= set(declared):
                answer = Eligibility(
                    name=entry.name,
                    eligible=False,
                    reason=CAPABILITY,
                    until=answer.until,
                    headroom=answer.headroom,
                    sunk=answer.sunk,
                    upstream=entry.upstream_model,
                )
            elif entry.upstream_model:
                answer = replace(answer, upstream=entry.upstream_model)
            answers[entry.key] = answer
            headroom[entry.key] = 1.0 if answer.headroom is None else answer.headroom
            if answer.reason == SPENT:
                excluded.add(entry.name)
        ctx.with_eligibility(answers)
        ctx.with_quota_headroom(headroom, frozenset(excluded))

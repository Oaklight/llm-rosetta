"""Continuous 429s: how a lane is sunk, then taken out for a while.

A single refusal is evidence about one request; a run of them is evidence about
the lane.  The rules, weakest first:

- the first refusal within the window **sinks** the provider: it moves to the
  back of the order but stays a candidate, so a route with nothing else still
  serves;
- ``after`` refusals within the window **disable** it for ``cooldown_s``;
- every repeated disable doubles the cooldown, up to ``cooldown_max_s``, and a
  success clears the count, the ladder and the sink alike.

State is keyed by ``(provider config key, credential fingerprint)``.  A 429
names the credential that earned it when the lane has keys, and the provider
itself when it does not, so the same rule covers a probed lane, a keyless one
and a lane with no probe at all.

Nothing is persisted.  The upstream still enforces its own limit, so losing the
count on a restart costs at most one extra refusal.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

#: What one counted refusal did to the lane.
Outcome = Literal["sink", "disable", "off"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """When a run of 429s takes a lane out, and for how long.

    Attributes:
        sink: Whether the first refusal moves the provider to the back of the
            order.  Off, refusals are only counted towards ``after``.
        after: Refusals within ``window_s`` that disable the lane.  ``0``
            never disables; the lane can then only be sunk.
        window_s: How long refusals count towards ``after``.
        cooldown_s: First disable's length.
        cooldown_max_s: Cap the doubling may not exceed.
    """

    sink: bool = True
    after: int = 3
    window_s: float = 300.0
    cooldown_s: float = 60.0
    cooldown_max_s: float = 900.0

    @classmethod
    def from_config(cls, raw: Any, *, label: str = "routing.rate_limit"):
        """Build a policy from a config block, refusing a malformed one.

        Args:
            raw: The ``routing.rate_limit`` object, or ``None`` for defaults.
            label: The config path, for the error message.

        Returns:
            The policy.

        Raises:
            TypeError: If *raw* is not an object.
            ValueError: If a field has the wrong type or an impossible value.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"config: {label} must be an object, got {raw!r}")
        unknown = set(raw) - {
            "sink",
            "after",
            "window_s",
            "cooldown_s",
            "cooldown_max_s",
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"config: {label} has unknown key(s): {names}")

        sink = raw.get("sink", True)
        if not isinstance(sink, bool):
            raise ValueError(
                f"config: {label}.sink must be true or false, got {sink!r}"
            )

        def number(key: str, default: float, *, minimum: float) -> float:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"config: {label}.{key} must be a number, got {value!r}"
                )
            if value < minimum:
                raise ValueError(
                    f"config: {label}.{key} must be at least {minimum}, got {value!r}"
                )
            return float(value)

        after = raw.get("after", 3)
        if isinstance(after, bool) or not isinstance(after, int):
            raise ValueError(f"config: {label}.after must be an integer, got {after!r}")
        if after < 0:
            raise ValueError(
                f"config: {label}.after must be 0 or greater, got {after!r}"
            )

        cooldown = number("cooldown_s", 60.0, minimum=1)
        cap = number("cooldown_max_s", 900.0, minimum=1)
        if cap < cooldown:
            raise ValueError(
                f"config: {label}.cooldown_max_s must be at least cooldown_s, "
                f"got {cap!r} < {cooldown!r}"
            )
        return cls(
            sink=sink,
            after=after,
            window_s=number("window_s", 300.0, minimum=1),
            cooldown_s=cooldown,
            cooldown_max_s=cap,
        )

    def as_dict(self) -> dict[str, Any]:
        """The policy as the admin panel reports it."""
        return {
            "sink": self.sink,
            "after": self.after,
            "window_s": self.window_s,
            "cooldown_s": self.cooldown_s,
            "cooldown_max_s": self.cooldown_max_s,
        }


@dataclass(frozen=True, slots=True)
class RateLimit:
    """What the latest refusals say about one lane.

    Attributes:
        refusals: Counted refusals in the current window.
        window_end: When the count stops being evidence.
        strikes: Disables since the last success, driving the doubling.
        disabled_until: When the lane may serve again, or ``None``.
        sink: Whether the count sinks the provider while the window holds.
    """

    refusals: int
    window_end: datetime
    strikes: int
    disabled_until: datetime | None = None
    sink: bool = True

    def is_sunk(self, now: datetime) -> bool:
        """Whether this record still sinks its provider at *now*."""
        return self.sink and now < self.window_end

    def is_disabled(self, now: datetime) -> bool:
        """Whether this record still disables its lane at *now*."""
        return self.disabled_until is not None and now < self.disabled_until

    def describe(self) -> dict[str, Any]:
        """The record as a JSON-ready dictionary."""
        return {
            "refusals": self.refusals,
            "strikes": self.strikes,
            "window_end": _iso(self.window_end),
            "disabled_until": _iso(self.disabled_until),
            "sink": self.sink,
        }


class RateLimitTracker:
    """The refusals of every lane, counted and graded.

    Two axes, because the two kinds of refusal are evidence about different
    things:

    - a **credential** refusal (a ``429``) is one key's state, so its records
      are keyed by ``(provider, credential fingerprint)``.  Every lane of the
      provider reads the same record: the provider sinks or leaves together,
      whatever lane answered, because they share the key.
    - a **capacity** refusal (the ``503`` family) is one upstream model's
      state, so its records are keyed by ``(provider, upstream model)``.  A
      sibling lane's success cannot clear them, and one lane's refusals cannot
      take its siblings out.

    Both ladders have the same shape (:class:`RateLimit`) and run the same
    counting code; they differ in the key and in the policy they are counted
    with.  The gateway runs one event loop, and a short critical section is
    enough for a read-modify-write of a counter; the same choice the quota
    state makes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str | None], RateLimit] = {}
        # Capacity refusals, by upstream model: ``(provider, upstream)``.
        # Separate from ``_records`` because "no credential" there means the
        # provider itself, while "no upstream" here is a real lane — the one
        # whose entry names no upstream model.
        self._upstream_records: dict[tuple[str, str | None], RateLimit] = {}

    def clear(self) -> None:
        """Drop every record.  Used by tests and by a full state reset."""
        with self._lock:
            self._records.clear()
            self._upstream_records.clear()

    # -- writing -----------------------------------------------------------

    def note(
        self,
        provider: str,
        credential: str | None,
        policy: RateLimitPolicy,
        *,
        now: datetime | None = None,
    ) -> Outcome:
        """Count one refusal against a lane and report what it did.

        Args:
            provider: The provider's config key.
            credential: The refusing credential's fingerprint, or ``None`` when
                the lane has no key (keyless) or the refusal was the provider's.
            policy: The rule in force.
            now: The moment of the refusal; tests pass one.

        Returns:
            ``"disable"`` when the lane is now out, ``"sink"`` when it is only
            ordered last, ``"off"`` when the policy counted nothing.
        """
        moment = now or _now()
        return self._note(self._records, (provider, credential), policy, moment)

    def note_upstream(
        self,
        provider: str,
        lane: str,
        policy: RateLimitPolicy,
        *,
        now: datetime | None = None,
    ) -> Outcome:
        """Count one capacity refusal against one upstream lane.

        The lane is the upstream model the attempt was made on, or — for a
        candidate that names none — the model the client asked for.  Either
        way it is never ``None``: two models routing through one provider are
        two lanes, and sharing one key would let a healthy model clear the
        failed one's evidence (``#1452`` review).

        The policy's ``sink`` applies as written: sinking a lane only reorders
        it behind its siblings, and a provider with nothing else still serves
        through it.

        Args:
            provider: The provider's config key.
            lane: The lane's key, as :func:`proxy.failover.attempted_lane`
                builds it.
            policy: The rule in force.
            now: The moment of the refusal; tests pass one.

        Returns:
            ``"disable"``, ``"sink"`` or ``"off"``, as :meth:`note` reports it.
        """
        moment = now or _now()
        return self._note(self._upstream_records, (provider, lane), policy, moment)

    def _note(
        self,
        store: dict[tuple[str, str | None], RateLimit],
        key: tuple[str, str | None],
        policy: RateLimitPolicy,
        moment: datetime,
    ) -> Outcome:
        with self._lock:
            self._sweep(store, moment)
            previous = store.get(key)
            if previous is not None and moment <= previous.window_end:
                refusals = previous.refusals + 1
                strikes = previous.strikes
            else:
                # A new window starts from zero; the strike ladder survives a
                # window boundary, because it is what the doubling counts.
                refusals = 1
                strikes = previous.strikes if previous is not None else 0

            window_end = moment + timedelta(seconds=policy.window_s)
            if policy.after and refusals >= policy.after:
                strikes += 1
                seconds = min(
                    policy.cooldown_s * (2 ** (strikes - 1)), policy.cooldown_max_s
                )
                store[key] = RateLimit(
                    refusals=0,
                    window_end=window_end,
                    strikes=strikes,
                    disabled_until=moment + timedelta(seconds=seconds),
                    sink=policy.sink,
                )
                return "disable"

            store[key] = RateLimit(
                refusals=refusals,
                window_end=window_end,
                strikes=strikes,
                disabled_until=None,
                sink=policy.sink,
            )
            return "sink" if policy.sink else "off"

    def clear_lane(self, provider: str, credential: str | None = None) -> None:
        """Forget one credential's count and strike ladder after a success."""
        with self._lock:
            self._records.pop((provider, credential), None)

    def clear_upstream(self, provider: str, lane: str) -> None:
        """Forget one upstream lane's capacity count after its own success.

        A sibling's success is not evidence about this lane: it says the
        provider can serve, not that the upstream model that refused has
        recovered.  Only a success *on this lane* clears it.
        """
        with self._lock:
            self._upstream_records.pop((provider, lane), None)

    def clear_provider(self, provider: str) -> None:
        """Forget every record of one provider, on both axes."""
        with self._lock:
            for store in (self._records, self._upstream_records):
                for key in [k for k in store if k[0] == provider]:
                    store.pop(key, None)

    # -- reading -----------------------------------------------------------

    def record(
        self,
        provider: str,
        credential: str | None = None,
        *,
        now: datetime | None = None,
    ) -> RateLimit | None:
        """The live record for one lane, or ``None`` when there is none."""
        moment = now or _now()
        return self._live(self._records, (provider, credential), moment)

    def _live(
        self,
        store: dict[tuple[str, str | None], RateLimit],
        key: tuple[str, str | None],
        moment: datetime,
    ) -> RateLimit | None:
        """The record at *key* while it still means something, else ``None``."""
        with self._lock:
            found = store.get(key)
            if found is None:
                return None
            if moment >= found.window_end and not found.is_disabled(moment):
                store.pop(key, None)
                return None
            return found

    def upstream_record(
        self,
        provider: str,
        lane: str,
        *,
        now: datetime | None = None,
    ) -> RateLimit | None:
        """The live capacity record for one upstream lane, or ``None``."""
        moment = now or _now()
        return self._live(self._upstream_records, (provider, lane), moment)

    def upstream_sunk(
        self,
        provider: str,
        lane: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Whether one upstream lane is sunk at *now*."""
        moment = now or _now()
        record = self.upstream_record(provider, lane, now=moment)
        return bool(record is not None and record.is_sunk(moment))

    def upstream_disabled(
        self,
        provider: str,
        lane: str,
        *,
        now: datetime | None = None,
    ) -> datetime | None:
        """When one upstream lane may serve again, or ``None``.

        A capacity disable is the lane's alone: its siblings keep serving.
        """
        moment = now or _now()
        record = self.upstream_record(provider, lane, now=moment)
        if record is None or not record.is_disabled(moment):
            return None
        return record.disabled_until

    def tracked(self) -> int:
        """How many records the tracker holds, on both axes.

        For diagnostics and tests: the sweep below is what keeps a lane that
        was renamed or removed from sitting here until the process ends.
        """
        with self._lock:
            return len(self._records) + len(self._upstream_records)

    @staticmethod
    def _sweep(
        store: dict[tuple[str, str | None], RateLimit], moment: datetime
    ) -> None:
        """Drop the records a write would otherwise never visit.

        ``_live`` expires a record when its own key is read, which is enough
        while the key keeps being asked about.  A lane that is renamed, or
        whose model is removed, is never asked about again — so a write sweeps
        the store instead of waiting for a read that will not come.
        """
        for key in [k for k, record in store.items() if moment >= record.window_end]:
            record = store[key]
            if not record.is_disabled(moment):
                store.pop(key, None)

    def disabled(
        self,
        provider: str,
        credential: str | None = None,
        *,
        now: datetime | None = None,
    ) -> datetime | None:
        """When *credential*'s lane may serve again, or ``None``.

        ``now`` is the moment to judge against; tests pass one so a fixed
        timeline stays fixed.  The caller with no opinion gets the clock.
        """
        moment = now or _now()
        record = self.record(provider, credential, now=moment)
        if record is None or not record.is_disabled(moment):
            return None
        return record.disabled_until

    def latest_disable(
        self, provider: str, *, now: datetime | None = None
    ) -> datetime | None:
        """The latest disable end across every lane of *provider*."""
        moment = now or _now()
        ends: list[datetime] = []
        with self._lock:
            for (owner, _credential), record in self._records.items():
                if owner == provider and record.is_disabled(moment):
                    assert record.disabled_until is not None
                    ends.append(record.disabled_until)
        return max(ends) if ends else None

    def sunk(self, provider: str, *, now: datetime | None = None) -> bool:
        """Whether any lane of *provider* is sunk at *now*.

        ``now`` is the moment to judge against; tests pass one so a fixed
        timeline does not have to run inside the policy's own window of its
        import to stay true.
        """
        moment = now or _now()
        with self._lock:
            return any(
                owner == provider and record.is_sunk(moment)
                for (owner, _credential), record in self._records.items()
            )

    def snapshot(self, provider: str) -> dict[str, Any]:
        """The provider's credential lanes, for the admin panel.

        Capacity lanes are not here: they are per upstream model, and the
        admin reads them through the selector's per-lane answer instead.
        """
        lanes: list[dict[str, Any]] = []
        with self._lock:
            for (owner, credential), record in self._records.items():
                if owner != provider:
                    continue
                lanes.append({"credential": credential, **record.describe()})
        return {
            "sunk": self.sunk(provider),
            "disabled_until": _iso(self.latest_disable(provider)),
            "lanes": lanes,
        }

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

    The gateway runs one event loop, and a short critical section is enough for
    a read-modify-write of a counter; the same choice the quota state makes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str | None], RateLimit] = {}

    def clear(self) -> None:
        """Drop every record.  Used by tests and by a full state reset."""
        with self._lock:
            self._records.clear()

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
        key = (provider, credential)
        with self._lock:
            previous = self._records.get(key)
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
                self._records[key] = RateLimit(
                    refusals=0,
                    window_end=window_end,
                    strikes=strikes,
                    disabled_until=moment + timedelta(seconds=seconds),
                    sink=policy.sink,
                )
                return "disable"

            self._records[key] = RateLimit(
                refusals=refusals,
                window_end=window_end,
                strikes=strikes,
                disabled_until=None,
                sink=policy.sink,
            )
            return "sink" if policy.sink else "off"

    def clear_lane(self, provider: str, credential: str | None = None) -> None:
        """Forget one lane's count and strike ladder after a success."""
        with self._lock:
            self._records.pop((provider, credential), None)

    def clear_provider(self, provider: str) -> None:
        """Forget every lane of one provider."""
        with self._lock:
            for key in [k for k in self._records if k[0] == provider]:
                self._records.pop(key, None)

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
        key = (provider, credential)
        with self._lock:
            found = self._records.get(key)
            if found is None:
                return None
            if moment >= found.window_end and not found.is_disabled(moment):
                self._records.pop(key, None)
                return None
            return found

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

    def sunk(self, provider: str) -> bool:
        """Whether any lane of *provider* is sunk right now."""
        moment = _now()
        with self._lock:
            return any(
                owner == provider and record.is_sunk(moment)
                for (owner, _credential), record in self._records.items()
            )

    def snapshot(self, provider: str) -> dict[str, Any]:
        """The provider's lanes, for the admin panel."""
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

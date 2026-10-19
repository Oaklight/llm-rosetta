"""Runtime quota state: the last reading per credential, and demotions.

Nothing here is persisted.  A restart loses the readings and re-learns them
from the next probe; losing a demotion is safe, because the upstream still
enforces the allowance and the in-band path re-learns it on the next 429.

State is keyed by ``(probe_id, credential fingerprint)`` rather than by
provider: two providers configured with the same key front one upstream
account and must share one allowance, while one key used on two lanes has two
different allowances and therefore two different probe ids.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import QuotaReading, credential_fingerprint


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Demotion:
    """A credential the quota layer has taken out of rotation.

    Attributes:
        until: When the credential may serve again.
        reason: Short description for logs and the admin surface.
        window: Vendor window that refused, when the signal named one.
    """

    until: datetime
    reason: str
    window: str | None = None

    def describe(self) -> dict:
        return {
            "until": self.until.isoformat().replace("+00:00", "Z"),
            "reason": self.reason,
            "window": self.window,
        }


class QuotaState:
    """Per-credential readings and temporary demotions.

    The gateway runs one event loop, but admin handlers and the proxy can
    touch this from different tasks; a lock keeps a read-modify-write pair
    from interleaving, and every operation is short enough that holding it is
    never a throughput concern.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readings: dict[tuple[str, str], QuotaReading] = {}
        self._demotions: dict[tuple[str, str], Demotion] = {}
        # Keyless lanes have no credential; their refusal is a cooldown of the
        # upstream account the provider fronts.
        self._provider_blocks: dict[str, Demotion] = {}

    def clear(self) -> None:
        """Drop every reading, demotion and provider cooldown.

        The state is process-wide and outlives a config reload by design, so
        tests are the one caller that needs to start from nothing.
        """
        with self._lock:
            self._readings.clear()
            self._demotions.clear()
            self._provider_blocks.clear()

    # -- readings ----------------------------------------------------------

    def put_reading(self, reading: QuotaReading) -> None:
        """Record (or replace) the latest reading for its credential."""
        key = (reading.probe, reading.credential)
        with self._lock:
            self._readings[key] = reading

    def get_reading(self, probe: str, credential: str) -> QuotaReading | None:
        """Return the latest reading, if one has been taken."""
        with self._lock:
            return self._readings.get((probe, credential))

    def reading_age(self, reading: QuotaReading) -> float:
        """Return the reading's age in seconds."""
        return max(0.0, (_now() - reading.read_at).total_seconds())

    # -- demotions ---------------------------------------------------------

    def demote(
        self,
        probe: str,
        credential: str,
        until: datetime,
        reason: str,
        window: str | None = None,
    ) -> None:
        """Take *credential* out of rotation until *until*."""
        key = (probe, credential)
        with self._lock:
            current = self._demotions.get(key)
            if current is not None and current.until > until:
                return
            self._demotions[key] = Demotion(until=until, reason=reason, window=window)

    def demote_for(
        self,
        probe: str,
        credential: str,
        seconds: int,
        reason: str,
        window: str | None = None,
    ) -> None:
        """Convenience wrapper: demote for *seconds* from now."""
        self.demote(
            probe,
            credential,
            _now() + timedelta(seconds=seconds),
            reason,
            window=window,
        )

    def demotion(self, probe: str, credential: str) -> Demotion | None:
        """Return the live demotion for *credential*, or ``None``."""
        key = (probe, credential)
        with self._lock:
            found = self._demotions.get(key)
            if found is None:
                return None
            if found.until <= _now():
                self._demotions.pop(key, None)
                return None
            return found

    def clear_demotion(self, probe: str, credential: str) -> None:
        """Drop any demotion for *credential* (a fresh reading says it is fine)."""
        with self._lock:
            self._demotions.pop((probe, credential), None)

    # -- keyless provider cooldowns ---------------------------------------

    def block_provider(self, identity: str, seconds: int, reason: str) -> None:
        """Stop routing to a keyless upstream for *seconds*."""
        self._provider_blocks[identity] = Demotion(
            until=_now() + timedelta(seconds=seconds), reason=reason
        )

    def provider_block(self, identity: str) -> Demotion | None:
        """Return the live cooldown for a keyless upstream, if any."""
        found = self._provider_blocks.get(identity)
        if found is None:
            return None
        if found.until <= _now():
            self._provider_blocks.pop(identity, None)
            return None
        return found

    # -- the predicate the key ring asks -----------------------------------

    def is_usable(self, probe: str, key: str) -> bool:
        """Whether *key* may serve a request under *probe* right now.

        A credential is unusable when it is demoted, or when its latest
        reading has an exhausted window with a reset time still in the
        future.  A reading that failed is not a demotion: an unreachable probe
        says nothing about the key's health.
        """
        credential = credential_fingerprint(key)
        if self.demotion(probe, credential) is not None:
            return False
        reading = self.get_reading(probe, credential)
        if reading is None:
            return True
        until = reading.exhausted_until
        return until is None or until <= _now()

    def demotion_for_key(self, probe: str, key: str) -> Demotion | None:
        """Return the demotion that applies to *key*, if any."""
        return self.demotion(probe, credential_fingerprint(key))

    def snapshot(self, probe: str, keys: list[str]) -> list[dict]:
        """Return readings and demotions for *keys*, keyed by fingerprint.

        Never includes a key: the caller passes keys only so the fingerprints
        can be derived here, which keeps the digest logic in one place.
        """
        out: list[dict] = []
        for key in keys:
            credential = credential_fingerprint(key)
            reading = self.get_reading(probe, credential)
            demotion = self.demotion(probe, credential)
            out.append(
                {
                    "credential": credential,
                    "reading": reading.to_dict() if reading else None,
                    "demotion": demotion.describe() if demotion else None,
                }
            )
        return out


# Process-wide state.  Config reloads rebuild ProviderInfo objects but must not
# lose what the probe learned, so the state is not owned by the config.
QUOTA_STATE = QuotaState()

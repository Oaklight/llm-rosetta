"""Normalized quota readings.

A reading describes one credential, because an upstream allowance belongs to
an account rather than to a gateway config.  The shapes here are deliberately
vendor-neutral: a probe translates whatever the vendor returns into these, and
the admin surface and the router consume only these.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from llm_rosetta.shims.quota_probe import InBandSignal, QuotaWindow

# Reading statuses.  ``ok`` means the probe answered and at least one window
# was understood; the others say why there is nothing to show.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_UNSUPPORTED = "unsupported"

__all__ = [
    "InBandSignal",
    "QuotaReading",
    "QuotaWindow",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNSUPPORTED",
    "credential_fingerprint",
]


def credential_fingerprint(key: str) -> str:
    """Return a stable, non-secret identity for a credential.

    The digest is truncated because it is only an identity: 64 bits is far
    more than enough to keep a key ring apart, and a short value is safe to
    put in an admin payload.

    Args:
        key: The raw API key or access token.

    Returns:
        The first 16 hex characters of the key's SHA-256 digest.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class QuotaReading:
    """The normalized result of probing one credential.

    Attributes:
        probe: Probe identifier that produced the reading.
        credential: Fingerprint of the credential, never the credential.
        windows: Allowance windows the vendor reported.
        plan: Plan name, when the vendor names one.
        status: One of :data:`STATUS_OK`, :data:`STATUS_ERROR` or
            :data:`STATUS_UNSUPPORTED`.
        error: Human-readable reason when the reading failed.
        read_at: When the reading was taken.
        demote_at: Used share that counts as spent for routing.  ``100``
            keeps the vendor's own rule (percent or status).
    """

    probe: str
    credential: str
    windows: list[QuotaWindow] = field(default_factory=list)
    plan: str | None = None
    status: str = STATUS_OK
    error: str | None = None
    read_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    demote_at: float = 100.0

    @property
    def exhausted_until(self) -> datetime | None:
        """Return when the last blocking window resets, or ``None``.

        The credential can serve again once *every* exhausted window has
        reset, so this is the latest reset among them.  A window with no
        reset time yields ``None``: the vendor did not say when it clears, and
        guessing would either strand the credential or send traffic at it.
        """
        resets: list[datetime] = []
        for window in self.windows:
            # A window carried by credits is spent but not blocking.
            if (window.status or "").lower() == "credits":
                continue
            over_threshold = (
                window.used_percent is not None
                and window.used_percent >= self.demote_at
            )
            if not (window.exhausted or over_threshold):
                continue
            if window.resets_at is None:
                return None
            resets.append(window.resets_at)
        return max(resets) if resets else None

    def to_dict(self) -> dict:
        """Return the JSON shape consumed by the admin panel."""
        return {
            "probe": self.probe,
            "credential": self.credential,
            "plan": self.plan,
            "status": self.status,
            "error": self.error,
            "read_at": _iso(self.read_at),
            "demote_at": self.demote_at,
            "windows": [w.to_dict() for w in self.windows],
        }

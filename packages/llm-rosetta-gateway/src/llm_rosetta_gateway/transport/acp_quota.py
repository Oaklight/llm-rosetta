"""What an ACP agent says about its own allowance.

The vendor CLIs behind the ``acp`` lanes hold the subscription, so the gateway
has no credential to probe.  The agent does report the account's windows as it
works: Claude's adapter sends a ``usage_update`` whose ``_meta`` carries
``_claude/rateLimit`` with the five-hour and weekly utilization.  This module
turns that notification into a normalized reading; the transport records it,
and the quota service reads it like any other.

The reading is kept here rather than in :mod:`llm_rosetta_gateway.quota` on
purpose: the transport is the foundation layer and may not import the state
layer.  The store is process-wide, so a config reload that rebuilds the
transport keeps the last reading.  The quota service, which may import this
module, converts the same reading into the admin payload.

Nothing here is vendor-specific beyond the field names it tolerates: a
``rateLimit`` object under any ``_meta`` key is read, and a window shape it
does not know still becomes a window with whatever share it reported.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from llm_rosetta.contracts.quota import QuotaWindow

#: Probe id the transport's observations are stored under.  One id is enough:
#: a reading is keyed by the lane's profile, so two entries never overwrite
#: each other even when they share an account.
ACP_QUOTA_PROBE = "acp-usage"

#: The one status a reading from an agent can have: the agent answered.
STATUS_OK = "ok"

#: Window names an adapter uses for its rolling limits, and their lengths.
_WINDOW_NAMES = {
    "five_hour": ("5 hours", 5 * 3600),
    "five_hours": ("5 hours", 5 * 3600),
    "5h": ("5 hours", 5 * 3600),
    "seven_day": ("7 days", 7 * 86400),
    "seven_days": ("7 days", 7 * 86400),
    "7d": ("7 days", 7 * 86400),
    "daily": ("day", 86400),
    "weekly": ("week", 7 * 86400),
    "monthly": ("month", None),
}


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def lane_identity(provider_info: Any) -> str:
    """A stable, non-secret identity for one lane's login profile.

    The profile directory is what the sign-in writes to, so a renamed provider
    entry that keeps ``acp.home`` keeps the same identity; two entries with
    different profiles are different accounts even when they name one CLI.
    """
    return str(
        getattr(provider_info, "acp_state_dir", None)
        or getattr(provider_info, "config_name", "")
        or ""
    )


def lane_credential(provider_info: Any) -> str:
    """The fingerprint the quota payload names for this lane.

    The digest rule is the one :func:`quota.models.credential_fingerprint`
    uses; the two must agree, and a test pins that.  The raw profile path is
    never part of a payload.
    """
    identity = lane_identity(provider_info)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class LaneReading:
    """The allowance one agent reported during a turn.

    The shape matches :class:`~llm_rosetta_gateway.quota.models.QuotaReading`
    so the quota service renders it without a translation step, but it lives
    here so the transport needs no state-layer import.
    """

    probe: str
    credential: str
    windows: list[QuotaWindow] = field(default_factory=list)
    plan: str | None = None
    status: str = STATUS_OK
    error: str | None = None
    read_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    demote_at: float = 100.0
    advisory: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape consumed by the admin panel."""
        return {
            "probe": self.probe,
            "credential": self.credential,
            "plan": self.plan,
            "status": self.status,
            "error": self.error,
            "read_at": _iso(self.read_at),
            "demote_at": self.demote_at,
            "advisory": self.advisory,
            "windows": [w.to_dict() for w in self.windows],
        }


_READINGS: dict[str, LaneReading] = {}
_READINGS_LOCK = threading.Lock()


def clear_lane_readings() -> None:
    """Drop every recorded reading.

    The store is process-wide, so tests are the one caller that needs to start
    from nothing.
    """
    with _READINGS_LOCK:
        _READINGS.clear()


def lanes_reading(provider_info: Any) -> LaneReading | None:
    """The last allowance this lane's agent reported, if any."""
    with _READINGS_LOCK:
        return _READINGS.get(lane_identity(provider_info))


def _body_of(update: dict[str, Any]) -> dict[str, Any] | None:
    """The ``session/update`` payload, whichever envelope it arrived in."""
    body = update.get("update") if isinstance(update.get("update"), dict) else update
    return body if isinstance(body, dict) else None


def _rate_limit_of(meta: Any) -> dict[str, Any] | None:
    """The first rate-limit object under an adapter's ``_meta`` mapping.

    Claude names it ``_claude/rateLimit``; the search is by substring so an
    adapter that renames the vendor prefix still works.  Values are passed
    through: this layer reads them, it does not interpret them.
    """
    if not isinstance(meta, dict):
        return None
    for key, value in meta.items():
        if isinstance(value, dict) and "ratelimit" in str(key).lower():
            return value
    return None


def observation_of(update: dict[str, Any]) -> dict[str, Any] | None:
    """Return the rate-limit object carried by a ``usage_update``, if any."""
    body = _body_of(update)
    if body is None:
        return None
    kind = body.get("sessionUpdate") or body.get("session_update") or ""
    if kind != "usage_update":
        return None
    return _rate_limit_of(body.get("_meta") or body.get("meta"))


def _reset_of(value: Any) -> datetime | None:
    """A window's reset as an aware datetime; unix seconds or ISO text."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _percent_of(spec: dict[str, Any]) -> float | None:
    """The used share as 0-100.

    ``utilization`` is a fraction (Claude reports ``0.02`` for two percent);
    a field already named a percent is taken as written.  A value above one
    in ``utilization`` is treated as a percentage too: no adapter reports more
    than one full window as a fraction, and refusing the reading would hide an
    exhausted account.
    """
    for key in ("usedPercent", "used_percent", "percent"):
        value = spec.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(100.0, float(value)))
    value = spec.get("utilization")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        share = float(value)
        if share <= 1.0:
            share *= 100.0
        return max(0.0, min(100.0, share))
    return None


def _window(name: str, spec: dict[str, Any], parent: dict[str, Any]) -> QuotaWindow:
    friendly, span = _WINDOW_NAMES.get(name.lower(), (name.replace("_", " "), None))
    return QuotaWindow(
        name=friendly,
        used_percent=_percent_of(spec),
        span_seconds=span,
        resets_at=_reset_of(
            spec.get("resetsAt")
            if spec.get("resetsAt") is not None
            else spec.get("resets_at", parent.get("resetsAt", parent.get("resets_at")))
        ),
        status=str(spec.get("status") or parent.get("status") or ""),
    )


def windows_of(limit: dict[str, Any]) -> list[QuotaWindow]:
    """The allowance windows carried by one rate-limit object.

    The adapter's ``unifiedWindows`` mapping is the normal source.  A
    rate-limit object that only names one window is still reported as one.
    """
    windows: list[QuotaWindow] = []
    unified = limit.get("unifiedWindows") or limit.get("unified_windows")
    if isinstance(unified, dict):
        for key, spec in unified.items():
            if isinstance(spec, dict):
                windows.append(_window(str(key), spec, limit))
    if not windows:
        name = str(
            limit.get("rateLimitType") or limit.get("rate_limit_type") or "allowance"
        )
        windows.append(_window(name, limit, limit))
    return [window for window in windows if window.used_percent is not None]


def reading_of(update: dict[str, Any], provider_info: Any) -> LaneReading | None:
    """Normalize one ``usage_update`` into a reading, or ``None``.

    A usage update that carries no rate-limit metadata is a token-count
    update only; it says nothing about the account's allowance.
    """
    limit = observation_of(update)
    if limit is None:
        return None
    windows = windows_of(limit)
    if not windows:
        return None
    return LaneReading(
        probe=ACP_QUOTA_PROBE,
        credential=lane_credential(provider_info),
        windows=windows,
        status=STATUS_OK,
        read_at=datetime.now(timezone.utc),
    )


def record_reading(update: dict[str, Any], provider_info: Any) -> bool:
    """Record the reading *update* carries, if it carries one.

    Returns:
        Whether the update reported an allowance.  A token-count-only update
        leaves the last reading untouched and returns ``False``.
    """
    reading = reading_of(update, provider_info)
    if reading is None:
        return False
    with _READINGS_LOCK:
        _READINGS[lane_identity(provider_info)] = reading
    return True

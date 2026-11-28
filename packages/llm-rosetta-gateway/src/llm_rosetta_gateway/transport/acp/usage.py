"""Decoding what an ACP agent says about its allowance.

The vendor CLIs behind the ``acp`` lanes hold the subscription, so the gateway
has no credential to probe.  The agent does report the account's windows as it
works: Claude's adapter sends a ``usage_update`` whose ``_meta`` carries
``_claude/rateLimit`` with the five-hour and weekly utilization.  This module
only reads that notification into :class:`~llm_rosetta.contracts.quota.QuotaWindow`
values.  It keeps nothing: the quota layer owns the readings
(:mod:`llm_rosetta_gateway.quota.lane_allowance`), and the transport hands each
update to the sink its provider is bound to.

Nothing here is vendor-specific beyond the field names it tolerates: a
``rateLimit`` object under any ``_meta`` key is read, and a window shape it
does not know still becomes a window with whatever share it reported.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from llm_rosetta.contracts.quota import QuotaWindow

#: Probe id the transport's observations are stored under.  One id is enough:
#: a reading is keyed by the lane's profile, so two entries never overwrite
#: each other even when they share an account.
ACP_QUOTA_PROBE = "acp-usage"

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


def windows_reported(update: dict[str, Any]) -> list[QuotaWindow]:
    """The allowance windows one ``session/update`` carries; empty when none.

    A usage update that carries no rate-limit metadata is a token-count update
    only; it says nothing about the account's allowance.
    """
    limit = observation_of(update)
    return windows_of(limit) if limit is not None else []

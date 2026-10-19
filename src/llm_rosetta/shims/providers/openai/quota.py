"""ChatGPT plan quota probe — the Codex backend's usage endpoint.

Loaded once for the ``openai`` group.  A regular API-key provider never uses
it: the probe is bound only when a shim (or a provider entry) declares
``quota.probe: openai_codex``, as the Sign in with ChatGPT shim will.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from llm_rosetta.shims.quota_probe import InBandSignal, QuotaWindow
    from llm_rosetta.shims.quota_probe import QuotaProbe


def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying it.

    The token comes from the operator's own credential store and is only read
    to find the account id the usage endpoint needs; signature verification
    happens upstream when the token is used.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + padding))
    except (ValueError, json.JSONDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


class OpenAICodexProbe:
    """ChatGPT plan windows as the Codex backend reports them."""

    id = "openai_codex"
    USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
    default_url = USAGE_URL
    _AUTH_CLAIM = "https://api.openai.com/auth"

    def match(self, base_url: str) -> str | None:
        parsed = urlparse(base_url)
        host = (parsed.hostname or "").lower()
        if host == "chatgpt.com" and "/backend-api" in parsed.path:
            return self.USAGE_URL
        return None

    def headers(self, key: str) -> dict[str, str]:
        auth = _jwt_claims(key).get(self._AUTH_CLAIM)
        if isinstance(auth, dict):
            account_id = auth.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id:
                return {"chatgpt-account-id": account_id}
        return {}

    def parse(self, body: Any) -> tuple[str | None, list[QuotaWindow]]:
        from dataclasses import replace

        from llm_rosetta.shims.quota_probe import QuotaWindow
        from llm_rosetta.shims.quota_probe import QuotaProbeError, vendor_message

        if not isinstance(body, dict):
            raise QuotaProbeError("Codex usage reply is not an object")
        rate_limit = body.get("rate_limit")
        if not isinstance(rate_limit, dict):
            raise QuotaProbeError(
                "Codex usage reply has no 'rate_limit' object "
                f"(got {vendor_message(body)})"
            )
        allowed = rate_limit.get("allowed")
        windows: list[QuotaWindow] = []
        for key in ("primary_window", "secondary_window"):
            entry = rate_limit.get(key)
            if isinstance(entry, dict):
                windows.append(self._window(entry))
        if not windows:
            raise QuotaProbeError("Codex usage reply named no window")

        credits = body.get("credits")
        has_credits = isinstance(credits, dict) and bool(
            credits.get("has_credits") or credits.get("unlimited")
        )
        if has_credits:
            # A spent plan window does not stop the account when credits
            # carry the next turn: the vendor's own composer stays open.
            windows = [
                replace(window, status="credits") if window.exhausted else window
                for window in windows
            ]
        if allowed is False and not has_credits:
            # ``allowed: false`` is the authoritative "this account cannot
            # serve now" flag.  It can arrive with a percent below 100, so it
            # must not be derived from the share.
            first = windows[0]
            windows[0] = QuotaWindow(
                name=first.name,
                used_percent=first.used_percent,
                span_seconds=first.span_seconds,
                resets_at=first.resets_at,
                status="rate-limited",
            )
        plan = body.get("plan_type")
        return (plan if isinstance(plan, str) else None), windows

    @staticmethod
    def _window(entry: dict[str, Any]) -> QuotaWindow:
        from llm_rosetta.shims.quota_probe import QuotaWindow

        span = entry.get("limit_window_seconds")
        span_seconds = int(span) if isinstance(span, (int, float)) and span else None
        name = "Allowance"
        if span_seconds:
            if span_seconds % 86400 == 0:
                name = f"{span_seconds // 86400} days"
            elif span_seconds % 3600 == 0:
                name = f"{span_seconds // 3600} hours"
        used = entry.get("used_percent")
        resets_at = None
        reset_at = entry.get("reset_at")
        if isinstance(reset_at, (int, float)) and reset_at > 0:
            resets_at = datetime.fromtimestamp(reset_at, timezone.utc)
        elif isinstance(entry.get("reset_after_seconds"), (int, float)):
            resets_at = datetime.now(timezone.utc) + timedelta(
                seconds=float(entry["reset_after_seconds"])
            )
        return QuotaWindow(
            name=name,
            used_percent=float(used) if isinstance(used, (int, float)) else None,
            span_seconds=span_seconds,
            resets_at=resets_at,
            status="" if entry.get("allowed", True) else "rate-limited",
        )

    def describe_error(self, status_code: int, body: Any) -> str:
        from llm_rosetta.shims.quota_probe import vendor_message

        message = vendor_message(body)
        if status_code == 401:
            return "the plan's access token does not authenticate (HTTP 401)"
        if status_code == 403:
            return f"the account is not eligible for plan usage (HTTP 403: {message})"
        return f"HTTP {status_code}: {message}"

    def inband(self, status_code: int, body: Any) -> InBandSignal | None:
        from llm_rosetta.shims.quota_probe import InBandSignal
        from llm_rosetta.shims.quota_probe import vendor_message

        if status_code not in (429, 403):
            return None
        text = vendor_message(body).lower()
        if "subscription_sharing_usage_limit_exceeded" in text or status_code == 429:
            return InBandSignal(
                window=None,
                cooldown_seconds=60,
                reason="ChatGPT plan usage limit exceeded",
            )
        return None


PROBES: tuple[QuotaProbe, ...] = (OpenAICodexProbe(),)

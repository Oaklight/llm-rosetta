"""OpenCode quota probe — the Go subscription's rolling windows.

Loaded once for the ``opencode`` group; each leaf shim selects it with
``quota.probe: opencode_go``.  Module scope deliberately imports only the
standard library: shims are loaded before the gateway package, so the gateway
types this probe returns are imported inside the methods that build them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from llm_rosetta.shims.quota_probe import InBandSignal, QuotaWindow
    from llm_rosetta.shims.quota_probe import QuotaProbe


class OpenCodeGoProbe:
    """OpenCode Go subscription windows (rolling / weekly / monthly)."""

    id = "opencode_go"
    default_url: str | None = None

    # (vendor key, display name, span in seconds).  The monthly window is
    # anchored on the subscription anniversary, so its span varies.
    _WINDOWS = (
        ("rolling", "5 hours", 5 * 3600),
        ("weekly", "7 days", 7 * 24 * 3600),
        ("monthly", "Month", None),
    )

    def match(self, base_url: str) -> str | None:
        parsed = urlparse(base_url)
        host = (parsed.hostname or "").lower()
        if not (host == "opencode.ai" or host.endswith(".opencode.ai")):
            return None
        base = base_url.rstrip("/")
        if parsed.path.rstrip("/").endswith("/zen/go/v1"):
            return f"{base}/usage"
        # The Zen lane has no keyed usage endpoint: /zen/v1/usage and both
        # /credits paths 404, and the console reads the balance with a session
        # cookie the gateway does not hold.
        if "/zen/go/" in parsed.path:
            return f"{base}/usage"
        return ""

    def headers(self, key: str) -> dict[str, str]:
        return {}

    def parse(self, body: Any) -> tuple[str | None, list[QuotaWindow]]:
        from llm_rosetta.shims.quota_probe import QuotaWindow
        from llm_rosetta.shims.quota_probe import (
            QuotaProbeError,
            parse_iso,
            vendor_message,
        )

        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            raise QuotaProbeError(
                "OpenCode Go's reply has no 'usage' object "
                f"(got {vendor_message(body)})"
            )
        windows: list[QuotaWindow] = []
        for key, name, span in self._WINDOWS:
            entry = usage.get(key)
            if not isinstance(entry, dict):
                continue
            percent = entry.get("percent")
            windows.append(
                QuotaWindow(
                    name=name,
                    used_percent=(
                        float(percent) if isinstance(percent, (int, float)) else None
                    ),
                    span_seconds=span,
                    resets_at=parse_iso(entry.get("resetsAt")),
                    status=str(entry.get("status") or ""),
                )
            )
        if not windows:
            raise QuotaProbeError(
                "OpenCode Go's reply named no rolling, weekly or monthly window "
                f"(got {vendor_message(body)})"
            )
        return None, windows

    def describe_error(self, status_code: int, body: Any) -> str:
        from llm_rosetta.shims.quota_probe import vendor_message

        if status_code == 401:
            return "the key does not authenticate (HTTP 401)"
        if status_code == 403:
            return "the key's member or workspace has no Go subscription (HTTP 403)"
        return f"HTTP {status_code}: {vendor_message(body)}"

    def inband(self, status_code: int, body: Any) -> InBandSignal | None:
        from llm_rosetta.shims.quota_probe import InBandSignal
        from llm_rosetta.shims.quota_probe import vendor_message

        if status_code != 429:
            return None
        window = None
        metadata = body.get("metadata") if isinstance(body, dict) else None
        if isinstance(metadata, dict):
            name = metadata.get("limitName") or metadata.get("limit_name")
            if isinstance(name, str):
                window = name
        text = vendor_message(body).lower()
        if window is None:
            for key, display, _span in self._WINDOWS:
                if key in text or display.lower() in text:
                    window = key
                    break
        cooldown = None
        if window is not None:
            for key, _display, span in self._WINDOWS:
                if key == window and span is not None:
                    cooldown = span
                    break
        return InBandSignal(
            window=window,
            cooldown_seconds=cooldown or 60,
            reason=f"OpenCode Go rate limit ({window or 'window unknown'})",
        )


PROBES: tuple[QuotaProbe, ...] = (OpenCodeGoProbe(),)

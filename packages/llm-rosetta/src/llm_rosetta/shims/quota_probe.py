"""The quota probe contract, owned by the shim layer.

A probe is a vendor fact — endpoint, auth header, body shape, error wording —
so every implementation ships inside the shim that declares it (``quota.py``
next to ``provider.yaml``).  The contract and the value shapes it returns live
here, in the translation core; the gateway owns the runtime state, the HTTP
call and the admin surface.

This direction is deliberate: ``shims`` must never import ``gateway``, and the
loader executes every shim's ``quota.py`` while building the registry.

A shim's ``quota.py`` exports ``PROBES`` (a tuple of probe objects).  The shim
loader attaches them to the :class:`~llm_rosetta.shims.provider_shim.ProviderShim`,
and the gateway resolves the one named by ``quota.probe`` against them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

# A vendor message longer than this is truncated before it is stored: it goes
# into an admin payload and a log line, and nothing needs more than the head.
_MAX_ERROR_CHARS = 300


class QuotaProbeError(Exception):
    """A probe could not decode the vendor's answer or the endpoint refused."""

    def __init__(self, message: str, *, status: str = "error") -> None:
        super().__init__(message)
        self.status = status


def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as ISO 8601 with a ``Z`` suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp, tolerating a trailing ``Z``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def vendor_message(body: Any) -> str:
    """Extract the vendor's own message from an error body, if any."""
    if isinstance(body, str):
        text = body
    else:
        text = json.dumps(body, ensure_ascii=False)
    try:
        parsed = json.loads(text)
    except ValueError:
        return text[:_MAX_ERROR_CHARS]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:_MAX_ERROR_CHARS]
        if isinstance(error, str):
            return error[:_MAX_ERROR_CHARS]
        if isinstance(parsed.get("message"), str):
            return parsed["message"][:_MAX_ERROR_CHARS]
    return text[:_MAX_ERROR_CHARS]


@dataclass(frozen=True, slots=True)
class QuotaWindow:
    """One allowance window reported by a vendor.

    Attributes:
        name: Human-readable window name (``"5 hours"``, ``"7 days"``).
        used_percent: Share of the window already used, 0–100, when the
            vendor reports a share.  ``None`` when it does not.
        span_seconds: Length of the window, when known.  ``None`` for a
            calendar-month window whose length varies.
        resets_at: When the window renews, when the vendor says.
        status: The vendor's own status word (``"ok"``, ``"rate-limited"``).
            Empty when the vendor reports none.
    """

    name: str
    used_percent: float | None = None
    span_seconds: int | None = None
    resets_at: datetime | None = None
    status: str = ""

    @property
    def exhausted(self) -> bool:
        """Whether this window currently blocks the credential.

        A window is exhausted when its status is not ``ok`` or when its used
        share has reached 100%.  A reset time in the past clears the reading
        even if the vendor still said ``rate-limited`` when it was taken.
        """
        if self.resets_at is not None and self.resets_at <= datetime.now(timezone.utc):
            return False
        # "credits" is the Codex case: the plan window is spent but the
        # account can still serve from purchased credits, so the window is
        # not a demotion — not even at 100%.
        if self.status:
            lowered = self.status.lower()
            if lowered == "credits":
                return False
            if lowered not in ("ok", "active", "allowed"):
                return True
        return self.used_percent is not None and self.used_percent >= 100

    def to_dict(self) -> dict:
        """Return the JSON shape consumed by the admin panel."""
        return {
            "name": self.name,
            "used_percent": self.used_percent,
            "remaining_percent": (
                None
                if self.used_percent is None
                else round(max(0.0, 100.0 - self.used_percent), 2)
            ),
            "span_seconds": self.span_seconds,
            "resets_at": _iso(self.resets_at),
            "status": self.status,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True, slots=True)
class InBandSignal:
    """A quota signal read from an upstream error response.

    Attributes:
        window: Vendor window name the error named, when it named one.
        cooldown_seconds: How long to demote the credential for.  ``None``
            means "until the probe's next reading says otherwise".
        reason: Short description for logs and the admin surface.
    """

    window: str | None = None
    cooldown_seconds: int | None = None
    reason: str = "upstream reported the allowance as spent"


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    """Where and how to read a credential's allowance."""

    url: str
    method: str = "GET"
    headers: dict[str, str] | None = None


@runtime_checkable
class QuotaProbe(Protocol):
    """The contract a shim's ``quota.py`` implements.

    Attributes:
        id: Stable identifier named by ``quota.probe``.
        default_url: Endpoint used when the probe is named explicitly and the
            provider's ``base_url`` does not identify the vendor (a plan token
            used against ``api.openai.com``, say).
    """

    @property
    def id(self) -> str:
        """Stable identifier named by ``quota.probe``."""
        ...

    @property
    def default_url(self) -> str | None:
        """Endpoint used when the probe is named explicitly and no URL is derivable."""
        ...

    def match(self, base_url: str) -> str | None:
        """Return the usage URL for *base_url*, or ``None`` if not this vendor.

        An empty string means the vendor is recognised but this lane has no
        keyed usage endpoint; the caller reports that instead of probing a
        URL that would 404.
        """
        ...

    def headers(self, key: str) -> dict[str, str]:
        """Return probe-specific headers for *key* (auth is added by caller)."""
        ...

    def parse(self, body: Any) -> tuple[str | None, list[Any]]:
        """Decode a successful answer into ``(plan, windows)``.

        Raises:
            QuotaProbeError: The answer carried no window this probe
                understands.
        """
        ...

    def describe_error(self, status_code: int, body: Any) -> str:
        """Return an operator-readable reason for a refused probe."""
        ...

    def inband(self, status_code: int, body: Any) -> Any | None:
        """Classify an inference error as a spent allowance, if it is one."""
        ...


@dataclass(frozen=True, slots=True)
class ProbeMatch:
    """The probe for a provider plus where it would read.

    Attributes:
        probe: The probe that owns this vendor.
        url: The usage URL, or ``None`` when the probe cannot read this lane.
        unsupported: Why the lane has no keyed reading, when ``url`` is None.
    """

    probe: QuotaProbe
    url: str | None
    unsupported: str | None = None


def probe_ids(probes: tuple[QuotaProbe, ...]) -> list[str]:
    """Return the identifiers of *probes*."""
    return [probe.id for probe in probes]


_IN_BAND_ONLY = "in-band signals only: this lane exposes no usage endpoint"


def resolve_probe(
    base_url: str,
    *,
    probes: tuple[QuotaProbe, ...],
    probe_id: str | None = None,
    url: str | None = None,
    in_band: bool = False,
) -> ProbeMatch | None:
    """Resolve which of *probes* reads *base_url* and where.

    Args:
        base_url: The provider's resolved base URL.
        probes: Probe implementations the shim attached, in declaration order.
        probe_id: Explicit probe name from provider config or shim.
        url: Explicit usage URL override.

    Returns:
        The match, or ``None`` when no probe knows this vendor.

    Raises:
        ValueError: *probe_id* names no probe the shim attached.
    """
    if probe_id:
        probe = next((p for p in probes if p.id == probe_id), None)
        if probe is None:
            known = ", ".join(probe_ids(probes)) or "none"
            raise ValueError(
                f"Unknown quota probe {probe_id!r}; this shim provides: {known}"
            )
        if in_band:
            # No endpoint to read: the probe is kept for its ``inband()``
            # classifier, and nothing polls it.
            return ProbeMatch(probe=probe, url=None, unsupported=_IN_BAND_ONLY)
        if url:
            return ProbeMatch(probe=probe, url=url)
        matched = probe.match(base_url)
        if matched == "":
            return ProbeMatch(
                probe=probe,
                url=None,
                unsupported="this lane has no keyed usage endpoint",
            )
        if matched is None:
            if probe.default_url:
                return ProbeMatch(probe=probe, url=probe.default_url)
            return ProbeMatch(
                probe=probe,
                url=None,
                unsupported=(
                    f"probe {probe.id!r} has no endpoint for this base_url; "
                    "set quota.url"
                ),
            )
        return ProbeMatch(probe=probe, url=matched)

    for probe in probes:
        if in_band:
            return ProbeMatch(probe=probe, url=None, unsupported=_IN_BAND_ONLY)
        matched = probe.match(base_url)
        if matched == "":
            return ProbeMatch(
                probe=probe,
                url=None,
                unsupported="this lane has no keyed usage endpoint",
            )
        if matched:
            return ProbeMatch(probe=probe, url=url or matched)
    return None


__all__ = [
    "InBandSignal",
    "ProbeMatch",
    "ProbeTarget",
    "QuotaProbe",
    "QuotaProbeError",
    "QuotaWindow",
    "parse_iso",
    "probe_ids",
    "resolve_probe",
    "vendor_message",
]

"""Per-request context extraction middleware for the gateway.

Consolidates duplicated context extraction (client IP, API format,
admin path detection, request ID) into a single ``before_request`` hook
that populates a :data:`request_context_var` contextvar.  Downstream
hooks and handlers read from the contextvar instead of re-extracting.

This module depends on :mod:`error_format` for ``detect_api_format``
and ``is_admin_path``, and on :mod:`headers` for ``get_request_id``.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from ipaddress import (
    IPv4Address,
    IPv4Network,
    IPv6Address,
    IPv6Network,
    ip_address,
    ip_network,
)
from typing import Any

from .error_format import detect_api_format, is_admin_path
from .headers import get_request_id


# ---------------------------------------------------------------------------
# RequestContext dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContext:
    """Immutable per-request context populated by the early middleware hook.

    Attributes:
        request_id: Unique identifier for the request, either extracted
            from the ``X-Request-Id`` header or auto-generated.
        client_ip: The client address, resolved once under the trusted-proxy
            policy (the TCP peer unless it is a trusted proxy that names one).
        https: Whether the request arrived over HTTPS, as the peer or a
            trusted proxy reports it.
        api_format: Detected API format based on the request path
            (``"openai"``, ``"anthropic"``, ``"google"``), or ``None``
            for paths where format detection is not meaningful (e.g.
            admin, health).
        is_admin: Whether the request targets an admin panel path.
        request_start: Monotonic timestamp captured when the request
            context is created, used to compute gateway-level TTFB.
            ``None`` for contexts created without timing (e.g. tests).
        user_agent: The ``User-Agent`` header, truncated.  ``None`` when the
            client sent none.  Kept for the request log: it names the agent
            behind a key better than the key label can when one key serves
            several clients.
    """

    request_id: str
    client_ip: str
    api_format: str | None
    is_admin: bool
    request_start: float | None = None
    user_agent: str | None = None
    https: bool = False


# ---------------------------------------------------------------------------
# Contextvar
# ---------------------------------------------------------------------------

request_context_var: contextvars.ContextVar[RequestContext | None] = (
    contextvars.ContextVar("request_context", default=None)
)


# ---------------------------------------------------------------------------
# Client IP extraction (canonical implementation)
# ---------------------------------------------------------------------------


#: A User-Agent longer than this is cut: it goes into a log row, and no agent
#: needs more than this to be recognised.
_USER_AGENT_MAX_CHARS = 200


def extract_user_agent(request: Any) -> str | None:
    """Return the request's ``User-Agent`` header, truncated, or ``None``."""
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    value = headers.get("User-Agent") or headers.get("user-agent")
    if not value:
        return None
    text = str(value).strip()
    return text[:_USER_AGENT_MAX_CHARS] or None


class TrustedProxies:
    """The set of proxy addresses whose forwarding headers are believed.

    Built from ``server.trusted_proxies``: a list of CIDRs (a bare address is a
    single-host network).  Empty trusts nobody.
    """

    __slots__ = ("_networks",)

    def __init__(self, networks: tuple[IPv4Network | IPv6Network, ...] = ()) -> None:
        self._networks = networks

    @classmethod
    def parse(cls, raw: Any, label: str = "server.trusted_proxies") -> TrustedProxies:
        """Read the configured list; raise ``ValueError`` naming a bad item."""
        if raw is None:
            return cls()
        if not isinstance(raw, (list, tuple)):
            raise ValueError(
                f"config: {label} must be a list of CIDRs, got {type(raw).__name__}"
            )
        networks: list[IPv4Network | IPv6Network] = []
        for index, item in enumerate(raw):
            try:
                if not isinstance(item, str):
                    raise ValueError("not a string")
                networks.append(ip_network(item.strip(), strict=False))
            except ValueError as exc:
                raise ValueError(
                    f"config: {label}[{index}] {item!r} is not a valid CIDR "
                    f"or address ({exc})"
                ) from exc
        return cls(tuple(networks))

    def __bool__(self) -> bool:
        return bool(self._networks)

    def __contains__(self, address: object) -> bool:
        ip = _address(address)
        return ip is not None and any(
            ip.version == net.version and ip in net for net in self._networks
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrustedProxies) and self._networks == other._networks

    __hash__ = None  # type: ignore[assignment]


def _address(value: object) -> IPv4Address | IPv6Address | None:
    """Parse *value* as an IP address (IPv4-mapped IPv6 becomes IPv4), or ``None``."""
    if not isinstance(value, str):
        return None
    try:
        ip = ip_address(value.strip())
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def resolve_client_ip(request: Any, trusted: TrustedProxies) -> str:
    """Return the client address of *request* under the trusted-proxy rule.

    The rule is written out in ``tests/gateway/test_trusted_proxies.py``: a peer
    outside *trusted* is the client; a trusted peer lets ``X-Forwarded-For``
    speak, read from the right, skipping trusted hops; ``X-Real-IP`` is used
    only when the chain is absent.  Anything that names no client yields the
    peer, and a request with no peer yields ``"unknown"``.
    """
    addr = getattr(request, "client_addr", None)
    peer_text = (
        addr[0] if isinstance(addr, (tuple, list)) and addr and addr[0] else None
    )
    if peer_text is None:
        return "unknown"
    peer_ip = _address(peer_text)
    peer = str(peer_ip) if peer_ip is not None else str(peer_text)
    if peer_ip is None or peer not in trusted:
        return peer
    forwarded = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        for entry in reversed(forwarded.split(",")):
            ip = _address(entry)
            if ip is None:
                return peer
            if str(ip) not in trusted:
                return str(ip)
        return peer
    real = _address(request.headers.get("x-real-ip") or "")
    if real is not None and str(real) not in trusted:
        return str(real)
    return peer


class ProxyPolicy:
    """The live trusted-proxy set, swapped when the config reloads.

    One instance per app; the request-context hook reads it, so the client
    address and the scheme are decided in one place.
    """

    __slots__ = ("trusted",)

    def __init__(self, trusted: TrustedProxies | None = None) -> None:
        self.trusted = trusted or TrustedProxies()

    def client_ip(self, request: Any) -> str:
        """The client address of *request* (see :func:`resolve_client_ip`)."""
        return resolve_client_ip(request, self.trusted)

    def is_https(self, request: Any) -> bool:
        """Whether *request* arrived over HTTPS, as a trusted proxy reports it."""
        addr = getattr(request, "client_addr", None)
        peer = addr[0] if isinstance(addr, (tuple, list)) and addr else None
        if peer not in self.trusted:
            return False
        proto = request.headers.get("x-forwarded-proto") or ""
        return proto.split(",")[0].strip().lower() == "https"


# ---------------------------------------------------------------------------
# Context setup hook
# ---------------------------------------------------------------------------


def setup_request_context(policy: ProxyPolicy) -> Any:
    """Return a ``before_request`` hook that populates :data:`request_context_var`.

    The returned hook should be registered as the **earliest**
    ``before_request`` handler so that auth, rate-limiting, and all
    downstream code can read the context.

    Args:
        policy: The trusted-proxy policy that decides the client address and
            the scheme; every consumer reads the result from the context.

    Returns:
        An async callable suitable for ``app.before_request()``.
    """

    async def _setup_request_context_hook(request: Any) -> None:
        path: str = request.path
        admin = is_admin_path(path)

        # For admin and public/health paths the api_format is not
        # meaningful — set to None so consumers can distinguish.
        api_format: str | None = None if admin else detect_api_format(path)

        ctx = RequestContext(
            request_id=get_request_id(request),
            client_ip=policy.client_ip(request),
            https=policy.is_https(request),
            api_format=api_format,
            is_admin=admin,
            request_start=time.monotonic(),
            user_agent=extract_user_agent(request),
        )
        request_context_var.set(ctx)

    return _setup_request_context_hook

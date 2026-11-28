"""Probe execution, caching and the bridge to routing.

This module is the only place that talks to a vendor's usage endpoint.  It
resolves the probe from the shim and provider config, probes each credential
in the provider's key ring, and keeps the last reading in
:data:`~.state.QUOTA_STATE` so the router and the admin panel read one truth.

The raw key never leaves this module: readings are stored and returned
against a fingerprint.

What routing and the admin panel read from that state is in :mod:`.allowance`;
what a live request reports back (quota errors, 429s, refusals, successes) is
in :mod:`.signals`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from llm_rosetta_gateway._vendor.httpclient import AsyncClient
from llm_rosetta_gateway._vendor.httpclient import Response as HttpResponse
from llm_rosetta_gateway.observability.sanitize import scrub_credential_patterns

# Callers import the quota functions from this module, and tests patch them
# here (``quota.service.provider_spent``, ``note_lane_refusal``, ...); most
# callers import inside the function, so a patch on this module reaches them.
# The read side lives in .allowance and the in-band signals in .signals; this
# module re-exports both so those imports and patches keep working.
#
# One exception: code that lives in .allowance itself reads these names from
# its own globals, so a patch on this module does not reach it.
# ``allowance.routing_facts`` calls ``provider_spent`` that way -- patching
# ``quota.service.provider_spent`` leaves its result unchanged, and only
# ``quota.allowance.provider_spent`` flips it.  A test that patches the facade
# and asserts on the routing snapshot would be green for the wrong reason;
# patch the defining module instead.
from ._common import provider_config_key  # noqa: F401
from .allowance import (  # noqa: F401
    READING_MAX_AGE_SECONDS,
    Allowance,
    provider_allowance,
    provider_cooldown,
    provider_exhausted_until,
    provider_headroom,
    provider_spent,
    provider_windows,
    reading_max_age,
    routing_facts,
)
from .models import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    QuotaReading,
    credential_fingerprint,
)
from .probes import (
    PROBE_ROLE_API_KEY,
    ProbeRefused,
    QuotaProbe,
    QuotaProbeError,
    probe_credential_role,
    probe_fetch,
    resolve_probe,
)
from .signals import (  # noqa: F401
    clear_upstream_rate_limit,
    note_lane_refusal,
    note_transient_refusal,
    note_upstream_quota_error,
    note_upstream_rate_limit,
    note_upstream_success,
    provider_offline_until,
)
from .state import QUOTA_STATE

logger = logging.getLogger("llm-rosetta-gateway")

# A page refresh should not hit the vendor every time, but a reading older than
# a minute is cheap to refresh and the windows move in minutes, not seconds.
DEFAULT_TTL_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 10.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _quota_declaration(config: Any, name: str) -> dict[str, Any]:
    """Return the merged quota declaration for a provider.

    The shim supplies vendor defaults; a ``quota`` object in the provider
    config overrides them field by field.
    """
    from llm_rosetta.api import get_shim

    cfg = getattr(config, "_raw_providers", {}).get(name, {}) or {}
    declared: dict[str, Any] = {}
    shim_name = getattr(config, "provider_shim_names", {}).get(name)
    shim = get_shim(shim_name) if shim_name else None
    shim_quota = getattr(shim, "quota", None)
    if shim_quota is not None:
        declared.update(shim_quota.as_dict())
    override = cfg.get("quota")
    if isinstance(override, dict):
        declared.update(override)
    return declared


def resolve_provider_probe(
    config: Any, name: str
) -> tuple[QuotaProbe | None, str | None, str | None]:
    """Return ``(probe, url, unsupported_reason)`` for a configured provider.

    ``(None, None, reason)`` covers every "nothing to read here" case: a
    keyless lane, an embedding provider, or a vendor no probe knows.
    """
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is None:
        return None, None, "provider is not loaded"
    if pinfo.keyless:
        return None, None, "the lane is keyless, so it has no per-key allowance"
    if not getattr(pinfo, "key_ring", None) or len(pinfo.key_ring) == 0:
        return None, None, "the provider holds no credential to probe"

    # Resolution happened when the config was built (the shim attaches its
    # probe implementations); this only reports what was bound.
    probe = getattr(pinfo, "quota_probe_impl", None)
    if probe is None:
        return (
            None,
            None,
            getattr(pinfo, "quota_unsupported", None)
            or "no quota probe is declared by this shim",
        )
    if pinfo.quota_usage_url is None:
        # The probe is bound for its in-band classifier only: say so, rather
        # than reporting a readable quota with no endpoint.
        return probe, None, pinfo.quota_unsupported or "no quota endpoint"
    return probe, pinfo.quota_usage_url, None


def _probe_headers(
    pinfo: Any,
    probe: QuotaProbe,
    credential: str,
    *,
    advisory: bool = False,
) -> dict[str, str]:
    """Build the probe request headers for one credential.

    An advisory read uses the probe's own headers only.  ``auth_headers_for``
    stamps whatever credential it is handed into the provider's own header —
    ``x-api-key`` for some vendors — and the control key is not a model
    credential: sending it that way describes the second key to the wrong
    plane, and the control plane documents one scheme.  The model credential
    is not sent either, because that plane answers it with 401.
    """
    headers = {"Accept": "application/json"}
    if not advisory:
        headers.update(pinfo.auth_headers_for(credential))
    headers.update(probe.headers(credential))
    return headers


async def _probe_secret(
    pinfo: Any,
    probe: QuotaProbe,
    key: str,
) -> tuple[str, bool, str | None]:
    """The credential *probe* sends, and whether its reading is advisory.

    A probe that declares the model's own role is read with the key the
    caller holds.  Any other role is a credential the shim's ``auth.py``
    serves, and a reading made with it is advisory: it describes the account
    and never decides the lane.

    Returns:
        ``(credential, advisory, unsupported)``.  ``credential`` is empty when
        the provider cannot serve the role, and ``unsupported`` says why.  No
        request is made in that case: the issue's rule is that a missing
        control credential reports the quota unsupported rather than probing
        with the wrong one.
    """
    from llm_rosetta.contracts.auth import (
        AuthExchangeError,
        InteractiveSetupRequired,
        credential_for,
    )

    role = probe_credential_role(probe)
    if role == PROBE_ROLE_API_KEY:
        return key, False, None
    exchange = getattr(pinfo, "auth_exchange", None)
    ctx = getattr(pinfo, "auth_context", None)
    if exchange is None or ctx is None:
        return (
            "",
            True,
            (
                f"the {role!r} credential is unavailable: this provider does not "
                "sign in through a shim exchange"
            ),
        )
    try:
        secret = await credential_for(exchange, ctx, role)
    except InteractiveSetupRequired as exc:
        return "", True, f"no {role} credential: {exc}"
    except AuthExchangeError as exc:
        return (
            "",
            True,
            f"no {role} credential: {scrub_credential_patterns(str(exc))}",
        )
    if not secret:
        return "", True, f"no {role} credential is stored"
    return secret, True, None


async def _probe_multi(
    client: Any,
    pinfo: Any,
    probe: QuotaProbe,
    url: str,
    secret: str,
    credential: str,
    *,
    demote_at: float,
    advisory: bool,
) -> QuotaReading:
    """Run a probe's own multi-step read and normalize the answer.

    The probe drives the calls through ``read(url, headers)``, which returns
    ``(status_code, body)`` and never raises for a non-2xx answer: describing
    a refusal is the probe's job, and it has the vendor's wording.
    """
    headers = _probe_headers(pinfo, probe, secret, advisory=advisory)

    async def read(target: str, extra: dict[str, str] | None = None) -> tuple[int, Any]:
        resp = await client.get(target, headers={**headers, **(extra or {})})
        assert isinstance(resp, HttpResponse)
        try:
            body: Any = resp.json()
        except Exception:
            body = resp.text
        return resp.status_code, body

    fetch = probe_fetch(probe)
    assert fetch is not None
    try:
        plan, windows = await fetch(read, secret, url)
    except ProbeRefused as exc:
        return QuotaReading(
            probe=probe.id,
            credential=credential,
            status=STATUS_ERROR,
            error=probe.describe_error(exc.status_code, exc.body),
            demote_at=demote_at,
            advisory=advisory,
        )
    except QuotaProbeError as exc:
        return QuotaReading(
            probe=probe.id,
            credential=credential,
            status=exc.status,
            error=str(exc),
            demote_at=demote_at,
            advisory=advisory,
        )
    return QuotaReading(
        probe=probe.id,
        credential=credential,
        windows=windows,
        plan=plan,
        status=STATUS_OK,
        demote_at=demote_at,
        advisory=advisory,
    )


async def _probe_one(
    pinfo: Any,
    probe: QuotaProbe,
    url: str,
    key: str,
    *,
    demote_at: float = 100.0,
) -> QuotaReading:
    """Read one credential and normalize the answer."""
    credential = credential_fingerprint(key)
    secret, advisory, unsupported = await _probe_secret(pinfo, probe, key)
    if not secret:
        return QuotaReading(
            probe=probe.id,
            credential=credential,
            status=STATUS_UNSUPPORTED,
            error=unsupported,
            demote_at=demote_at,
            advisory=advisory,
        )
    proxy = getattr(pinfo, "proxy_url", None)
    fetch = probe_fetch(probe)
    try:
        async with AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS, proxy=proxy) as client:
            if fetch is not None:
                return await _probe_multi(
                    client,
                    pinfo,
                    probe,
                    url,
                    secret,
                    credential,
                    demote_at=demote_at,
                    advisory=advisory,
                )
            headers = _probe_headers(pinfo, probe, secret, advisory=advisory)
            resp = await client.get(url, headers=headers)
            assert isinstance(resp, HttpResponse)
            body: Any
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            if resp.status_code >= 400:
                return QuotaReading(
                    probe=probe.id,
                    credential=credential,
                    status=STATUS_ERROR,
                    error=probe.describe_error(resp.status_code, body),
                    demote_at=demote_at,
                    advisory=advisory,
                )
            try:
                plan, windows = probe.parse(body)
            except QuotaProbeError as exc:
                return QuotaReading(
                    probe=probe.id,
                    credential=credential,
                    status=exc.status,
                    error=str(exc),
                    demote_at=demote_at,
                    advisory=advisory,
                )
            return QuotaReading(
                probe=probe.id,
                credential=credential,
                windows=windows,
                plan=plan,
                status=STATUS_OK,
                demote_at=demote_at,
                advisory=advisory,
            )
    except Exception as exc:  # noqa: BLE001 - report, never raise into admin
        logger.warning(
            "Quota probe %s failed for %s: %s",
            probe.id,
            pinfo.name,
            scrub_credential_patterns(str(exc)),
        )
        return QuotaReading(
            probe=probe.id,
            credential=credential,
            status=STATUS_ERROR,
            error=f"probe request failed: {scrub_credential_patterns(str(exc))}",
            demote_at=demote_at,
        )


def _lane_reports_its_own_quota(pinfo: Any) -> bool:
    """Whether this provider's agent reports its allowance during a turn.

    A non-HTTP keyless lane does: its agent sends the windows, or a declared
    local command prints them.  A keyless HTTP lane (a local server, say) has
    no endpoint this module knows how to read, but a command it declares makes
    the lane its own source too.
    """
    if not bool(getattr(pinfo, "keyless", False)):
        return False
    return bool(
        (getattr(pinfo, "transport", "http") or "http") != "http"
        or getattr(pinfo, "quota_local_argv", ())
    )


async def _lane_quota_payload(
    name: str, pinfo: Any, ttl: float, *, force: bool = False
) -> dict[str, Any]:
    """The quota payload of a lane whose allowance comes from its own agent.

    Two sources exist.  An agent may send its windows during a turn (Claude's
    ``usage_update`` metadata); that reading is already in the store.  A
    vendor whose ACP surface says nothing about the account can instead
    declare a local command (Antigravity's ``/usage``); the gateway runs it
    here, with the lane's profile, and the shim probe decodes the output.

    A local command is read when the stored reading is older than *ttl* or a
    refresh was asked for.  Its failure is stored too, so a broken command is
    not rerun on every card open.
    """
    from ..transport.acp.quota import (
        ACP_QUOTA_PROBE,
        STATUS_ERROR,
        lane_credential,
        lane_lock,
        lanes_reading,
        refresh_local,
    )

    credential = lane_credential(pinfo)
    local_argv = tuple(getattr(pinfo, "quota_local_argv", ()) or ())
    probe = getattr(pinfo, "quota_probe_impl", None)
    refreshed = False
    warning: str | None = None
    reading = lanes_reading(pinfo)
    if local_argv and probe is None:
        warning = (
            "quota.local is declared without a probe to decode it; "
            "set quota.probe on this entry or in its shim"
        )
    elif local_argv:
        # One command per profile at a time: two cards, or the poll and a
        # card, would otherwise run the vendor CLI against one profile folder.
        async with lane_lock(pinfo):
            reading = lanes_reading(pinfo)
            fresh = (
                reading is not None
                and not force
                and (datetime.now(timezone.utc) - reading.read_at).total_seconds() < ttl
            )
            if not fresh:
                result = await refresh_local(
                    pinfo,
                    probe,
                    local_argv,
                    getattr(pinfo, "quota_local_env", None),
                    timeout=float(getattr(pinfo, "quota_local_timeout", 30.0)),
                )
                kept = lanes_reading(pinfo)
                if (
                    result.status == STATUS_ERROR
                    and kept is not None
                    and kept.status != STATUS_ERROR
                ):
                    # The agent's own turn is still the better reading; the
                    # failed command is a warning beside it, not a replacement.
                    reading = kept
                    warning = result.error
                else:
                    reading = result
                refreshed = True
    probe_id = getattr(probe, "id", None) if local_argv else None
    payload: dict[str, Any] = {
        "provider": name,
        "supported": True,
        "probe": str(reading.probe if reading else (probe_id or ACP_QUOTA_PROBE)),
        "status": (
            "unknown"
            if reading is None
            else (STATUS_ERROR if reading.status == STATUS_ERROR else STATUS_OK)
        ),
        "refreshed": refreshed,
        "ttl_seconds": ttl,
        "credentials": [],
        "routing": routing_facts(pinfo, name),
    }
    if reading is None:
        payload["reason"] = (
            "This lane's agent has not reported its allowance yet. Send one "
            "request through it, or press Test, then open Quota again."
        )
    else:
        rendered = reading.to_dict()
        payload["credentials"] = [
            {"credential": credential, "reading": rendered, "demotion": None}
        ]
        payload["read_at"] = rendered.get("read_at")
        if reading.status == STATUS_ERROR and reading.error:
            payload["reason"] = reading.error
    if warning:
        payload["warning"] = warning
    block = provider_cooldown(name, pinfo)
    if block is not None:
        payload["provider_block"] = block
    return payload


async def read_provider_quota(
    config: Any,
    name: str,
    *,
    force: bool = False,
    ttl: float = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    """Return the normalized quota of every credential of a provider.

    Args:
        config: The live :class:`~gateway.config.GatewayConfig`.
        name: Provider config key.
        force: Ignore cached readings.
        ttl: Maximum age of a cached reading, in seconds.

    Returns:
        A JSON-ready payload.  ``supported`` is ``False`` with a ``reason``
        when there is nothing to probe; the credential list never contains a
        key, only its fingerprint.
    """
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is not None and _lane_reports_its_own_quota(pinfo):
        # A keyless agent lane is its own quota source: the agent sends its
        # windows during a turn, or a declared local command prints them.
        # There is no key ring and no HTTP endpoint to probe.
        return await _lane_quota_payload(name, pinfo, ttl, force=force)

    probe, url, unsupported = resolve_provider_probe(config, name)
    if probe is None or url is None:
        payload: dict[str, Any] = {
            "provider": name,
            "supported": False,
            "status": STATUS_UNSUPPORTED,
            "reason": unsupported,
            "credentials": [],
        }
        # A blocked lane has something to say even with nothing to read: an
        # in-band provider is out of rotation for a bounded time, and this is
        # the only place the operator can see it.
        if pinfo is not None:
            block = provider_cooldown(name, pinfo)
            if block is not None:
                payload["provider_block"] = block
            payload["routing"] = routing_facts(pinfo, name)
        return payload

    pinfo = config.providers[name]
    declaration = _quota_declaration(config, name)
    demote_at = float(declaration.get("demote_at", 100.0))
    keys = pinfo.key_ring.keys()
    refreshed = False
    for key in keys:
        credential = credential_fingerprint(key)
        cached = QUOTA_STATE.get_reading(probe.id, credential)
        if not force and cached is not None and QUOTA_STATE.reading_age(cached) < ttl:
            continue
        reading = await _probe_one(pinfo, probe, url, key, demote_at=demote_at)
        QUOTA_STATE.put_reading(reading)
        refreshed = True
        if (
            reading.status == STATUS_OK
            and reading.exhausted_until is None
            and not reading.advisory
        ):
            # A fresh reading that says the credential can serve clears an
            # in-band demotion taken before the window reset.  An advisory
            # reading does not: it was made with another credential.
            QUOTA_STATE.clear_demotion(probe.id, credential)
            # The same reading is the one signal that lifts a permanent park:
            # the vendor now says the account can pay.  A dated cooldown is
            # left alone — it belongs to the rate-limit ladder, not the
            # balance.
            identity = getattr(pinfo, "quota_identity", None) or f"@{name}"
            block = QUOTA_STATE.provider_block(identity)
            if block is not None and block.permanent:
                QUOTA_STATE.clear_provider_block(identity)
                logger.warning(
                    "Quota: %s reports the account available; park lifted",
                    name,
                )

    payload: dict[str, Any] = {
        "provider": name,
        "supported": True,
        "probe": probe.id,
        "url": scrub_credential_patterns(url),
        "status": STATUS_OK,
        "refreshed": refreshed,
        "ttl_seconds": ttl,
        "credentials": QUOTA_STATE.snapshot(probe.id, keys),
        # The same numbers selection read, so the panel describes the router's
        # decision rather than re-deriving it.
        "routing": routing_facts(pinfo, name),
    }
    # A lane can be readable *and* cooling down; the panel needs both to avoid
    # painting a live reading on a provider the router is skipping.
    block = provider_cooldown(name, pinfo)
    if block is not None:
        payload["provider_block"] = block
    if payload["credentials"]:
        payload["read_at"] = (
            payload["credentials"][0]["reading"]["read_at"]
            if payload["credentials"][0]["reading"]
            else None
        )
    return payload


def quota_intervals(config: Any) -> dict[str, int]:
    """Return ``{provider: seconds}`` for providers that want a background poll.

    Only providers whose declaration sets ``quota.interval`` above zero are
    polled; the default is on demand (the admin card) plus in-band demotion.
    """
    out: dict[str, int] = {}
    for name in getattr(config, "providers", {}):
        declaration = _quota_declaration(config, name)
        try:
            interval = int(declaration.get("interval", 0) or 0)
        except (TypeError, ValueError):
            continue
        if interval > 0:
            out[name] = interval
    return out


def has_usable_alternative(pinfo: Any) -> bool:
    """Whether another credential of *pinfo* is currently usable.

    Used to decide whether a quota refusal is worth retrying immediately:
    with one credential, or with every credential demoted, a retry would go
    out on the same key and only spend another request.
    """
    probe_id = getattr(pinfo, "quota_probe", None)
    if not probe_id:
        return False
    keys = pinfo.key_ring.keys()
    if len(keys) < 2:
        return False
    return any(QUOTA_STATE.is_usable(probe_id, key) for key in keys)


def bind_provider_quota(name: str, pinfo: Any) -> None:
    """Attach the process-wide quota state to a freshly built provider.

    Called by the config loader after every :class:`ProviderInfo` is built.
    The provider keeps only predicates and an identifier: it never holds the
    state object, so a config reload cannot create a second source of truth.
    """
    # A provider that mints its credential has no ``credential_token`` until a
    # record exists, so the identity falls back to ``@<name>`` and becomes
    # ``acct:<digest>`` on the reload after sign-in.  Both are stable per
    # provider — which is what a block needs — but neither follows the account
    # to another provider entry.
    identity = getattr(pinfo, "credential_token", None) or f"@{name}"
    pinfo.quota_identity = identity
    # A keyless lane has no per-key allowance but can still be cooling down
    # from a 429, so the provider-level predicate is bound for it too.
    pinfo.quota_provider_blocked = lambda: (
        QUOTA_STATE.provider_block(identity) is not None
    )
    probe, _url, _unsupported = resolve_provider_probe_for_info(name, pinfo)
    probe_id = probe.id if probe is not None else None
    if probe_id is not None:
        pinfo.quota_probe = probe_id

    def _usable(key: str) -> bool:
        """One credential's availability: quota demotion *and* 429 disable.

        Bound for every provider, not only one with a probe: a lane that
        answers 429 has a credential that earned it whether or not anything
        can read its allowance, and the ring must skip that credential.
        """
        if probe_id is not None and not QUOTA_STATE.is_usable(probe_id, key):
            return False
        return (
            QUOTA_STATE.rate_limits.disabled(name, credential_fingerprint(key)) is None
        )

    pinfo.key_ring.set_usable(_usable)


def resolve_provider_probe_for_info(
    name: str, pinfo: Any
) -> tuple[QuotaProbe | None, str | None, str | None]:
    """Resolve a probe from the shim's implementations and the declaration.

    Config reloads build the provider before the config object exists, so
    everything needed is carried on the ``ProviderInfo``: the shim attached
    ``quota_probes``, the declaration names one (``quota_probe_id``), and the
    provider may override the URL.
    """
    if pinfo.keyless and not getattr(pinfo, "quota_local_argv", ()):
        pinfo.quota_unsupported = "the lane is keyless"
        return None, None, pinfo.quota_unsupported
    try:
        match = resolve_probe(
            pinfo.base_url,
            probes=pinfo.quota_probes,
            probe_id=pinfo.quota_probe_id,
            url=pinfo.quota_url,
            in_band=bool(getattr(pinfo, "quota_in_band", False)),
        )
    except ValueError as exc:
        pinfo.quota_unsupported = str(exc)
        return None, None, pinfo.quota_unsupported
    if match is None:
        pinfo.quota_unsupported = "no quota probe is declared by this shim"
        return None, None, pinfo.quota_unsupported
    if match.url is None:
        # Nothing to poll, but the probe stays bound: its ``inband()`` is the
        # only signal this lane gives, or a declared local command prints the
        # reading without any endpoint.
        pinfo.quota_probe_impl = match.probe
        if getattr(pinfo, "quota_local_argv", ()):
            pinfo.quota_unsupported = None
            return match.probe, None, None
        pinfo.quota_unsupported = match.unsupported or "no quota endpoint"
        return match.probe, None, pinfo.quota_unsupported
    pinfo.quota_probe_impl = match.probe
    pinfo.quota_usage_url = match.url
    return match.probe, match.url, None

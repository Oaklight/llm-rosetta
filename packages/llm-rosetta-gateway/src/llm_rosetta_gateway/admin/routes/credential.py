"""Provider credentials: read, refresh, test, install, remove.

A provider mints its own credential in one of two ways: an argv
``token_command`` (declared by the shim or by the provider's own config) or a
shim-shipped ``auth.py`` exchange.  This module is the admin surface for both.

The credential section of the provider editor asks four questions of the same
endpoints — where the credential comes from, whether the last refresh worked,
what the operator can do about it, and which fields are editable.  A record
produced elsewhere (an OAuth profile) is installed through the exchange; the
gateway never parses it and never renders a token, only a digest.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.observability.sanitize import scrub_credential_patterns
from llm_rosetta_gateway.providers import _account_token_for
from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL
from llm_rosetta_gateway.transport.token_refresh import (
    force_refresh,
    public_token_status,
)

from ._shared import (
    exchange_can_host_login,
    exchange_field_text,
    exchange_fields,
    parse_json_body,
)
from .config.payload import _get_gateway_config


class _NotSupported(Exception):
    """The provider has no usable credential channel.

    Carries the status the caller deserves: a lookup failure is a 404, "this
    provider has no credential channel" is a 409 — a client should be able to
    tell "no such provider" from "this one cannot be handed a record".
    """

    def __init__(self, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _provider(request: Any, name: str) -> Any:
    """Resolve *name* to its live ProviderInfo.

    Raises:
        _NotSupported: No config loaded, or unknown (or disabled) provider.
    """
    config = _get_gateway_config(request)
    if config is None:
        raise _NotSupported("No config loaded", status=404)
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is None:
        raise _NotSupported(f"Provider '{name}' not found", status=404)
    return pinfo


def _provider_and_exchange(request: Any, name: str) -> tuple[Any, Any]:
    """Resolve *name* to its live ProviderInfo and the exchange it declares.

    Raises:
        _NotSupported: No config, unknown provider, or no exchange.
    """
    pinfo = _provider(request, name)
    exchange = getattr(pinfo, "auth_exchange", None)
    if exchange is None:
        raise _NotSupported(
            f"Provider '{name}' mints no credential of its own: its shim ships "
            "no auth.py, and a static key is configured in the provider entry"
        )
    return pinfo, exchange


def _error(exc: _NotSupported) -> Response:
    return JSONResponse({"error": str(exc)}, status_code=exc.status)


def exchange_can_install(exchange: Any) -> bool:
    """Whether the exchange accepts a record produced elsewhere.

    ``BaseAuthExchange`` defines ``install`` and raises, so presence is not the
    question — whether the exchange overrode it is.  Callable rather than
    present because a shim could expose either name as a property, and a
    ``TypeError`` raised while rendering the panel would be a 500.
    """
    install = getattr(exchange, "install", None)
    can_install = getattr(exchange, "can_install", None)
    return install is not None and (can_install() if callable(can_install) else True)


def exchange_can_begin_login(exchange: Any) -> bool:
    """Whether the exchange supports a server-side browser sign-in."""
    begin = getattr(exchange, "begin_login", None)
    capability = getattr(exchange, "can_begin_login", None)
    return begin is not None and (capability() if callable(capability) else True)


def exchange_can_complete_login(exchange: Any) -> bool:
    """Whether the exchange accepts a pasted browser callback."""
    complete = getattr(exchange, "complete_login", None)
    capability = getattr(exchange, "can_complete_login", None)
    return complete is not None and (capability() if callable(capability) else True)


def exchange_can_revoke(exchange: Any) -> bool:
    """Whether sign-out can reach the vendor and remove the record.

    Same split as :func:`exchange_can_install`: the base class raises, so an
    exchange that did not override ``revoke`` cannot sign out.
    """
    revoke = getattr(exchange, "revoke", None)
    can_revoke = getattr(exchange, "can_revoke", None)
    return revoke is not None and (can_revoke() if callable(can_revoke) else True)


#: What a sign-in snapshot may carry to a browser, and how much of it.
_LOGIN_STATES = frozenset(
    {"idle", "running", "verifying", "succeeded", "failed", "cancelled"}
)
_LOGIN_LINES = 40
_LOGIN_TEXT = 400


#: Query parameters a URL shown to an operator must not carry: an
#: authorization *request* has none of them, a redirect that holds one is the
#: sign-in's own secret.
_SECRET_PARAMS = frozenset(
    {"code", "token", "access_token", "refresh_token", "id_token", "client_secret"}
)


def _safe_login_url(url: Any) -> bool:
    """Whether *url* may be rendered as a link: ``https``, bounded, no secret."""
    if not isinstance(url, str) or len(url) > 2000:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.netloc:
        return False
    if any(
        name.lower() in _SECRET_PARAMS
        for name, _ in parse_qsl(parts.query, keep_blank_values=True)
    ):
        return False
    return scrub_credential_patterns(url) == url


def public_login_status(raw: Any) -> dict[str, Any]:
    """The sign-in snapshot an operator's browser may receive.

    The exchange has already clipped and scrubbed what it prints; this is the
    second line of defence, because a vendor program's output is not under our
    control and an operator may be on the far side of a network.  An allow-list
    of keys, a scrub over every string, a bound on the number and length of
    lines, and a URL only when scrubbing leaves it untouched.
    """
    data = raw if isinstance(raw, dict) else {}
    state = data.get("state")
    url = data.get("url")
    if not _safe_login_url(url):
        url = None
    lines = [
        scrub_credential_patterns(str(line))[:_LOGIN_TEXT]
        for line in (data.get("lines") or [])
        if isinstance(line, str)
    ][-_LOGIN_LINES:]
    out: dict[str, Any] = {
        "state": state if state in _LOGIN_STATES else "idle",
        "url": url,
        "lines": lines,
        "message": scrub_credential_patterns(str(data.get("message") or ""))[
            :_LOGIN_TEXT
        ],
    }
    for key in ("started_at", "timeout"):
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out


async def _restart_lane(request: Any, pinfo: Any) -> None:
    """Drop a local-agent provider's processes so the next turn re-reads who is signed in.

    An ACP agent keeps its sign-in for the life of the process; after a
    sign-in or sign-out the running one answers for the *previous* account.
    """
    if getattr(pinfo, "transport", None) != "acp":
        return
    transport = (getattr(request.app, "transports", None) or {}).get("acp")
    close = getattr(transport, "aclose_provider", None)
    if close is not None:
        await close(pinfo.config_name)


async def _off_loop(func: Any, *args: Any) -> Any:
    """Run a blocking exchange call on the executor.

    Installing a record verifies an ID token, which fetches the issuer's
    discovery document and JWKS — network I/O that must not block the gateway's
    event loop.
    """
    return await asyncio.get_running_loop().run_in_executor(None, func, *args)


def _public_status(pinfo: Any) -> dict[str, Any]:
    """The refresh log the panel may render.

    The text is already clipped where it is written; clipping again here is a
    second line of defence for a ``token_status`` built by older code or set
    directly on the provider.  It never hides why the mint failed — the
    sentence is bounded, not removed.
    """
    return public_token_status(pinfo.token_status)


def _command_payload(pinfo: Any, name: str) -> dict[str, Any]:
    """The credential view of a provider whose token comes from a command.

    The panel renders the same section for a shim-declared command and one the
    operator wrote into the config; ``command_editable`` is what separates
    them.  ``state`` answers the operator's first question — is a usable token
    in the ring, still arriving, or failing to arrive — without leaking a
    token.
    """
    status = _public_status(pinfo)
    # A failed mint is the headline even while the ring still holds the pending
    # sentinel: "waiting for the first token" would bury the error that says
    # why it has not arrived.
    if status.get("last_error"):
        state = "failing"
    elif pinfo.credentials_pending:
        state = "initializing"
    else:
        state = "connected"
    return {
        "provider": name,
        "kind": "command",
        "source": pinfo.credential_source,
        "state": state,
        "installed": not pinfo.credentials_pending,
        "needs_interactive_setup": False,
        "token_status": status,
        # A token in the ring and a refresher that will replace it are two
        # facts.  ``connected`` stays true when the refresher is off; the
        # panel can say so instead of implying the token renews itself.
        "refresher_enabled": status.get("enabled", True) is not False,
        # The digest, not the token: ``cmd:<sha256>`` is how the gateway
        # names this credential everywhere else.
        "credential": pinfo.credential_token,
        "command": list(pinfo.token_command or []),
        "command_env": pinfo.token_command_env,
        "refresh_interval": pinfo.token_refresh_interval,
        "command_editable": pinfo.credential_source == "config_command",
        # A command provider has no record to install and nothing to sign out
        # of; the panel's exchange actions are not offered for it.
        "can_install": False,
        "can_revoke": False,
    }


async def get_provider_credential(request: Any, name: str) -> Response:
    """Report where the provider's credential comes from, and its state."""
    try:
        pinfo = _provider(request, name)
    except _NotSupported as exc:
        return _error(exc)

    if pinfo.credential_source in ("config_command", "shim_command"):
        return JSONResponse(_command_payload(pinfo, name))

    exchange = getattr(pinfo, "auth_exchange", None)
    if exchange is None:
        # A static key is edited in the key box, not here.  Saying "no channel"
        # keeps the endpoint's contract: it reports records, not ordinary keys.
        return _error(
            _NotSupported(
                f"Provider '{name}' mints no credential of its own: its shim "
                "ships no auth.py, and a static key is configured in the "
                "provider entry"
            )
        )

    ctx = pinfo.auth_context
    describe = getattr(exchange, "describe", None)
    summary = None
    unreadable = False
    if describe is not None:
        try:
            summary = await _off_loop(describe, ctx)
        except Exception:  # noqa: BLE001 - a broken record is a state, not a 500
            # A record that exists but cannot be read is neither "connected"
            # nor "needs a sign-in"; saying either would send the operator to
            # the wrong action.
            unreadable = True
    status = _public_status(pinfo)
    # A record can exist and still be unusable: a refused grant leaves the
    # refresh token in the store, so ``describe`` keeps answering.  Trusting
    # that answer alone reported "connected" and offered Refresh for a token
    # that can never mint again.  The refresh log knows better — a mint that
    # needs a human outranks the record, and any other failure outranks it
    # too, because Refresh may recover a timeout but only a sign-in recovers
    # a refused grant.
    if unreadable:
        state = "unreadable"
    elif status.get("interactive_setup_required"):
        state = "needs_setup"
    elif status.get("last_error"):
        state = "failing"
    elif summary is not None:
        state = "connected"
    else:
        state = "needs_setup"
    return JSONResponse(
        {
            "provider": name,
            "kind": getattr(exchange, "kind", "exchange"),
            "source": pinfo.credential_source,
            "installed": summary is not None,
            "state": state,
            "needs_interactive_setup": bool(exchange.needs_interactive_setup(ctx)),
            # The hash, not the subject: the gateway names an account by
            # digest everywhere else, and a raw subject would be one surface
            # that can be correlated across providers.  ``summary`` is where
            # the exchange may say which account this is, in its own words.
            "account": _account_token_for(exchange.account_id(ctx)),
            "summary": summary,
            "credential": pinfo.credential_token,
            "refresh_interval": pinfo.token_refresh_interval,
            "token_status": status,
            "refresher_enabled": status.get("enabled", True) is not False,
            # The panel renders install/sign-out only when the action would
            # not come back 501.  The exchange owns both answers.
            "can_install": exchange_can_install(exchange),
            "can_revoke": exchange_can_revoke(exchange),
            "can_begin_login": exchange_can_begin_login(exchange),
            "can_complete_login": exchange_can_complete_login(exchange),
            # A sign-in the vendor's own program performs on this host: the
            # panel starts it and polls ``login``'s snapshot.
            "can_host_login": exchange_can_host_login(exchange),
            "login": (
                await _login_snapshot(pinfo, exchange)
                if exchange_can_host_login(exchange)
                else None
            ),
            # The inputs install() takes.  Non-empty means the panel renders a
            # form; empty keeps the record-textarea path.
            "fields": exchange_fields(exchange),
        }
    )


async def begin_provider_login(request: Any, name: str) -> Response:
    """Start browser OAuth and return only its authorization URL."""
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)
    if not exchange_can_begin_login(exchange):
        return JSONResponse(
            {"error": f"Provider '{name}' has no browser sign-in flow"},
            status_code=501,
        )
    try:
        result = await _off_loop(exchange.begin_login, pinfo.auth_context)
    except Exception as exc:  # noqa: BLE001 - vendor errors are operator-facing
        return JSONResponse(
            {"error": scrub_credential_patterns(str(exc))}, status_code=400
        )
    if exchange_can_host_login(exchange):
        # The program was started on this host; what comes back is its first
        # snapshot, not a URL to open.  Progress is read from the status route.
        return JSONResponse(
            {
                "provider": name,
                "pending": True,
                "host": True,
                "login": public_login_status(result),
            }
        )
    if not isinstance(result, dict) or not isinstance(
        result.get("authorization_url"), str
    ):
        return JSONResponse(
            {"error": "the exchange returned no authorization URL"}, status_code=502
        )
    # Keep the response allow-list explicit: a shim must never accidentally
    # expose a verifier or token by adding a field to its private result.
    payload = {
        "provider": name,
        "pending": True,
        "authorization_url": result["authorization_url"],
    }
    for key in ("callback_url", "expires_at"):
        if key in result:
            payload[key] = result[key]
    # What the operator pastes back is not the same for every exchange: a
    # loopback redirect hands over a callback URL, Globus hands over a code on
    # its own page.  The exchange says which, in its own words, and the panel
    # labels the box with it instead of always asking for a URL.
    for key in ("paste_label", "paste_placeholder"):
        text = exchange_field_text(result.get(key))
        if text is not None:
            payload[key] = text
    return JSONResponse(payload)


async def _login_snapshot(pinfo: Any, exchange: Any) -> dict[str, Any] | None:
    """The scrubbed snapshot of the exchange's sign-in run, or ``None``."""
    try:
        return public_login_status(
            await _off_loop(exchange.login_status, pinfo.auth_context)
        )
    except Exception:  # noqa: BLE001 - a status read must not become a 500
        return None


async def get_provider_login_status(request: Any, name: str) -> Response:
    """Report a host sign-in's progress.

    When the run has just succeeded, this is also where the gateway adopts it:
    the key ring is re-seeded, the lane's agent processes are replaced so they
    answer for the account that is signed in now, and the provider is made
    ready.  Once per run — polling must not start a process each time.
    """
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)
    if not exchange_can_host_login(exchange):
        return JSONResponse(
            {"error": f"Provider '{name}' signs in through no program on this host"},
            status_code=501,
        )
    raw = await _off_loop(exchange.login_status, pinfo.auth_context)
    login = public_login_status(raw)
    payload: dict[str, Any] = {"provider": name, "login": login}
    started = login.get("started_at")
    applied = getattr(pinfo, "login_applied_at", None)
    if login["state"] == "succeeded" and started is not None and started != applied:
        pinfo.login_applied_at = started
        error = await _reseed(pinfo)
        await _restart_lane(request, pinfo)
        await _activate_after_reseed(request, name, pinfo, error)
        payload["connected"] = error is None
        if error is not None:
            payload["error"] = f"the sign-in finished but is not in use: {error}"
    return JSONResponse(payload)


async def cancel_provider_login(request: Any, name: str) -> Response:
    """Stop a host sign-in that is still running."""
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)
    cancel = getattr(exchange, "cancel_login", None)
    if not exchange_can_host_login(exchange) or cancel is None:
        return JSONResponse(
            {"error": f"Provider '{name}' signs in through no program on this host"},
            status_code=501,
        )
    cancelled = bool(await _off_loop(cancel, pinfo.auth_context))
    return JSONResponse(
        {
            "provider": name,
            "cancelled": cancelled,
            "login": await _login_snapshot(pinfo, exchange),
        }
    )


async def complete_provider_login(request: Any, name: str) -> Response:
    """Complete browser OAuth from one pasted callback URL."""
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)
    if not exchange_can_complete_login(exchange):
        return JSONResponse(
            {"error": f"Provider '{name}' has no browser callback flow"},
            status_code=501,
        )
    body, err = parse_json_body(request)
    if err:
        return err
    if (
        not isinstance(body, dict)
        or set(body) != {"callback_url"}
        or not isinstance(body.get("callback_url"), str)
    ):
        return JSONResponse(
            {"error": "only a callback_url is accepted"}, status_code=400
        )
    try:
        summary = await _off_loop(
            exchange.complete_login,
            pinfo.auth_context,
            body["callback_url"],
        )
    except Exception as exc:  # noqa: BLE001 - exchange owns safe error wording
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not isinstance(summary, dict):
        return JSONResponse(
            {"error": "the exchange returned no summary"}, status_code=502
        )
    error = await _reseed(pinfo)
    await _activate_after_reseed(request, name, pinfo, error)
    payload: dict[str, Any] = {
        "provider": name,
        "connected": error is None,
        "summary": summary,
    }
    if error is not None:
        payload["error"] = f"the sign-in is saved but not in use: {error}"
    return JSONResponse(payload)


async def refresh_provider_credential(request: Any, name: str) -> Response:
    """Mint a fresh credential now, without waiting for the scheduled refresh.

    The 401 path debounces bursts; the operator's button must not.  A failed
    mint keeps the previous credential in the ring — that is
    :func:`force_refresh`'s contract — and reports why.
    """
    try:
        pinfo = _provider(request, name)
    except _NotSupported as exc:
        return _error(exc)

    if not pinfo.mints_credential:
        return JSONResponse(
            {"error": f"Provider '{name}' mints no credential of its own"},
            status_code=409,
        )

    refreshed = await force_refresh(pinfo, debounce=False)
    status = _public_status(pinfo)
    payload: dict[str, Any] = {
        "provider": name,
        "ok": refreshed,
        "refreshed": refreshed,
        "token_status": status,
    }
    if not refreshed:
        payload["error"] = (
            status.get("last_error") or "the credential could not be refreshed"
        )
    return JSONResponse(payload)


async def test_provider_credential(request: Any, name: str) -> Response:
    """Refresh the credential, then let the upstream accept or reject it.

    A minted token proves only that the command exited 0.  The models listing
    is the cheapest authenticated call every format shares, and it already
    knows the provider's declared path and headers, so the connectivity probe
    is reused rather than copied.
    """
    try:
        pinfo = _provider(request, name)
    except _NotSupported as exc:
        return _error(exc)

    if not pinfo.mints_credential:
        return JSONResponse(
            {"error": f"Provider '{name}' mints no credential of its own"},
            status_code=409,
        )

    refreshed = await force_refresh(pinfo, debounce=False)
    status = _public_status(pinfo)
    if not refreshed:
        # No new token, so there is nothing to put to the upstream: probing
        # would test the credential that just failed to renew.
        return JSONResponse(
            {
                "provider": name,
                "ok": False,
                "refreshed": False,
                "accepted": False,
                "token_status": status,
                "error": (
                    status.get("last_error") or "the credential could not be refreshed"
                ),
            }
        )

    from .connectivity import test_provider_connectivity

    probe = await test_provider_connectivity(request, name)
    connectivity = json.loads(probe.body)
    models = (connectivity.get("endpoints") or {}).get("models") or {}
    accepted = bool(models.get("ok"))

    payload: dict[str, Any] = {
        "provider": name,
        "ok": accepted,
        "refreshed": True,
        "accepted": accepted,
        "reachable": connectivity.get("reachable"),
        "token_status": status,
        "connectivity": connectivity,
    }
    if not accepted:
        if connectivity.get("reachable") is False:
            # The token was minted; it was the host that did not answer.  A
            # "rejected" message would send the operator to redo a credential
            # that is fine.
            reason = connectivity.get("base_error") or "no answer from the host"
            payload["error"] = f"the upstream is unreachable: {reason}"
        else:
            reason = models.get("error") or models.get("status") or "no answer"
            payload["error"] = f"the upstream rejected the credential: {reason}"
    return JSONResponse(payload)


async def put_provider_credential(request: Any, name: str) -> Response:
    """Install a credential record produced elsewhere.

    The body is the record the exchange wrote on the machine that signed in.
    It is passed through untouched: only the exchange knows what its own record
    looks like, and validating it here would be a second, drifting copy of that
    knowledge.
    """
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)

    install = getattr(exchange, "install", None)
    # ``BaseAuthExchange`` defines ``install`` and raises, so presence is not
    # the question — whether the exchange overrode it is.
    usable = exchange_can_install(exchange)
    if not usable:
        return JSONResponse(
            {
                "error": (
                    f"Provider '{name}' cannot be handed a credential record: "
                    "its exchange does not implement install()"
                )
            },
            status_code=501,
        )

    body, err = parse_json_body(request)
    if err:
        return err
    if not body:
        return JSONResponse(
            {"error": "A credential record is required"}, status_code=400
        )

    try:
        summary = await _off_loop(install, pinfo.auth_context, body)
    except NotImplementedError:
        return JSONResponse(
            {"error": f"Provider '{name}' cannot install a credential record"},
            status_code=501,
        )
    except Exception as exc:  # noqa: BLE001 - every failure is the caller's to read
        # The exchange's message names what is wrong with the record; it never
        # contains the record itself.
        return JSONResponse({"error": str(exc)}, status_code=400)

    payload = {"provider": name, "installed": True, "summary": summary}
    error = await _reseed(pinfo)
    await _activate_after_reseed(request, name, pinfo, error)
    if error is not None:
        payload["error"] = f"the record is installed but not in use: {error}"
    return JSONResponse(payload)


async def delete_provider_credential(request: Any, name: str) -> Response:
    """Forget the credential, revoking it upstream when the exchange can."""
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)

    revoke = getattr(exchange, "revoke", None)
    # ``BaseAuthExchange.revoke`` raises, so presence is not the question —
    # whether the exchange overrode it is.
    revocable = exchange_can_revoke(exchange)
    revoked = False
    error = None
    if revocable:
        try:
            revoked = bool(await _off_loop(revoke, pinfo.auth_context))
        except Exception as exc:  # noqa: BLE001 - the record is removed regardless
            # The exchange removes the record even when the network refuses;
            # report the revoke failure without pretending the sign-out failed.
            error = str(exc)
    # The key ring must not keep what it was holding: a "signed out" provider
    # that still sends the revoked token until it expires is the one thing this
    # endpoint must not do.
    await _reseed(pinfo)
    await _restart_lane(request, pinfo)
    removed = bool(exchange.needs_interactive_setup(pinfo.auth_context))
    if removed:
        deferred = getattr(request.app, "deferred_startup", None)
        deactivate = getattr(deferred, "provider_credential_removed", None)
        if deactivate is not None:
            await deactivate(name, pinfo)
    if not removed and not revocable:
        # "Sign out" and "did nothing" must not look alike: the record is still
        # there because this exchange has no way to remove it.  Same split as
        # install's 501.
        return JSONResponse(
            {
                "error": f"Provider '{name}' cannot sign out: its exchange does not "
                "implement revoke(), so the credential is still installed"
            },
            status_code=501,
        )
    payload = {"provider": name, "removed": removed, "revoked": revoked}
    if error is not None:
        payload["error"] = error
    return JSONResponse(payload)


async def _activate_after_reseed(
    request: Any, name: str, pinfo: Any, error: str | None
) -> None:
    """Notify the application lifecycle when a credential became usable."""
    if error is not None:
        return
    deferred = getattr(request.app, "deferred_startup", None)
    activate = getattr(deferred, "provider_credential_ready", None)
    if activate is not None:
        await activate(name, pinfo)


async def _reseed(pinfo: Any) -> str | None:
    """Point the live provider at the credential that is now on disk.

    The key ring still holds what the previous seed produced.  Re-reading the
    record is cheap and makes the next request use it without waiting for a
    restart or a scheduled refresh.  ``credential()`` is a coroutine that
    already runs its blocking work on a thread.

    Every failure empties the ring, not just "there is no credential".  An
    admin mutation has just changed what is on disk, so whatever the ring holds
    may belong to a *different account* — an install that switched accounts and
    then failed to mint would otherwise keep authenticating as the old one
    while the panel says the new one is installed.  A provider that cannot
    produce its credential is not ready; saying so is the truth.

    Returns:
        The failure message when the credential could not be minted.  After an
        install that is news worth reporting — the record is in and has not
        taken effect.  After a sign-out it is the expected outcome (there is no
        record to mint from), so that caller ignores it.
    """
    try:
        token = await pinfo.auth_exchange.credential(pinfo.auth_context)
    except Exception as exc:  # noqa: BLE001 - the ring is emptied below either way
        pinfo.key_ring.refresh(TOKEN_PENDING_SENTINEL)
        return str(exc)
    pinfo.key_ring.refresh(token)
    return None

"""Shim-declared credentials: read, install, remove.

A provider whose shim ships ``auth.py`` has no key to type into the editor —
its credential is a record the exchange produces.  This endpoint is the way in
for a record made somewhere else: the operator runs the sign-in on a machine
with a browser and hands the finished record to the gateway over its
authenticated admin API.

The gateway never parses the record and never stores it itself.  It hands the
object to the exchange (:meth:`AuthExchange.install`), which owns the shape and
the location, and returns only what the exchange says is safe to render — a
subject and an expiry, never a token.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.providers import _account_token_for
from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL

from ._shared import parse_json_body
from .config.payload import _get_gateway_config


class _NotSupported(Exception):
    """The provider has no usable credential exchange.

    Carries the status the caller deserves: a lookup failure is a 404, "this
    provider has no credential channel" is a 409 — a client should be able to
    tell "no such provider" from "this one cannot be handed a record".
    """

    def __init__(self, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _provider_and_exchange(request: Any, name: str) -> tuple[Any, Any]:
    """Resolve *name* to its live ProviderInfo and the exchange it declares.

    Raises:
        _NotSupported: No config, unknown provider, or no exchange.
    """
    config = _get_gateway_config(request)
    if config is None:
        raise _NotSupported("No config loaded", status=404)
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is None:
        raise _NotSupported(f"Provider '{name}' not found", status=404)
    exchange = getattr(pinfo, "auth_exchange", None)
    if exchange is None:
        raise _NotSupported(
            f"Provider '{name}' mints no credential of its own: its shim ships "
            "no auth.py, and a static key is configured in the provider entry"
        )
    return pinfo, exchange


def _error(exc: _NotSupported) -> Response:
    return JSONResponse({"error": str(exc)}, status_code=exc.status)


async def _off_loop(func: Any, *args: Any) -> Any:
    """Run a blocking exchange call on the executor.

    Installing a record verifies an ID token, which fetches the issuer's
    discovery document and JWKS — network I/O that must not block the gateway's
    event loop.
    """
    return await asyncio.get_running_loop().run_in_executor(None, func, *args)


async def get_provider_credential(request: Any, name: str) -> Response:
    """Report whether the provider has a credential, and whose it is."""
    try:
        pinfo, exchange = _provider_and_exchange(request, name)
    except _NotSupported as exc:
        return _error(exc)

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
    if summary is not None:
        state = "connected"
    elif unreadable:
        state = "unreadable"
    else:
        state = "needs_setup"
    return JSONResponse(
        {
            "provider": name,
            "kind": getattr(exchange, "kind", "exchange"),
            "installed": summary is not None,
            "state": state,
            "needs_interactive_setup": bool(exchange.needs_interactive_setup(ctx)),
            # The hash, not the subject: the gateway names an account by
            # digest everywhere else, and a raw subject would be one surface
            # that can be correlated across providers.  ``summary`` is where
            # the exchange may say which account this is, in its own words.
            "account": _account_token_for(exchange.account_id(ctx)),
            "summary": summary,
        }
    )


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
    can_install = getattr(exchange, "can_install", None)
    # ``BaseAuthExchange`` defines ``install`` and raises, so presence is not
    # the question — whether the exchange overrode it is.
    usable = install is not None and (can_install() if can_install else True)
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
    revoked = False
    error = None
    if revoke is not None:
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
    payload = {
        "provider": name,
        "removed": bool(exchange.needs_interactive_setup(pinfo.auth_context)),
        "revoked": revoked,
    }
    if error is not None:
        payload["error"] = error
    return JSONResponse(payload)


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

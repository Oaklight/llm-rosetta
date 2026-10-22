"""Sign in with ChatGPT — the shim's own credential exchange.

The plan route is OAuth, and none of it fits in an argv: the first run needs
a browser, the refresh token rotates on every use, what is stored is a record
rather than a string, and one installation may hold several accounts.  So the
exchange is code, shipped beside the shim's ``provider.yaml`` and reached
through :class:`llm_rosetta.shims.auth_exchange.AuthExchange` — the gateway
only decides *when* to ask for a credential.

What this module owns, and the gateway does not: dynamic client registration,
PKCE + state + nonce, the loopback callback, the code exchange, ID-token
verification against the discovery JWKS, the credential store, refresh-token
rotation and revocation, and the host id that tells OpenAI this is the same
installation across sign-ins.

Two rules shape the code:

* **The standard library only.** The core distribution has no dependencies
  and the shim layer loads before the gateway, so RS256 is verified with
  ``hashlib`` and Python's integers rather than a JOSE library, and HTTP is
  ``urllib`` on a worker thread.  Verification is not skipped: a signature
  check we cannot do is a profile we must not store.
* **A refresh token is spent when it is used.** Every read-modify-write of the
  record takes an exclusive file lock, because two concurrent refreshes would
  rotate the token twice and lose the account.

The record lives under :attr:`AuthContext.state_dir` — the gateway supplies
``<data_dir>/shim-auth/<provider>``, so two configured accounts never share a
file — and the host id beside it, where it survives a re-sign-in.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from llm_rosetta.shims.auth_exchange import (
    AuthContext,
    AuthExchangeError,
    BaseAuthExchange,
    InteractiveSetupRequired,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

ISSUER = "https://auth.openai.com"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"

#: The client id used for the *first* registration.  It is a request, not a
#: result: the exchange that comes back issues an ``oaiapp_…`` id, and that is
#: the one to save and reuse.  Sending this placeholder on a returning sign-in
#: would register a second client every time.
DYNAMIC_CLIENT = "dynamic_agent_client"

SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
RESOURCE = "https://api.openai.com/v1"
CALLBACK_PATH = "/auth/callback"

#: Wall-clock budget for one outbound call, and the cap on what we will read
#: from it.  A token endpoint that streams forever must not fill memory.
TIMEOUT = 180.0
MAX_BYTES = 16 * 1024 * 1024

#: Clock skew tolerated on ``exp``/``iat``/``nbf``.
SKEW = 60

#: How long the discovery document and its JWKS are reused.  OpenAI's own
#: ``cache-control`` is honoured when it is longer.
DISCOVERY_TTL = 24 * 3600.0

RECORD_NAME = "credential.json"
HOST_ID_NAME = "host-id"

_SHA256_DIGESTINFO = bytes.fromhex("3031300d060960864801650304020105000420")


# ---------------------------------------------------------------------------
# HTTP — stdlib, no redirects, capped
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse 3xx instead of following it.

    A token endpoint that answers with a redirect is either misconfigured or
    under someone else's control; following it would hand the authorization
    code, and later the refresh token, to whatever host it names.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise AuthExchangeError(
            f"{req.full_url} answered {code} → {newurl}; refusing to follow a redirect"
        )


_OPENER = urllib.request.build_opener(_NoRedirect)


def _request(
    url: str,
    *,
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Call *url* and decode a JSON object from the answer.

    Raises:
        AuthExchangeError: The call failed, was refused, redirected, or
            answered something that is not a JSON object.  A 429 is reported
            as-is and never retried here — the caller owns its back-off.
    """
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    if body is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with _OPENER.open(req, timeout=TIMEOUT) as resp:
            raw = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        raise AuthExchangeError(f"{url} answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AuthExchangeError(f"{url} unreachable: {exc.reason}") from exc
    if len(raw) > MAX_BYTES:
        raise AuthExchangeError(f"{url} answered more than {MAX_BYTES} bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthExchangeError(f"{url} did not answer JSON") from exc
    if not isinstance(payload, dict):
        raise AuthExchangeError(
            f"{url} answered {type(payload).__name__}, not an object"
        )
    return payload


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_DISCOVERY: tuple[float, dict[str, Any]] | None = None


def _discover() -> dict[str, Any]:
    """Return the issuer's discovery document, cached."""
    global _DISCOVERY
    now = time.time()
    if _DISCOVERY is not None and now - _DISCOVERY[0] < DISCOVERY_TTL:
        return _DISCOVERY[1]
    doc = _request(DISCOVERY_URL)
    for field in ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not isinstance(doc.get(field), str) or not doc[field]:
            raise AuthExchangeError(f"discovery document has no {field}")
    _DISCOVERY = (now, doc)
    return doc


def _reset_discovery_cache() -> None:
    """Drop the cached discovery document (for tests)."""
    global _DISCOVERY
    _DISCOVERY = None


# ---------------------------------------------------------------------------
# JWKS and RS256
# ---------------------------------------------------------------------------

_JWKS: dict[str, tuple[float, dict[str, Any]]] = {}


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _jwks(uri: str, *, refresh: bool = False) -> dict[str, Any]:
    """Return the key set at *uri*, keyed by ``kid``, cached."""
    now = time.time()
    cached = _JWKS.get(uri)
    if cached is not None and not refresh and now - cached[0] < DISCOVERY_TTL:
        return cached[1]
    doc = _request(uri)
    keys = doc.get("keys")
    if not isinstance(keys, list) or not keys:
        raise AuthExchangeError(f"{uri} published no keys")
    by_kid = {k["kid"]: k for k in keys if isinstance(k, dict) and k.get("kid")}
    if not by_kid:
        raise AuthExchangeError(f"{uri} published no key with a kid")
    _JWKS[uri] = (now, by_kid)
    return by_kid


def _emsa_pkcs1_v15_sha256(signing_input: bytes, size: int) -> bytes:
    """Build the EMSA-PKCS1-v1_5 block an RS256 signature must decrypt to."""
    digest_info = _SHA256_DIGESTINFO + hashlib.sha256(signing_input).digest()
    if size < len(digest_info) + 11:
        raise AuthExchangeError(f"RSA modulus too small for RS256 ({size} bytes)")
    padding = b"\xff" * (size - len(digest_info) - 3)
    return b"\x00\x01" + padding + b"\x00" + digest_info


def _verify_rs256(signing_input: bytes, signature: bytes, jwk: dict[str, Any]) -> bool:
    """Verify an RS256 signature against a JWK, with no crypto dependency.

    Textbook RSA verification: ``m = sig^e mod n``, then compare the recovered
    block with the one PKCS#1 v1.5 prescribes for a SHA-256 digest.  The
    comparison is constant-time; the padding check is not secret.
    """
    if jwk.get("kty") != "RSA":
        raise AuthExchangeError(f"unsupported key type {jwk.get('kty')!r}")
    try:
        n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
        e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthExchangeError("malformed RSA key in the key set") from exc
    # A public exponent below 3 — or an even one — makes the "signature" the
    # plaintext block (``e = 1``) or trivially invertible.  No real issuer
    # publishes such a key, so refusing it costs nothing and closes a forgery
    # for anyone who can influence the key set.
    if e < 3 or e % 2 == 0:
        raise AuthExchangeError(f"implausible RSA exponent {e}")
    size = (n.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    recovered = pow(int.from_bytes(signature, "big"), e, n).to_bytes(size, "big")
    return hmac.compare_digest(recovered, _emsa_pkcs1_v15_sha256(signing_input, size))


def _verify_claims(
    claims: dict[str, Any],
    *,
    issuer: str,
    client_id: str,
    nonce: str | None,
    now: float,
) -> None:
    """Check the claims that make a verified token *this* sign-in.

    Raises:
        AuthExchangeError: Any claim is missing, stale, or belongs to someone
            else.
    """
    if claims.get("iss") != issuer:
        raise AuthExchangeError(
            f"ID token issuer is {claims.get('iss')!r}, not {issuer!r}"
        )
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if not any(
        isinstance(a, str) and hmac.compare_digest(a, client_id) for a in audiences
    ):
        raise AuthExchangeError("ID token was issued for a different client")
    if claims.get("azp") not in (None, client_id):
        raise AuthExchangeError("ID token authorized party is a different client")
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)) or exp < now - SKEW:
        raise AuthExchangeError("ID token has expired")
    iat = claims.get("iat")
    if isinstance(iat, (int, float)) and iat > now + SKEW:
        raise AuthExchangeError("ID token was issued in the future")
    if nonce is not None:
        got = claims.get("nonce")
        if not isinstance(got, str) or not hmac.compare_digest(got, nonce):
            raise AuthExchangeError("ID token nonce does not match this sign-in")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise AuthExchangeError("ID token carries no subject")


def _verify_id_token(
    token: str,
    *,
    jwks_uri: str,
    issuer: str,
    client_id: str,
    nonce: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify an ID token and return its claims.

    Signature first, then the claims that make the token *this* sign-in:
    issuer, audience, expiry, and the nonce when the caller has one to check.

    Raises:
        AuthExchangeError: Any check failed.  A token that cannot be verified
            is never stored.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthExchangeError("ID token is not a three-part JWT")
    try:
        header = json.loads(_b64url_decode(parts[0]))
        claims = json.loads(_b64url_decode(parts[1]))
        signature = _b64url_decode(parts[2])
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthExchangeError("ID token is not decodable") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise AuthExchangeError("ID token header or payload is not an object")
    if header.get("alg") != "RS256":
        raise AuthExchangeError(
            f"ID token is signed with {header.get('alg')!r}, not RS256"
        )

    signing_input = f"{parts[0]}.{parts[1]}".encode()
    kid = header.get("kid")
    keys = _jwks(jwks_uri)
    if kid not in keys:
        # A rotation we have not seen yet: refetch once before refusing.
        keys = _jwks(jwks_uri, refresh=True)
    jwk = keys.get(kid)
    if jwk is None:
        raise AuthExchangeError(f"ID token names an unknown key {kid!r}")
    if not _verify_rs256(signing_input, signature, jwk):
        raise AuthExchangeError("ID token signature does not verify")

    _verify_claims(
        claims,
        issuer=issuer,
        client_id=client_id,
        nonce=nonce,
        now=time.time() if now is None else now,
    )
    return claims


# ---------------------------------------------------------------------------
# The credential store
# ---------------------------------------------------------------------------


def _state_dir(ctx: AuthContext) -> Path:
    """Return the directory this provider's record lives in.

    The gateway supplies one per configured provider; an exchange that has to
    choose must key it on ``ctx.provider``, or two accounts share a file.
    """
    if ctx.state_dir:
        return Path(ctx.state_dir)
    return Path.home() / ".config" / "llm-rosetta" / "siwc" / _slug(ctx.provider)


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in value) or "profile"


def _record_path(ctx: AuthContext) -> Path:
    return _state_dir(ctx) / RECORD_NAME


def _host_id_path(ctx: AuthContext) -> Path:
    return _state_dir(ctx) / HOST_ID_NAME


def _prepare_dir(path: Path) -> None:
    """Create *path* with owner-only permissions, tightening an existing one."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, readable only by its owner.

    The temporary file is created with ``0600`` before anything is written, so
    the record is never briefly world-readable; ``os.replace`` then makes the
    swap atomic, so a reader sees either the old record or the new one.
    """
    _prepare_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextmanager
def _record_lock(ctx: AuthContext) -> Iterator[None]:
    """Hold an exclusive lock across one read-modify-write of the record.

    The refresh token rotates: two processes refreshing at once would spend it
    twice, and the second write would leave the account unreachable.  The lock
    is a separate file so it can be taken before the record exists.
    """
    import fcntl

    _prepare_dir(_state_dir(ctx))
    lock_path = _state_dir(ctx) / ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _read_record(ctx: AuthContext) -> dict[str, Any]:
    path = _record_path(ctx)
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except FileNotFoundError as exc:
        raise InteractiveSetupRequired(
            f"no Sign in with ChatGPT profile at {path}; run the sign-in once"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthExchangeError(
            f"credential record at {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(record, dict) or not record.get("access_token"):
        raise AuthExchangeError(f"credential record at {path} holds no access token")
    return record


def _save_record(ctx: AuthContext, record: dict[str, Any]) -> None:
    _write_private(
        _record_path(ctx), json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def _host_id(ctx: AuthContext) -> str:
    """Return this installation's host id, creating it once.

    An opaque identifier, not a credential — but it must outlive a re-sign-in,
    because it is what tells OpenAI this is the same installation rather than
    a new one.
    """
    path = _host_id_path(ctx)
    try:
        with open(path, encoding="utf-8") as handle:
            existing = handle.read().strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    value = f"urn:uuid:{uuid.uuid4()}"
    _write_private(path, value + "\n")
    return value


# ---------------------------------------------------------------------------
# The OAuth dance
# ---------------------------------------------------------------------------


def _pkce() -> tuple[str, str]:
    """Return a PKCE ``(verifier, challenge)`` pair (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _authorize_url(
    doc: dict[str, Any],
    *,
    redirect_uri: str,
    state: str,
    nonce: str,
    challenge: str,
    client_id: str | None,
    host_id: str,
    id_token_hint: str | None = None,
) -> str:
    """Build the authorize URL for a first or a returning sign-in.

    The two differ in exactly one way: a returning sign-in names the client id
    OpenAI issued (``oaiapp_…``) and drops the registration hint.  Sending the
    placeholder again would register a new client on every sign-in.
    """
    params = {
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "ext_agent_host_id": host_id,
    }
    if client_id:
        params["client_id"] = client_id
        if id_token_hint:
            params["id_token_hint"] = id_token_hint
    else:
        params["client_id"] = DYNAMIC_CLIENT
        params["agent_name_hint"] = "llm-rosetta"
    return f"{doc['authorization_endpoint']}?{urllib.parse.urlencode(params)}"


def _token_call(doc: dict[str, Any], data: dict[str, str]) -> dict[str, Any]:
    return _request(doc["token_endpoint"], data=data)


def _record_from_token(
    ctx: AuthContext,
    doc: dict[str, Any],
    payload: dict[str, Any],
    *,
    client_id: str,
    previous: dict[str, Any] | None = None,
    nonce: str | None = None,
    host_id: str | None = None,
) -> dict[str, Any]:
    """Verify a token response and merge it into a credential record.

    Raises:
        AuthExchangeError: The response carries no access token, no verifiable
            ID token, or no refresh token where one is required.
    """
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise AuthExchangeError("token response carries no access token")
    id_token = payload.get("id_token")
    claims: dict[str, Any] = {}
    if isinstance(id_token, str) and id_token:
        claims = _verify_id_token(
            id_token,
            jwks_uri=doc["jwks_uri"],
            issuer=doc["issuer"],
            client_id=client_id,
            nonce=nonce,
        )
    elif previous is None:
        raise AuthExchangeError("token response carries no ID token to verify")

    expires_in = payload.get("expires_in")
    lifetime = float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0
    now = time.time()
    # A rotating refresh token: the new one replaces the old, and its absence
    # on a refresh means the grant was not extended — keep what we have rather
    # than lose the account.
    refresh = payload.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        refresh = (previous or {}).get("refresh_token", "")
    if not refresh:
        raise AuthExchangeError("token response carries no refresh token")
    scopes = payload.get("scope")
    if isinstance(scopes, str):
        scope_list = scopes.split()
    else:
        scope_list = (previous or {}).get("scopes") or SCOPES.split()

    record: dict[str, Any] = {
        "issuer": doc["issuer"],
        "client_id": client_id,
        "ext_agent_host_id": host_id or (previous or {}).get("ext_agent_host_id", ""),
        "access_token": access,
        "refresh_token": refresh,
        "token_type": payload.get("token_type") or "Bearer",
        "expires_in": int(lifetime),
        "expires_at": now + lifetime,
        # Refresh a little early: the token must still be valid when the
        # upstream sees it, not when we minted it.
        "earliest_refresh_at": now + max(60.0, lifetime * 0.8),
        "scopes": scope_list,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }
    if isinstance(id_token, str) and id_token:
        record["id_token"] = id_token
    elif previous is not None and previous.get("id_token"):
        record["id_token"] = previous["id_token"]
    subject = claims.get("sub") or (previous or {}).get("subject")
    if isinstance(subject, str) and subject:
        record["subject"] = subject
    return record


def _refresh_record(
    ctx: AuthContext, record: dict[str, Any], *, doc: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Exchange the stored refresh token for a fresh record, and save it."""
    doc = doc or _discover()
    client_id = record.get("client_id")
    if not isinstance(client_id, str) or not client_id or client_id == DYNAMIC_CLIENT:
        # A record without an issued client id cannot be refreshed: the grant
        # belongs to the client that asked for it.
        raise InteractiveSetupRequired(
            "the stored profile has no issued client id; sign in again"
        )
    try:
        payload = _token_call(
            doc,
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": record["refresh_token"],
                "resource": RESOURCE,
            },
        )
    except AuthExchangeError as exc:
        if "invalid_grant" in str(exc):
            # The grant is gone (revoked, expired, or already rotated by
            # someone else).  Only a new sign-in can fix it, and saying so is
            # more useful than a 401 on every request.
            raise InteractiveSetupRequired(
                f"the stored Sign in with ChatGPT grant was refused ({exc}); sign in again"
            ) from exc
        raise
    fresh = _record_from_token(ctx, doc, payload, client_id=client_id, previous=record)
    _save_record(ctx, fresh)
    return fresh


def revoke(ctx: AuthContext) -> bool:
    """Revoke the stored grant upstream, and delete the record.

    Returns ``True`` when the issuer accepted the revocation.  The record is
    removed either way — a credential the operator asked to be rid of must not
    stay on disk because the network was down.
    """
    try:
        record = _read_record(ctx)
    except (InteractiveSetupRequired, AuthExchangeError):
        return False
    client_id = record.get("client_id")
    accepted = False
    endpoint = _discover().get("revocation_endpoint")
    if isinstance(endpoint, str) and endpoint and isinstance(client_id, str):
        try:
            _request(
                endpoint,
                data={
                    "token": record.get("refresh_token", ""),
                    "token_type_hint": "refresh_token",
                    "client_id": client_id,
                },
            )
            accepted = True
        except AuthExchangeError as exc:
            logger.warning("Revoking the Sign in with ChatGPT grant failed: %s", exc)
    try:
        _record_path(ctx).unlink()
    except FileNotFoundError:
        pass
    return accepted


# ---------------------------------------------------------------------------
# Interactive sign-in — a loopback listener for the callback
# ---------------------------------------------------------------------------

_CALLBACK_PAGE = (
    "<!doctype html><meta charset=utf-8><title>llm-rosetta</title>"
    "<body style='font:16px system-ui;margin:4rem auto;max-width:32rem'>"
    "<h1>{title}</h1><p>{body}</p>"
)


def _loopback_handler(result: dict[str, Any]):
    """Build a request handler that accepts exactly one callback.

    The listener is bound to ``127.0.0.1`` and is reachable by anything on the
    machine, so it checks the ``Host`` header as well as the path: a browser
    tricked into sending the callback to another port, or a DNS name resolving
    here, is not the sign-in we started.
    """
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # One sign-in, one answer: a second request to the listener is not the
        # callback we are waiting for, and must not overwrite what arrived.
        answered = False

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            if Handler.answered:
                self._answer(404, "Not found", "This sign-in has already finished.")
                return
            parsed = urllib.parse.urlparse(self.path)
            address = getattr(self.server, "server_address", None)
            port = address[1] if isinstance(address, tuple) else 0
            expected_host = f"127.0.0.1:{port}"
            if (
                self.headers.get("Host") != expected_host
                or parsed.path != CALLBACK_PATH
            ):
                self._answer(404, "Not found", "This is not the sign-in callback.")
                return
            query = urllib.parse.parse_qs(parsed.query)
            result.update({k: v[0] for k, v in query.items() if v})
            error = result.get("error")
            if error:
                self._answer(
                    400,
                    "Sign-in refused",
                    f"{error}: {result.get('error_description', '')}".strip(),
                )
                return
            self._answer(
                200, "Signed in", "You can close this tab and return to the terminal."
            )

        def _answer(self, status: int, title: str, body: str) -> None:
            Handler.answered = True
            page = _CALLBACK_PAGE.format(title=title, body=body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Keep the callback out of the gateway's log."""

    return Handler


def login(
    ctx: AuthContext,
    *,
    open_browser: bool = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run the interactive sign-in and store the credential record.

    Binds a loopback listener, opens the browser at the authorize URL, waits
    for the callback, then exchanges the code.  Returns the stored record.

    This is the *local* path: the operator runs it where a browser exists.  A
    remote gateway does not bind its loopback for someone else's browser — the
    record is produced here and handed to the gateway (the admin surface for
    that is a separate slice).

    Raises:
        AuthExchangeError: The sign-in failed, was refused, or timed out.
    """
    from http.server import HTTPServer

    doc = _discover()
    _prepare_dir(_state_dir(ctx))
    host_id = _host_id(ctx)
    try:
        previous = _read_record(ctx)
    except (InteractiveSetupRequired, AuthExchangeError):
        previous = None
    client_id = previous.get("client_id") if previous else None
    id_token_hint = previous.get("id_token") if previous else None

    result: dict[str, Any] = {}
    server = HTTPServer(("127.0.0.1", 0), _loopback_handler(result))
    # The port is only known after the bind, which is why the handler reads it
    # from the server rather than being told.
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}{CALLBACK_PATH}"

    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    url = _authorize_url(
        doc,
        redirect_uri=redirect_uri,
        state=state,
        nonce=nonce,
        challenge=challenge,
        client_id=client_id,
        host_id=host_id,
        id_token_hint=id_token_hint,
    )

    server.timeout = timeout
    try:
        if open_browser:
            webbrowser.open(url)
        logger.info(
            "Sign in with ChatGPT: waiting for the callback on %s", redirect_uri
        )
        print(url, flush=True)
        server.handle_request()
    finally:
        server.server_close()

    if not result:
        raise AuthExchangeError(f"no callback arrived within {timeout:.0f}s")
    got_state = result.get("state", "")
    if not hmac.compare_digest(got_state, state):
        raise AuthExchangeError("callback state does not match this sign-in")
    code = result.get("code")
    if not isinstance(code, str) or not code:
        raise AuthExchangeError("callback carried no authorization code")

    payload = _token_call(
        doc,
        {
            "grant_type": "authorization_code",
            "client_id": client_id or DYNAMIC_CLIENT,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        },
    )
    # The issued client id is the one to keep: it is what a refresh must name,
    # and it is what the ID token's audience must equal.
    issued = payload.get("client_id")
    if not isinstance(issued, str) or not issued:
        issued = client_id or DYNAMIC_CLIENT
    if issued == DYNAMIC_CLIENT:
        raise AuthExchangeError(
            "the issuer returned the registration placeholder as the client id"
        )
    record = _record_from_token(
        ctx,
        doc,
        payload,
        client_id=issued,
        previous=previous,
        nonce=nonce,
        host_id=host_id,
    )
    _save_record(ctx, record)
    return record


# ---------------------------------------------------------------------------
# The exchange the gateway calls
# ---------------------------------------------------------------------------


def _is_fresh(record: dict[str, Any], *, now: float | None = None) -> bool:
    """Whether the record's access token is still worth sending.

    Both hints are consulted and the earlier wins: ``earliest_refresh_at`` is
    when we would *like* to refresh, ``expires_at`` is when the token stops
    working, and a record whose first field sat later than the second would
    otherwise send a token we already know is dead.
    """
    moment = time.time() if now is None else now
    limits = []
    earliest = record.get("earliest_refresh_at")
    if isinstance(earliest, (int, float)):
        limits.append(float(earliest))
    expires_at = record.get("expires_at")
    if isinstance(expires_at, (int, float)):
        limits.append(float(expires_at) - SKEW)
    return bool(limits) and moment < min(limits)


def _credential(ctx: AuthContext) -> str:
    """Return a usable access token, refreshing it if it is near expiry."""
    record = _read_record(ctx)
    if _is_fresh(record):
        return record["access_token"]
    with _record_lock(ctx):
        # Another process may have refreshed while this one waited for the
        # lock; re-read rather than spend the refresh token a second time.
        record = _read_record(ctx)
        if _is_fresh(record):
            return record["access_token"]
        return _refresh_record(ctx, record)["access_token"]


class SignInWithChatGPT(BaseAuthExchange):
    """The plan route's credential exchange, as the gateway sees it."""

    kind = "oauth"
    consumes_key = False

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        """Whether a human still has to sign in for this provider."""
        return not _record_path(ctx).exists()

    async def credential(self, ctx: AuthContext) -> str:
        """Return the access token, refreshing it on a worker thread.

        The exchange is synchronous stdlib I/O; running it on a thread keeps
        the gateway's event loop free while a token endpoint answers.
        """
        return await asyncio.to_thread(_credential, ctx)

    def account_id(self, ctx: AuthContext) -> str | None:
        """Return the signed-in subject — the account, not the token.

        Two profiles signed into the same account are one account, and a
        re-sign-in does not change it; the access token rotates, the subject
        does not.
        """
        try:
            record = _read_record(ctx)
        except (InteractiveSetupRequired, AuthExchangeError):
            return None
        subject = record.get("subject")
        return subject if isinstance(subject, str) and subject else None


EXCHANGE = SignInWithChatGPT()

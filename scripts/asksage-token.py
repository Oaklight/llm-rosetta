#!/usr/bin/env python3
"""Mint AskSage access tokens for the gateway.

Zero external dependencies — stdlib only.

AskSage's inference endpoints do not accept the account API key.  They take a
short-lived access token, which you get by exchanging the API key::

    POST {host}/user/get-token-with-api-key
    {"email": "<account email>", "api_key": "<account api key>"}
    -> {"response": {"access_token": "..."}, "status": 200}

This script performs that exchange and prints the token on stdout, so it drops
straight into a provider's ``token_command``::

    {
      "providers": {
        "asksage": {
          "type": "asksage--open_responses",
          "base_url": "https://api.asksage.anl.gov/server/openai/v1",
          "token_command": ["python3", "/scripts/asksage-token.py"]
        }
      }
    }

``base_url`` is not optional here.  The shim defaults to
``https://api.asksage.ai/server/openai/v1`` while this script mints against
``https://api.asksage.anl.gov``, and a token from one host is refused by the
other — leaving the line out reintroduces exactly the mismatch described under
`Hosts` below.

Every diagnostic goes to stderr, so ``$(python3 scripts/asksage-token.py)``
captures the token and nothing else.

Hosts
-----
AskSage is reachable on two hosts that serve the same model roster::

    https://api.asksage.anl.gov   Argonne deployment
    https://api.asksage.ai        public deployment

Their credentials are not interchangeable.  An Argonne API key is refused by
the public host's token exchange::

    {"response": "Key doesnt exist", "status": 400}

Both hosts do answer ``GET {host}/server/openai/v1/models`` with the same
49-model list (``GET {host}/models`` is a 404), but that route appears not to
check credentials at all — it answers with a bogus, empty, or absent auth
header — so a working model list is not evidence that a key works.

``--host`` selects where the token is minted, and ``--probe`` confirms the
token is actually accepted for inference on that host; the shim's ``base_url``
has to point at the same one.

Environment
-----------
Read from the environment, never from argv — a key passed as a flag is visible
to every process on the machine via the process table::

    ASKSAGE_TOKEN_EMAIL    account email    (or --email)
    ASKSAGE_TOKEN_API_KEY  account API key
    ASKSAGE_HOST           default https://api.asksage.anl.gov

The account API key falls back to ``ASK_SAGE_API`` (the name the ANL sample
client uses) and then to ``ASKSAGE_API_KEY``.

Those dedicated names exist because of a collision.  Every ``asksage--*`` shim
declares ``connection.api_key_env: ASKSAGE_API_KEY``, and the gateway fills a
provider's ``api_key`` from that variable *before* it considers
``token_command`` (``gateway/providers.py``: ``_apply_shim_defaults`` runs
ahead of ``_resolve_token_command``), which then rejects the pair as mutually
exclusive.  So as long as ``ASKSAGE_API_KEY`` is in the gateway's environment,
the config shown above cannot be built at all::

    ValueError: Provider 'open_responses': token_command and api_key are
    mutually exclusive

Put the account key under ``ASKSAGE_TOKEN_API_KEY`` or ``ASK_SAGE_API`` and
leave ``ASKSAGE_API_KEY`` unset for that process.

Other flags
-----------
::

    --status          Mint and report the token's expiry; print no token
    --probe [MODEL]   Mint, then send one minimal /responses request

The default probe model is ``gpt-4.1``; several models on the roster are not
served over ``/responses`` at all.

``--probe`` only covers the OpenAI-compatible path
(``{host}/server/openai/v1``).  ``asksage--messages`` is served under
``/server/anthropic`` and ``asksage--google_generate`` under
``/server/google/v1beta``, so a passing probe is indirect evidence for those
two rather than proof that the path they use accepts the token.

Docker
------
The gateway image ships no credentials, so pass them into the container and
mount this script read-only::

    # docker-compose.yaml
    #   environment:
    #     ASKSAGE_TOKEN_EMAIL: you@example.gov
    #     ASKSAGE_TOKEN_API_KEY: ${ASKSAGE_API_KEY}
    #   volumes:
    #     - ./asksage-token.py:/scripts/asksage-token.py:ro

Note the variable names on either side: the host variable may be anything, but
the container must see the key under ``ASKSAGE_TOKEN_API_KEY``, not
``ASKSAGE_API_KEY`` — see `Environment` above.

Standalone download (no git clone needed)::

    curl -fsSL -o asksage-token.py \
      https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/scripts/asksage-token.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "https://api.asksage.anl.gov"
USER_PATH = "/user"
SERVER_PATH = "/server/openai/v1"
DEFAULT_PROBE_MODEL = "gpt-4.1"

# Kept below the gateway's own kill budget on purpose.  ``token_command`` runs
# under ``COMMAND_TIMEOUT = 30`` (gateway/transport/token_refresh.py) and the
# process is killed when that expires, so a longer socket timeout means a mint
# that would have finished between this script's deadline and the gateway's is
# thrown away, and the operator sees the gateway's timeout instead of this
# script's diagnostic.  ``scripts/alcf-token.py`` uses 30 for the same reason;
# this one probes as well, and the budget covers the whole process.
TIMEOUT = 25


def _log(msg: str) -> None:
    print(f"asksage-token: {msg}", file=sys.stderr)


def _resolve_host(override: str | None) -> str:
    """Return the AskSage host to mint against, without a trailing slash."""
    host = override or os.environ.get("ASKSAGE_HOST") or DEFAULT_HOST
    return host.rstrip("/")


def _resolve_email(override: str | None) -> str:
    """Return the account email, or exit when it is unset."""
    email = override or os.environ.get("ASKSAGE_TOKEN_EMAIL", "")
    if not email:
        _log("No account email. Set ASKSAGE_TOKEN_EMAIL or pass --email.")
        sys.exit(1)
    return email


def _resolve_api_key() -> str:
    """Return the account API key from the environment, or exit.

    Prefers the names an ``asksage--*`` shim does not claim.  Falling back to
    ``ASKSAGE_API_KEY`` still mints a token standalone, but that variable is
    the shims' ``api_key_env``, and the gateway turns it into ``api_key``
    before it reads ``token_command`` — see the module docstring.
    """
    for name in ("ASKSAGE_TOKEN_API_KEY", "ASK_SAGE_API"):
        value = os.environ.get(name)
        if value:
            return value

    fallback = os.environ.get("ASKSAGE_API_KEY")
    if fallback:
        _log(
            "Using ASKSAGE_API_KEY. That is the asksage--* shims' api_key_env, "
            "so a gateway reading this same environment will reject the "
            "token_command provider as mutually exclusive with api_key. Use "
            "ASKSAGE_TOKEN_API_KEY (or ASK_SAGE_API) for token_command."
        )
        return fallback

    _log("No API key. Set ASKSAGE_TOKEN_API_KEY (or ASK_SAGE_API).")
    sys.exit(1)


def _exchange(host: str, email: str, api_key: str) -> str:
    """Exchange the account API key for an access token.

    Args:
        host: AskSage host, without a trailing slash.
        email: Account email.
        api_key: Account API key.

    Returns:
        The access token.

    Raises:
        SystemExit: The exchange failed or returned no token.
    """
    url = f"{host}{USER_PATH}/get-token-with-api-key"
    payload = json.dumps({"email": email, "api_key": api_key}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            envelope = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        _log(f"Exchange failed: HTTP {exc.code}")
        _log(exc.read().decode(errors="replace")[:300])
        sys.exit(1)
    except urllib.error.URLError as exc:
        _log(f"Cannot reach {url}: {exc.reason}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        _log(f"Exchange returned a non-JSON body: {exc}")
        sys.exit(1)

    # AskSage answers HTTP 200 even when the exchange is refused, putting the
    # failure in the envelope instead:
    #   {"response": "Token is invalid [2]", "status": 400}
    body = envelope.get("response") if isinstance(envelope, dict) else None
    token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        # A non-string or empty token must not reach stdout: token_command
        # treats whatever the process prints as the key, so a literal "None"
        # would be seeded as a credential and every later request would 401.
        _log(f"Exchange rejected: {json.dumps(envelope)[:300]}")
        sys.exit(1)
    return token


def _token_expiry(token: str) -> float | None:
    """Return the token's ``exp`` claim, or None when it is not a JWT.

    The signature is not verified.  This only reads the claim so the operator
    can see when the token dies; nothing is trusted from it.

    Args:
        token: The access token.

    Returns:
        The expiry as a Unix timestamp, or ``None`` if unavailable.
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict):
        # Valid JSON, but not a claims object (``123``, ``"abc"``, ``[...]``).
        # Same answer as a token that is not a JWT at all, rather than an
        # AttributeError from ``.get``.
        return None
    exp = claims.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def _probe(host: str, token: str, model: str) -> bool:
    """Send one minimal ``/responses`` request to confirm the token works.

    Args:
        host: AskSage host, without a trailing slash.
        token: Access token to send.
        model: Model id to request.

    Returns:
        True when the upstream answered 200.
    """
    url = f"{host}{SERVER_PATH}/responses"
    payload = json.dumps(
        {
            "model": model,
            "input": "Reply with the single word: ok",
            "max_output_tokens": 16,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            code = response.getcode()
            body = response.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        code = exc.code
        body = exc.read().decode(errors="replace")
    except urllib.error.URLError as exc:
        _log(f"Probe {model} failed: {exc.reason}")
        return False

    if code == 200:
        _log(f"Probe {model}: OK")
        return True
    _log(f"Probe {model}: HTTP {code} {body[:200]}")
    return False


def main() -> None:
    """Exchange the API key and print the resulting access token."""
    parser = argparse.ArgumentParser(
        description="Mint an AskSage access token for a gateway token_command.",
    )
    parser.add_argument(
        "--email",
        help="AskSage account email (default: $ASKSAGE_TOKEN_EMAIL)",
    )
    parser.add_argument(
        "--host",
        help=f"AskSage host to mint against (default: $ASKSAGE_HOST or {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report the token's expiry instead of printing the token.",
    )
    parser.add_argument(
        "--probe",
        nargs="?",
        const=DEFAULT_PROBE_MODEL,
        metavar="MODEL",
        help="Send one minimal /responses request to verify the token "
        f"(default model: {DEFAULT_PROBE_MODEL}).",
    )
    args = parser.parse_args()

    host = _resolve_host(args.host)
    token = _exchange(host, _resolve_email(args.email), _resolve_api_key())

    # ``is not None`` rather than truthiness: ``--probe ""`` used to read as
    # "no probe", so the token was printed as though the probe had passed.
    if args.probe is not None:
        model = args.probe.strip()
        if not model:
            _log(
                "--probe got an empty model. Omit the value to use "
                f"{DEFAULT_PROBE_MODEL}."
            )
            sys.exit(1)
        if not _probe(host, token, model):
            sys.exit(1)

    if args.status:
        exp = _token_expiry(token)
        if exp is None:
            _log(f"Minted a token against {host}; no readable expiry claim.")
        else:
            remaining = exp - time.time()
            if remaining <= 0:
                _log(f"Token minted against {host} is already expired.")
            else:
                hours, rest = divmod(int(remaining), 3600)
                minutes = rest // 60
                _log(f"Token valid for {hours}h {minutes}m (host={host})")
        return

    print(token)


if __name__ == "__main__":
    main()

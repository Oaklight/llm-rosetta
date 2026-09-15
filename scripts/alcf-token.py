#!/usr/bin/env python3
"""Mint / refresh Globus Bearer tokens for the ALCF Inference Service.

Zero external dependencies — stdlib only.  Replaces the need for
``uvx alcf-ai auth get-access-token`` when you just want a valid token
for the gateway.

Single-user (default)
---------------------
::

    # First time — headless-safe, prints a URL you open on any browser:
    python3 scripts/alcf-token.py --login

    # After that, auto-refreshes and prints the token:
    export ALCF_API_KEY=$(python3 scripts/alcf-token.py)

Multi-user (shared gateway)
---------------------------
Multiple ALCF users can pool their tokens.  Each user logs in to their
own token file; the gateway aggregates them (comma-separated for KeyRing
round-robin)::

    # Each user logs in once:
    python3 scripts/alcf-token.py --login --token-file /shared/alcf-tokens/alice.json
    python3 scripts/alcf-token.py --login --token-file /shared/alcf-tokens/bob.json

    # Gateway startup reads all of them:
    export ALCF_API_KEY=$(python3 scripts/alcf-token.py --tokens-dir /shared/alcf-tokens/)

``--tokens-dir`` refreshes each file independently and outputs one
comma-separated string.  Files that fail to refresh are skipped with
a warning.

Other flags
-----------
::

    --status          Show token expiry (per-file with --tokens-dir)
    --force           Refresh even if not expired
    --probe CLUSTER   Verify the token against sophia / metis / minerva

ALCF Clusters
-------------
===========================  ============  ========================================
Cluster                      Framework     Base URL suffix
===========================  ============  ========================================
Sophia  (NVIDIA A100)        vLLM          /resource_server/sophia/vllm/v1
Metis   (SambaNova SN40L)    SambaNova     /resource_server/metis/api/v1
Minerva (NVIDIA B200)        API           /resource_server/minerva/api/v1
===========================  ============  ========================================

All clusters share the same Globus Bearer token.

Docker
------
When running the gateway in Docker, login on the **host** first, then
mount the token file and this script into the container::

    # 1. Login on the host (one-time):
    python3 scripts/alcf-token.py --login

    # 2. Add volume mounts to docker-compose.yaml:
    #   - ~/.globus:/home/appuser/.globus            # token file (rw)
    #   - ./scripts/alcf-token.py:/scripts/alcf-token.py:ro

    # 3. Add the ALCF provider via admin panel or config.jsonc:
    #   "token_command": ["python3", "/scripts/alcf-token.py"]

The container's ``appuser`` (uid 1000) needs read-write access to the
mounted ``~/.globus`` directory so the refresh token can be updated
in-place.  If your host uid differs, set ``PUID``/``PGID`` in the
compose file or ``chown 1000:1000`` the ``.globus`` directory.

See ``docker/docker-compose.yaml`` for the full example.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Globus constants (from alcf-ai auth.py — public, not secrets)
# ---------------------------------------------------------------------------
AUTH_CLIENT_ID = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
GATEWAY_RESOURCE_SERVER = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
GATEWAY_SCOPE = f"https://auth.globus.org/scopes/{GATEWAY_RESOURCE_SERVER}/action_all"
OPENID_SCOPE = "openid"
AUTH_BASE = "https://auth.globus.org"
TOKEN_ENDPOINT = f"{AUTH_BASE}/v2/oauth2/token"
AUTHORIZE_ENDPOINT = f"{AUTH_BASE}/v2/oauth2/authorize"
SESSION_POLICY = "83732ff2-9c42-4548-b5ce-17e498c84f6a"

DEFAULT_TOKENS_DIR = os.path.join(
    os.path.expanduser("~"),
    ".globus",
    "app",
    AUTH_CLIENT_ID,
    "inference_app",
)
DEFAULT_TOKENS_PATH = os.path.join(DEFAULT_TOKENS_DIR, "tokens.json")

EXPIRY_BUFFER = 120  # seconds

# ---------------------------------------------------------------------------
# ALCF clusters
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, str] = {
    "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "metis": "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
    "minerva": "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1",
}

PROBE_MODELS: dict[str, str] = {
    "sophia": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "metis": "gpt-oss-120b",
    "minerva": "inkling-bf16",
}


def _log(msg: str) -> None:
    print(f"alcf-token: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Token file I/O  (path-parameterised — no global state)
# ---------------------------------------------------------------------------


def _read_tokens(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_tokens(path: str, data: dict) -> None:
    dirname = os.path.dirname(path)
    os.makedirs(dirname, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _get_gateway_entry(data: dict) -> dict | None:
    try:
        return data["data"]["DEFAULT"][GATEWAY_RESOURCE_SERVER]
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# OAuth2 native-app login (headless-safe)
# ---------------------------------------------------------------------------


def _do_login(path: str) -> None:
    import base64
    import hashlib
    import secrets

    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    redirect_uri = f"{AUTH_BASE}/v2/web/auth-code"
    scopes = f"{OPENID_SCOPE} {GATEWAY_SCOPE}"

    params = urllib.parse.urlencode(
        {
            "client_id": AUTH_CLIENT_ID,
            "response_type": "code",
            "scope": scopes,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "session_required_policies": SESSION_POLICY,
        }
    )
    authorize_url = f"{AUTHORIZE_ENDPOINT}?{params}"

    _log("Open this URL in any browser (can be a different machine):")
    print(f"\n  {authorize_url}\n", file=sys.stderr)
    auth_code = input("Paste the Authorization Code here: ").strip()
    if not auth_code:
        _log("No auth code entered.")
        sys.exit(1)

    form = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "client_id": AUTH_CLIENT_ID,
            "code_verifier": verifier,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _log(f"Token exchange failed: HTTP {e.code}")
        _log(body)
        sys.exit(1)

    entries: dict[str, dict] = {}
    _store_token_entry(entries, result)
    for other in result.get("other_tokens", []):
        _store_token_entry(entries, other)

    data = {
        "data": {"DEFAULT": entries},
        "format_version": "2.0",
        "globus-sdk.version": "standalone",
    }
    _write_tokens(path, data)
    _log(f"Login successful. Token file written to {path}")

    if GATEWAY_RESOURCE_SERVER in entries:
        expires_in = entries[GATEWAY_RESOURCE_SERVER].get(
            "expires_at_seconds", 0
        ) - int(time.time())
        h, m = divmod(max(0, expires_in), 3600)
        _log(f"Inference token valid for {h}h {m // 60}m")
    else:
        _log(
            "Warning: no inference gateway token in response. "
            "The requested scope may not have been granted."
        )


def _store_token_entry(entries: dict, token_response: dict) -> None:
    rs = token_response.get("resource_server", "auth.globus.org")
    entry: dict = {
        "resource_server": rs,
        "scope": token_response.get("scope", ""),
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token", ""),
        "expires_at_seconds": int(time.time())
        + token_response.get("expires_in", 172800),
        "token_type": token_response.get("token_type", "Bearer"),
    }
    if "identity_id" in token_response:
        entry["identity_id"] = token_response["identity_id"]
    entries[rs] = entry


# ---------------------------------------------------------------------------
# OAuth2 token refresh (RFC 6749 §6)
# ---------------------------------------------------------------------------


def _refresh_token(entry: dict) -> dict:
    form = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": entry["refresh_token"],
            "client_id": AUTH_CLIENT_ID,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Single-file: ensure fresh token, return access_token or None on failure
# ---------------------------------------------------------------------------


def _ensure_fresh(path: str, *, force: bool = False) -> str | None:
    """Read *path*, refresh if needed, return the access_token (or None)."""
    label = os.path.basename(path)
    try:
        data = _read_tokens(path)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"{label}: cannot read ({e})")
        return None

    entry = _get_gateway_entry(data)
    if entry is None:
        _log(f"{label}: no gateway entry, skipping")
        return None

    remaining = entry.get("expires_at_seconds", 0) - time.time()
    need_refresh = force or (remaining < EXPIRY_BUFFER)

    if need_refresh:
        if not entry.get("refresh_token"):
            _log(f"{label}: no refresh token, skipping")
            return None
        try:
            result = _refresh_token(entry)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            _log(
                f"{label}: refresh failed ({e}), using existing token ({int(remaining)}s remaining)"
            )
            return entry.get("access_token")
        entry["access_token"] = result["access_token"]
        # OAuth servers may rotate the refresh token, but they may also omit
        # it when the existing refresh token remains valid.
        if result.get("refresh_token"):
            entry["refresh_token"] = result["refresh_token"]
        entry["expires_at_seconds"] = int(time.time()) + result["expires_in"]
        try:
            _write_tokens(path, data)
        except OSError as e:
            _log(f"{label}: could not write back ({e})")
        new_h, new_m = divmod(result["expires_in"], 3600)
        _log(f"{label}: refreshed, valid for {new_h}h {new_m // 60}m")

    return entry.get("access_token")


# ---------------------------------------------------------------------------
# Multi-file: collect tokens from a directory
# ---------------------------------------------------------------------------


def _collect_tokens_dir(tokens_dir: str, *, force: bool = False) -> list[str]:
    """Read all .json files in *tokens_dir*, refresh each, return tokens."""
    if not os.path.isdir(tokens_dir):
        _log(f"Not a directory: {tokens_dir}")
        sys.exit(1)

    files = sorted(
        f
        for f in os.listdir(tokens_dir)
        if f.endswith(".json") and not f.startswith(".")
    )
    if not files:
        _log(f"No .json token files found in {tokens_dir}")
        sys.exit(1)

    tokens: list[str] = []
    for fname in files:
        path = os.path.join(tokens_dir, fname)
        tok = _ensure_fresh(path, force=force)
        if tok:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------


def _probe_cluster(cluster: str, token: str) -> bool:
    base = CLUSTERS.get(cluster)
    if not base:
        _log(f"Unknown cluster '{cluster}'. Known: {', '.join(CLUSTERS)}")
        return False

    model = PROBE_MODELS.get(cluster, "test")
    url = f"{base}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        _log(f"Probe {cluster} failed: {e}")
        return False

    if code == 200:
        _log(f"Probe {cluster}: OK (model={model})")
        return True
    else:
        _log(f"Probe {cluster}: HTTP {code} (model={model})")
        return False


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def _show_status(path: str) -> None:
    label = os.path.basename(path)
    try:
        data = _read_tokens(path)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"{label}: cannot read ({e})")
        return

    entry = _get_gateway_entry(data)
    if entry is None:
        _log(f"{label}: no gateway entry")
        return

    identity = entry.get("identity_id", "unknown")
    remaining = entry.get("expires_at_seconds", 0) - time.time()
    if remaining > 0:
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        _log(f"{label}: valid for {h}h {m}m {s}s (identity={identity})")
    else:
        _log(f"{label}: expired {-int(remaining)}s ago (identity={identity})")


def _show_status_dir(tokens_dir: str) -> None:
    if not os.path.isdir(tokens_dir):
        _log(f"Not a directory: {tokens_dir}")
        sys.exit(1)
    files = sorted(
        f
        for f in os.listdir(tokens_dir)
        if f.endswith(".json") and not f.startswith(".")
    )
    if not files:
        _log(f"No .json token files found in {tokens_dir}")
        sys.exit(1)
    for fname in files:
        _show_status(os.path.join(tokens_dir, fname))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mint/refresh Globus Bearer tokens for ALCF Inference.",
    )
    ap.add_argument(
        "--login",
        action="store_true",
        help="Run the initial Globus login flow (headless-safe).",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Show token expiry status; don't print tokens.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force a refresh even if tokens are still valid.",
    )
    ap.add_argument(
        "--probe",
        metavar="CLUSTER",
        help=f"Verify the token against a cluster ({', '.join(CLUSTERS)}).",
    )

    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--token-file",
        metavar="PATH",
        help="Use a specific token file instead of the default Globus path.",
    )
    group.add_argument(
        "--tokens-dir",
        metavar="DIR",
        help="Read all .json token files in DIR; output comma-separated "
        "tokens for KeyRing round-robin.",
    )

    args = ap.parse_args()
    token_file = args.token_file or DEFAULT_TOKENS_PATH

    # --- login ---
    if args.login:
        if args.tokens_dir:
            _log(
                "--login cannot be used with --tokens-dir. "
                "Use --token-file to login to a specific file."
            )
            sys.exit(1)
        _do_login(token_file)
        return

    # --- status ---
    if args.status:
        if args.tokens_dir:
            _show_status_dir(args.tokens_dir)
        else:
            if not os.path.isfile(token_file):
                _log(f"Token file not found: {token_file}")
                _log("Run:  python3 scripts/alcf-token.py --login")
                sys.exit(1)
            _show_status(token_file)
        sys.exit(0)

    # --- multi-token ---
    if args.tokens_dir:
        tokens = _collect_tokens_dir(args.tokens_dir, force=args.force)
        if not tokens:
            _log("No valid tokens found.")
            sys.exit(1)
        _log(f"{len(tokens)} token(s) loaded from {args.tokens_dir}")
        if args.probe:
            _probe_cluster(args.probe, tokens[0])
        print(",".join(tokens))
        return

    # --- single-token ---
    if not os.path.isfile(token_file):
        _log(f"Token file not found: {token_file}")
        _log("Run:  python3 scripts/alcf-token.py --login")
        sys.exit(1)

    tok = _ensure_fresh(token_file, force=args.force)
    if not tok:
        _log("Failed to obtain a valid token.")
        sys.exit(1)

    if args.probe:
        if not _probe_cluster(args.probe, tok):
            sys.exit(1)

    print(tok)


if __name__ == "__main__":
    main()

"""Sign in with ChatGPT — the shim's OAuth credential exchange.

The exchange is a shim module (``providers/openai_chatgpt/open_responses/auth.py``)
loaded by path, so these tests load it the same way the shim loader does and
drive it through a fake transport.  Nothing here reaches the network: the
issuer, its JWKS and the token endpoint are all answered locally, and the ID
tokens are signed with a throwaway RSA key.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.shims.auth_exchange import AuthContext, InteractiveSetupRequired

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "packages/llm-rosetta/src/llm_rosetta/shims/providers/openai_chatgpt/open_responses/auth.py"
)

# A throwaway 2048-bit key, generated for these tests only.  The exchange
# verifies RS256 with ``pow`` and ``hashlib``, so the tests sign with the same
# arithmetic: nothing here needs a crypto library either.
_KEY_N = (
    "sv7iuEejSNNm-vCYDSqHzzOs6bsd_YdSZIvq6TPzK0ZNPMjUzW6RvIveHXic6B8AhSh3QsUI2wdeZ9TpY3syInyS"
    "yDfvzW92OeDLzAYKiSrZYcBczuRWqH1B0V2CHg6E4fg1SUJg0khdSIwUyjeUCyuluecgXYxQd8argjWqPwWurrO"
    "2Lxg4EIw7LPdCvffHCOsv0vdIUMbih2FGW1feTii8gyFKOqJ9hWys2aSbL0Oz7ZWRtnPoAlVtndvs1m8r48LPkL"
    "fFStbqxev9XbHiUS7mgJAOYzK3KsZO3HDN06UmttUXvdcWU4BAdC2xzGGbk3isIE-p4rHnioCQF9vJsw"
)
_KEY_D = (
    "UK8nOIUGO-4bJfNsvsaUmFhFUchoYN9uSudtGPizXKbgaw1mK0HzjIeJtEQpilJrNBrFA4J4lu6P5gLpv0-Jj"
    "lUT_w-D11kudDF5-_ofJqx7d54z14jC7Ukk8YCShgTqbkepkV_L8A3xiqGmP27lG1Ht7soZ40E8Bnm11shk-O"
    "nPqD4X-Q_drD9LqA0__U6PScbjEPssiejFXVFPOdI869Ir33AlsbFKy6HUCgVnuHhHJoRJ5v2cd9MjCXbZL2IJ"
    "pSFheOz9V4LgjwK5C1mNbp6l5uR8P_cfhIcJ4bo-ac0iNI0ctqP6GaCjPCJN2UP8LSHfHmarO2_03AvAZ9r6DQ"
)
_KEY_E = 65537
_KID = "test-key-1"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _int(value: str) -> int:
    return int.from_bytes(
        base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)), "big"
    )


def _sign(signing_input: bytes) -> str:
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420")
    digest_info += hashlib.sha256(signing_input).digest()
    size = (_int(_KEY_N).bit_length() + 7) // 8
    block = (
        b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    )
    signature = pow(int.from_bytes(block, "big"), _int(_KEY_D), _int(_KEY_N))
    return _b64url(signature.to_bytes(size, "big"))


def id_token(
    claims: dict[str, Any] | None = None,
    *,
    kid: str = _KID,
    alg: str = "RS256",
    tamper: bool = False,
) -> str:
    """Build a signed ID token whose claims these tests control."""
    now = int(time.time())
    body = {
        "iss": "https://auth.openai.com",
        "aud": "oaiapp_test",
        "sub": "user-1",
        "iat": now,
        "exp": now + 3600,
        "nonce": "n",
    }
    body.update(claims or {})
    header = _b64url(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(body).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = _sign(signing_input)
    if tamper:
        signature = _b64url(bytes(len(signature) * 3 // 4))
    return f"{header}.{payload}.{signature}"


@pytest.fixture()
def siwc():
    """The shim module, loaded the way the shim loader loads it."""
    spec = importlib.util.spec_from_file_location("_siwc_auth_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._reset_discovery_cache()
    return module


@pytest.fixture()
def ctx(tmp_path):
    return AuthContext(provider="chatgpt-personal", state_dir=str(tmp_path / "state"))


class FakeIssuer:
    """A stand-in for auth.openai.com: discovery, JWKS and the token endpoint."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.tokens: list[dict[str, Any]] = []
        self.fail_with: str | None = None
        self.id_token_claims: dict[str, Any] = {}
        self.omit_refresh = False
        self.issued_client = "oaiapp_test"
        self.revoked: list[dict[str, str]] = []
        self.keys: list[dict[str, Any]] = [
            {
                "kty": "RSA",
                "kid": _KID,
                "n": _KEY_N,
                "e": _b64url(_KEY_E.to_bytes(3, "big")),
            }
        ]
        module._request = self._request

    def _request(self, url, *, data=None, headers=None, method=None):
        self.calls.append((url, dict(data or {})))
        if url.endswith("openid-configuration"):
            return {
                "issuer": "https://auth.openai.com",
                "authorization_endpoint": "https://auth.openai.com/api/accounts/authorize",
                "token_endpoint": "https://auth.openai.com/api/accounts/oauth/token",
                "revocation_endpoint": "https://auth.openai.com/api/accounts/oauth/revoke",
                "jwks_uri": "https://auth.openai.com/.well-known/jwks.json",
            }
        if url.endswith("jwks.json"):
            return {"keys": self.keys}
        if url.endswith("/revoke"):
            self.revoked.append(dict(data or {}))
            return {}
        if self.fail_with is not None:
            raise self.module.AuthExchangeError(f"token endpoint: {self.fail_with}")
        payload = {
            "access_token": f"access-{len(self.tokens)}",
            "refresh_token": f"refresh-{len(self.tokens)}",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": self.module.SCOPES,
            "id_token": id_token(self.id_token_claims),
            "client_id": self.issued_client,
        }
        if self.omit_refresh:
            payload.pop("refresh_token")
        self.tokens.append(payload)
        return payload

    def token_calls(self) -> list[dict[str, str]]:
        return [d for url, d in self.calls if url.endswith("/oauth/token")]


@pytest.fixture()
def issuer(siwc):
    return FakeIssuer(siwc)


def write_record(ctx: AuthContext, **overrides: Any) -> dict[str, Any]:
    """Write a credential record as a completed sign-in would."""
    now = time.time()
    record = {
        "issuer": "https://auth.openai.com",
        "client_id": "oaiapp_test",
        "ext_agent_host_id": "urn:uuid:test",
        "access_token": "access-0",
        "refresh_token": "refresh-0",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expires_at": now + 3600,
        "earliest_refresh_at": now + 3000,
        "scopes": ["openid"],
        "subject": "user-1",
        "id_token": id_token(),
        "saved_at": "2026-10-22T00:00:00Z",
    }
    record.update(overrides)
    path = Path(ctx.state_dir or "")
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    (path / "credential.json").write_text(json.dumps(record))
    os.chmod(path / "credential.json", 0o600)
    return record


# ---------------------------------------------------------------------------
# The authorize URL
# ---------------------------------------------------------------------------


class TestAuthorizeUrl:
    def test_a_first_sign_in_registers(self, siwc):
        url = siwc._authorize_url(
            {
                "authorization_endpoint": "https://auth.openai.com/api/accounts/authorize"
            },
            redirect_uri="http://127.0.0.1:1234/auth/callback",
            state="s",
            nonce="n",
            challenge="c",
            client_id=None,
            host_id="urn:uuid:h",
        )
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)

        assert params["client_id"] == ["dynamic_agent_client"]
        assert params["agent_name_hint"] == ["llm-rosetta"]
        assert params["ext_agent_host_id"] == ["urn:uuid:h"]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"] == ["c"]
        assert params["response_type"] == ["code"]
        assert params["resource"] == ["https://api.openai.com/v1"]
        assert "chatgpt.tokens.use.direct" in params["scope"][0]
        assert "id_token_hint" not in params

    def test_a_returning_sign_in_reuses_the_issued_client(self, siwc):
        url = siwc._authorize_url(
            {
                "authorization_endpoint": "https://auth.openai.com/api/accounts/authorize"
            },
            redirect_uri="http://127.0.0.1:1234/auth/callback",
            state="s",
            nonce="n",
            challenge="c",
            client_id="oaiapp_issued",
            host_id="urn:uuid:h",
            id_token_hint="previous-id-token",
        )
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)

        assert params["client_id"] == ["oaiapp_issued"]
        # The placeholder must never be saved as the issued id, nor sent again.
        assert "agent_name_hint" not in params
        assert params["id_token_hint"] == ["previous-id-token"]

    def test_pkce_challenge_is_the_verifier_digest(self, siwc):
        verifier, challenge = siwc._pkce()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )

        assert challenge == expected
        assert len(verifier) >= 43


# ---------------------------------------------------------------------------
# The credential the gateway asks for
# ---------------------------------------------------------------------------


class TestCredential:
    def test_a_fresh_record_is_returned_as_is(self, siwc, ctx, issuer):
        write_record(ctx)

        assert siwc._credential(ctx) == "access-0"
        assert issuer.token_calls() == []

    def test_a_near_expiry_record_is_refreshed(self, siwc, ctx, issuer):
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        assert siwc._credential(ctx) == "access-0"
        assert issuer.token_calls()[0]["grant_type"] == "refresh_token"
        stored = json.loads((Path(ctx.state_dir) / "credential.json").read_text())
        assert stored["refresh_token"] == "refresh-0"

    def test_a_rotated_refresh_token_is_persisted(self, siwc, ctx, issuer):
        """Every successful refresh replaces the refresh token."""
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        siwc._credential(ctx)

        stored = json.loads((Path(ctx.state_dir) / "credential.json").read_text())
        assert stored["refresh_token"] == "refresh-0"
        assert stored["access_token"] == "access-0"
        assert stored["earliest_refresh_at"] > time.time()

    def test_a_refresh_without_a_new_token_keeps_the_stored_one(
        self, siwc, ctx, issuer
    ):
        """Losing the grant because the vendor omitted a field is not an option."""
        issuer.omit_refresh = True
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        siwc._credential(ctx)

        stored = json.loads((Path(ctx.state_dir) / "credential.json").read_text())
        assert stored["refresh_token"] == "refresh-0"

    def test_a_refused_grant_asks_for_a_new_sign_in(self, siwc, ctx, issuer):
        """``invalid_grant`` is not a transient failure; only a human fixes it."""
        issuer.fail_with = "invalid_grant"
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        with pytest.raises(InteractiveSetupRequired, match="sign in again"):
            siwc._credential(ctx)

    def test_an_expired_token_is_refreshed_even_if_the_hint_says_otherwise(
        self, siwc, ctx, issuer
    ):
        """``expires_at`` wins over a later ``earliest_refresh_at``.

        Both are server-supplied; a record whose refresh hint sat past the
        expiry would otherwise send a token we already know is dead.
        """
        write_record(
            ctx,
            earliest_refresh_at=time.time() + 3600,
            expires_at=time.time() - 10,
        )

        siwc._credential(ctx)

        assert len(issuer.token_calls()) == 1

    def test_no_record_asks_for_the_interactive_run(self, siwc, ctx, issuer):
        with pytest.raises(InteractiveSetupRequired, match="sign-in"):
            siwc._credential(ctx)

    def test_concurrent_refreshes_spend_the_token_once(self, siwc, ctx, issuer):
        """The refresh token rotates, so two refreshes would lose the account."""
        write_record(ctx, earliest_refresh_at=time.time() - 1)
        started = threading.Barrier(2)
        tokens: list[str] = []

        def worker() -> None:
            started.wait()
            tokens.append(siwc._credential(ctx))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(issuer.token_calls()) == 1
        assert tokens == ["access-0", "access-0"]


# ---------------------------------------------------------------------------
# ID-token verification
# ---------------------------------------------------------------------------


class TestIdTokenVerification:
    def _verify(self, siwc, token: str, **kw):
        return siwc._verify_id_token(
            token,
            jwks_uri="https://auth.openai.com/.well-known/jwks.json",
            issuer="https://auth.openai.com",
            client_id="oaiapp_test",
            **kw,
        )

    def test_a_good_token_verifies(self, siwc, issuer):
        claims = self._verify(siwc, id_token())

        assert claims["sub"] == "user-1"

    def test_a_tampered_signature_is_refused(self, siwc, issuer):
        with pytest.raises(siwc.AuthExchangeError, match="signature"):
            self._verify(siwc, id_token(tamper=True))

    def test_an_unsigned_algorithm_is_refused(self, siwc, issuer):
        with pytest.raises(siwc.AuthExchangeError, match="not RS256"):
            self._verify(siwc, id_token(alg="none"))

    def test_an_unknown_key_is_refused(self, siwc, issuer):
        with pytest.raises(siwc.AuthExchangeError, match="unknown key"):
            self._verify(siwc, id_token(kid="other-key"))

    @pytest.mark.parametrize(
        ("claims", "message"),
        [
            ({"iss": "https://evil.example"}, "issuer"),
            ({"aud": "someone-else"}, "different client"),
            ({"exp": int(time.time()) - 3600}, "expired"),
            ({"iat": int(time.time()) + 3600}, "future"),
            ({"azp": "another-client"}, "authorized party"),
            ({"nonce": "wrong"}, "nonce"),
            ({"sub": ""}, "subject"),
        ],
    )
    def test_claims_are_checked(self, siwc, issuer, claims, message):
        with pytest.raises(siwc.AuthExchangeError, match=message):
            self._verify(siwc, id_token(claims), nonce="n")

    def test_the_nonce_must_match_the_sign_in(self, siwc, issuer):
        with pytest.raises(siwc.AuthExchangeError, match="nonce"):
            self._verify(siwc, id_token({"nonce": "n"}), nonce="different")

    @pytest.mark.parametrize("exponent", [1, 2])
    def test_an_implausible_exponent_is_refused(self, siwc, issuer, exponent):
        """``e = 1`` makes the signature the plaintext block — no forgery for free."""
        issuer.keys = [
            {
                "kty": "RSA",
                "kid": _KID,
                "n": _KEY_N,
                "e": _b64url(exponent.to_bytes(1, "big")),
            }
        ]

        with pytest.raises(siwc.AuthExchangeError, match="implausible RSA exponent"):
            self._verify(siwc, id_token())


# ---------------------------------------------------------------------------
# The credential store
# ---------------------------------------------------------------------------


class TestCredentialStore:
    def test_the_record_is_owner_only(self, siwc, ctx, issuer):
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        siwc._credential(ctx)

        mode = stat.S_IMODE(os.stat(Path(ctx.state_dir) / "credential.json").st_mode)
        assert mode == 0o600
        assert stat.S_IMODE(os.stat(ctx.state_dir).st_mode) == 0o700

    def test_the_write_leaves_no_temporary_behind(self, siwc, ctx, issuer):
        write_record(ctx, earliest_refresh_at=time.time() - 1)

        siwc._credential(ctx)

        leftovers = [
            p.name for p in Path(ctx.state_dir).iterdir() if p.name.startswith(".")
        ]
        assert leftovers == [".lock"]

    def test_each_provider_gets_its_own_directory(self, siwc, tmp_path):
        first = AuthContext(provider="team-a", state_dir=str(tmp_path / "a"))
        second = AuthContext(provider="team-b", state_dir=str(tmp_path / "b"))
        write_record(first, access_token="a-token")
        write_record(second, access_token="b-token")

        assert siwc._credential(first) == "a-token"
        assert siwc._credential(second) == "b-token"

    def test_a_shim_chosen_default_is_namespaced_by_provider(self, siwc):
        """Without a gateway data dir the exchange still separates accounts."""
        first = AuthContext(provider="team-a")
        second = AuthContext(provider="team-b")

        assert siwc._record_path(first) != siwc._record_path(second)
        assert "team-a" in str(siwc._record_path(first))

    def test_the_host_id_is_stable_and_private(self, siwc, ctx):
        first = siwc._host_id(ctx)
        second = siwc._host_id(ctx)

        assert first == second
        assert first.startswith("urn:uuid:")
        path = Path(ctx.state_dir) / "host-id"
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# ---------------------------------------------------------------------------
# The contract the gateway sees
# ---------------------------------------------------------------------------


class TestTheExchangeContract:
    def test_interactive_setup_is_reported_until_a_record_exists(self, siwc, ctx):
        assert siwc.EXCHANGE.needs_interactive_setup(ctx) is True

        write_record(ctx)

        assert siwc.EXCHANGE.needs_interactive_setup(ctx) is False

    def test_the_account_is_the_subject(self, siwc, ctx):
        write_record(ctx, subject="user-42")

        assert siwc.EXCHANGE.account_id(ctx) == "user-42"

    def test_no_record_means_no_identity(self, siwc, ctx):
        assert siwc.EXCHANGE.account_id(ctx) is None

    def test_the_exchange_does_not_consume_the_operator_key(self, siwc):
        assert siwc.EXCHANGE.consumes_key is False
        assert siwc.EXCHANGE.kind == "oauth"

    def test_credential_is_awaited_on_a_worker_thread(self, siwc, ctx, issuer):
        import asyncio

        write_record(ctx)

        assert asyncio.run(siwc.EXCHANGE.credential(ctx)) == "access-0"


# ---------------------------------------------------------------------------
# The interactive run
# ---------------------------------------------------------------------------


class TestInteractiveSignIn:
    def _login_in_background(self, siwc, ctx, **kw):
        """Start ``login`` on a thread and return it, so the callback can be sent."""
        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                outcome["record"] = siwc.login(
                    ctx, open_browser=False, timeout=10, **kw
                )
            except BaseException as exc:  # noqa: BLE001 - reported to the test
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, outcome

    def test_the_callback_completes_the_sign_in(self, siwc, ctx, issuer, capsys):
        thread, outcome = self._login_in_background(siwc, ctx)
        url = self._await_authorize_url(capsys)
        # The issuer echoes the nonce it was sent, which is what the exchange
        # checks the ID token against.
        issuer.id_token_claims = {"nonce": self._param(url, "nonce")}

        self._send_callback(
            url, {"code": "the-code", "state": self._param(url, "state")}
        )
        thread.join(timeout=10)

        assert "error" not in outcome, outcome.get("error")
        stored = json.loads((Path(ctx.state_dir) / "credential.json").read_text())
        assert stored["client_id"] == "oaiapp_test"
        assert stored["access_token"] == "access-0"
        exchange = issuer.token_calls()[0]
        assert exchange["grant_type"] == "authorization_code"
        assert exchange["code"] == "the-code"
        assert exchange["code_verifier"]

    def test_a_wrong_state_is_refused(self, siwc, ctx, issuer, capsys):
        thread, outcome = self._login_in_background(siwc, ctx)
        url = self._await_authorize_url(capsys)

        self._send_callback(url, {"code": "the-code", "state": "not-the-state"})
        thread.join(timeout=10)

        assert isinstance(outcome.get("error"), siwc.AuthExchangeError)
        assert "state" in str(outcome["error"])
        assert not (Path(ctx.state_dir) / "credential.json").exists()

    def _await_authorize_url(self, capsys) -> str:
        for _ in range(100):
            out = capsys.readouterr().out
            for line in out.splitlines():
                if line.startswith("https://auth.openai.com/api/accounts/authorize"):
                    return line
            time.sleep(0.05)
        raise AssertionError("the authorize URL was never printed")

    def _param(self, url: str, name: str) -> str:
        return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)[name][0]

    def _send_callback(self, url: str, params: dict[str, str]) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        port = urllib.parse.urlparse(query["redirect_uri"][0]).port
        target = (
            f"http://127.0.0.1:{port}/auth/callback?{urllib.parse.urlencode(params)}"
        )
        try:
            urllib.request.urlopen(target, timeout=5).read()
        except urllib.error.HTTPError:
            pass


def test_the_shim_loader_loads_the_shipped_exchange():
    """The module is a shim's ``auth.py``, so the loader must find it there.

    #1058 adds the ``provider.yaml`` beside it; until then the directory has
    no shim, and the built-in scan must skip it rather than fail on it.
    """
    from llm_rosetta.shims.providers import _load_auth_exchange

    exchange = _load_auth_exchange(MODULE_PATH)

    assert exchange is not None
    assert exchange.kind == "oauth"
    assert callable(exchange.credential)


class TestInstallingARecord:
    """A record produced on a machine with a browser, handed to the gateway."""

    def _record(self, **overrides: Any) -> dict[str, Any]:
        record = {
            "client_id": "oaiapp_test",
            "access_token": "access-uploaded",
            "refresh_token": "refresh-uploaded",
            "id_token": id_token(),
            "token_type": "Bearer",
            "expires_in": 3600,
            "scopes": ["openid", "offline_access"],
        }
        record.update(overrides)
        return record

    def test_a_verified_record_is_stored(self, siwc, ctx, issuer):
        summary = siwc.EXCHANGE.install(ctx, self._record())

        stored = json.loads((Path(ctx.state_dir) / "credential.json").read_text())
        assert stored["access_token"] == "access-uploaded"
        assert stored["refresh_token"] == "refresh-uploaded"
        assert stored["subject"] == "user-1"
        assert stored["expires_at"] > time.time()
        assert summary["connected"] is True
        assert summary["subject"] == "user-1"

    def test_the_summary_carries_no_token(self, siwc, ctx, issuer):
        summary = siwc.EXCHANGE.install(ctx, self._record())

        rendered = json.dumps(summary)
        assert "access-uploaded" not in rendered
        assert "refresh-uploaded" not in rendered

    def test_the_installed_record_is_usable_and_private(self, siwc, ctx, issuer):
        siwc.EXCHANGE.install(ctx, self._record())

        assert siwc._credential(ctx) == "access-uploaded"
        assert issuer.token_calls() == []
        mode = stat.S_IMODE(os.stat(Path(ctx.state_dir) / "credential.json").st_mode)
        assert mode == 0o600

    def test_the_subject_is_the_verified_one(self, siwc, ctx, issuer):
        """A caller cannot claim to be someone else by putting a subject in."""
        siwc.EXCHANGE.install(
            ctx,
            self._record(subject="someone-else", id_token=id_token({"sub": "user-9"})),
        )

        assert siwc.EXCHANGE.account_id(ctx) == "user-9"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"id_token": ""}, "no id_token"),
            ({"refresh_token": ""}, "no refresh token"),
            ({"access_token": ""}, "no access token"),
            ({"client_id": "dynamic_agent_client"}, "client id the issuer granted"),
            ({"client_id": ""}, "client id the issuer granted"),
            ({"id_token": id_token({"aud": "someone-else"})}, "different client"),
            ({"id_token": id_token(tamper=True)}, "signature"),
        ],
    )
    def test_a_record_that_does_not_verify_is_refused(
        self, siwc, ctx, issuer, overrides, message
    ):
        with pytest.raises(siwc.AuthExchangeError, match=message):
            siwc.EXCHANGE.install(ctx, self._record(**overrides))
        assert not (Path(ctx.state_dir) / "credential.json").exists()

    def test_describe_reports_the_signed_in_account(self, siwc, ctx, issuer):
        assert siwc.EXCHANGE.describe(ctx) is None

        siwc.EXCHANGE.install(ctx, self._record())

        described = siwc.EXCHANGE.describe(ctx)
        assert described["connected"] is True
        assert described["subject"] == "user-1"
        assert described["expires_at"] > time.time()

    def test_revoke_removes_the_record(self, siwc, ctx, issuer):
        siwc.EXCHANGE.install(ctx, self._record())

        assert siwc.EXCHANGE.revoke(ctx) is True

        assert not (Path(ctx.state_dir) / "credential.json").exists()
        assert issuer.revoked and issuer.revoked[0]["token"] == "refresh-uploaded"

"""The admin endpoint for shim-declared credentials.

A provider whose shim ships ``auth.py`` has no key to type in: its credential
is a record the exchange owns.  The endpoint is how a record produced on a
machine with a browser reaches the gateway — and it must stay a courier: the
exchange decides the shape, the gateway never renders a token.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any


from llm_rosetta_gateway.admin.routes.credential import (
    delete_provider_credential,
    get_provider_credential,
    put_provider_credential,
)
from llm_rosetta_gateway.transport.provider_info import (
    TOKEN_PENDING_SENTINEL,
    ProviderInfo,
    openai_auth,
)
from llm_rosetta.shims.auth_exchange import AuthContext, AuthExchangeError

RECORD = {"access_token": "a", "refresh_token": "r", "id_token": "i", "client_id": "c"}


class _Exchange:
    """A credential exchange that records what the endpoint did to it."""

    kind = "oauth"
    consumes_key = False

    def __init__(self, *, fail_with: str | None = None, credential_fails: bool = False):
        self.fail_with = fail_with
        self.credential_fails = credential_fails
        self.installed: list[dict[str, Any]] = []
        self.revoked = 0
        self._record: dict[str, Any] | None = None

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        return self._record is None

    async def credential(self, ctx: AuthContext) -> str:
        if self.credential_fails:
            raise AuthExchangeError("the token endpoint is unreachable")
        if self._record is None:
            from llm_rosetta.shims.auth_exchange import InteractiveSetupRequired

            raise InteractiveSetupRequired("nothing installed")
        return "minted-token"

    def account_id(self, ctx: AuthContext) -> str | None:
        return (self._record or {}).get("subject")

    def describe(self, ctx: AuthContext) -> dict[str, Any] | None:
        if self._record is None:
            return None
        return {"connected": True, "subject": self._record["subject"]}

    def install(self, ctx: AuthContext, record: Any) -> dict[str, Any]:
        if self.fail_with:
            raise AuthExchangeError(self.fail_with)
        if not isinstance(record, dict):
            raise AuthExchangeError("a credential record must be an object")
        self.installed.append(record)
        self._record = {"subject": "user-1", **record}
        return {"connected": True, "subject": "user-1"}

    def revoke(self, ctx: AuthContext) -> bool:
        self.revoked += 1
        self._record = None
        return True


class _LegacyExchange:
    """An exchange written before ``install()`` existed."""

    kind = "oauth"
    consumes_key = False

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        return True

    async def credential(self, ctx: AuthContext) -> str:
        raise AuthExchangeError("nothing installed")

    def account_id(self, ctx: AuthContext) -> str | None:
        return None


def _pinfo(exchange: Any, *, key: str = TOKEN_PENDING_SENTINEL) -> ProviderInfo:
    return ProviderInfo(
        name="open_responses",
        api_key=key,
        base_url="https://api.openai.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/responses",
        auth_exchange=exchange,
        auth_context=(
            AuthContext(provider="chatgpt", state_dir="/tmp/none") if exchange else None
        ),
    )


class _Request:
    """The slice of the request object these handlers read."""

    def __init__(self, config: Any, body: dict[str, Any] | None = None) -> None:
        self.app = SimpleNamespace(gateway_config=config)
        self._body = body or {}

    def json(self) -> dict[str, Any]:
        if not self._body:
            raise ValueError("no body")
        return self._body


def _config(*, exchange: Any = None, key: str = TOKEN_PENDING_SENTINEL):
    return SimpleNamespace(providers={"chatgpt": _pinfo(exchange, key=key)})


def _call(handler: Any, request: Any, name: str = "chatgpt") -> Any:
    return asyncio.run(handler(request, name))


class TestGet:
    def test_a_provider_without_an_exchange_is_refused(self):
        response = _call(get_provider_credential, _Request(_config(key="sk-static")))

        assert response.status_code == 409
        assert b"mints no credential of its own" in response.body

    def test_an_unknown_provider_is_not_found(self):
        response = _call(get_provider_credential, _Request(_config()), name="nope")

        assert response.status_code == 404
        assert b"not found" in response.body

    def test_a_provider_waiting_for_a_sign_in_says_so(self):
        response = _call(
            get_provider_credential, _Request(_config(exchange=_Exchange()))
        )
        payload = json.loads(response.body)

        assert payload["installed"] is False
        assert payload["state"] == "needs_setup"
        assert payload["needs_interactive_setup"] is True
        assert payload["account"] is None
        assert payload["kind"] == "oauth"

    def test_a_record_that_cannot_be_read_is_its_own_state(self):
        """Neither "connected" nor "sign in" — the operator would take the
        wrong action for either."""
        exchange: Any = _Exchange()

        def boom(ctx):
            raise AuthExchangeError("the record is corrupt")

        exchange.describe = boom

        payload = json.loads(
            _call(get_provider_credential, _Request(_config(exchange=exchange))).body
        )

        assert payload["state"] == "unreadable"
        assert payload["installed"] is False

    def test_the_summary_never_carries_a_token(self):
        exchange = _Exchange()
        exchange.install(AuthContext(provider="chatgpt"), dict(RECORD))

        payload = json.loads(
            _call(get_provider_credential, _Request(_config(exchange=exchange))).body
        )

        assert payload["installed"] is True
        # The account identity is a digest, never the raw subject.
        assert payload["account"].startswith("acct:")
        assert "user-1" not in payload["account"]
        assert "a" not in json.dumps(payload.get("summary"))  # no access token


class TestPut:
    def test_a_record_is_handed_to_the_exchange_and_the_provider_reseeded(self):
        exchange = _Exchange()
        pinfo = _pinfo(exchange)
        request = _Request(SimpleNamespace(providers={"chatgpt": pinfo}), dict(RECORD))

        response = _call(put_provider_credential, request)
        payload = json.loads(response.body)

        assert response.status_code == 200
        assert payload["installed"] is True
        assert exchange.installed == [RECORD]
        # The key ring now holds what the exchange minted, without a restart.
        assert pinfo.key_ring.next() == "minted-token"
        assert pinfo.ready is True

    def test_the_record_is_passed_through_untouched(self):
        """The exchange owns the shape; the gateway is a courier."""
        exchange = _Exchange()
        extra = {**RECORD, "vendor_specific": {"nested": [1, 2]}}

        _call(
            put_provider_credential,
            _Request(_config(exchange=exchange), dict(extra)),
        )

        assert exchange.installed == [extra]

    def test_a_refused_record_is_a_400_naming_the_reason(self):
        exchange = _Exchange(fail_with="the record carries no refresh token")

        response = _call(
            put_provider_credential, _Request(_config(exchange=exchange), dict(RECORD))
        )

        assert response.status_code == 400
        assert b"no refresh token" in response.body

    def test_an_empty_body_is_refused(self):
        response = _call(
            put_provider_credential, _Request(_config(exchange=_Exchange()))
        )

        assert response.status_code == 400

    def test_an_exchange_that_cannot_install_says_so(self):
        """A duck-typed exchange without the method at all."""
        response = _call(
            put_provider_credential,
            _Request(_config(exchange=_LegacyExchange()), dict(RECORD)),
        )

        assert response.status_code == 501

    def test_an_exchange_written_against_the_base_class_says_so_too(self):
        """``BaseAuthExchange.install`` raises; presence is not the question."""
        from llm_rosetta.shims.auth_exchange import BaseAuthExchange

        class _BaseOnly(BaseAuthExchange):
            async def credential(self, ctx):  # pragma: no cover - never called
                return "unused"

        response = _call(
            put_provider_credential,
            _Request(_config(exchange=_BaseOnly()), dict(RECORD)),
        )

        assert response.status_code == 501
        assert b"does not implement install()" in response.body

    def test_a_reseed_that_fails_still_empties_the_ring(self):
        """The old token may belong to the account the operator just replaced.

        The record is installed, so the panel would say the new account is in
        use — while the ring still authenticates as the old one.  Emptied is
        the only honest state.
        """
        exchange = _Exchange(credential_fails=True)
        pinfo = _pinfo(exchange, key="old-account-token")
        request = _Request(SimpleNamespace(providers={"chatgpt": pinfo}), dict(RECORD))

        response = _call(put_provider_credential, request)

        assert response.status_code == 200
        assert exchange.installed == [RECORD]
        assert pinfo.key_ring.contains_only(TOKEN_PENDING_SENTINEL)
        assert pinfo.ready is False
        # The record is in; only the minting failed, and the caller is told.
        payload = json.loads(response.body)
        assert payload["installed"] is True
        assert "not in use" in payload["error"]

    def test_a_provider_with_no_exchange_is_refused(self):
        response = _call(
            put_provider_credential, _Request(_config(key="sk-static"), dict(RECORD))
        )

        assert response.status_code == 409


class TestDeleteWithoutRevoke:
    """``revoke`` exists on the base class and raises; presence is not the
    question, so a base-only exchange must not be called at all."""

    @staticmethod
    def _base_only(path):
        from llm_rosetta.shims.auth_exchange import BaseAuthExchange

        class _BaseOnly(BaseAuthExchange):
            def needs_interactive_setup(self, ctx):
                return not path.exists()

            async def credential(self, ctx):
                raise AuthExchangeError("nothing installed")

        return _BaseOnly()

    def test_nothing_installed_is_an_idempotent_success(self, tmp_path):
        pinfo = _pinfo(self._base_only(tmp_path / "rec.json"), key="live-token")
        request = _Request(SimpleNamespace(providers={"chatgpt": pinfo}))

        response = _call(delete_provider_credential, request)
        payload = json.loads(response.body)

        assert response.status_code == 200
        assert payload == {"provider": "chatgpt", "removed": True, "revoked": False}
        assert pinfo.key_ring.contains_only(TOKEN_PENDING_SENTINEL)

    def test_a_record_it_cannot_remove_is_not_a_sign_out(self, tmp_path):
        """200 with ``removed: false`` would make "signed out" and "did
        nothing" look alike; the 501 says which one happened."""
        record = tmp_path / "rec.json"
        record.write_text("{}")
        pinfo = _pinfo(self._base_only(record), key="live-token")
        request = _Request(SimpleNamespace(providers={"chatgpt": pinfo}))

        response = _call(delete_provider_credential, request)

        assert response.status_code == 501
        assert b"cannot sign out" in response.body
        assert record.exists()


class TestDelete:
    def test_sign_out_takes_the_live_token_out_of_use(self):
        """The ring must not keep sending a token the operator just revoked."""
        exchange = _Exchange()
        exchange.install(AuthContext(provider="chatgpt"), dict(RECORD))
        # A provider that has already served requests: the ring holds its token.
        pinfo = _pinfo(exchange, key="live-access-token")
        request = _Request(SimpleNamespace(providers={"chatgpt": pinfo}))
        assert pinfo.key_ring.next() == "live-access-token"

        payload = json.loads(_call(delete_provider_credential, request).body)

        assert payload == {"provider": "chatgpt", "removed": True, "revoked": True}
        assert exchange.revoked == 1
        assert pinfo.key_ring.contains_only(TOKEN_PENDING_SENTINEL)
        assert pinfo.ready is False

    def test_a_provider_with_no_exchange_is_refused(self):
        response = _call(delete_provider_credential, _Request(_config(key="sk-static")))

        assert response.status_code == 409

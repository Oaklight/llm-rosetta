"""The credential-exchange contract, owned by the shim layer.

A shim's ``connection.token_command`` describes an exchange that fits in an
argv: the gateway runs a program and reads a string off its stdout.  That is
enough for a key that is swapped for a short-lived token (AskSage) and for a
credential a helper already keeps on disk (ALCF), but not for an OAuth
profile: an interactive first run, a rotating refresh token, a record rather
than one string, and several identities per installation.

So a shim that needs more ships ``auth.py`` next to its ``provider.yaml`` —
the same trust level and the same loading path as ``transforms.py`` and
``quota.py`` — and exports ``EXCHANGE``.  The module owns the vendor's facts
(registration, PKCE, the credential store, rotation); the gateway owns *when*
the credential is minted (startup seed, scheduled refresh, reactive retry on
401) and where it goes (the key ring, and from there the auth header).

One call, not three
~~~~~~~~~~~~~~~~~~~

There is a single ``credential()`` rather than separate ``seed`` and
``refresh`` entry points because the gateway never distinguishes them: a
credential is either usable *now* or it is not.  Splitting the two would let a
shim implement one and forget the other, which is exactly the failure the
contract exists to prevent.  Rotating a refresh token, if the vendor does,
happens inside the same call.

Sync versus async
~~~~~~~~~~~~~~~~~

``credential()`` is a coroutine: an exchange that talks to a token endpoint
uses whatever HTTP client the shim chooses (the core distribution has no
dependencies).  ``account_id()`` and ``needs_interactive_setup()`` are
synchronous and must read local state only — the gateway calls them while
building a config, outside any event loop.

This direction is deliberate: ``shims`` must never import ``gateway``, and the
loader executes every shim's ``auth.py`` while building the registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class AuthExchangeError(Exception):
    """The exchange could not produce a credential right now.

    Raised for a vendor refusal, an unreadable record, a failed refresh — the
    conditions a later attempt may get past.
    """


class InteractiveSetupRequired(AuthExchangeError):
    """No credential exists yet, and a human has to complete a sign-in.

    Distinguishable from a transient failure so the gateway can report "this
    provider needs an interactive sign-in" instead of retrying, or letting
    every request fail upstream with a 401.
    """


@dataclass(frozen=True, slots=True)
class AuthContext:
    """What the gateway tells an exchange about the provider it serves.

    Attributes:
        provider: The gateway provider name (the config key), so a shim can
            keep one record per configured provider.
        config: The provider's own config entry, read-only.  An exchange may
            read vendor-specific knobs the gateway does not model.
        state_dir: A directory the exchange may write its credential record
            to.  ``None`` when the gateway has no data directory configured;
            the exchange then picks its own default — which **must** include
            :attr:`provider`, because two providers of one shim are two
            accounts and that directory is the only thing keeping their
            records apart.
        input_key: The operator's ``api_key``, for an exchange that consumes
            it (see :attr:`AuthExchange.consumes_key`).  Empty otherwise.
    """

    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)
    state_dir: str | None = None
    input_key: str = ""


@runtime_checkable
class AuthExchange(Protocol):
    """The contract a shim's ``auth.py`` implements.

    Attributes:
        kind: Short label for the exchange form, e.g. ``"oauth"`` or
            ``"command"``.  Reported to the operator; no behavior depends on
            it.
        consumes_key: Whether :attr:`AuthContext.input_key` is an *input* to
            the exchange.  When ``True`` the operator's ``api_key`` is legal
            beside the exchange and is handed to it; when ``False`` the two
            are alternative sources for one slot and a config that sets both
            is rejected.
    """

    kind: str
    consumes_key: bool

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        """Whether a human must sign in before a credential can exist.

        Must read local state only.  Called on the config-build path.
        """
        ...

    async def credential(self, ctx: AuthContext) -> str:
        """Return a usable credential, minting or refreshing as needed.

        Returns:
            The credential to send upstream — for a bearer-token upstream, the
            access token itself.  An exchange that keeps a richer record
            (refresh token, expiry, scopes) stores it under
            :attr:`AuthContext.state_dir` and returns the access token.

        Raises:
            InteractiveSetupRequired: No record exists and a sign-in is
                needed.
            AuthExchangeError: The exchange failed for another reason.
        """
        ...

    def account_id(self, ctx: AuthContext) -> str | None:
        """Return an opaque identity of the account this provider fronts.

        Two providers with the same non-empty identity are assumed to share
        one upstream account.  Must read local state only, and must not be the
        credential itself (the gateway hashes what this returns).  ``None``
        means "no stable identity", which keeps the provider out of account
        grouping.
        """
        ...

    def install(self, ctx: AuthContext, record: Mapping[str, Any]) -> Mapping[str, Any]:
        """Adopt a credential record produced somewhere else.

        The interactive run cannot always happen on the gateway host: an
        operator signing in on their own machine has a browser and the
        gateway's loopback listener does not.  So the gateway accepts the
        finished record over its authenticated admin API and hands it to the
        exchange, which alone knows the record's shape and where it lives.

        Implementations must validate what they are given and must never
        return a token.  This is optional: an exchange that cannot be handed a
        record simply does not implement it, and the gateway refuses for it.

        Returns:
            A summary safe to render in a UI — identity, expiry, scopes — and
            never the credential.

        Raises:
            AuthExchangeError: The record is malformed or was refused.
        """
        ...

    def describe(self, ctx: AuthContext) -> Mapping[str, Any] | None:
        """Return the stored credential's token-free summary, or ``None``.

        Optional, like :meth:`install`.  What an admin panel can show without
        touching the credential: who is signed in, and until when.
        """
        ...

    def revoke(self, ctx: AuthContext) -> bool:
        """Sign out: remove the record, revoking upstream when the vendor allows.

        The other side of :meth:`login`.  A vendor that issues revocable
        tokens should call its revocation endpoint before dropping the record;
        one that does not simply forgets it.  What must not happen is a record
        removed locally while the grant stays live upstream, which is why this
        returns whether the vendor side was actually revoked — a caller can say
        "signed out" or "forgotten locally" and mean it.

        Implementations must remove the record **before** attempting the
        network call, and under whatever lock protects it from a concurrent
        refresh: a refresh that read the old record just before a delete would
        otherwise write it back afterwards, leaving a signed-out provider
        holding a live grant.  Removal must happen even when the network call
        fails — the operator asked for it gone.

        ``needs_interactive_setup()`` is how the gateway decides whether a
        sign-out removed anything (`removed` in the response).  An exchange
        whose record bookkeeping it does not reflect should override it
        alongside this member; otherwise the endpoint under-reports a
        successful sign-out as "nothing removed".

        Optional, like :meth:`install`: an exchange that keeps no record of its
        own does not implement it.

        Returns:
            ``True`` when the grant was revoked upstream, ``False`` when only
            local state was removed.

        Raises:
            AuthExchangeError: The record could not be removed at all.
        """
        ...

    def login(self, ctx: AuthContext) -> Mapping[str, Any]:
        """Run the interactive sign-in and return the finished record.

        The other half of :meth:`install`: a record has to be *produced*
        somewhere a browser exists, and that is not the gateway host.  This
        call is blocking and needs an operator watching — a browser opens and
        a loopback listener waits — so the gateway never calls it; the CLI
        does, on the operator's machine, and what it returns is what the admin
        API accepts.

        Optional, like :meth:`install`: an exchange with no browser flow does
        not implement it.

        Returns:
            The record, token-free only if the vendor's flow allows it — it
            *is* the credential.  The caller hands it straight to
            :meth:`install` and must not log it.

        Raises:
            AuthExchangeError: The sign-in failed, was refused, or timed out.
                The message is shown to the operator, so it must never carry
                the record — a vendor's error body is not under our control,
                and a message is a much easier thing to leak than a file.
        """
        ...


class BaseAuthExchange:
    """Defaults for the optional half of :class:`AuthExchange`.

    A shim's ``EXCHANGE`` may implement the protocol directly; subclassing
    this is only a convenience for an exchange that has nothing to say about
    interactive setup or account identity.
    """

    kind = "exchange"
    consumes_key = False

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        """Return ``False``; override when sign-in is required."""
        return False

    def account_id(self, ctx: AuthContext) -> str | None:
        """Return ``None``; override for a provider with an account identity."""
        return None

    def describe(self, ctx: AuthContext) -> Mapping[str, Any] | None:
        """Return ``None``; override to show who is signed in."""
        return None

    def install(
        self, ctx: AuthContext, record: Mapping[str, Any]
    ) -> Mapping[str, Any]:  # pragma: no cover
        """Refuse: this exchange cannot adopt a record from elsewhere."""
        raise NotImplementedError

    def login(self, ctx: AuthContext) -> Mapping[str, Any]:  # pragma: no cover
        """Refuse: this exchange has no browser sign-in."""
        raise NotImplementedError

    def revoke(self, ctx: AuthContext) -> bool:  # pragma: no cover
        """Refuse: this exchange keeps no record to revoke."""
        raise NotImplementedError

    def can_install(self) -> bool:
        """Whether :meth:`install` was overridden.

        ``install`` exists on this class and raises, so "has the attribute" is
        the wrong question; this is the one to ask.
        """
        return type(self).install is not BaseAuthExchange.install

    def can_login(self) -> bool:
        """Whether :meth:`login` was overridden."""
        return type(self).login is not BaseAuthExchange.login

    def can_revoke(self) -> bool:
        """Whether :meth:`revoke` was overridden."""
        return type(self).revoke is not BaseAuthExchange.revoke

    async def credential(self, ctx: AuthContext) -> str:  # pragma: no cover
        """Mint a credential.  Every exchange must implement this."""
        raise NotImplementedError

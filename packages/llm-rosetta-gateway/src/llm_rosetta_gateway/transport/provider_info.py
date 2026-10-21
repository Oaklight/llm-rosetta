"""Provider runtime configuration — connection info, auth, and key rotation.

This module contains the data classes that describe *how* to talk to an
upstream provider at the transport level:

* :class:`KeyRing` — round-robin API key selector.
* :class:`ProviderInfo` — base URL, auth headers, URL templates.
* Auth header builder functions (``openai_auth``, ``anthropic_auth``,
  ``google_auth``).

Higher-level factory logic (shim resolution, config parsing) stays in
``gateway.providers``.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

# Type alias for auth-header builder callables
AuthHeaderFn = Callable[[str], dict[str, str]]

# Sentinel API key used when token_command will be resolved asynchronously
# after the event loop starts.  Deliberately not a plausible key format so
# upstream providers reject it immediately if it leaks past the readiness
# guard.
TOKEN_PENDING_SENTINEL = "__token_pending__"

# The credential a request actually went out with.  ``auth_headers()`` is
# synchronous and is called once per upstream request from the task that will
# also see the response, so a context variable carries the key from the header
# build to the in-band quota handler without leaking across concurrent
# requests.  ``None`` for a keyless request.
_ACTIVE_CREDENTIAL: ContextVar[str | None] = ContextVar(
    "rosetta_active_credential", default=None
)

# Characters used by OpenCode's identifier encoding (base62).
_BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Per-upstream-request identifier sequence.  Process-scoped so the minted id
# differs between requests even though ProviderInfo copies are per request.
_REQUEST_SEQ = itertools.count(1)


# ---------------------------------------------------------------------------
# API key rotation (round-robin)
# ---------------------------------------------------------------------------


class KeyRing:
    """Round-robin API key selector.

    Accepts a single key string **or** a comma-separated list of keys.
    Each call to :meth:`next` returns the next key in rotation.
    """

    def __init__(
        self,
        keys_csv: str,
        *,
        usable: Callable[[str], bool] | None = None,
    ) -> None:
        self._keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        self._idx = 0
        self._usable = usable

    def set_usable(self, usable: Callable[[str], bool] | None) -> None:
        """Install (or clear) the predicate that says a key may be used.

        The predicate is consulted on every selection, so quota demotions
        take effect without rebuilding the ring.
        """
        self._usable = usable

    def keys(self) -> tuple[str, ...]:
        """Return the configured keys in order (never logged or returned)."""
        return tuple(self._keys)

    def _is_usable(self, key: str) -> bool:
        return self._usable is None or self._usable(key)

    def refresh(self, keys_csv: str) -> int | None:
        """Replace all keys atomically via reference swap.

        No lock needed: both ``next()`` and ``refresh()`` run on the
        same asyncio event-loop thread.  CPython's GIL guarantees that
        the list reference assignment (``self._keys = new``) is atomic
        for concurrent readers.  A ``threading.Lock`` would be wrong
        here — ``next()`` is synchronous (called from ``auth_headers``)
        and cannot ``await`` an ``asyncio.Lock``, while a threading lock
        risks blocking the event loop.

        Returns the previous key count if it changed (so the caller can
        warn about affinity disruption), or ``None`` if unchanged.
        """
        new = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if not new:
            return None
        old_count = len(self._keys)
        self._keys = new
        self._idx = self._idx % len(new)
        return old_count if old_count != len(new) else None

    def next(self) -> str:
        """Return the next usable API key.

        A key the configured predicate rejects (for example one a quota probe
        demoted until its window resets) is skipped exactly like an absent
        key.  If every key is rejected the ring falls back to plain
        round-robin: a demotion is advisory, and refusing here would take the
        whole provider out of service on a stale reading.
        """
        if not self._keys:
            raise ValueError("No API keys configured")
        n = len(self._keys)
        for offset in range(n):
            idx = (self._idx + offset) % n
            key = self._keys[idx]
            if self._is_usable(key):
                self._idx = (idx + 1) % n
                return key
        key = self._keys[self._idx]
        self._idx = (self._idx + 1) % n
        return key

    def select(self, identity: str) -> str:
        """Return the key with the highest rendezvous hash score.

        Uses rendezvous hashing (highest random weight) so that
        adding or removing a key only disrupts ~1/n of existing
        mappings, preserving prompt-cache locality across key
        refreshes.

        Unlike :meth:`next`, this does not advance the round-robin
        counter.
        """
        if not self._keys:
            raise ValueError("No API keys configured")
        candidates = [k for k in self._keys if self._is_usable(k)] or self._keys
        if len(candidates) == 1:
            return candidates[0]
        best_key = candidates[0]
        best_score = hashlib.sha256(f"{identity}\0{candidates[0]}".encode()).digest()
        for key in candidates[1:]:
            score = hashlib.sha256(f"{identity}\0{key}".encode()).digest()
            if score > best_score:  # bytes comparison is lexicographic
                best_score = score
                best_key = key
        return best_key

    def contains_only(self, value: str) -> bool:
        """True when the ring holds exactly one key equal to *value*."""
        return len(self._keys) == 1 and self._keys[0] == value

    def __len__(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------


# Version-prefix segments that may appear at the end of base_url AND
# at the start of a url_template path, causing duplication.
_VERSION_SUFFIXES = re.compile(r"/v\d+(?:beta\d?)?$", re.IGNORECASE)


def normalize_base_url(base_url: str, url_template: str) -> str:
    """Strip trailing version prefix from *base_url* when it would
    duplicate the start of *url_template*.

    Example: ``base_url="https://api.openai.com/v1"`` with
    ``url_template="{base_url}/v1/embeddings"`` → strips ``/v1`` from
    base_url to avoid ``/v1/v1/embeddings``.

    Templates like ``"{base_url}/chat/completions"`` (no version prefix
    after ``{base_url}``) are left alone — the ``/v1`` in base_url is
    needed.
    """
    m = _VERSION_SUFFIXES.search(base_url)
    if not m:
        return base_url
    suffix = m.group()  # e.g. "/v1", "/v1beta", "/v3"
    # Check if the template path (after {base_url}) starts with the same segment
    tpl_path = (
        url_template.split("{base_url}", 1)[-1] if "{base_url}" in url_template else ""
    )
    if tpl_path.startswith(suffix + "/") or tpl_path == suffix:
        return base_url[: m.start()]
    return base_url


class ProviderInfo:
    """Runtime representation of a single configured provider.

    Encapsulates base_url, key rotation, auth-header construction,
    and upstream URL building.
    """

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        auth_header_fn: AuthHeaderFn,
        url_template: str,
        stream_url_template: str | None = None,
        proxy_url: str | None = None,
        timeout: float | None = None,
        token_command: list[str] | None = None,
        token_refresh_interval: int = 3600,
        keyless: bool = False,
        session_header: str | None = None,
        credential_token: str | None = None,
        extra_headers: dict[str, str] | None = None,
        request_header: str | None = None,
        session_id_format: str | None = None,
        request_id_format: str | None = None,
        stream_required: bool = False,
        quota_probe_id: str | None = None,
        quota_url: str | None = None,
        quota_probes: tuple[Any, ...] = (),
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Provider '{name}': base_url must start with http:// or https://, "
                f"got '{base_url}'"
            )
        self.name = name
        self.base_url = normalize_base_url(base_url.rstrip("/"), url_template)
        self.key_ring = KeyRing(api_key)
        self._auth_header_fn = auth_header_fn
        self._url_template = url_template
        self._stream_url_template = stream_url_template
        self.proxy_url = proxy_url
        self.timeout = timeout
        self._affinity_identity: str | None = None
        self.token_command = token_command
        self.token_refresh_interval = token_refresh_interval
        self.token_status: dict | None = None
        self.keyless = keyless
        # Per-conversation session header (name declared by the shim, value
        # derived per request in gateway.proxy).  ``session_id`` stays None
        # until the request pipeline sets it; both must be present to emit.
        self.session_header = session_header
        self.session_id: str | None = None
        # Opaque digest of the credential source, or ``None`` when the provider
        # has no stable one.  Compared by :meth:`account_token` to detect two
        # providers that front the same upstream account.  Never the raw key.
        self.credential_token = credential_token

        # Static headers a shim adds to every upstream request (a client
        # fingerprint, for instance) and the per-request identifier header.
        self.extra_headers = dict(extra_headers or {})
        self.request_header = request_header
        # ``"opencode"`` mints OpenCode's canonical identifiers; ``None`` keeps
        # the plain sha256 session id the session stage derives.
        self.session_id_format = session_id_format
        self.request_id_format = request_id_format
        # A shim whose upstream only answers SSE: the non-streaming handler
        # forces streaming and reassembles the reply.
        self.stream_required = stream_required

        # Quota probe declaration and the shim's implementations.  The
        # declaration names one probe; ``bind_provider_quota`` resolves it to
        # ``quota_probe_impl`` / ``quota_usage_url`` (or records why not).
        self.quota_probe_id = quota_probe_id
        self.quota_url = quota_url
        self.quota_probes = tuple(quota_probes)
        self.quota_probe: str | None = None
        self.quota_probe_impl: Any | None = None
        self.quota_usage_url: str | None = None
        self.quota_unsupported: str | None = None
        self.quota_identity = credential_token
        # Set by ``gateway.quota.service.bind_provider_quota``: says whether
        # the provider (not one key) is cooling down.
        self.quota_provider_blocked: Callable[[], bool] | None = None

    # -- readiness ----------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether this provider can accept traffic.

        Returns ``False`` when the key ring contains only the deferred
        token-command sentinel, meaning the initial token fetch has not
        yet completed, and when a keyless upstream is in a quota cooldown.
        """
        if self.keyless:
            return not (
                self.quota_provider_blocked is not None
                and self.quota_provider_blocked()
            )
        if self.key_ring.contains_only(TOKEN_PENDING_SENTINEL):
            return False
        return True

    # -- public helpers used by the proxy -----------------------------------

    def account_token(self, provider_name: str) -> str:
        """Return an opaque identity for this provider's upstream account.

        Two providers whose tokens are equal front the same account.  A
        provider with no stable credential (keyless, or an unresolved key)
        gets a unique token keyed on *provider_name*, so it never groups.

        Args:
            provider_name: The gateway provider's config key.  Used only for
                the ungroupable fallback; ``self.name`` holds the base type.

        Returns:
            The credential digest, or ``@{provider_name}`` when there is none.
        """
        if not self.credential_token:
            return f"@{provider_name}"
        return self.credential_token

    def auth_headers(self) -> dict[str, str]:
        """Return auth headers using the next rotated or affinity-selected key.

        A keyless provider builds its headers without the key ring, so it
        never raises "No API keys configured".
        """
        if self.keyless:
            _ACTIVE_CREDENTIAL.set(None)
            return self._auth_header_fn("")
        if self._affinity_identity is not None:
            key = self.key_ring.select(self._affinity_identity)
        else:
            key = self.key_ring.next()
        _ACTIVE_CREDENTIAL.set(key)
        return self._auth_header_fn(key)

    def auth_headers_for(self, key: str) -> dict[str, str]:
        """Build auth headers for one specific key.

        Used by the quota probe, which must read every key rather than the
        one the next request would rotate to.  Does not touch the
        per-request active-credential record.
        """
        if self.keyless:
            return self._auth_header_fn("")
        return self._auth_header_fn(key)

    def active_key(self) -> str | None:
        """Return the credential this request went out with, if any."""
        return _ACTIVE_CREDENTIAL.get()

    def request_headers(self) -> dict[str, str]:
        """Return the per-request identifier header, if the shim declared one.

        The value is minted per upstream request, not per conversation: an
        upstream that validates it expects a fresh one each time.
        """
        if not self.request_header:
            return {}
        seq = next(_REQUEST_SEQ)
        seed = f"{self.session_id or ''}:{seq}"
        if self.request_id_format == "opencode":
            value = opencode_identifier("msg", seed, counter=seq)
        else:
            value = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
        return {self.request_header: value}

    def session_headers(self) -> dict[str, str]:
        """Return the per-conversation session header, if configured.

        Empty until both the shim-declared header name and a derived
        ``session_id`` are present, so a provider that needs no session header
        (or a request that never reached derivation) contributes nothing.
        """
        if self.session_header and self.session_id:
            return {self.session_header: self.session_id}
        return {}

    def upstream_url(self, model: str, *, stream: bool = False) -> str:
        """Build the upstream URL for the given model."""
        tpl = (
            self._stream_url_template
            if (stream and self._stream_url_template)
            else self._url_template
        )
        return tpl.format(base_url=self.base_url, model=model)

    def with_url_templates(
        self,
        url_template: str | None = None,
        stream_url_template: str | None = None,
    ) -> ProviderInfo:
        """Return a shallow copy with overridden URL template(s).

        The new instance shares the same :class:`KeyRing` (round-robin
        state is preserved).  Fields not overridden keep their current
        values.
        """
        import copy

        clone = copy.copy(self)
        if url_template is not None:
            clone._url_template = url_template
        if stream_url_template is not None:
            clone._stream_url_template = stream_url_template
        return clone

    def with_timeout(self, timeout: float | None) -> ProviderInfo:
        """Return a shallow copy with an overridden timeout.

        The new instance shares the same :class:`KeyRing`.
        Returns ``self`` unchanged if *timeout* is ``None``.
        """
        if timeout is None:
            return self
        import copy

        clone = copy.copy(self)
        clone.timeout = timeout
        return clone

    def with_affinity(self, identity: str | None) -> ProviderInfo:
        """Return a shallow copy with affinity-based key selection.

        When *identity* is set, :meth:`auth_headers` uses
        :meth:`KeyRing.select` (rendezvous hashing) instead of
        round-robin :meth:`KeyRing.next`.
        The new instance shares the same :class:`KeyRing`.
        Returns ``self`` unchanged if *identity* is ``None`` or empty.
        """
        if not identity:
            return self
        import copy

        clone = copy.copy(self)
        clone._affinity_identity = identity
        return clone

    def with_session(self, session_id: str | None) -> ProviderInfo:
        """Return a shallow copy with a derived per-conversation session id.

        The new instance shares the same :class:`KeyRing` and affinity state.
        Returns ``self`` unchanged when *session_id* is empty or the provider
        declares no session header, so callers can apply it unconditionally.
        """
        if not session_id or not self.session_header:
            return self
        import copy

        clone = copy.copy(self)
        clone.session_id = session_id
        return clone


# ---------------------------------------------------------------------------
# Per-provider auth header builders
# ---------------------------------------------------------------------------


def opencode_identifier(prefix: str, seed: str, *, counter: int = 1) -> str:
    """Mint an OpenCode-shaped identifier (``ses_``/``msg_``).

    OpenCode's free lane validates the identifier's *value*: it decodes the
    first six bytes back into a timestamp and refuses one outside a recent
    window.  The layout is ``~((unix_ms << 12) | counter)`` truncated to 48
    bits, hex-encoded, followed by 14 base62 characters.  The timestamp is
    anchored to the current UTC day minus a seed-derived offset, so the value
    is stable for every request of one conversation within a day and stays
    inside the window as time passes.

    Args:
        prefix: ``"ses"`` or ``"msg"``.
        seed: Conversation seed; only its digest is used.
        counter: 12-bit counter mixed into the timestamp half.

    Returns:
        The identifier, e.g. ``ses_02f1c3d4e5ffeeAb12Cd34Ef56Gh``.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    day_start_ms = (int(_time_now() // 86400) - 1 - (digest[0] % 25)) * 86400 * 1000
    payload = ~(((day_start_ms // 1000 * 1000) << 12) | (counter & 0xFFF))
    payload &= (1 << 48) - 1
    head = payload.to_bytes(6, "big").hex()
    suffix = "".join(_BASE62[b % 62] for b in digest[:14])
    return f"{prefix}_{head}{suffix}"


def _time_now() -> float:
    """Return the current epoch seconds (indirection keeps the mint testable)."""
    import time

    return time.time()


def no_auth(_api_key: str) -> dict[str, str]:
    """Auth header builder for keyless upstreams: sends no credential."""
    return {}


def openai_auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def anthropic_auth(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def google_auth(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key}

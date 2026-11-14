"""Per-key limits: a model allowlist and a request quota (#1502 §13).

A key used to be all-or-nothing: any key reached every model.  Both limits are
properties of the key, so they live on the key record and apply with or
without the deployment's global rate limiting.  The quota is written the way
the settings fields write theirs — ``60/m`` or ``30/m, 500/h`` — so the two
limits are read and validated at one place.

The cases here are the ones an implementation can quietly get wrong:
a default that silently restricts, an allowlist that survives a rotation, a
window that is not rebuilt when the key's quota changes, a hook that only
enforces the cap when global limiting happens to be on, and a spec that is
stored without being parsed.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta_gateway.app.routes_llm import model_allowed
from llm_rosetta_gateway.keystore import KeyContext, KeyStore


class _Ctx:
    """The fields the allowlist reads."""

    def __init__(self, models: set[str] | None) -> None:
        self.allowed_models = frozenset(models) if models is not None else frozenset()


@pytest.mark.asyncio
async def test_a_new_key_is_unrestricted(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    _, raw = await store.create_key(label="plain")
    found = store.validate(raw)
    assert found is not None
    _, ctx = found
    assert ctx.allowed_models == frozenset({"*"})
    assert ctx.rate_limit == ""


@pytest.mark.asyncio
async def test_the_limits_round_trip_through_the_store(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    _, raw = await store.create_key(
        label="limited",
        allowed_models=["gpt-4o", "claude-3-5-sonnet"],
        rate_limit="60/m",
    )
    found = store.validate(raw)
    assert found is not None
    key_id, ctx = found
    assert ctx.allowed_models == frozenset({"gpt-4o", "claude-3-5-sonnet"})
    assert ctx.rate_limit == "60/m"

    listed = {k["id"]: k for k in await store.list_keys()}[key_id]
    assert listed["allowed_models"] == ["claude-3-5-sonnet", "gpt-4o"]
    assert listed["rate_limit"] == "60/m"

    # An update re-reads the cache, so the next request sees the new values.
    assert await store.update(
        key_id, allowed_models=["gpt-4o"], rate_limit="5/m, 100/h"
    )
    found = store.validate(raw)
    assert found is not None
    _, ctx = found
    assert ctx.allowed_models == frozenset({"gpt-4o"})
    assert ctx.rate_limit == "5/m, 100/h"


@pytest.mark.asyncio
async def test_a_blank_quota_clears_the_limit(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    key_id, raw = await store.create_key(rate_limit="60/m")
    found = store.validate(raw)
    assert found is not None and found[1].rate_limit == "60/m"
    await store.update(key_id, rate_limit="")
    found = store.validate(raw)
    assert found is not None and found[1].rate_limit == ""


@pytest.mark.asyncio
async def test_an_unreadable_stored_quota_means_no_limit(tmp_path: Any) -> None:
    """The column once held a bare integer; a leftover must not raise on a
    request, it must simply not restrict."""
    store = await KeyStore.create(tmp_path / "keys.db")
    key_id, raw = await store.create_key(label="legacy", rate_limit="60/m")
    await store._conn.execute(
        "UPDATE api_keys SET rate_limit = ? WHERE id = ?", (2, key_id)
    )
    await store._conn.commit()
    await store._refresh_cache()
    found = store.validate(raw)
    assert found is not None and found[1].rate_limit == ""
    listed = {k["id"]: k for k in await store.list_keys()}[key_id]
    assert listed["rate_limit"] == ""


def test_the_allowlist_predicate() -> None:
    assert model_allowed(None, "anything")
    assert model_allowed(_Ctx({"*"}), "anything")
    assert model_allowed(_Ctx({"gpt-4o"}), "gpt-4o")
    assert not model_allowed(_Ctx({"gpt-4o"}), "gpt-4o-mini")
    # Empty is the state a damaged record reads as: it allows nothing, because
    # widening an authorization control on unreadable data fails the wrong way.
    assert not model_allowed(_Ctx(None), "anything")


@pytest.mark.asyncio
async def test_a_damaged_allowlist_denies_rather_than_widens(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    scoped, raw_scoped = await store.create_key(label="scoped", allowed_models=["m-a"])
    open_key, raw_open = await store.create_key(label="open")
    await store._conn.execute(
        "UPDATE api_keys SET allowed_models = ? WHERE id = ?", ("not json", scoped)
    )
    await store._conn.execute(
        "UPDATE api_keys SET allowed_models = ? WHERE id = ?", ("[]", open_key)
    )
    await store._conn.commit()
    await store._refresh_cache()

    scoped_ctx = store.validate(raw_scoped)
    open_ctx = store.validate(raw_open)
    assert scoped_ctx is not None and open_ctx is not None
    assert scoped_ctx[1].allowed_models == frozenset()
    assert open_ctx[1].allowed_models == frozenset()
    assert not model_allowed(scoped_ctx[1], "m-a")


def test_the_allowlist_is_a_set_on_a_frozen_context() -> None:
    ctx = KeyContext(label="x", allowed_shims=frozenset({"*"}))
    assert ctx.allowed_models == frozenset({"*"})


class _Req:
    def __init__(self, path: str = "/v1/chat/completions", body: Any = None) -> None:
        self.path = path
        self.method = "POST"
        self.headers: dict[str, str] = {}
        self._body = body or {}

    def json(self) -> Any:
        return self._body


@pytest.mark.asyncio
async def test_the_cap_refuses_the_request_over_it() -> None:
    from llm_rosetta_gateway.middleware.auth import api_key_context_var
    from llm_rosetta_gateway.middleware.ratelimit import (
        RateLimitState,
        create_rate_limit_hook,
    )

    state = RateLimitState()
    state.rebuild(cast(Any, _NoConfig()))
    assert not state.enabled, "the deployment's own limiting is off in this case"

    ctx = KeyContext(
        label="tight",
        allowed_shims=frozenset({"*"}),
        rate_limit="2/m",
        key_hash="k" * 64,
    )
    token = api_key_context_var.set(ctx)
    try:
        hook = create_rate_limit_hook(state)
        assert await hook(_Req()) is None
        assert await hook(_Req()) is None
        denied = await hook(_Req())
    finally:
        api_key_context_var.reset(token)

    assert denied is not None
    assert denied.status_code == 429
    assert denied.headers["Retry-After"].isdigit()


@pytest.mark.asyncio
async def test_a_changed_cap_starts_a_fresh_window() -> None:
    from llm_rosetta_gateway.middleware.auth import api_key_context_var
    from llm_rosetta_gateway.middleware.ratelimit import (
        RateLimitState,
        create_rate_limit_hook,
    )

    state = RateLimitState()
    state.rebuild(cast(Any, _NoConfig()))
    hook = create_rate_limit_hook(state)

    tight = KeyContext(
        label="t", allowed_shims=frozenset({"*"}), rate_limit="1/m", key_hash="a" * 64
    )
    token = api_key_context_var.set(tight)
    try:
        assert await hook(_Req()) is None
        assert await hook(_Req()) is not None
    finally:
        api_key_context_var.reset(token)

    loose = KeyContext(
        label="t", allowed_shims=frozenset({"*"}), rate_limit="5/m", key_hash="a" * 64
    )
    token = api_key_context_var.set(loose)
    try:
        # A new limit is a new window, not the old one's spent quota.
        assert await hook(_Req()) is None
    finally:
        api_key_context_var.reset(token)


def test_idle_key_windows_are_pruned(monkeypatch: Any) -> None:
    """A key that stops calling must not hold a limiter for the process life."""
    import time as time_module

    from llm_rosetta_gateway.middleware.ratelimit import RateLimitState

    state = RateLimitState()
    state.rebuild(cast(Any, _NoConfig()))
    first = KeyContext(
        label="a", allowed_shims=frozenset({"*"}), rate_limit="1/m", key_hash="a" * 64
    )
    state.key_limiter(first)
    assert len(state._key_limits) == 1

    later = time_module.monotonic() + RateLimitState.KEY_LIMIT_IDLE_S + 1
    monkeypatch.setattr(
        "llm_rosetta_gateway.middleware.ratelimit.time.monotonic", lambda: later
    )
    state.rebuild(cast(Any, _NoConfig()))
    second = KeyContext(
        label="b", allowed_shims=frozenset({"*"}), rate_limit="1/m", key_hash="b" * 64
    )
    state.key_limiter(second)

    assert list(state._key_limits) == [("b" * 64, "1/m")]


@pytest.mark.asyncio
async def test_every_window_of_a_quota_is_enforced() -> None:
    """A spec may name more than one window; the loosest minute does not
    excuse the hour."""
    from llm_rosetta_gateway.middleware.auth import api_key_context_var
    from llm_rosetta_gateway.middleware.ratelimit import (
        RateLimitState,
        create_rate_limit_hook,
    )

    state = RateLimitState()
    state.rebuild(cast(Any, _NoConfig()))
    hook = create_rate_limit_hook(state)

    ctx = KeyContext(
        label="two-windows",
        allowed_shims=frozenset({"*"}),
        rate_limit="5/m, 2/h",
        key_hash="b" * 64,
    )
    token = api_key_context_var.set(ctx)
    try:
        first, second = await hook(_Req()), await hook(_Req())
        third = await hook(_Req())
    finally:
        api_key_context_var.reset(token)

    assert first is None and second is None
    assert third is not None and third.status_code == 429


def test_the_admin_refuses_a_malformed_limit_before_storing_it() -> None:
    from llm_rosetta_gateway.admin.routes.keys import _key_models, _key_quota

    assert _key_models(None) == (None, None)
    assert _key_models("gpt-4o, gpt-4o claude-3-5-sonnet") == (
        ["gpt-4o", "claude-3-5-sonnet"],
        None,
    )
    models, err = _key_models(["gpt-4o", "bad name"])
    assert models is None and "bad name" in (err or "")
    # The panel sends ["*"] back for "every model", so the validator accepts it.
    assert _key_models(["*"]) == (["*"], None)
    assert _key_models(["*", "gpt-4o"]) == (["*"], None)

    assert _key_quota("") == ("", None)
    assert _key_quota("30/m, 500/h") == ("30/m, 500/h", None)
    spec, err = _key_quota("60x")
    assert spec == "60x" and "60x" in (err or "")
    # Zero is not "no limit" — the limiter refuses to be built with it — and
    # the settings form says blank for none, so it is refused with the reason.
    for zero in ("0/m", "5/m, 0/h"):
        spec, err = _key_quota(zero)
        assert spec == zero and "at least one request" in (err or "")


class _NoConfig:
    """The subset of GatewayConfig the limiter reads, with no limits set."""

    rate_limit_algorithm = "fixed"
    rate_limit_global = None
    rate_limit_per_ip = None
    rate_limit_per_key = None
    rate_limit_per_model = None
    rate_limit_exclude = ["/health", "/admin"]
    rate_limit_trust_proxy = False
    rate_limit_enabled = False

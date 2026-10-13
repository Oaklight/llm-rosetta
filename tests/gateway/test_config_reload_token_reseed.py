"""A hot config reload must re-seed ``token_command`` providers (#976).

The deferred token seeder runs once, at process start.  A reload swaps in a
fresh :class:`GatewayConfig` whose ``token_command`` providers hold the pending
sentinel, so without an explicit re-seed they never obtain a token: routing
skips them, and the admin model-listing button — which does not go through the
readiness gate — sends the sentinel as a bearer and reports the upstream 401 as
"this provider may not support model listing".
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

from llm_rosetta.gateway.admin.routes._shared import _reload_gateway_config
from llm_rosetta.gateway.admin.routes.config import fetch_upstream_models
from llm_rosetta.gateway.admin.routes.connectivity import (
    test_provider_connectivity as _provider_connectivity,
)
from llm_rosetta.gateway.deferred_startup import DeferredStartup, ProviderInitState
from llm_rosetta.gateway.shim_sources import shims_signature
from llm_rosetta.gateway.transport.provider_info import (
    TOKEN_PENDING_SENTINEL,
    ProviderInfo,
    openai_auth,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _pinfo(
    name: str = "alcf",
    *,
    api_key: str = TOKEN_PENDING_SENTINEL,
    token_command: list[str] | None = None,
) -> ProviderInfo:
    return ProviderInfo(
        name=name,
        api_key=api_key,
        base_url="https://example.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        token_command=token_command if token_command is not None else ["echo", "x"],
        token_refresh_interval=3600,
    )


def _plain_pinfo(name: str = "openai") -> ProviderInfo:
    """A provider with a static key and no ``token_command``."""
    return ProviderInfo(
        name=name,
        api_key="sk-real",
        base_url="https://example.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
        token_command=None,
        token_refresh_interval=3600,
    )


class _Config:
    """Minimal GatewayConfig stub."""

    def __init__(
        self,
        providers: dict[str, ProviderInfo],
        raw_providers: dict[str, Any] | None = None,
    ) -> None:
        self.providers = providers
        self.provider_types = {n: "chat_completions" for n in providers}
        self._raw_providers: dict[str, Any] = raw_providers or {}
        self.models: dict[str, Any] = {}
        self._deferred_startup: Any = None


class _App:
    def __init__(self) -> None:
        self._token_refresh_tasks: list = []
        self.config_path: str | None = None
        self.config_io: Any = None
        self.shim_source_signature: str | None = None
        self.gateway_config: Any = None
        self.deferred_startup: Any = None
        self._prev_gateway_config: Any = None


class _Request:
    def __init__(self, app: Any, name: str | None = None) -> None:
        self.app = app
        self.path_params = {"name": name} if name is not None else {}


# ---------------------------------------------------------------------------
# DeferredStartup.apply_config_reload
# ---------------------------------------------------------------------------


class TestApplyConfigReload:
    def test_reseeds_token_command_provider(self):
        """The reloaded provider must leave the pending sentinel behind."""
        old = _Config({"alcf": _pinfo("alcf")})
        app = _App()
        ds = DeferredStartup(old, app)

        reloaded = _pinfo("alcf")
        new_config = _Config({"alcf": reloaded})

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    new_callable=AsyncMock,
                    return_value="fresh-token",
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ):
                await ds.apply_config_reload(new_config)
                await asyncio.gather(*ds._tasks)

        asyncio.run(_run())

        assert reloaded.ready is True
        assert reloaded.key_ring.next() == "fresh-token"
        assert ds.provider_state("alcf") == ProviderInitState.READY

    def test_reattaches_deferred_startup_to_new_config(self):
        """``resolve()`` reads states off the config it routes through."""
        ds = DeferredStartup(_Config({}), _App())
        new_config = _Config({})

        asyncio.run(ds.apply_config_reload(new_config))

        assert new_config._deferred_startup is ds

    def test_cancels_superseded_refresh_tasks(self):
        """Tasks bound to the replaced ProviderInfo must not keep running."""
        ds = DeferredStartup(_Config({}), _App())
        app = ds._app

        async def _run():
            async def _never() -> None:
                await asyncio.sleep(3600)

            stale = asyncio.create_task(_never())
            app._token_refresh_tasks = [stale]

            # Ask whether the reload returns within a window, rather than
            # asserting on the task afterwards.  wait_for would cancel the
            # awaited task itself on timeout, and the reload swallows its own
            # CancelledError, so a removed cancel loop would cancel `stale`
            # through the back door and pass.  asyncio.wait leaves the pending
            # work alone, so a missing cancel blocks the reload and fails here.
            reload_task = asyncio.create_task(ds.apply_config_reload(_Config({})))
            done, _ = await asyncio.wait({reload_task}, timeout=2)
            assert done, "reload blocked on a refresher it never cancelled"
            await reload_task

            assert stale.cancelled()
            assert app._token_refresh_tasks == []

        asyncio.run(_run())

    def test_no_token_command_providers_is_a_noop(self):
        ds = DeferredStartup(_Config({}), _App())
        new_config = _Config({"openai": _plain_pinfo("openai")})

        async def _run():
            await ds.apply_config_reload(new_config)

        asyncio.run(_run())

        assert ds._provider_states == {}
        assert "token_seed" not in ds._pending_tasks

    def test_reload_supersedes_an_in_flight_seeding(self):
        """A pass started before the reload must not finish against the new
        config, nor register a second set of refresh loops."""
        ds = DeferredStartup(_Config({"alcf": _pinfo("alcf")}), _App())
        # Long enough to park the failing first pass in its retry sleep, which
        # is the window a reload can land in.
        ds._RETRY_DELAY = 60

        calls = {"n": 0}
        refresher_calls: list[set[str]] = []

        async def _run_token(argv: list[str]) -> str:  # noqa: ARG001
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first pass fails")
            return "fresh-token"

        def _fake_refreshers(providers: dict) -> list:
            refresher_calls.append(set(providers))

            async def _noop() -> None:
                await asyncio.sleep(0)

            return [asyncio.create_task(_noop())]

        reloaded = _pinfo("alcf")
        new_config = _Config({"alcf": reloaded})

        async def _run():
            with (
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.run_token_command",
                    side_effect=_run_token,
                ),
                patch(
                    "llm_rosetta.gateway.transport.token_refresh.start_token_refreshers",
                    new=AsyncMock(side_effect=_fake_refreshers),
                ),
            ):
                await ds.start()
                await asyncio.sleep(0.05)
                first = ds._seed_task
                assert first is not None and not first.done()

                # Same discriminator as the refresher case: the reload must
                # come back on its own, not because something else cancelled
                # the pass it is waiting on.
                reload_task = asyncio.create_task(ds.apply_config_reload(new_config))
                done, _ = await asyncio.wait({reload_task}, timeout=2)
                assert done, "reload blocked on the seeding pass it never cancelled"
                await reload_task
                await asyncio.sleep(0.05)

                assert first.cancelled()
                # Exactly one registration: the reload's pass.  A stale pass
                # surviving the swap would add a second loop over the same
                # provider, minting tokens on the same interval.
                assert refresher_calls == [{"alcf"}]
                assert len(ds._app._token_refresh_tasks) == 1
                assert reloaded.ready is True

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Reload wiring
# ---------------------------------------------------------------------------


class _StaticConfigIO:
    def __init__(self, raw: dict) -> None:
        self._raw = raw

    def load(self, path: str) -> dict:  # noqa: ARG002 - signature parity
        return self._raw


class _RecordingDeferred:
    def __init__(self) -> None:
        self.reloaded_with: list[Any] = []

    async def apply_config_reload(self, config: Any) -> None:
        self.reloaded_with.append(config)


def test_reload_config_reseeds_providers():
    """``_reload_gateway_config`` must hand the new config to the deferred seeder."""
    import llm_rosetta.gateway.app as app_mod

    raw: dict[str, Any] = {"providers": {}, "models": {}}
    app = _App()
    app.config_path = "/tmp/does-not-exist.jsonc"
    app.config_io = _StaticConfigIO(raw)
    app.shim_source_signature = shims_signature(raw)
    app.gateway_config = None
    app.deferred_startup = _RecordingDeferred()

    original = getattr(app_mod, "_config", None)
    try:
        new_config = asyncio.run(
            _reload_gateway_config(_Request(app), "/tmp/does-not-exist.jsonc")
        )
    finally:
        app_mod._config = original

    assert app.deferred_startup.reloaded_with == [new_config]


# ---------------------------------------------------------------------------
# Admin model listing guard
# ---------------------------------------------------------------------------


def test_fetch_models_reports_startup_when_not_ready():
    """An unseeded provider must not be reported as an unsupported upstream."""
    pinfo = _pinfo("alcf")
    config = _Config({"alcf": pinfo})
    request = _Request(_App(), "alcf")
    request.app.gateway_config = config

    resp = asyncio.run(fetch_upstream_models(request))
    body = json.loads(resp.body)

    assert "starting up" in body["error"].lower()
    assert "not support" not in body["error"].lower()


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status


_PROBES: list[tuple[str, str, dict]] = []


class _FakeUpstreamClient:
    """AsyncClient stand-in that records every probe's auth headers."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> _FakeUpstreamClient:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get(self, url: str, headers: dict | None = None) -> _Resp:
        _PROBES.append(("GET", url, headers or {}))
        return _Resp(200)

    async def post(
        self, url: str, headers: dict | None = None, json: Any = None
    ) -> _Resp:
        _PROBES.append(("POST", url, headers or {}))
        return _Resp(401)


def test_connectivity_skips_every_authenticated_probe_when_not_ready():
    """The sentinel must not reach the POST probes either.

    They are checked after the models GET and used to run unconditionally, so
    the sentinel still went on the wire and the resulting 401 was recorded as
    ``ok`` by a helper that only rejects 404.
    """
    pinfo = _pinfo("alcf")
    raw = {
        "base_url": "https://example.com/v1",
        "embedding_format": "openai",
        "rerank_format": "openai",
    }
    config = _Config({"alcf": pinfo}, {"alcf": raw})
    request = _Request(_App(), "alcf")
    request.app.gateway_config = config

    _PROBES.clear()
    with patch("llm_rosetta._vendor.httpclient.AsyncClient", new=_FakeUpstreamClient):
        resp = asyncio.run(_provider_connectivity(request, "alcf"))

    body = json.loads(resp.body)

    sent = [h for _, _, h in _PROBES if any(k.lower() == "authorization" for k in h)]
    assert sent == []
    assert body["endpoints"]["models"]["skipped"] == "provider not ready"
    assert body["endpoints"]["embedding"]["skipped"] == "provider not ready"
    assert body["endpoints"]["rerank"]["skipped"] == "provider not ready"
    assert any("skipped" in w for w in body["warnings"])

"""A typed endpoint forwards the credential its provider lane minted (#1401).

A provider entry mints its credential when it names a ``token_command`` or
when its shim ships an ``auth.py``.  Its embeddings / rerank / decisions
endpoints are *second endpoints of the same entry*, not second providers, so
they read the entry's key ring rather than holding a copy: one mint, one
seeding pass and one refresh loop serve every endpoint, and a key the quota
state demoted is skipped by all of them.

The legacy ``<type>_providers`` sections have no entry in ``providers:`` and
therefore no lane to share — those keep the old refusal, and its reason says
where to move the provider.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.maintenance.startup import DeferredStartup, ProviderInitState
from llm_rosetta_gateway.transport.token_refresh import reset_reactive_state

TYPED_TYPES = ("embeddings", "rerank", "decisions")


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The quota state is process-wide; no test may inherit another's."""
    from llm_rosetta_gateway.quota.state import QUOTA_STATE

    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


class _FakeApp:
    """The slots ``DeferredStartup`` reads off the app."""

    def __init__(self) -> None:
        self.metrics = None
        self.persistence = None
        self.keystore = None
        self.gateway_config = None
        self._counter_rebuild_needed = False
        self._token_refresh_tasks: list[Any] = []


def _minting_raw(counter: Path) -> dict[str, Any]:
    """One provider, four endpoints, one credential source."""
    return {
        "providers": {
            "acme": {
                "type": "chat_completions",
                "base_url": "https://api.example.com/v1",
                "embeddings_format": "openai",
                "rerank_format": "jina",
                "decisions_format": "openai_decisions",
                "token_command": [
                    "sh",
                    "-c",
                    f"printf 'run\\n' >> {counter}; printf 'tok-1'",
                ],
            }
        },
        "models": {},
    }


def _typed_info(cfg: GatewayConfig, type_name: str, name: str = "acme") -> Any:
    return cfg.typed_pool(type_name)[1][name]


class TestTheEndpointJoinsThePool:
    def test_a_minting_provider_serves_every_typed_type(self, tmp_path: Path) -> None:
        raw = _minting_raw(tmp_path / "runs")

        cfg = GatewayConfig(raw)

        for type_name in TYPED_TYPES:
            assert "acme" in cfg.typed_pool(type_name)[0], type_name

    def test_no_endpoint_is_auto_disabled_for_minting(self, tmp_path: Path) -> None:
        cfg = GatewayConfig(_minting_raw(tmp_path / "runs"))

        assert not cfg.raw_providers["acme"].get("_auto_disabled_endpoints")


class TestOneCredentialSource:
    def test_the_endpoint_shares_the_lane_key_ring(self, tmp_path: Path) -> None:
        """Identity, not equality: a copied key set would pass ``==``.

        The construction order is enforced elsewhere — the typed loop reads
        ``self.providers`` without a fallback, so moving it earlier fails the
        load with an AttributeError.  This assertion pins the sharing itself.
        """
        cfg = GatewayConfig(_minting_raw(tmp_path / "runs"))
        owner = cfg.providers["acme"]

        for type_name in TYPED_TYPES:
            info = _typed_info(cfg, type_name)
            assert info.key_ring is owner.key_ring, type_name
            assert info.credential_owner is owner, type_name

    def test_the_endpoint_reads_the_owners_credential_state(
        self, tmp_path: Path
    ) -> None:
        cfg = GatewayConfig(_minting_raw(tmp_path / "runs"))
        owner = cfg.providers["acme"]
        info = _typed_info(cfg, "embeddings")

        owner.token_status = {"last_refresh": "2026-11-08T00:00:00+00:00"}

        # Read through, not copied at build time: the panel renders the
        # credential as it is now.
        assert info.token_status == owner.token_status

    async def test_seeding_mints_once_and_every_endpoint_sees_it(
        self, tmp_path: Path
    ) -> None:
        counter = tmp_path / "runs"
        cfg = GatewayConfig(_minting_raw(counter))
        app = _FakeApp()
        ds = DeferredStartup(cfg, app)

        assert cfg.providers["acme"].credentials_pending is True
        assert _typed_info(cfg, "embeddings").credentials_pending is True

        try:
            await ds._seed_tokens()

            assert ds.provider_state("acme") == ProviderInitState.READY
            # The command ran exactly once for the provider, not once per
            # endpoint: the endpoints forward what the lane minted.
            assert counter.read_text().count("run") == 1
            owner = cfg.providers["acme"]
            assert owner.auth_headers()["Authorization"] == "Bearer tok-1"
            for type_name in TYPED_TYPES:
                info = _typed_info(cfg, type_name)
                assert info.credentials_pending is False, type_name
                assert info.auth_headers()["Authorization"] == "Bearer tok-1"
            # One refresh loop for the one credential source.
            assert len(app._token_refresh_tasks) == 1
        finally:
            for task in app._token_refresh_tasks:
                task.cancel()
            await asyncio.gather(*app._token_refresh_tasks, return_exceptions=True)
            reset_reactive_state()

    async def test_a_failed_mint_names_the_command_and_keeps_the_lanes_unserved(
        self, tmp_path: Path
    ) -> None:
        raw: dict[str, Any] = {
            "providers": {
                "acme": {
                    "type": "chat_completions",
                    "base_url": "https://api.example.com/v1",
                    "embeddings_format": "openai",
                    "token_command": ["sh", "-c", "echo 'no ticket' >&2; exit 3"],
                }
            },
            "models": {},
        }
        cfg = GatewayConfig(raw)
        app = _FakeApp()
        ds = DeferredStartup(cfg, app)
        ds._RETRY_DELAY = 0

        try:
            with patch(
                "llm_rosetta_gateway.transport.token_refresh.start_token_refreshers",
                new_callable=AsyncMock,
                return_value=[],
            ):
                await ds._seed_tokens()
        finally:
            reset_reactive_state()

        info = _typed_info(cfg, "embeddings")
        assert ds.provider_state("acme") == ProviderInitState.FAILED
        # The endpoint is configured, and says why it cannot serve: the same
        # status object the panel renders, naming the command and its error.
        assert info.credentials_pending is True
        assert info.ready is False
        status = info.token_status
        assert status is not None
        assert "exit" in status["last_error"]
        assert "no ticket" in status["last_error"]
        assert "sh" in status["last_error"]

    def test_a_minting_typed_only_provider_gets_an_owner_too(
        self, tmp_path: Path
    ) -> None:
        """No chat lane declared, still one provider entry, still one ring."""
        cfg = GatewayConfig(
            {
                "providers": {
                    "emb-only": {
                        "base_url": "https://api.example.com/v1",
                        "embeddings_format": "openai",
                        "token_command": ["printf", "%s", "tok-2"],
                    }
                },
                "models": {},
            }
        )

        assert "emb-only" in cfg.providers
        assert (
            _typed_info(cfg, "embeddings", "emb-only").key_ring
            is cfg.providers["emb-only"].key_ring
        )


class TestSharedQuotaState:
    def test_a_demoted_key_is_skipped_by_the_endpoint_too(self) -> None:
        """Same credential, same allowance: a demotion covers every endpoint."""
        from datetime import datetime, timedelta, timezone

        from llm_rosetta_gateway.quota.models import credential_fingerprint
        from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        cfg = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        "type": "chat_completions",
                        "base_url": "https://api.example.com/v1",
                        "embeddings_format": "openai",
                        "api_key": "k-bad,k-good",
                    }
                },
                "models": {},
            }
        )
        # ``GatewayConfig`` bound the provider's ring predicate; demote one
        # credential through the same state a 429 or a probe writes to.
        QUOTA_STATE.rate_limits.note(
            "acme",
            credential_fingerprint("k-bad"),
            RateLimitPolicy(after=1, cooldown_s=60),
            now=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        assert (
            QUOTA_STATE.rate_limits.disabled("acme", credential_fingerprint("k-bad"))
            is not None
        )

        owner = cfg.providers["acme"]
        info = _typed_info(cfg, "embeddings")
        assert owner.auth_headers()["Authorization"] == "Bearer k-good"
        for _ in range(4):
            assert info.auth_headers()["Authorization"] == "Bearer k-good"


class TestTheLegacySectionStaysRefused:
    def test_the_reason_names_the_section_and_the_fix(self) -> None:
        raw: dict[str, Any] = {
            "embeddings_providers": {
                "p": {
                    "format": "openai",
                    "base_url": "https://api.example.com/v1",
                    "token_command": ["printf", "%s", "tok-3"],
                }
            }
        }

        cfg = GatewayConfig(raw)

        assert "p" not in cfg.embeddings_providers
        marker: Any = raw["embeddings_providers"]["p"]["_auto_disabled_endpoints"]
        reason = marker["embeddings"]["reason"]
        assert "mints its credential with a token_command" in reason
        assert "embeddings_providers" in reason
        assert "`providers:`" in reason

    def test_an_exchange_shim_in_the_legacy_section_is_refused_too(self) -> None:
        from llm_rosetta.shims.provider_shim import (
            ConnectionConfig,
            ProviderShim,
            register_shim,
        )

        class Exchange:
            kind = "oauth"

            async def credential(self, ctx):  # pragma: no cover - never called
                return "x"

        register_shim(
            ProviderShim(
                name="minty-legacy--systemone",
                format="systemone",
                model_type="decisions",
                connection=ConnectionConfig(base_url="https://minty.example"),
                auth=Exchange(),
            )
        )
        raw: dict[str, Any] = {
            "decisions_providers": {"p": {"decisions_shim": "minty-legacy--systemone"}}
        }

        cfg = GatewayConfig(raw)

        assert "p" not in cfg.decisions_providers
        marker: Any = raw["decisions_providers"]["p"]["_auto_disabled_endpoints"]
        assert "oauth exchange" in marker["decisions"]["reason"]


class TestTheExchangeShim:
    def test_a_provider_entry_with_an_exchange_shim_shares_its_ring(self) -> None:
        from llm_rosetta.shims.provider_shim import (
            ConnectionConfig,
            ProviderShim,
            register_shim,
        )

        class Exchange:
            kind = "oauth"
            consumes_key = False

            def account_id(self, ctx):  # pragma: no cover - record absent
                return None

            async def credential(self, ctx):  # pragma: no cover - not minted here
                return "minted"

        register_shim(
            ProviderShim(
                name="minty-entry--systemone",
                format="systemone",
                model_type="decisions",
                connection=ConnectionConfig(base_url="https://minty.example"),
                auth=Exchange(),
            )
        )
        cfg = GatewayConfig(
            {
                "providers": {
                    "p": {
                        "decisions_shim": "minty-entry--systemone",
                        "base_url": "https://minty.example",
                    }
                },
                "models": {},
            }
        )

        owner = cfg.providers["p"]
        assert owner.mints_credential is True
        assert owner.credentials_pending is True
        info = _typed_info(cfg, "decisions", "p")
        assert info.key_ring is owner.key_ring
        assert info.auth_exchange is owner.auth_exchange
        assert info.credentials_pending is True


class TestTheAdminPayload:
    """The panel learns the credential state from the provider entry.

    The typed endpoint has no separate credential state to report: it belongs
    to the provider entry, and that entry is what the payload renders.  So a
    typed-only minting provider shows its source and its pending status on the
    same card that carries the endpoint — and no endpoint marker.
    """

    def _payload(self, monkeypatch, cfg: GatewayConfig, raw: dict[str, Any]) -> Any:
        import json
        from types import SimpleNamespace

        from llm_rosetta_gateway.admin.routes.config import payload

        monkeypatch.setattr(payload, "_get_config_path", lambda request: "/tmp/c.jsonc")
        monkeypatch.setattr(
            payload,
            "_get_config_io",
            lambda request: SimpleNamespace(load_raw=lambda path: raw),
        )
        monkeypatch.setattr(payload, "_shim_status_payload", lambda request: {})
        monkeypatch.setattr(payload, "_shim_removal_advisories", lambda r, doc: [])
        monkeypatch.setattr(payload, "_shim_source_presets_payload", lambda: [])
        cfg.credential_visible = False
        cfg.api_keys_db = None
        cfg.admin_session_idle_minutes = 30
        cfg.admin_session_remember_days = 30
        request = SimpleNamespace(
            app=SimpleNamespace(
                gateway_config=cfg,
                auth_state=SimpleNamespace(admin_password=None),
            )
        )
        return json.loads(asyncio.run(payload.get_config(request)).body)

    def test_the_endpoint_and_its_source_reach_the_panel(self, monkeypatch) -> None:
        raw: dict[str, Any] = {
            "providers": {
                "emb-only": {
                    "base_url": "https://api.example.com/v1",
                    "embeddings_format": "openai",
                    "token_command": ["printf", "%s", "tok-4"],
                }
            },
            "models": {},
        }
        cfg = GatewayConfig(raw)

        entry = self._payload(monkeypatch, cfg, raw)["providers"]["emb-only"]

        assert entry["mints_credential"] is True
        assert entry["credential_source"] == "config_command"
        assert entry["credentials_pending"] is True
        assert "_auto_disabled_endpoints" not in entry


class TestTheTyped401Retry:
    """A typed request reacts to its own 401, once and only once.

    The chat route re-mints on a 401 through ``proxy.errors._handle_401``; a
    typed request never passes through that handler, so before this it waited
    for the next scheduled refresh while every request failed.
    """

    @staticmethod
    def _config(counter: Path, *, name: str, command: str) -> GatewayConfig:
        return GatewayConfig(
            {
                "providers": {
                    name: {
                        "type": "chat_completions",
                        "base_url": "https://api.example.com/v1",
                        "embeddings_format": "openai",
                        "token_command": ["sh", "-c", command],
                    }
                },
                "models": {},
            }
        )

    @staticmethod
    def _request(body: dict[str, Any], transport: Any) -> Any:
        from types import SimpleNamespace

        # A plain namespace rather than a MagicMock: the pipeline reads
        # ``request.app.ops`` through ``getattr(..., None)``, and a mock would
        # answer with another mock instead of "no persistence".
        return SimpleNamespace(
            json=lambda: body,
            headers={},
            app=SimpleNamespace(
                transport=transport, metrics=None, request_log=None, state=None
            ),
        )

    @staticmethod
    def _upstream_401() -> Any:
        from llm_rosetta_gateway.transport._base import UpstreamResponse

        return UpstreamResponse(
            status_code=401,
            body={"error": {"message": "token expired"}},
            raw_content=b'{"error":{"message":"token expired"}}',
        )

    @staticmethod
    def _upstream_200() -> Any:
        from llm_rosetta_gateway.transport._base import UpstreamResponse

        body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "m",
        }
        return UpstreamResponse(status_code=200, body=body, raw_content=b"{}")

    @pytest.fixture(autouse=True)
    def _clean_reactive(self):
        reset_reactive_state()
        yield
        reset_reactive_state()

    async def test_a_401_re_mints_once_and_the_retry_succeeds(
        self, tmp_path: Path
    ) -> None:
        from llm_rosetta_gateway.pipelines.embeddings import handle_embeddings

        counter = tmp_path / "runs"
        cfg = self._config(
            counter,
            name="retry-ok",
            command=f"printf 'run\\n' >> {counter}; printf 'fresh'",
        )
        owner = cfg.providers["retry-ok"]
        owner.key_ring.refresh("expired")

        seen: list[str] = []

        async def send(pinfo, url, body, **kwargs):
            seen.append(pinfo.auth_headers()["Authorization"])
            return self._upstream_401() if len(seen) == 1 else self._upstream_200()

        transport = MagicMock()
        transport.send = AsyncMock(side_effect=send)
        request = self._request({"model": "m", "input": "hi"}, transport)
        cfg.embeddings_models["m"] = "retry-ok"

        response = await handle_embeddings(request, cfg)

        assert response.status_code == 200
        assert seen == ["Bearer expired", "Bearer fresh"]
        # Exactly one mint for the one 401 — not one per endpoint or per retry.
        assert counter.read_text().count("run") == 1

    async def test_a_second_401_is_reported_not_retried_again(
        self, tmp_path: Path
    ) -> None:
        from llm_rosetta_gateway.pipelines.embeddings import handle_embeddings

        counter = tmp_path / "runs"
        cfg = self._config(
            counter,
            name="retry-twice",
            command=f"printf 'run\\n' >> {counter}; printf 'fresh'",
        )
        cfg.providers["retry-twice"].key_ring.refresh("expired")
        transport = MagicMock()
        transport.send = AsyncMock(return_value=self._upstream_401())
        request = self._request({"model": "m", "input": "hi"}, transport)
        cfg.embeddings_models["m"] = "retry-twice"

        response = await handle_embeddings(request, cfg)

        assert response.status_code == 401
        # Exactly one retry, and one mint: a refusal that survives a fresh
        # credential is the answer.
        assert transport.send.await_count == 2
        assert counter.read_text().count("run") == 1

    async def test_a_failed_re_mint_returns_the_401_without_retrying(
        self, tmp_path: Path
    ) -> None:
        from llm_rosetta_gateway.pipelines.embeddings import handle_embeddings

        counter = tmp_path / "runs"
        cfg = self._config(
            counter,
            name="retry-fail",
            command=f"printf 'run\\n' >> {counter}; echo nope >&2; exit 3",
        )
        owner = cfg.providers["retry-fail"]
        owner.key_ring.refresh("expired")
        transport = MagicMock()
        transport.send = AsyncMock(return_value=self._upstream_401())
        request = self._request({"model": "m", "input": "hi"}, transport)
        cfg.embeddings_models["m"] = "retry-fail"

        response = await handle_embeddings(request, cfg)

        assert response.status_code == 401
        assert transport.send.await_count == 1
        # The failure is reported where the credential lives, not swallowed.
        status = owner.token_status
        assert status is not None and "exit" in status["last_error"]

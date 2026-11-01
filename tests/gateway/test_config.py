"""Tests for gateway configuration parsing and validation."""

from __future__ import annotations

import asyncio
import os

import pytest

from llm_rosetta_gateway.config import (
    GatewayConfig,
    async_config_lock,
    config_lock,
)


def _minimal_raw(**server_overrides) -> dict:
    """Return a minimal valid config dict with optional server overrides."""
    raw = {
        "providers": {
            "test": {
                "api_key": "sk-test",
                "base_url": "https://api.example.com",
                "type": "openai",
            }
        },
        "models": {"gpt-test": "test"},
        "server": {},
    }
    raw["server"].update(server_overrides)
    return raw


class TestAdminPasswordUnresolvedEnvVar:
    """admin_password must not contain unresolved ${...} placeholders."""

    def test_reject_unresolved_placeholder(self):
        raw = _minimal_raw(admin_password="${ADMIN_PASSWORD}")
        with pytest.raises(ValueError, match="unresolved"):
            GatewayConfig(raw)

    def test_reject_partial_placeholder(self):
        raw = _minimal_raw(admin_password="prefix-${SOME_VAR}-suffix")
        with pytest.raises(ValueError, match="unresolved"):
            GatewayConfig(raw)

    def test_accept_literal_password(self):
        raw = _minimal_raw(admin_password="my-secret-password")
        cfg = GatewayConfig(raw)
        assert cfg.admin_password == "my-secret-password"

    def test_accept_none(self):
        raw = _minimal_raw()
        cfg = GatewayConfig(raw)
        assert cfg.admin_password is None


class TestUpdateSwitches:
    """server.update_check and server.self_update gate the console's update UI."""

    def test_both_are_on_by_default(self):
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.update_check is True
        assert cfg.self_update is True

    def test_each_switch_turns_off(self):
        cfg = GatewayConfig(_minimal_raw(update_check=False, self_update=False))
        assert cfg.update_check is False
        assert cfg.self_update is False

    def test_a_quoted_value_does_not_enable_a_switch(self):
        # Only a literal true enables the probe or the pip run; a string in a
        # hand-written file must not turn either on by accident.
        cfg = GatewayConfig(_minimal_raw(update_check="true", self_update="false"))
        assert cfg.update_check is False
        assert cfg.self_update is False


class TestOpenOnNoKeys:
    """server.open_on_no_keys controls anonymous access when no keys exist."""

    def test_defaults_to_false(self):
        # Secure by default: absent flag → closed.
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.open_on_no_keys is False

    def test_explicit_true(self):
        cfg = GatewayConfig(_minimal_raw(open_on_no_keys=True))
        assert cfg.open_on_no_keys is True

    def test_explicit_false(self):
        cfg = GatewayConfig(_minimal_raw(open_on_no_keys=False))
        assert cfg.open_on_no_keys is False

    def test_coerced_to_bool(self):
        # Truthy/falsy JSON values are normalised to real bools.
        assert GatewayConfig(_minimal_raw(open_on_no_keys=1)).open_on_no_keys is True
        assert GatewayConfig(_minimal_raw(open_on_no_keys=0)).open_on_no_keys is False


# ---------------------------------------------------------------------------
# Config lock tests
# ---------------------------------------------------------------------------


class TestConfigLock:
    """Sync config_lock acquires file lock and serializes access."""

    def test_creates_lock_file(self, tmp_path):
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        with config_lock(str(cfg)):
            lock_file = str(cfg) + ".lock"
            assert os.path.exists(lock_file)

    def test_lock_released_after_exit(self, tmp_path):
        """Lock file is released after context manager exits."""
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        with config_lock(str(cfg)):
            pass
        # Should be able to reacquire
        with config_lock(str(cfg)):
            pass


class TestAsyncConfigLock:
    """async_config_lock must not block the event loop."""

    async def test_basic_acquire_release(self, tmp_path):
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        async with async_config_lock(str(cfg)):
            lock_file = str(cfg) + ".lock"
            assert os.path.exists(lock_file)

    async def test_serializes_concurrent_access(self, tmp_path):
        """Two concurrent async_config_lock calls should not overlap."""
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        order: list[str] = []

        async def worker(name: str, delay: float):
            async with async_config_lock(str(cfg)):
                order.append(f"{name}-enter")
                await asyncio.sleep(delay)
                order.append(f"{name}-exit")

        await asyncio.gather(worker("a", 0.05), worker("b", 0.01))
        assert order[0] == "a-enter"
        assert order[1] == "a-exit"
        assert order[2] == "b-enter"
        assert order[3] == "b-exit"

    async def test_does_not_block_event_loop(self, tmp_path):
        """A background task should run while the lock is held."""
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        bg_ran = False

        async def background():
            nonlocal bg_ran
            bg_ran = True

        async with async_config_lock(str(cfg)):
            task = asyncio.create_task(background())
            await asyncio.sleep(0)  # yield to let bg task run
            await task

        assert bg_ran

    async def test_lock_released_on_exception(self, tmp_path):
        """Lock must be released even if body raises."""
        cfg = tmp_path / "config.jsonc"
        cfg.write_text("{}")
        with pytest.raises(ValueError, match="boom"):
            async with async_config_lock(str(cfg)):
                raise ValueError("boom")

        # Should be able to reacquire
        async with async_config_lock(str(cfg)):
            pass


class TestShimConnectionDefaults:
    """A provider configured via a shim inherits the shim's connection
    defaults at runtime.

    Regression: ``GatewayConfig`` used to pass the base converter type to
    ``build_provider_info`` and drop the shim name, so ``get_shim()`` found
    nothing and every shim's ``connection.*`` defaults (base_url, custom
    auth header) were silently ignored.
    """

    def test_shim_base_url_applied(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        raw = {
            "providers": {
                "openrouter": {
                    "type": "openrouter--chat_completions",
                    "api_key": "sk-x",
                }
            },
            "models": {"m": "openrouter"},
            "server": {},
        }
        cfg = GatewayConfig(raw)
        info = cfg.providers["openrouter"]
        assert info.base_url == "https://openrouter.ai/api/v1"
        # ProviderInfo.name keeps the base type (unchanged semantics)
        assert info.name == "chat_completions"

    def test_shim_custom_auth_header_applied(self, tmp_path):
        from llm_rosetta.shims import unregister_shim
        from llm_rosetta.shims.providers import load_providers_from_dir

        leaf = tmp_path / "acme"
        leaf.mkdir()
        (leaf / "provider.yaml").write_text(
            "name: acme--google_generate\n"
            "model_type: llm\n"
            "format: google_generate\n"
            "connection:\n"
            "  base_url: https://example.invalid/v1\n"
            "  auth_header: x-access-tokens\n"
        )
        load_providers_from_dir(tmp_path)
        try:
            raw = {
                "providers": {
                    "acme": {"type": "acme--google_generate", "api_key": "k"}
                },
                "models": {"m": "acme"},
                "server": {},
            }
            cfg = GatewayConfig(raw)
            assert cfg.providers["acme"].auth_headers() == {"x-access-tokens": "k"}
        finally:
            unregister_shim("acme--google_generate")

    def test_explicit_base_url_wins_over_shim_default(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        raw = {
            "providers": {
                "openrouter": {
                    "type": "openrouter--chat_completions",
                    "api_key": "sk-x",
                    "base_url": "https://proxy.internal/v1",
                }
            },
            "models": {"m": "openrouter"},
            "server": {},
        }
        cfg = GatewayConfig(raw)
        assert cfg.providers["openrouter"].base_url == "https://proxy.internal/v1"

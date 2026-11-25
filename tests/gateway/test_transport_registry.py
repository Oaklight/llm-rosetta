"""The transport table: what the gateway ships and what a package adds.

Discovery reads entry-point metadata, so config validation can name an
installed transport without importing it.  The plugin loads once, at app
creation.  A plugin that claims a built-in id is refused rather than allowed
to shadow it.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.config.providers import build_provider_info
from llm_rosetta_gateway.transport import registry

ANTIGRAVITY = "antigravity--acp"


class _Distribution:
    name = "fake-transports"
    version = "0.1.0"


class _EntryPoint:
    """The part of ``importlib.metadata.EntryPoint`` the registry reads."""

    def __init__(self, name: str, factory: object, dist: object | None = None) -> None:
        self.name = name
        self.value = f"fake_transports:{name}"
        self.dist = dist or _Distribution()
        self._factory = factory

    def load(self) -> object:
        if isinstance(self._factory, BaseException):
            raise self._factory
        return self._factory


@pytest.fixture(autouse=True)
def _no_installed_plugins(monkeypatch):
    """Default to the environment this gateway ships with."""
    monkeypatch.setattr(registry, "_entry_points", lambda: ())


class TestWhatIsInstalled:
    def test_the_builtins_are_listed(self):
        assert registry.installed_transport_names() == (
            "acp",
            "claude_cli",
            "http",
        )

    def test_a_plugin_name_is_known_without_loading_it(self, monkeypatch):
        loaded: list[bool] = []

        def factory() -> object:
            loaded.append(True)
            return object()

        monkeypatch.setattr(
            registry, "_entry_points", lambda: (_EntryPoint("carrier-pigeon", factory),)
        )

        assert "carrier-pigeon" in registry.installed_transport_names()
        assert loaded == []


class TestBuildingTheLaneTable:
    def test_http_is_not_a_lane_transport(self):
        built = registry.build_lane_transports(timeout=60.0)

        assert "http" not in built
        assert isinstance(built["acp"], registry.AcpTransport)
        assert isinstance(built["claude_cli"], registry.ClaudeCliTransport)

    def test_a_plugin_receives_the_server_timeout(self, monkeypatch):
        timeouts: list[float | None] = []
        transport = object()

        def factory(*, timeout: float = 1.0) -> object:
            timeouts.append(timeout)
            return transport

        monkeypatch.setattr(
            registry, "_entry_points", lambda: (_EntryPoint("carrier-pigeon", factory),)
        )

        built = registry.build_lane_transports(timeout=42.0)

        assert built["carrier-pigeon"] is transport
        assert timeouts == [42.0]

    def test_no_timeout_leaves_the_factory_its_own_default(self, monkeypatch):
        calls: list[tuple[object, ...]] = []

        def factory(*args: object) -> object:
            calls.append(args)
            return object()

        monkeypatch.setattr(
            registry, "_entry_points", lambda: (_EntryPoint("carrier-pigeon", factory),)
        )

        registry.build_lane_transports(timeout=None)

        assert calls == [()]

    def test_a_plugin_cannot_shadow_a_builtin(self, monkeypatch):
        monkeypatch.setattr(
            registry,
            "_entry_points",
            lambda: (_EntryPoint("acp", object()),),
        )

        with pytest.raises(registry.TransportPluginError, match="already ships"):
            registry.build_lane_transports(timeout=60.0)

    def test_a_broken_plugin_names_the_entry_point(self, monkeypatch):
        monkeypatch.setattr(
            registry,
            "_entry_points",
            lambda: (_EntryPoint("broken", ImportError("no module named acme")),),
        )

        with pytest.raises(
            registry.TransportPluginError,
            match=r"'broken'.*failed to load: ImportError",
        ):
            registry.build_lane_transports(timeout=60.0)

    def test_a_non_callable_plugin_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            registry, "_entry_points", lambda: (_EntryPoint("bad", 42),)
        )

        with pytest.raises(registry.TransportPluginError, match="not callable"):
            registry.build_lane_transports(timeout=60.0)

    def test_an_override_covers_every_lane_without_loading_a_plugin(self, monkeypatch):
        loaded: list[bool] = []

        def factory() -> object:
            loaded.append(True)
            return object()

        monkeypatch.setattr(
            registry, "_entry_points", lambda: (_EntryPoint("carrier-pigeon", factory),)
        )
        override = object()

        built = registry.build_lane_transports(timeout=60.0, override=override)

        assert set(built) == {"acp", "claude_cli", "carrier-pigeon"}
        assert all(value is override for value in built.values())
        assert loaded == []


class TestWhatConfigAccepts:
    def test_an_installed_transport_name_carries_the_lane(
        self, monkeypatch, moved_vendors
    ):
        monkeypatch.setattr(
            registry,
            "installed_transport_names",
            lambda: ("acp", "carrier-pigeon", "http"),
        )

        info = build_provider_info(
            "acp", {"transport": "carrier-pigeon"}, shim_name=ANTIGRAVITY
        )

        assert info.transport == "carrier-pigeon"

    def test_an_uninstalled_name_is_still_refused(self, moved_vendors):
        with pytest.raises(ValueError, match="unknown transport 'carrier-pigeon'"):
            build_provider_info(
                "acp", {"transport": "carrier-pigeon"}, shim_name=ANTIGRAVITY
            )

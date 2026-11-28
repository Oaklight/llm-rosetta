"""Audited lifecycle events: admin setup, startup, shutdown and config reload."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_ADMIN_SETUP,
    EVENT_CONFIG_RELOAD,
    EVENT_SHUTDOWN,
    EVENT_STARTUP,
    SOURCE_ADMIN,
    SOURCE_CONFIG,
    SOURCE_GATEWAY,
)

from .base import OpsBase, OpsContext


class OpsAdminSetup(OpsBase):
    """Record that the admin panel finished wiring."""

    event_type: ClassVar[str] = EVENT_ADMIN_SETUP
    source: ClassVar[str] = SOURCE_ADMIN

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return "Admin panel initialized"

    def _details(self, result: Any) -> dict[str, Any]:
        return {}


class OpsGatewayStartup(OpsBase):
    """Record the boot line: the address and how many providers and models serve."""

    event_type: ClassVar[str] = EVENT_STARTUP
    source: ClassVar[str] = SOURCE_GATEWAY

    __slots__ = ("_host", "_port", "_config")

    def __init__(
        self, ctx: OpsContext, host: str, port: int, config: Any = None
    ) -> None:
        super().__init__(ctx)
        self._host = host
        self._port = port
        self._config = config

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return f"Gateway starting on {self._host}:{self._port}"

    def _details(self, result: Any) -> dict[str, Any]:
        config = self._config
        return {
            "host": self._host,
            "port": self._port,
            "provider_count": len(config.providers) if config else 0,
            "model_count": len(config.models) if config else 0,
        }


class OpsGatewayShutdown(OpsBase):
    """Record the shutdown line, without the pruning pass that would delay close."""

    event_type: ClassVar[str] = EVENT_SHUTDOWN
    source: ClassVar[str] = SOURCE_GATEWAY
    skip_prune: ClassVar[bool] = True

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return "Gateway shutting down"

    def _details(self, result: Any) -> dict[str, Any]:
        return {}


class OpsConfigReload(OpsBase):
    """Record a hot config reload and the size of the config now served."""

    event_type: ClassVar[str] = EVENT_CONFIG_RELOAD
    source: ClassVar[str] = SOURCE_CONFIG

    __slots__ = ("_config",)

    def __init__(self, ctx: OpsContext, config: Any) -> None:
        super().__init__(ctx)
        self._config = config

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return (
            f"Config reloaded ({len(self._config.providers)} providers, "
            f"{len(self._config.models)} models)"
        )

    def _details(self, result: Any) -> dict[str, Any]:
        return {
            "provider_count": len(self._config.providers),
            "model_count": len(self._config.models),
        }

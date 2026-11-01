"""Security event operations — password, token, session."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_PASSWORD_CHANGED,
    EVENT_SELF_UPDATE,
    EVENT_SESSION_LOGOUT_ALL,
    EVENT_TOKEN_ROTATED,
    SOURCE_ADMIN,
    SOURCE_AUTH,
)

from .base import OpsBase, OpsContext


class OpsPasswordChange(OpsBase):
    """Record an admin password change."""

    event_type: ClassVar[str] = EVENT_PASSWORD_CHANGED
    source: ClassVar[str] = SOURCE_AUTH

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return "Admin password changed"

    def _details(self, result: Any) -> dict[str, Any]:
        return {}


class OpsTokenRotate(OpsBase):
    """Record an internal proxy token rotation."""

    event_type: ClassVar[str] = EVENT_TOKEN_ROTATED
    source: ClassVar[str] = SOURCE_AUTH

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return "Internal proxy token rotated"

    def _details(self, result: Any) -> dict[str, Any]:
        return {}


class OpsSessionLogoutAll(OpsBase):
    """Record a bulk session invalidation."""

    event_type: ClassVar[str] = EVENT_SESSION_LOGOUT_ALL
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_count",)

    def __init__(self, ctx: OpsContext, *, count: int) -> None:
        super().__init__(ctx)
        self._count = count

    async def _run(self) -> dict[str, Any]:
        return {"sessions_cleared": self._count}

    def _message(self, result: Any) -> str:
        return f"All admin sessions invalidated ({self._count} cleared)"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"sessions_cleared": self._count}


class OpsSelfUpdate(OpsBase):
    """Record a self-update: the install that replaced the running code."""

    event_type: ClassVar[str] = EVENT_SELF_UPDATE
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_version",)

    def __init__(self, ctx: OpsContext, *, version: str) -> None:
        super().__init__(ctx)
        self._version = version

    async def _run(self) -> dict[str, Any]:
        return {"version": self._version}

    def _message(self, result: Any) -> str:
        return f"Installed {self._version} and restarting"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"version": self._version}

"""Operations on installed extensions: installing the program one drives (#1820),
removing an extension and updating it (#1829)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, ClassVar

from llm_rosetta.api import (
    CliInstallError,
    ExtensionInstallError,
    install_cli,
    remove_with_programs,
)

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_EXTENSION_CLI_INSTALL,
    EVENT_EXTENSION_REMOVE,
    EVENT_EXTENSION_UPDATE,
    SOURCE_ADMIN,
)

from .base import OpsBase, OpsContext


class OpsExtensionCliInstall(OpsBase):
    """Install the external program an extension declares, at its pinned version.

    The administrator names only the extension and the program; the version,
    the source and the checksum are the ones the extension's manifest pins.  The
    audit record holds the extension id, the program, the pinned version and the
    outcome, and on a failure the installer's one-line reason: never the
    environment or the install's output.
    """

    event_type: ClassVar[str] = EVENT_EXTENSION_CLI_INSTALL
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_ext_id", "_program", "_version")

    def __init__(self, ctx: OpsContext, *, ext_id: str, program: Any) -> None:
        super().__init__(ctx)
        self._ext_id = ext_id
        self._program = program
        self._version = program.install.version

    async def _run(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            done = await loop.run_in_executor(
                None, lambda: install_cli(self._ext_id, self._program)
            )
        except CliInstallError:
            raise
        except Exception as exc:  # noqa: BLE001 - the record must not carry its text
            raise CliInstallError(
                f"the install stopped unexpectedly ({type(exc).__name__})"
            ) from None
        return {"state": done.state, "found": done.found}

    def _message(self, result: Any) -> str:
        name = f"{self._ext_id}/{self._program.name} {self._version}"
        if result is None:
            return f"Install of {name} failed"
        return (
            f"Installed {name}"
            if result["state"] == "installed"
            else (f"{name} was already installed")
        )

    def _details(self, result: Any) -> dict[str, Any]:
        details: dict[str, Any] = {
            "extension": self._ext_id,
            "program": self._program.name,
            "version": self._version,
        }
        if result is not None:
            details.update(result)
        return details


class OpsExtensionRemove(OpsBase):
    """Remove an extension with the programs it installed.

    The record holds the extension id and version, the paths deleted, the paths
    that could not be deleted (with the reason) and the shared paths that were
    kept; no file contents and no environment.  ``replace`` marks a removal that
    is the first half of a replacement.
    """

    event_type: ClassVar[str] = EVENT_EXTENSION_REMOVE
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_ext_id", "_plan", "_replace")

    def __init__(
        self, ctx: OpsContext, *, ext_id: str, plan: Any, replace: bool = False
    ) -> None:
        super().__init__(ctx)
        self._ext_id = ext_id
        self._plan = plan
        self._replace = replace

    async def _run(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            done = await loop.run_in_executor(
                None,
                lambda: remove_with_programs(
                    self._ext_id,
                    path=self._plan.delete[0].path if self._plan.delete else None,
                ),
            )
        except ExtensionInstallError:
            raise
        except Exception as exc:  # noqa: BLE001 - the record must not carry its text
            raise ExtensionInstallError(
                f"the removal stopped unexpectedly ({type(exc).__name__})"
            ) from None
        return {"deleted": list(done.deleted), "failed": [list(f) for f in done.failed]}

    def _message(self, result: Any) -> str:
        name = f"{self._ext_id} {self._plan.version or ''}".strip()
        if result is None:
            return f"Removal of {name} failed"
        if result["failed"]:
            return (
                f"Removed {name}; {len(result['failed'])} path(s) could not be deleted"
            )
        return f"Removed {name}"

    def _details(self, result: Any) -> dict[str, Any]:
        details: dict[str, Any] = {
            "extension": self._ext_id,
            "version": self._plan.version,
            "replace": self._replace,
            "kept": [entry.path for entry in self._plan.keep],
        }
        if result is not None:
            details.update(result)
        return details


class OpsExtensionUpdate(OpsBase):
    """Install a newer (or a chosen) release of an extension.

    ``work`` does the install and returns the version now installed.  The
    record holds the extension id, the version before and after, and on a
    failure the installer's one-line reason.
    """

    event_type: ClassVar[str] = EVENT_EXTENSION_UPDATE
    source: ClassVar[str] = SOURCE_ADMIN

    __slots__ = ("_ext_id", "_before", "_work")

    def __init__(
        self,
        ctx: OpsContext,
        *,
        ext_id: str,
        before: str | None,
        work: Callable[[], str | None],
    ) -> None:
        super().__init__(ctx)
        self._ext_id = ext_id
        self._before = before
        self._work = work

    async def _run(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            after = await loop.run_in_executor(None, self._work)
        except ExtensionInstallError:
            raise
        except Exception as exc:  # noqa: BLE001 - the record must not carry its text
            raise ExtensionInstallError(
                f"the update stopped unexpectedly ({type(exc).__name__})"
            ) from None
        return {"after": after}

    def _message(self, result: Any) -> str:
        if result is None:
            return f"Update of {self._ext_id} failed"
        return f"Updated {self._ext_id} {self._before or '?'} -> {result['after']}"

    def _details(self, result: Any) -> dict[str, Any]:
        details: dict[str, Any] = {"extension": self._ext_id, "before": self._before}
        if result is not None:
            details["after"] = result["after"]
        return details

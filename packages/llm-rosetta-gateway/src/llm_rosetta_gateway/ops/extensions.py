"""Operations on installed extensions: installing the program one drives (#1820)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from llm_rosetta.api import CliInstallError, install_cli

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_EXTENSION_CLI_INSTALL,
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

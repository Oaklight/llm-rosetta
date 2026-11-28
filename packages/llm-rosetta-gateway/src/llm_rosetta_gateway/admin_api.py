"""The surface the admin panel depends on.

``admin/`` is served by the gateway but must not reach into its internals.
Everything it needs from the running gateway is declared here as a
:class:`~typing.Protocol`, grouped by capability, and implemented by one
adapter (``app/admin_context.py``) that ``create_app`` exposes as
``app.admin``.

Only members the admin code uses today belong here.  A new need adds a member
to the matching capability (or a new capability) and its adapter
implementation; it does not add a gateway import to ``admin/``.
``tests/gateway/test_admin_import_boundary.py`` enforces the boundary and
keeps the list of not-yet-migrated imports shrinking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .observability.logging import get_logger

if TYPE_CHECKING:
    from .config import GatewayConfig

__all__ = [
    "AdminConfigAccess",
    "AdminContext",
    "AdminExtensions",
    "AdminShimSources",
    "ExtensionInstallRefused",
    "admin_context",
]

logger = get_logger()

#: Set once the stock-adapter fallback has been reported.  The fallback is taken
#: on every admin request of such an app, so a per-call warning would flood the
#: log; one line per process is enough to make a missing wiring visible.
_fallback_reported = False


@runtime_checkable
class AdminConfigAccess(Protocol):
    """Read and replace the config the gateway is serving."""

    def served_config(self) -> GatewayConfig | None:
        """Return the config the gateway serves, or ``None`` if none is set."""
        ...

    def publish_config(self, config: GatewayConfig) -> None:
        """Make *config* the one the gateway serves from now on."""
        ...


@runtime_checkable
class AdminShimSources(Protocol):
    """What the manifest shim sources offer (#1660 S9)."""

    def shim_source_catalogue(
        self, config: Any, report: Any, *, refresh: bool = False
    ) -> dict[str, Any]:
        """Every manifest source *config* reads, with every vendor it publishes.

        Returns ``{"core": {surface, ir, version}, "sources": [...],
        "install": [...]}``.  Reads the manifests through the installer's
        cache; *refresh* refetches them.  Blocking: call it off the loop.
        """
        ...


class ExtensionInstallRefused(Exception):
    """An install the panel asked for did not happen.

    Attributes:
        status: The HTTP status that fits: 404 (no such extension or program),
            400 (the program must be named), 409 (nothing to install, or one
            install of that extension is already running) or 502 (the install
            ran and failed).
        reason: A short code the panel can switch on.
        message: One sentence for the operator.
    """

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


@runtime_checkable
class AdminExtensions(Protocol):
    """What the installed extensions look like (epic #1703)."""

    def extension_listing(self) -> list[dict[str, Any]]:
        """Every extension found or installed, one JSON-safe record each.

        A record holds ``id``, ``version``, ``source`` (``entry point``,
        ``namespace`` or ``extensions directory``), ``state`` (``loaded``,
        ``skipped``, ``failed`` or ``installed`` when it is on disk and not yet
        loaded), ``reason``, ``path``, ``maturity``, ``shims`` (the names it
        registered), ``transports``, ``external`` (one status per program it
        drives) and ``missing_dependencies``.  A program's status holds
        ``state`` (the check's: ``ok``, ``missing``, ``broken``, ``too_old``,
        ``unknown``), ``status`` (the card's: ``ready``, ``missing``, ``broken``,
        ``too_old``, ``unknown``), ``found``, ``minimum``, ``install`` (the pinned
        ``{version, source}`` or ``None``) and ``install_command`` (the
        ``llm-rosetta ext install-cli`` line that installs it, or ``None`` when
        it is ready or declares no install).  The check also looks in the
        directory ``ext install-cli`` fills.  Blocking: it runs the external
        programs' version checks, so call it off the loop.
        """
        ...

    async def install_extension_cli(
        self, ext_id: str, program: str | None
    ) -> dict[str, Any]:
        """Install a program an extension declares, at the version it pins.

        Only the extension id and the program's name come from the caller; the
        version, the source and the checksum are the manifest's.  One install
        per extension runs at a time.  The attempt is recorded in the ops log
        (extension, program, version and outcome; never the environment or the
        install's output).

        Returns:
            ``{"extension", "program", "state", "version", "found"}``; *state*
            is ``installed`` or ``unchanged``.

        Raises:
            ExtensionInstallRefused: The request names nothing installable, one
                install of that extension is running, or the install failed.
        """
        ...

    def extension_removal_plan(self, ext_id: str) -> dict[str, Any] | None:
        """What removing the extension deletes and keeps, or ``None`` if unknown.

        ``{"id", "version", "origin", "removable", "delete", "keep", "hint"}``;
        *delete* and *keep* are lists of ``{path, size, note}``.  *origin* is
        ``directory`` (removable) or ``image`` (provided by the image, e.g. a pip
        package; *hint* says how to remove it).  Blocking: it measures sizes.
        """
        ...

    async def remove_extension(
        self, ext_id: str, *, replace: bool = False
    ) -> dict[str, Any]:
        """Remove an extension and the programs it installed, as the plan says.

        One operation per extension runs at a time.  The attempt is recorded in
        the ops log (extension, version, paths deleted or failed, paths kept).

        Returns:
            ``{"extension", "deleted", "failed"}``.

        Raises:
            ExtensionInstallRefused: The extension is unknown (404), provided by
                the image (409 ``not_removable``), busy (409) or the removal
                failed (502).
        """
        ...

    async def check_extension_updates(
        self, config: Any, report: Any, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Compare installed extensions with the manifest sources, refetching them.

        One record per extension: ``id``, ``current``, ``available`` (the newest
        release this core runs), ``newest``, ``source``, ``versions`` and
        ``status`` (``update``, ``current``, ``not_published``,
        ``no_compatible`` or ``failed``) with ``error`` for a failure.  A source
        that cannot be read is named in the error, never skipped silently.
        """
        ...

    async def update_extension(
        self, ext_id: str, config: Any, report: Any, version: str | None = None
    ) -> dict[str, Any]:
        """Install the newest release (or *version*) of an extension from its source.

        Recorded in the ops log (extension, version before and after).

        Returns:
            ``{"extension", "before", "after"}``.

        Raises:
            ExtensionInstallRefused: Unknown extension or version (404/400),
                provided by the image (409), busy (409) or the install failed
                (502).
        """
        ...


@runtime_checkable
class AdminContext(AdminConfigAccess, AdminShimSources, AdminExtensions, Protocol):
    """Everything the admin panel may ask of the running gateway.

    Capabilities are added as base classes, one per concern: config,
    providers and credentials, shims, routing and quota, keys and auth,
    observability and data, jobs, maintenance.  Only config is migrated so far.
    """


def admin_context(request: Any) -> AdminContext:
    """Return the :class:`AdminContext` the app of *request* exposes.

    ``create_app`` sets ``app.admin``.  An app object built by hand (a test
    double, an embedding host) may lack it; it then gets the stock adapter,
    which reads the same process state.
    """
    context = getattr(request.app, "admin", None)
    if context is not None:
        return context  # type: ignore[no-any-return]
    from .app.admin_context import GatewayAdminContext

    global _fallback_reported
    if not _fallback_reported:
        _fallback_reported = True
        logger.warning(
            "app.admin is not set; admin falls back to the stock "
            "GatewayAdminContext (create_app sets app.admin, so this app was "
            "built by hand)"
        )
    return GatewayAdminContext()

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
        ``state`` (the check's: ``ok``, ``missing``, ``too_old``, ``unknown``),
        ``status`` (the card's: ``ready``, ``missing``, ``too_old``,
        ``unknown``), ``found``, ``minimum``, ``install`` (the pinned
        ``{version, source}`` or ``None``) and ``install_command`` (the
        ``llm-rosetta ext install-cli`` line that installs it, or ``None`` when
        it is ready or declares no install).  The check also looks in the
        directory ``ext install-cli`` fills.  Blocking: it runs the external
        programs' version checks, so call it off the loop.
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

"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from ..admin_api import app_admin_context

if TYPE_CHECKING:
    from ..admin_api import AdminContext, ConfigIO, GatewayConfig

__all__ = [
    "setup_admin",
    "apply_admin_theme",
    "apply_theme",
]

logger = logging.getLogger("llm-rosetta-gateway")


_VALID_TABS = frozenset(
    {
        "providers",
        "shims",
        "models",
        "routing",
        "keys",
        "dashboard",
        "logs",
        "jobs",
        "settings",
    }
)


def _admin_theme_style(css: str | None) -> str:
    """Wrap an operator theme override in a ``<style>`` block.

    A literal ``</style`` would close the block early, and CSS has no use for
    one, so it is escaped.  An empty override produces no block at all.

    Args:
        css: The ``server.admin_custom_css`` text, or ``None``.

    Returns:
        A ``<style>`` element, or an empty string.
    """
    text = (css or "").strip()
    if not text:
        return ""
    text = re.sub(r"</(style)", r"<\\/\1", text, flags=re.IGNORECASE)
    return f'<style id="admin-custom-css">\n{text}\n</style>'


def apply_admin_theme(app: Any, css: str | None) -> None:
    """Rebuild ``app.admin_custom_head`` from the stored prefix and *css*.

    Called at setup and again after a config reload, so a theme saved from the
    admin panel reaches the next page load without restarting the gateway.
    The override goes last: it must be able to beat both the built-in schemes
    and a downstream ``custom_head`` fragment.

    Args:
        app: The httpserver application.
        css: The operator's CSS override, or ``None``.
    """
    apply_theme(app_admin_context(app), css)


def apply_theme(context: AdminContext, css: str | None) -> None:
    """Rebuild the custom head held by *context* from its prefix and *css*."""
    parts = [context.admin_head_prefix(), _admin_theme_style(css)]
    context.set_admin_custom_head("\n".join(part for part in parts if part))


async def setup_admin(
    app: Any,
    config: GatewayConfig,
    config_path: str | None,
    config_io: ConfigIO | None = None,
    disabled_tabs: list[str] | None = None,
    custom_head: str | None = None,
    branding: dict[str, Any] | None = None,
    data_dir: str | None = None,
) -> None:
    """Initialize admin panel state on the app.

    Args:
        app: The httpserver application.
        config: Parsed gateway configuration.
        config_path: Path to the config file on disk (``None`` disables
            persistence and config editing).
        config_io: Optional I/O adapter for reading/writing the config
            file.  Defaults to :class:`JsoncConfigIO` when ``None``.
            Supply a custom implementation for non-JSONC formats
            (e.g. YAML in argo-proxy).
        disabled_tabs: Optional list of tab identifiers to hide from the
            admin panel.  Valid values: ``"providers"``, ``"shims"``,
            ``"models"``, ``"routing"``, ``"keys"``, ``"dashboard"``,
            ``"logs"``, ``"jobs"``, ``"settings"``.  Disabling
            ``"providers"`` hides ``"shims"`` too: the shims page is where a
            provider is added, and its routes are guarded by the providers
            tab.  Disabling ``"logs"`` hides ``"jobs"`` too: the job board is
            part of observability.  Disabled tabs are hidden via CSS, their
            data-fetching JS functions are suppressed, and their API routes
            return 404.
        custom_head: Optional HTML fragment (``<style>``, ``<script>``,
            etc.) injected before ``</head>`` when serving the admin
            panel.  Allows downstream projects to override UI behavior
            without modifying the reference ``admin.html``.
        branding: Optional dict to customize the admin panel identity.
            Supported keys::

                title       — header title (default: "llm-rosetta")
                subtitle    — header subtitle (default: "gateway admin")
                version     — version shown in settings footer
                links       — list of {"label", "url", "icon"} for footer buttons
                              icon: "github" | "pypi" | "docker" | "docs" | None
                attribution — "Powered by ..." line in settings footer

    Routes are registered separately via ``register_admin_routes`` before
    calling this function.
    """
    disabled = frozenset(disabled_tabs or ())
    unknown = disabled - _VALID_TABS
    if unknown:
        logger.warning(
            "Ignoring unknown disabled_tabs: %s (valid: %s)",
            ", ".join(sorted(unknown)),
            ", ".join(sorted(_VALID_TABS)),
        )
        disabled = disabled & _VALID_TABS
    app_admin_context(app).set_disabled_tabs(disabled)
    await app_admin_context(app).start_services(
        config, config_path, config_io, data_dir
    )
    # Build custom_head: disabled-tab CSS + user fragment + branding injection
    parts: list[str] = []
    # Two child pages go with their parent: the shims page is where a
    # provider is added and its routes carry the providers guard, and the job
    # board's routes are guarded by ``logs``.  One set feeds both the hide CSS
    # and the branding the client reads.
    hidden = set(disabled)
    if "providers" in hidden:
        hidden.add("shims")
    if {"logs", "observability"} & hidden:
        hidden.add("jobs")
    if hidden:
        selectors = ", ".join(f'.tab[data-tab="{t}"], #tab-{t}' for t in sorted(hidden))
        parts.append(f"<style>{selectors} {{ display: none !important; }}</style>")
    if custom_head:
        parts.append(custom_head)
    merged_branding = dict(branding) if branding else {}
    if hidden:
        merged_branding["disabled_tabs"] = sorted(hidden)
        merged_branding["all_tabs"] = sorted(_VALID_TABS)
    if merged_branding:
        import json as _json

        serialized = _json.dumps(merged_branding).replace("</", r"<\/")
        parts.append(f"<script>window.__branding={serialized};</script>")
    # Kept so a config reload can rebuild the head with a new theme
    # override without repeating the disabled-tab and branding fragments.
    app_admin_context(app).set_admin_head_prefix("\n".join(parts))
    apply_admin_theme(app, getattr(config, "admin_custom_css", ""))

    # Record admin setup event (after all wiring is complete)
    await app_admin_context(app).record_admin_setup()

"""The transport table: what this gateway ships and what a package adds.

A provider names its transport in ``connection.transport``.  The gateway
ships ``http`` (the proxy's own upstream call) and ``acp`` (a local agent
CLI over stdio).  Any other lane is supplied by an installed Python package
that publishes an entry point in :data:`TRANSPORT_ENTRY_POINT_GROUP`; the
entry-point name is the id a provider writes.

Discovery reads package metadata only.  Config validation therefore knows
every installed id without importing a plugin, and the plugin module loads
once, at app creation, where a broken one fails startup instead of a
request.

The factory contract is a callable that accepts the server's upstream
timeout as a keyword and returns an :class:`UpstreamTransport`::

    def __init__(self, *, timeout: float = 300.0) -> None: ...

When the gateway has no timeout to pass, it calls the factory with no
arguments and the factory keeps its own default.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from importlib import metadata
from typing import Any

from ._base import UpstreamTransport
from .acp import AcpTransport
from .claude_cli import ClaudeCliTransport
from .http import HttpTransport

logger = logging.getLogger("llm-rosetta-gateway")

#: Entry-point group a transport package publishes.
TRANSPORT_ENTRY_POINT_GROUP = "llm_rosetta_gateway.transports"

#: What an entry point must resolve to.
TransportFactory = Callable[..., UpstreamTransport]

#: The transports this distribution ships.  ``http`` is not a lane transport:
#: the app keeps it in ``app.transport`` so an embedder can replace it.
BUILTIN_TRANSPORTS: dict[str, TransportFactory] = {
    "http": HttpTransport,
    "acp": AcpTransport,
    "claude_cli": ClaudeCliTransport,
}


class TransportPluginError(RuntimeError):
    """An installed package cannot supply the transport it claims."""


@lru_cache(maxsize=1)
def _entry_points() -> tuple[metadata.EntryPoint, ...]:
    """The installed transport entry points, in metadata order."""
    found = metadata.entry_points()
    if hasattr(found, "select"):
        return tuple(found.select(group=TRANSPORT_ENTRY_POINT_GROUP))
    return tuple(found.get(TRANSPORT_ENTRY_POINT_GROUP, ()))


def clear_transport_cache() -> None:
    """Forget the metadata scan, so the next call sees a new install.

    Tests call this after faking the entry-point table; a long-running
    process calls it if it installs a transport package at runtime.
    """
    _entry_points.cache_clear()


def installed_transport_names() -> tuple[str, ...]:
    """Every transport id a provider may name.

    The names come from package metadata; no plugin module is imported.

    Returns:
        The built-in ids and the installed entry-point names, sorted.
    """
    names = set(BUILTIN_TRANSPORTS)
    names.update(entry_point.name for entry_point in _entry_points())
    return tuple(sorted(names))


def _plugin_source(entry_point: metadata.EntryPoint) -> str:
    """Name the distribution behind an entry point, for an error message."""
    dist = getattr(entry_point, "dist", None)
    if dist is not None and getattr(dist, "name", None):
        return f"{dist.name} {getattr(dist, 'version', '')}".strip()
    return getattr(entry_point, "value", entry_point.name)


def _build(factory: TransportFactory, timeout: float | None) -> UpstreamTransport:
    """Call a factory with the server timeout when there is one."""
    if timeout is None:
        return factory()
    return factory(timeout=timeout)


def build_lane_transports(
    *, timeout: float | None = None, override: Any | None = None
) -> dict[str, UpstreamTransport]:
    """Build the transport instance each non-HTTP lane id resolves to.

    ``http`` is deliberately absent: that slot is ``app.transport``, which a
    caller (or a test) may replace after the app is built, and a registry copy
    would keep routing to the object it replaced.

    A plugin that claims a built-in id is refused: silently shadowing the
    gateway's own transport would change what every provider of that id
    reaches.

    Args:
        timeout: The server's upstream timeout; ``None`` leaves each factory's
            own default in place.
        override: When set, every lane id maps to this object and no plugin is
            imported.  An embedder that supplies its own transport supplies it
            for every lane too: the gateway does not start local processes
            behind its back.

    Returns:
        One instance per lane id, excluding ``http``.

    Raises:
        TransportPluginError: A plugin claims a built-in id, fails to import,
            or does not resolve to a callable.
    """
    if override is not None:
        return {
            name: override for name in installed_transport_names() if name != "http"
        }

    transports: dict[str, UpstreamTransport] = {
        name: _build(factory, timeout)
        for name, factory in BUILTIN_TRANSPORTS.items()
        if name != "http"
    }
    for entry_point in _entry_points():
        name = entry_point.name
        if name in BUILTIN_TRANSPORTS:
            raise TransportPluginError(
                f"Transport package {_plugin_source(entry_point)} claims {name!r}, "
                "which this gateway already ships; refusing to let it shadow "
                "the built-in transport"
            )
        try:
            factory = entry_point.load()
        except Exception as exc:  # noqa: BLE001 - name the package that broke
            raise TransportPluginError(
                f"Transport entry point {name!r} from "
                f"{_plugin_source(entry_point)} failed to load: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(factory):
            raise TransportPluginError(
                f"Transport entry point {name!r} from "
                f"{_plugin_source(entry_point)} resolved to {factory!r}, "
                "which is not callable"
            )
        transports[name] = _build(factory, timeout)
        logger.debug("Loaded transport %r from %s", name, _plugin_source(entry_point))
    return transports

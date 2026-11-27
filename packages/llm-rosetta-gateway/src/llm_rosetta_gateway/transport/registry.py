"""The transport table: what this gateway ships and what an extension adds.

A provider names its transport in ``connection.transport``.  The gateway ships
``http`` (the proxy's own upstream call) and ``acp`` (a local agent CLI over
stdio).  Any other lane is supplied by an installed *extension*: its
``plugin.yaml`` lists the ids under ``provides.transports`` and its module
``llm_rosetta_ext.<id>.transport`` builds them.

Scope, not a global table.  A shim's transport resolves from the shim's own
extension: the extension's own transports first, then a gateway built-in, then
the transports of the extensions it lists under ``requires.extensions``
(:func:`llm_rosetta.contracts.extension.resolve_transport`).  Two extensions may declare the same
id and each shim reaches its own; an extension may not claim a built-in id; a
shim that is not part of an extension reaches the built-ins only.  Resolution
reads manifests, so config validation (:func:`reachable_transport_names`,
:func:`resolve_lane`) imports no extension code.

The factory convention.  The transport module (a module or a package) exposes::

    TRANSPORTS: dict[str, Callable[..., UpstreamTransport]]

mapping every transport id the manifest declares to a callable that accepts the
server's upstream timeout as ``timeout=`` and returns an
:class:`UpstreamTransport`::

    def __init__(self, *, timeout: float = 300.0) -> None: ...

When the gateway has no timeout to pass it calls the factory with no arguments.
:func:`build_lane_transports` imports the module once, at app creation, for each
loaded extension that a shim or a configured provider references, so a broken
module fails startup (naming the extension) instead of a request.  The instance
is keyed ``"<extension id>:<transport id>"`` (:func:`lane_key`); a built-in
keeps its bare id.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rosetta.api import (
    KindLoadError,
    get_shim,
    import_extension_module,
    list_extensions,
    register_kind_handler,
)
from llm_rosetta.contracts.extension import (
    TRANSPORT_MODULE,
    TRANSPORTS_ATTRIBUTE,
    ExtensionManifest,
    UnresolvedTransport,
    resolve_transport,
)

from ._base import UpstreamTransport
from .acp import AcpTransport
from .http import HttpTransport

logger = logging.getLogger("llm-rosetta-gateway")

#: What a ``TRANSPORTS`` entry must be.
TransportFactory = Callable[..., UpstreamTransport]

#: The transports this distribution ships.  ``http`` is not a lane transport:
#: the app keeps it in ``app.transport`` so an embedder can replace it.
BUILTIN_TRANSPORTS: dict[str, TransportFactory] = {
    "http": HttpTransport,
    "acp": AcpTransport,
}


class TransportPluginError(RuntimeError):
    """An extension cannot supply the transport it claims."""


@dataclass(frozen=True)
class LaneRef:
    """The transport one lane resolved to.

    Attributes:
        name: The ``connection.transport`` value.
        owner: The extension that provides it; ``None`` for a built-in.
    """

    name: str
    owner: str | None = None

    @property
    def key(self) -> str:
        """The key the lane's instance has in ``app.transports``."""
        return lane_key(self.name, self.owner)


def lane_key(name: str, owner: str | None = None) -> str:
    """Return the ``app.transports`` key of a transport.

    Args:
        name: The transport id.
        owner: The providing extension, or ``None`` for a built-in.

    Returns:
        ``name`` for a built-in, ``"<owner>:<name>"`` otherwise.
    """
    return name if owner is None else f"{owner}:{name}"


def _loaded() -> dict[str, tuple[ExtensionManifest, tuple[str, ...]]]:
    """The loaded extensions: id -> (manifest, shim names it registered)."""
    return {
        item.id: (item.manifest, item.shims)
        for item in list_extensions()
        if item.state == "loaded" and item.manifest is not None
    }


def loaded_manifests() -> dict[str, ExtensionManifest]:
    """Return the manifest of every extension the last load accepted."""
    return {ext_id: manifest for ext_id, (manifest, _) in _loaded().items()}


def owning_extension(shim_name: str | None) -> str | None:
    """Return the extension that registered a shim, or ``None``.

    Built-in shims and shims from a directory source have no owner.
    """
    if not shim_name:
        return None
    for ext_id, (_, shims) in _loaded().items():
        if shim_name in shims:
            return ext_id
    return None


def _claim_check(manifest: ExtensionManifest, folder: Path) -> list[object]:
    """The ``transports`` kind handler: refuse an extension that claims a built-in id.

    The extension fails at load time, so it never reaches the lane table and its
    shims are not registered.  Building a transport stays in
    :func:`build_lane_transports`.
    """
    claimed = sorted(set(manifest.transports) & set(BUILTIN_TRANSPORTS))
    if claimed:
        raise KindLoadError(
            f"claims transport {', '.join(map(repr, claimed))}, which this gateway "
            "already ships; refusing to let it shadow the built-in transport"
        )
    return []


register_kind_handler("transports", _claim_check)


def resolve_lane(name: str, shim_name: str | None = None) -> LaneRef:
    """Resolve a ``connection.transport`` value from the shim's own scope.

    Args:
        name: The transport id the shim (or the provider's config) names.
        shim_name: The shim; its extension is the scope.  ``None`` or a shim
            that is not part of an extension reaches the built-ins only.

    Returns:
        The lane reference.

    Raises:
        UnresolvedTransport: Nothing the shim's extension may reach provides
            *name*.
    """
    manifests = loaded_manifests()
    owner_id = owning_extension(shim_name)
    if owner_id is None:
        if name in BUILTIN_TRANSPORTS:
            return LaneRef(name)
        raise UnresolvedTransport(
            f"transport {name!r} is not one this gateway ships "
            f"({', '.join(sorted(BUILTIN_TRANSPORTS))}); only the shims of an "
            "extension can name an extension's transport"
        )
    owner = resolve_transport(name, manifests[owner_id], manifests, BUILTIN_TRANSPORTS)
    return LaneRef(name, owner.extension)


def reachable_transport_names(shim_name: str | None = None) -> tuple[str, ...]:
    """Return the transport ids a shim could name, from manifests only.

    No extension code is imported.

    Args:
        shim_name: The shim whose scope is asked for; ``None`` asks for the
            built-ins alone.

    Returns:
        The built-in ids, the shim's extension's own ids and the ids of the
        extensions it requires (when loaded), sorted.
    """
    names = set(BUILTIN_TRANSPORTS)
    manifests = loaded_manifests()
    owner_id = owning_extension(shim_name)
    if owner_id is not None:
        manifest = manifests[owner_id]
        names.update(manifest.transports)
        for dependency in manifest.requires.extensions:
            if dependency in manifests:
                names.update(manifests[dependency].transports)
    return tuple(sorted(names))


def _build(
    factory: TransportFactory, timeout: float | None, *, label: str
) -> UpstreamTransport:
    """Call a factory with the server timeout when there is one.

    A factory that cannot start (a constructor that does not take ``timeout=``,
    a missing program) is reported with the transport it belongs to, so the
    operator is not left with a bare ``TypeError`` from a plugin.
    """
    try:
        if timeout is None:
            return factory()
        return factory(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - name the transport that broke
        raise TransportPluginError(
            f"Transport {label} failed to start: {type(exc).__name__}: {exc}"
        ) from exc


def _factory(ext_id: str, name: str) -> TransportFactory:
    """Import ``llm_rosetta_ext.<id>.transport`` and return the factory for *name*."""
    module_name = f"llm_rosetta_ext.{ext_id}.{TRANSPORT_MODULE}"
    try:
        module = import_extension_module(ext_id, TRANSPORT_MODULE)
    except Exception as exc:  # noqa: BLE001 - name the extension that broke
        raise TransportPluginError(
            f"Extension {ext_id!r}: transport module {module_name} failed to "
            f"import: {type(exc).__name__}: {exc}"
        ) from exc
    table = getattr(module, TRANSPORTS_ATTRIBUTE, None)
    if not isinstance(table, Mapping):
        raise TransportPluginError(
            f"Extension {ext_id!r}: {module_name} does not expose a "
            f"{TRANSPORTS_ATTRIBUTE} mapping of transport id to factory"
        )
    if name not in table:
        raise TransportPluginError(
            f"Extension {ext_id!r} declares transport {name!r}, but "
            f"{module_name}.{TRANSPORTS_ATTRIBUTE} has no entry for it"
        )
    factory = table[name]
    if not callable(factory):
        raise TransportPluginError(
            f"Extension {ext_id!r}: transport {name!r} resolved to {factory!r}, "
            "which is not callable"
        )
    return factory


def _referenced_by_shims(
    loaded: Mapping[str, tuple[ExtensionManifest, tuple[str, ...]]],
) -> set[tuple[str, str]]:
    """The (extension, transport) pairs the loaded shims name by default."""
    manifests = {ext_id: manifest for ext_id, (manifest, _) in loaded.items()}
    found: set[tuple[str, str]] = set()
    for ext_id, (manifest, shim_names) in loaded.items():
        for shim_name in shim_names:
            shim = get_shim(shim_name)
            name = shim.connection.transport if shim is not None else "http"
            if name in BUILTIN_TRANSPORTS:
                continue
            try:
                owner = resolve_transport(name, manifest, manifests, BUILTIN_TRANSPORTS)
            except UnresolvedTransport as exc:
                logger.warning("Shim %r: %s", shim_name, exc)
                continue
            if owner.extension is not None:
                found.add((owner.extension, name))
    return found


def build_lane_transports(
    *,
    timeout: float | None = None,
    override: Any | None = None,
    wanted: Iterable[tuple[str, str]] = (),
) -> dict[str, UpstreamTransport]:
    """Build the transport instance each non-HTTP lane resolves to.

    ``http`` is deliberately absent: that slot is ``app.transport``, which a
    caller (or a test) may replace after the app is built, and a registry copy
    would keep routing to the object it replaced.

    One instance exists per (extension id, transport id).  An extension's
    transport module is imported only when the extension is loaded and a loaded
    shim of it (or a dependent's), or an entry of *wanted*, names the transport.

    Args:
        timeout: The server's upstream timeout; ``None`` leaves each factory's
            own default in place.
        override: When set, every lane key maps to this object and no extension
            code is imported.  An embedder that supplies its own transport
            supplies it for every lane too: the gateway does not start local
            processes behind its back.
        wanted: Further ``(extension id, transport id)`` pairs to build, such
            as the ones the configured providers resolved to.

    Returns:
        One instance per lane key, excluding ``http``.

    Raises:
        TransportPluginError: An extension fails to
            import its transport module, or does not expose a callable factory
            for a transport it declares.
    """
    loaded = _loaded()
    manifests = {ext_id: manifest for ext_id, (manifest, _) in loaded.items()}
    if override is not None:
        keys = [name for name in BUILTIN_TRANSPORTS if name != "http"]
        keys += [
            lane_key(name, ext_id)
            for ext_id, manifest in manifests.items()
            for name in manifest.transports
        ]
        return {key: override for key in keys}

    transports: dict[str, UpstreamTransport] = {
        name: _build(factory, timeout, label=repr(name))
        for name, factory in BUILTIN_TRANSPORTS.items()
        if name != "http"
    }
    pairs = _referenced_by_shims(loaded) | {
        (ext_id, name) for ext_id, name in wanted if ext_id in manifests
    }
    for ext_id, name in sorted(pairs):
        if name not in manifests[ext_id].transports:
            continue
        transports[lane_key(name, ext_id)] = _build(
            _factory(ext_id, name), timeout, label=f"{name!r} of extension {ext_id!r}"
        )
        logger.debug("Built transport %r of extension %r", name, ext_id)
    return transports

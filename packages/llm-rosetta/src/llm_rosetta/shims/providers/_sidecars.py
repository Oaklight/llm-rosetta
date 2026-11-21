"""Sidecar Python modules beside a ``provider.yaml``: plugin namespaces, transforms, auth, quota."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


from ...contracts import (
    AUTH_EXCHANGE,
    IR_TRANSFORMS,
    POST_IR_TRANSFORMS,
    PRE_IR_TRANSFORMS,
    QUOTA_PROBES,
    RESPONSE_BODY_TRANSFORMS,
    TRANSFORMS_MODULE,
)
from ..provider_shim import (
    Transform,
)
from ._constants import (
    logger,
)


# Synthetic package namespace for plugin (non-builtin) shim transforms.  The
# namespace is keyed per providers root so two external sources with the same
# group/leaf names do not collide in ``sys.modules``.
_PLUGIN_NS_PREFIX = "_llm_rosetta_plugin_shims"
_plugin_namespaces: dict[str, list[str]] = {}


def _ensure_pkg(name: str, dir_path: Path) -> None:
    """Register *name* as a package whose ``__path__`` points at *dir_path*.

    External shim transforms are executed by path rather than imported as part
    of a real package, so a relative import such as ``from .helpers import x``
    would otherwise raise ``ModuleNotFoundError``.  Giving each directory level
    a synthetic package with a ``__path__`` makes relative imports resolve
    through the normal path finder.  (A *bare* ``import helpers`` still fails —
    the directory is not on ``sys.path`` — so plugin transforms must use
    relative or fully-qualified plugin imports.)
    """
    locs = _plugin_namespaces.get(name)
    if locs is None:
        locs = []
        _plugin_namespaces[name] = locs
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = locs
        mod = types.ModuleType(name)
        mod.__spec__ = spec
        mod.__package__ = name
        sys.modules[name] = mod
    if str(dir_path) not in locs:
        locs.append(str(dir_path))
    sys.modules[name].__path__ = locs


def _clear_plugin_namespaces() -> None:
    """Drop the synthetic plugin packages and their submodules (on reload)."""
    for name in [m for m in sys.modules if m.startswith(_PLUGIN_NS_PREFIX)]:
        sys.modules.pop(name, None)
    _plugin_namespaces.clear()


def _register_plugin_chain(
    provider_dir: Path, group: str | None, root: Path | None
) -> str:
    """Register the synthetic package chain for a plugin shim's transforms.

    The namespace is keyed on the *resolved* providers root so the same
    directory reached by different paths (relative, symlink, …) shares one
    namespace.

    Returns:
        The dotted name to load ``transforms.py`` under.
    """
    if root is None:
        # Walk up past the group directories so the namespace is keyed on the
        # providers root even when the caller does not pass one.
        depth = len(group.split(".")) if group else 0
        root = provider_dir.parents[depth]
    resolved = Path(root).resolve()
    ns = f"{_PLUGIN_NS_PREFIX}_{hashlib.sha1(str(resolved).encode()).hexdigest()[:8]}"
    cur, cur_dir = ns, resolved
    chain = [(ns, resolved)]
    for part in group.split(".") if group else []:
        cur_dir = cur_dir / part
        cur = f"{cur}.{part}"
        chain.append((cur, cur_dir))
    leaf = f"{cur}.{provider_dir.name}"
    chain.append((leaf, provider_dir.resolve()))
    for name, path in chain:
        _ensure_pkg(name, path)
    return f"{leaf}.transforms"


def _rollback_plugin_load(module_name: str, before: frozenset[str]) -> None:
    """Undo a failed load, removing only what *this* load registered.

    Drops the half-initialised module plus any package/submodule it imported
    before raising, but keeps modules an earlier successful load of the same
    root registered: the namespace is shared per root, so a blanket namespace
    wipe would silently evict a working shim's modules.  The
    ``_plugin_namespaces`` bookkeeping is pruned for whatever was removed, so
    the next ``_ensure_pkg`` rebuilds instead of raising ``KeyError``.
    """
    for name in [
        m
        for m in sys.modules
        if m not in before and (m == module_name or m.startswith(_PLUGIN_NS_PREFIX))
    ]:
        sys.modules.pop(name, None)
    for pkg_name in [k for k in _plugin_namespaces if k not in sys.modules]:
        _plugin_namespaces.pop(pkg_name, None)


def _load_quota_probes(path: Path) -> tuple[Any, ...]:
    """Execute a shim's ``quota.py`` and return the probes it exports.

    A shim that serves an upstream allowance ships the probe beside its
    ``provider.yaml`` (group-level for a vendor spanning several formats,
    leaf-level for a one-off).  The module may import the standard library and
    :mod:`llm_rosetta.contracts.quota` at module scope; it must not import
    ``gateway`` — the shim layer loads before the gateway, and the core
    boundary forbids it.

    Args:
        path: The ``quota.py`` to execute.  A missing file yields ``()``.

    Returns:
        The module's ``PROBES`` (a tuple, or a mapping of id → probe).
    """
    if not path.exists():
        return ()
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    module_name = f"_llm_rosetta_shim_quota_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load %s", path)
        return ()
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    probes = getattr(mod, QUOTA_PROBES, ())
    if isinstance(probes, dict):
        probes = tuple(probes.values())
    return tuple(probes)


def _load_auth_exchange(path: Path) -> Any | None:
    """Execute a shim's ``auth.py`` and return the object it exports.

    A shim whose credential is more than one argv ships the exchange beside
    its ``provider.yaml``: the OAuth profile needs an interactive first run, a
    rotating refresh token and a stored record, and only the vendor knows how
    those work.  The module may import the standard library and
    :mod:`llm_rosetta.contracts.auth` at module scope; it must not import
    ``gateway`` — the shim layer loads before the gateway, and the core
    boundary forbids it.

    Args:
        path: The ``auth.py`` to execute.  A missing file yields ``None``.

    Returns:
        The module's ``EXCHANGE`` object, or ``None`` when the shim declares
        no exchange.

    Raises:
        ValueError: The module exports an object without ``credential()``.
        Exception: Whatever the module raised while executing (the caller
            decides whether a broken shim is fatal).
    """
    if not path.exists():
        return None
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    module_name = f"_llm_rosetta_shim_auth_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    exchange = getattr(mod, AUTH_EXCHANGE, None)
    if exchange is None:
        return None
    if not callable(getattr(exchange, "credential", None)):
        raise ValueError(
            f"{path}: {AUTH_EXCHANGE} must implement credential(); "
            "see llm_rosetta.contracts.auth.AuthExchange"
        )
    return exchange


def _load_transforms(
    provider_dir: Path,
    *,
    group: str | None = None,
    _builtin: bool = True,
    root: Path | None = None,
) -> tuple[tuple, tuple, tuple, tuple, Any]:
    """Import transforms.py if present, return (pre, post, ir, response_body, module).

    Reads ``pre_ir_transforms``, ``post_ir_transforms``, ``ir_transforms`` and
    ``response_body_transforms`` from the transforms module.

    The fourth element is the loaded module object (or ``None`` when no
    ``transforms.py`` exists).  Callers can inspect it for convention-based
    hooks such as ``model_list_transform``.

    Args:
        provider_dir: Path to the leaf provider directory.
        group: Name of the parent group folder, if this is a grouped shim.
        _builtin: Whether this is a built-in provider directory.  Plugin
            transforms use a separate module namespace to avoid collisions
            with built-in modules in ``sys.modules``.
    """
    tf_path = provider_dir / TRANSFORMS_MODULE
    if not tf_path.exists():
        return (), (), (), (), None
    # Snapshot before the synthetic packages are registered: the rollback only
    # undoes what this load added, and those packages count as added.
    before = frozenset(sys.modules)
    if _builtin:
        prefix = "llm_rosetta.shims.providers"
        module_name = (
            f"{prefix}.{group}.{provider_dir.name}.transforms"
            if group is not None
            else f"{prefix}.{provider_dir.name}.transforms"
        )
    else:
        module_name = _register_plugin_chain(provider_dir, group, root)
    spec = importlib.util.spec_from_file_location(module_name, tf_path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load %s", tf_path)
        return (), (), (), (), None
    mod = importlib.util.module_from_spec(spec)
    if not _builtin:
        # Built-in transforms resolve through their real package, so they need
        # no sys.modules entry; only plugin loads register one (and roll it back
        # on failure).
        sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
        pre, post, ir, response_body = _read_transform_exports(mod, tf_path)
    except BaseException:
        _rollback_plugin_load(module_name, before)
        raise
    return (pre, post, ir, response_body, mod)


def _declared_transforms(
    mod: Any, name: str, tf_path: Path
) -> tuple[Transform, ...] | None:
    """Return one export of a transforms module as a tuple, or ``None``.

    The pipeline iterates these exports, so a module that exports a *function*
    instead loads cleanly and then fails on the first request.  Refusing the
    shape here puts the error beside the file that has it.
    """
    value = getattr(mod, name, None)
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"{tf_path}: {name} must be a tuple of transforms, got "
            f"{type(value).__name__}"
        )
    return tuple(value)


def _read_transform_exports(mod: Any, tf_path: Path) -> tuple:
    """Return ``(pre, post, ir, response_body)`` from a transforms module.

    ``is None`` rather than truthiness, because an empty tuple is a valid
    declaration.  The 0.13 names ``from_transforms`` and ``to_transforms``
    were removed in 0.14.0; a module that still exports one is refused rather
    than loaded with its transforms dropped.

    Raises:
        TypeError: The module exports a removed name, or an export is not a
            tuple of transforms.
    """
    for removed, replacement in (
        ("from_transforms", PRE_IR_TRANSFORMS),
        ("to_transforms", POST_IR_TRANSFORMS),
    ):
        if hasattr(mod, removed):
            raise TypeError(
                f"{tf_path}: {removed} was removed in 0.14.0; "
                f"rename it to {replacement}"
            )
    pre = _declared_transforms(mod, PRE_IR_TRANSFORMS, tf_path) or ()
    post = _declared_transforms(mod, POST_IR_TRANSFORMS, tf_path) or ()
    return (
        pre,
        post,
        _declared_transforms(mod, IR_TRANSFORMS, tf_path) or (),
        _declared_transforms(mod, RESPONSE_BODY_TRANSFORMS, tf_path) or (),
    )

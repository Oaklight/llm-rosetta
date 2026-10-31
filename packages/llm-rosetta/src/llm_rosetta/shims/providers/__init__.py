"""Provider shim loading — built-in directory scan + plugin entry points.

Shim lifecycle
--------------
**Registration** (startup, once):

1. ``load_providers()`` scans the built-in ``providers/`` directory via
   ``load_providers_from_dir()``, registering each shim found.
2. It then discovers ``llm_rosetta.shim_providers`` entry points and
   calls each plugin callable, which may register additional shims.

**Usage** (per request):

- ``get_shim(name)`` looks up a registered shim by name.
- The gateway / ``convert()`` injects the shim's reasoning config
  and applies transforms around the converter.

Directory layout
----------------
Each subdirectory that contains a ``provider.yaml`` is treated as a leaf
provider definition.  An optional ``transforms.py`` alongside the YAML
may export ``pre_ir_transforms`` and/or ``post_ir_transforms`` tuples
(legacy names ``from_transforms`` / ``to_transforms`` are also accepted).

**Grouped directories** are also supported: a child directory that does
NOT contain ``provider.yaml`` but DOES contain subdirectories with one is
treated as a *group folder* (e.g. ``vendor/messages/``, ``vendor/chat_completions/``).

**Model-type directories** name a category, using the endpoint spelling:
``llm/``, ``embeddings/``, ``rerank/`` and ``decisions/``.  Under one, either
the directory itself holds the leaf (``vendor/embeddings/provider.yaml``, the
usual non-chat shape) or it groups format directories
(``vendor/llm/chat_completions/provider.yaml``, because a chat vendor serves
several dialects).  A shim filed under a model-type directory must declare the
matching ``model_type``; a mismatch is refused at load time.

Plugin shims
------------
Downstream packages register shims via entry points::

    # pyproject.toml
    [project.entry-points."llm_rosetta.shim_providers"]
    my_provider = "my_package.shims:register_shims"

The callable receives no arguments.  Most plugins simply call
``load_providers_from_dir()`` to scan their own YAML directory::

    from pathlib import Path
    from llm_rosetta.shims import load_providers_from_dir

    def register_shims():
        return load_providers_from_dir(Path(__file__).parent / "providers")

For advanced use cases (conditional registration, dynamic shims),
call ``register_shim()`` directly instead of scanning a directory.
The callable may optionally return ``list[ProviderShim]`` for inclusion
in ``load_providers()``'s combined result.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import logging
import sys
import types
from importlib.metadata import entry_points
from pathlib import Path
from collections.abc import Callable
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load

from ..provider_shim import (
    ConnectionConfig,
    ProviderShim,
    QuotaDeclaration,
    ReasoningCapability,
    SoftErrorPattern,
    ToolsConfig,
    register_shim,
)

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent

#: Model-type directory names, keyed to the registry category they name.
#: The spelling follows the endpoint a type serves (``/v1/embeddings``,
#: ``/v1/rerank``, ``/v1/decisions``) so a path reads like the route it
#: configures; ``llm`` keeps its registry name because chat has no single
#: endpoint.  A plural type name cannot collide with a format name
#: (``embedding`` is a format; ``embeddings`` is not), so legacy layouts that
#: use the format name as a directory keep loading unchanged.
_TYPE_DIRS: dict[str, str] = {
    "llm": "llm",
    "embeddings": "embedding",
    "rerank": "rerank",
    "decisions": "decision",
}

# Convention-based model list transforms registered by shim transforms modules.
# Each entry maps a shim name to a callable that converts upstream model
# entries into (display_ids, upstream_map).
ModelListTransform = Callable[[list[dict[str, Any]]], tuple[list[str], dict[str, str]]]

_model_list_transforms: dict[str, ModelListTransform] = {}

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


def get_model_list_transform(
    shim_name: str,
) -> ModelListTransform | None:
    """Return the model_list_transform for *shim_name*, or ``None``.

    Transforms are registered under the canonical shim name; a legacy spelling
    (e.g. ``"kilo--chat_completions"``) is normalised before lookup so both resolve
    to the same hook.
    """
    transform = _model_list_transforms.get(shim_name)
    if transform is not None:
        return transform
    from llm_rosetta.provider_names import (
        canonical_provider_name,
        canonical_shim_name,
    )

    canonical = canonical_shim_name(canonical_provider_name(shim_name))
    if canonical == shim_name:
        return None
    return _model_list_transforms.get(canonical)


def register_model_list_transform(name: str, transform: ModelListTransform) -> None:
    """Register a model_list_transform for providers without a shim directory."""
    _model_list_transforms[name] = transform


# --------------- Built-in model list transforms ---------------


def _jina_model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Strip the ``jina-ai/`` org prefix from Jina model IDs."""
    ids: list[str] = []
    upstream_map: dict[str, str] = {}
    for m in entries:
        raw_id = m.get("id", "")
        if not raw_id:
            continue
        display = raw_id.removeprefix("jina-ai/")
        ids.append(display)
        if display != raw_id:
            upstream_map[display] = raw_id
    return ids, upstream_map


def _cohere_model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Parse Cohere's ``models[].name`` format."""
    ids: list[str] = []
    for m in entries:
        name = m.get("name", "")
        if name:
            ids.append(name)
    return ids, {}


def _parse_reasoning_cap(
    cfg: dict,
    *,
    base: ReasoningCapability | None = None,
) -> ReasoningCapability:
    """Parse a reasoning config dict into a :class:`ReasoningCapability`.

    When *base* is provided (model_overrides), unset fields inherit from it.
    """
    _d = base  # shorthand for fallback defaults

    # effort_range: YAML gives [floor, ceiling] list → tuple
    raw_range = cfg.get("effort_range", _d.effort_range if _d else None)
    effort_range = tuple(raw_range) if isinstance(raw_range, list) else raw_range

    return ReasoningCapability(
        thinking_modes=cfg.get("thinking_modes", _d.thinking_modes if _d else None),
        thinking_default=cfg.get(
            "thinking_default", _d.thinking_default if _d else None
        ),
        effort_field=cfg.get(
            "effort_field", _d.effort_field if _d else "reasoning_effort"
        ),
        effort_range=effort_range,
        budget_ratio=cfg.get("budget_ratio", _d.budget_ratio if _d else None),
        visibility_modes=cfg.get(
            "visibility_modes", _d.visibility_modes if _d else None
        ),
        replay_missing=cfg.get("replay_missing", _d.replay_missing if _d else "as_is"),
        unsigned_blocks=cfg.get(
            "unsigned_blocks", _d.unsigned_blocks if _d else "as_is"
        ),
    )


def _load_quota_probes(path: Path) -> tuple[Any, ...]:
    """Execute a shim's ``quota.py`` and return the probes it exports.

    A shim that serves an upstream allowance ships the probe beside its
    ``provider.yaml`` (group-level for a vendor spanning several formats,
    leaf-level for a one-off).  The module may import the standard library and
    :mod:`llm_rosetta.shims.quota_probe` at module scope; it must not import
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
    probes = getattr(mod, "PROBES", ())
    if isinstance(probes, dict):
        probes = tuple(probes.values())
    return tuple(probes)


def _load_auth_exchange(path: Path) -> Any | None:
    """Execute a shim's ``auth.py`` and return the object it exports.

    A shim whose credential is more than one argv ships the exchange beside
    its ``provider.yaml``: the OAuth profile needs an interactive first run, a
    rotating refresh token and a stored record, and only the vendor knows how
    those work.  The module may import the standard library and
    :mod:`llm_rosetta.shims.auth_exchange` at module scope; it must not import
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
    exchange = getattr(mod, "EXCHANGE", None)
    if exchange is None:
        return None
    if not callable(getattr(exchange, "credential", None)):
        raise ValueError(
            f"{path}: EXCHANGE must implement credential(); "
            "see llm_rosetta.shims.auth_exchange.AuthExchange"
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

    Accepts both new names (``pre_ir_transforms``, ``post_ir_transforms``)
    and legacy names (``from_transforms``, ``to_transforms``) from the
    transforms module.  New names take precedence if both are present.

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
    tf_path = provider_dir / "transforms.py"
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
    except BaseException:
        _rollback_plugin_load(module_name, before)
        raise
    # New names take precedence; fall back to legacy names.
    # Use `is None` (not `or`) since empty tuple () is falsy but valid.
    pre = getattr(mod, "pre_ir_transforms", None)
    if pre is None:
        pre = getattr(mod, "from_transforms", ())
        if pre and hasattr(mod, "from_transforms"):
            logger.info(
                "%s uses deprecated 'from_transforms'; rename to 'pre_ir_transforms'",
                tf_path,
            )
    post = getattr(mod, "post_ir_transforms", None)
    if post is None:
        post = getattr(mod, "to_transforms", ())
        if post and hasattr(mod, "to_transforms"):
            logger.info(
                "%s uses deprecated 'to_transforms'; rename to 'post_ir_transforms'",
                tf_path,
            )
    return (
        pre,
        post,
        getattr(mod, "ir_transforms", ()),
        getattr(mod, "response_body_transforms", ()),
        mod,
    )


def _resolve_shim_pair(cfg: dict[str, Any], yaml_path: Path) -> tuple[str, str]:
    """Return the canonical ``(format, model_type)`` a shim declares.

    ``format`` is the preferred key; ``base`` is its deprecated alias.  When
    ``base`` names a model category (the old decision shims declared
    ``base: decision``), the category is taken from it and a ``format`` is
    still required.  The pair is validated against the converter registry so
    a shim cannot claim a format another category owns.

    Raises:
        ValueError: With a message naming the shim file and the expected
            formats.
    """
    # Imported lazily: the converter package imports shims.provider_shim at
    # module scope (for ReasoningCapability), so a module-level import here
    # would close the cycle while the shims package is initialising.  The
    # vocabulary module is data-only and safe to import at any time.
    from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE, MODEL_TYPES
    from llm_rosetta.provider_names import canonical_provider_name

    declared_type = cfg.get("model_type")
    legacy_base = cfg.get("base")
    if not isinstance(legacy_base, str):
        legacy_base = None

    # ``format`` is the new spelling and the schema requires the category with
    # it: without ``model_type`` the pair would silently default to ``llm``, so
    # a future embedding shim that forgot the field would load as a chat shim.
    # A shim that carries ``base`` as well is mid-migration, not a legacy
    # shim: only a declaration with ``base`` and no ``format`` may omit the
    # category (the base name then infers it, else defaults to ``llm``).
    if declared_type is None and cfg.get("format") is not None:
        raise ValueError(
            f"{yaml_path}: model_type is required when 'format' is declared "
            f"(one of {', '.join(MODEL_TYPES)}); a legacy shim may use "
            f"'base' instead"
        )

    model_type = declared_type or "llm"
    if declared_type is None and legacy_base in MODEL_TYPES:
        # Old-style paradigm shim: ``base`` named the category.
        model_type = legacy_base

    raw_format = cfg.get("format")
    if raw_format is None:
        if legacy_base in MODEL_TYPES:
            raise ValueError(
                f"{yaml_path}: base={legacy_base!r} names a model type; "
                f"add a 'format' (available: "
                f"{', '.join(FORMATS_BY_TYPE[legacy_base])})"
            )
        raw_format = legacy_base
    if not isinstance(raw_format, str):
        raise ValueError(f"{yaml_path}: a shim needs 'format' (or the legacy 'base')")
    if legacy_base in MODEL_TYPES and declared_type not in (None, legacy_base):
        raise ValueError(
            f"{yaml_path}: base={legacy_base!r} names a model type but "
            f"model_type={declared_type!r} disagrees"
        )

    fmt = canonical_provider_name(raw_format)
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"{yaml_path}: unknown model_type {model_type!r}; "
            f"expected one of {', '.join(MODEL_TYPES)}"
        )
    if fmt not in FORMATS_BY_TYPE[model_type]:
        raise ValueError(
            f"{yaml_path}: format {fmt!r} is not a {model_type} format; "
            f"available: {', '.join(FORMATS_BY_TYPE[model_type])}"
        )
    return fmt, model_type


def _parse_connection_token_command(
    raw: Any, provider_dir: Path, name: str
) -> tuple[str, ...] | None:
    """Parse ``connection.token_command`` from a shim declaration.

    A relative argument that names a file in the shim's own directory is
    rewritten to its absolute path, so a shim can ship the exchange beside its
    ``provider.yaml``; anything else (``python3``, a flag, an absolute path) is
    left for the gateway, which resolves bare names on ``PATH``.

    A malformed declaration is ignored with a warning rather than rejected:
    the shim's other facts are still usable, and a broken command must not
    stop the gateway from loading an otherwise valid config.
    """
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(part, str) and part for part in raw)
    ):
        logger.warning(
            "Shim %s: connection.token_command must be a non-empty list of "
            "strings; ignoring it",
            name,
        )
        return None
    # Resolve every argument that names a file in the shim's directory — both
    # ``[token.py]`` and the more common ``[python3, token.py]`` ship a script
    # beside the declaration.  A bare binary name (``python3``) and any flag
    # are left untouched, since no such file exists here.
    argv = []
    for part in raw:
        candidate = provider_dir / part
        argv.append(
            str(candidate.resolve())
            if not Path(part).is_absolute() and candidate.is_file()
            else part
        )
    return tuple(argv)


def _load_single_provider(
    provider_dir: Path,
    *,
    group: str | None = None,
    _builtin: bool = True,
    root: Path | None = None,
    quota_probes: tuple[Any, ...] = (),
    expected_type: str | None = None,
) -> ProviderShim | None:
    """Load a single provider from *provider_dir* and register it.

    Args:
        provider_dir: Directory containing ``provider.yaml``.
        group: Name of the parent group folder (``None`` for top-level shims).
        _builtin: Whether this is a built-in provider directory.
        expected_type: The model type the enclosing model-type directory
            names.  When set, the shim's own ``model_type`` must match it.

    Returns:
        The registered :class:`ProviderShim`, or ``None`` on failure.
    """
    yaml_path = provider_dir / "provider.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml_load(f.read())
    if (
        not isinstance(cfg, dict)
        or "name" not in cfg
        or ("format" not in cfg and "base" not in cfg)
    ):
        logger.warning(
            "Skipping %s: missing 'name' or 'format' (legacy 'base')", yaml_path
        )
        return None

    fmt, model_type = _resolve_shim_pair(cfg, yaml_path)
    if expected_type is not None and model_type != expected_type:
        raise ValueError(
            f"{yaml_path}: directory {provider_dir.name!r} is the "
            f"{expected_type!r} model-type level, but this shim declares "
            f"model_type={model_type!r}; move it under the {model_type!r} "
            f"directory or correct the declaration"
        )

    pre_t, post_t, ir_t, resp_t, transforms_mod = _load_transforms(
        provider_dir, group=group, _builtin=_builtin, root=root
    )

    # A leaf-level quota.py complements the group's: the group module carries
    # a vendor's shared lanes, a leaf module a one-off.  Both are attached to
    # every shim in the leaf so a config can select either by probe id.
    leaf_quota_probes = _load_quota_probes(provider_dir / "quota.py")
    all_quota_probes = tuple(quota_probes) + leaf_quota_probes

    # A credential exchange that does not fit in an argv.  It is the module
    # form of the same declaration as ``connection.token_command``, so a shim
    # that ships both leaves the loader with no single answer: reject it here
    # rather than let one silently shadow the other.
    exchange = _load_auth_exchange(provider_dir / "auth.py")

    # Parse optional reasoning capability config from YAML.
    reasoning_cfg = cfg.get("reasoning")
    reasoning_cap: ReasoningCapability | None = None
    if isinstance(reasoning_cfg, dict):
        reasoning_cap = _parse_reasoning_cap(reasoning_cfg)

    # Parse per-model reasoning overrides (inherit provider defaults).
    model_reasoning: dict[str, ReasoningCapability] | None = None
    if isinstance(reasoning_cfg, dict) and isinstance(
        reasoning_cfg.get("model_overrides"), dict
    ):
        model_reasoning = {}
        for model_id, overrides in reasoning_cfg["model_overrides"].items():
            if not isinstance(overrides, dict):
                continue
            assert reasoning_cap is not None  # model_overrides requires reasoning
            model_reasoning[model_id] = _parse_reasoning_cap(
                overrides, base=reasoning_cap
            )

    # Parse connection config (grouped or flat fallback).
    conn_raw = cfg.get("connection")
    if isinstance(conn_raw, dict):
        connection = ConnectionConfig(
            base_url=conn_raw.get("base_url"),
            api_key_env=conn_raw.get("api_key_env"),
            models_path=conn_raw.get("models_path"),
            default_path=conn_raw.get("default_path"),
            model_id_field=conn_raw.get("model_id_field"),
            auth_header=conn_raw.get("auth_header"),
            keyless=conn_raw.get("keyless", False),
            session_header=conn_raw.get("session_header"),
            headers=conn_raw.get("headers"),
            request_header=conn_raw.get("request_header"),
            session_id_format=conn_raw.get("session_id_format"),
            request_id_format=conn_raw.get("request_id_format"),
            stream_required=bool(conn_raw.get("stream_required", False)),
            token_command=_parse_connection_token_command(
                conn_raw.get("token_command"), provider_dir, cfg.get("name", "")
            ),
            token_refresh_interval=conn_raw.get("token_refresh_interval"),
            token_command_env=conn_raw.get("token_command_env"),
            manage_usage_url=conn_raw.get("manage_usage_url"),
        )
    else:
        connection = ConnectionConfig(
            base_url=cfg.get("default_base_url"),
            api_key_env=cfg.get("default_api_key_env"),
            models_path=cfg.get("models_path"),
            default_path=cfg.get("default_path"),
            model_id_field=cfg.get("model_id_field"),
            keyless=cfg.get("keyless", False),
            session_header=cfg.get("session_header"),
            headers=cfg.get("headers"),
            request_header=cfg.get("request_header"),
            session_id_format=cfg.get("session_id_format"),
            request_id_format=cfg.get("request_id_format"),
            stream_required=bool(cfg.get("stream_required", False)),
            token_command=_parse_connection_token_command(
                cfg.get("token_command"), provider_dir, cfg.get("name", "")
            ),
            token_refresh_interval=cfg.get("token_refresh_interval"),
            token_command_env=cfg.get("token_command_env"),
            manage_usage_url=cfg.get("manage_usage_url"),
        )

    if exchange is not None and connection.token_command:
        raise ValueError(
            f"{yaml_path}: declares both auth.py and connection.token_command; "
            f"a shim ships one form of its credential exchange"
        )

    # Parse the quota declaration: which probe reads this lane, and how often
    # the gateway should read it.  Parsers live in the gateway; the shim only
    # names one.
    quota_raw = cfg.get("quota")
    quota_decl: QuotaDeclaration | None = None
    if isinstance(quota_raw, dict):
        quota_decl = QuotaDeclaration(
            probe=quota_raw.get("probe"),
            url=quota_raw.get("url"),
            interval=int(quota_raw.get("interval", 0) or 0),
            demote_at=float(quota_raw.get("demote_at", 100.0)),
            in_band=bool(quota_raw.get("in_band", False)),
        )

    # Parse tools config (grouped or flat fallback).
    tools_raw = cfg.get("tools")
    if isinstance(tools_raw, dict):
        tools = ToolsConfig(
            custom_tools=tools_raw.get("custom_tools", False),
            max_description_length=tools_raw.get("max_description_length"),
            search_mode=tools_raw.get("search_mode", "disabled"),
            multimodal_result=tools_raw.get("multimodal_result"),
            tool_groups=tools_raw.get("tool_groups", False),
            intrinsic_tools=tuple(tools_raw.get("intrinsic_tools") or ()),
        )
    else:
        tools = ToolsConfig(
            custom_tools=cfg.get("supports_custom_tools", False),
            max_description_length=cfg.get("max_tool_description_length"),
            search_mode=cfg.get("tool_search_mode", "disabled"),
            multimodal_result=cfg.get("multimodal_tool_result"),
            tool_groups=cfg.get("tool_groups", False),
            intrinsic_tools=tuple(cfg.get("intrinsic_tools") or ()),
        )

    # Parse soft-error patterns for 200-but-error detection.
    raw_patterns = cfg.get("soft_error_patterns", [])
    _soft_list: list[SoftErrorPattern] = []
    for p in raw_patterns:
        if not isinstance(p, dict):
            logger.warning("Skipping non-dict soft_error_pattern in %s", yaml_path)
            continue
        sep = SoftErrorPattern(
            pattern=p["pattern"],
            status_code=p["status_code"],
            message=p["message"],
        )
        sep.compiled  # eagerly validate regex at load time
        _soft_list.append(sep)
    soft_errors = tuple(_soft_list)

    shim = ProviderShim(
        name=cfg["name"],
        format=fmt,
        model_type=model_type,
        logo=cfg.get("logo"),
        display_name=cfg.get("display_name"),
        group=cfg.get("group"),
        group_label=cfg.get("group_label"),
        free_source=cfg.get("free_source", False),
        connection=connection,
        auth=exchange,
        quota=quota_decl,
        quota_probes=all_quota_probes,
        tools=tools,
        pre_ir_transforms=pre_t,
        post_ir_transforms=post_t,
        ir_transforms=ir_t,
        response_body_transforms=resp_t,
        reasoning=reasoning_cap,
        model_reasoning=model_reasoning,
        response_id_prefix=cfg.get("response_id_prefix", ""),
        preserve_fields=tuple(cfg.get("preserve_fields") or ()),
        request_fields=tuple(cfg.get("request_fields") or ()),
        store_default=cfg.get("store_default"),
        hoist_system_messages=cfg.get("hoist_system_messages", True),
        soft_error_patterns=soft_errors,
    )
    register_shim(shim)

    # Register convention-based model_list_transform if the transforms
    # module exports one (e.g. shims that slug-ify display names).
    if transforms_mod is not None:
        mlt = getattr(transforms_mod, "model_list_transform", None)
        if mlt is not None:
            _model_list_transforms[shim.name] = mlt

    logger.debug(
        "Registered provider shim: %s (model_type=%s format=%s)",
        shim.name,
        shim.model_type,
        shim.format,
    )
    return shim


def _load_provider_tolerant(
    directory: Path,
    group: str | None,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
    expected_type: str | None = None,
) -> ProviderShim | None:
    """Load one provider directory, optionally downgrading errors to a warning.

    With ``on_error="warn"`` a malformed ``provider.yaml`` or a failing
    ``transforms.py`` in *directory* is logged and skipped instead of aborting
    the whole scan.
    """
    try:
        return _load_single_provider(
            directory,
            group=group,
            _builtin=builtin,
            root=root,
            quota_probes=quota_probes,
            expected_type=expected_type,
        )
    except Exception as exc:
        if on_error == "warn":
            logger.warning("Skipping shim directory %s: %s", directory, exc)
            return None
        raise


def _load_type_folder(
    type_dir: Path,
    group: str | None,
    expected_type: str,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
) -> list[ProviderShim]:
    """Load the leaves under a model-type directory.

    The directory itself may be the leaf (``vendor/embeddings/provider.yaml``)
    or group format directories (``vendor/llm/chat_completions/provider.yaml``).
    Every leaf is checked against the type the directory names.
    """
    if (type_dir / "provider.yaml").exists():
        shim = _load_provider_tolerant(
            type_dir,
            group,
            builtin=builtin,
            root=root,
            on_error=on_error,
            quota_probes=quota_probes,
            expected_type=expected_type,
        )
        return [shim] if shim is not None else []

    child_group = f"{group}.{type_dir.name}" if group else type_dir.name
    loaded: list[ProviderShim] = []
    for sub in sorted(type_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        if (sub / "provider.yaml").exists():
            shim = _load_provider_tolerant(
                sub,
                child_group,
                builtin=builtin,
                root=root,
                on_error=on_error,
                quota_probes=quota_probes,
                expected_type=expected_type,
            )
            if shim is not None:
                loaded.append(shim)
    return loaded


def _load_group_folder(
    group_dir: Path,
    group: str | None,
    *,
    builtin: bool,
    root: Path,
    on_error: str,
    quota_probes: tuple[Any, ...] = (),
) -> list[ProviderShim]:
    """Load every leaf provider inside a *group_dir*.

    A child is a leaf (``<group>/<leaf>/provider.yaml``) or a model-type
    directory (``<group>/llm/...``, ``<group>/embeddings/...``); a child that
    is neither is not a provider directory and is skipped.
    """
    child_group = f"{group}.{group_dir.name}" if group else group_dir.name
    loaded: list[ProviderShim] = []
    for sub in sorted(group_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        expected_type = _TYPE_DIRS.get(sub.name)
        if expected_type is not None:
            loaded.extend(
                _load_type_folder(
                    sub,
                    child_group,
                    expected_type,
                    builtin=builtin,
                    root=root,
                    on_error=on_error,
                    quota_probes=quota_probes,
                )
            )
            continue
        if (sub / "provider.yaml").exists():
            shim = _load_provider_tolerant(
                sub,
                child_group,
                builtin=builtin,
                root=root,
                on_error=on_error,
                quota_probes=quota_probes,
            )
            if shim is not None:
                loaded.append(shim)
    return loaded


def load_providers_from_dir(
    providers_dir: Path, *, group: str | None = None, on_error: str = "raise"
) -> list[ProviderShim]:
    """Scan *providers_dir* for provider shims and register them.

    This is the public entry point for loading shims from an arbitrary
    directory.  Downstream packages (e.g. argo-proxy) can call this to
    load their own shim directories alongside the built-in ones.

    Plugin transforms use a separate module namespace
    (``_llm_rosetta_plugin_shims.*``) to avoid collisions with built-in
    modules in ``sys.modules``.

    Supports three layouts:

    * **Flat** — a direct child with ``provider.yaml`` (e.g. ``openai/``).
    * **Grouped** — a child WITHOUT ``provider.yaml`` whose own children
      each contain one (e.g. ``vendor/messages/``, ``vendor/chat_completions/``).
    * **Model-typed** — a child WITHOUT ``provider.yaml`` whose name is a
      model-type directory (``llm``, ``embeddings``, ``rerank``, ``decisions``):
      the directory itself holds the leaf, or it groups format directories.
      Every shim below it must declare the matching ``model_type``.

    Args:
        providers_dir: Root directory to scan for provider subdirectories.
        group: Optional group name prefix for all shims loaded from this
            directory.  When ``None``, the directory structure determines
            grouping automatically.
        on_error: ``"raise"`` (default) propagates a bad ``provider.yaml`` or
            a failing ``transforms.py``; ``"warn"`` logs and skips that
            provider so one bad shim cannot abort the whole scan.  External
            sources use ``"warn"`` so a third-party repo cannot take down
            startup; built-ins keep the strict default.

    Returns:
        List of registered :class:`ProviderShim` instances.
    """
    builtin = providers_dir == _PROVIDERS_DIR
    shims: list[ProviderShim] = []
    for d in sorted(providers_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        if (d / "provider.yaml").exists():
            shim = _load_provider_tolerant(
                d, group, builtin=builtin, root=providers_dir, on_error=on_error
            )
            if shim is not None:
                shims.append(shim)
        else:
            # Potential group folder — load the vendor's shared quota probes
            # once, then scan its leaf providers.
            group_probes: tuple[Any, ...] = ()
            try:
                group_probes = _load_quota_probes(d / "quota.py")
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning("Skipping quota probes in %s: %s", d, exc)
            shims.extend(
                _load_group_folder(
                    d,
                    group,
                    builtin=builtin,
                    root=providers_dir,
                    on_error=on_error,
                    quota_probes=group_probes,
                )
            )
    return shims


def load_providers() -> list[ProviderShim]:
    """Load built-in provider shims and any plugin shims.

    1. Scans the built-in ``providers/`` directory.
    2. Discovers entry points in the ``llm_rosetta.shim_providers`` group
       and calls each one to let plugins register their own shims.
    3. Registers built-in model list transforms for providers without
       shim directories (Jina, Cohere).

    Returns:
        Combined list of all registered :class:`ProviderShim` instances
        (built-in + plugin).
    """
    # 1. Built-in shims
    shims = load_providers_from_dir(_PROVIDERS_DIR)

    # 2. Plugin shims via entry points
    shims.extend(_load_plugin_shims())

    # 3. Built-in transforms for providers without shim directories
    register_model_list_transform("jina", _jina_model_list_transform)
    register_model_list_transform("cohere", _cohere_model_list_transform)

    return shims


def _load_plugin_shims() -> list[ProviderShim]:
    """Discover and invoke ``llm_rosetta.shim_providers`` entry points.

    Each entry point should be a callable that registers shims via
    :func:`register_shim` when called (no arguments).  The callable may
    optionally return a ``list[ProviderShim]`` for inclusion in the
    combined result of :func:`load_providers`.

    Errors in individual plugins are logged and do not prevent other
    plugins from loading.
    """
    registered: list[ProviderShim] = []

    eps = entry_points()
    # Python 3.12+: eps.select(); Python 3.10–3.11: dict-style
    if hasattr(eps, "select"):
        plugin_eps = eps.select(group="llm_rosetta.shim_providers")
    else:
        plugin_eps = eps.get("llm_rosetta.shim_providers", [])

    for ep in plugin_eps:
        try:
            loader = ep.load()
            result = loader()
            if isinstance(result, list):
                registered.extend(result)
            logger.info("Loaded plugin shims from entry point: %s", ep.name)
        except Exception:
            logger.warning(
                "Failed to load plugin shim entry point: %s", ep.name, exc_info=True
            )

    return registered

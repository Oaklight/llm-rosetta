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
treated as a *group folder* (e.g. ``argo/anthropic/``, ``argo/openai_chat/``).

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

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path
from collections.abc import Callable
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load

from ..provider_shim import ProviderShim, ReasoningCapability, register_shim

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent

# Convention-based model list transforms registered by shim transforms modules.
# Each entry maps a shim name to a callable that converts upstream model
# entries into (display_ids, upstream_map).
ModelListTransform = Callable[[list[dict[str, Any]]], tuple[list[str], dict[str, str]]]

_model_list_transforms: dict[str, ModelListTransform] = {}


def get_model_list_transform(
    shim_name: str,
) -> ModelListTransform | None:
    """Return the model_list_transform for *shim_name*, or ``None``."""
    return _model_list_transforms.get(shim_name)


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


register_model_list_transform("jina", _jina_model_list_transform)
register_model_list_transform("cohere", _cohere_model_list_transform)


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
        unsigned_blocks=cfg.get(
            "unsigned_blocks", _d.unsigned_blocks if _d else "as_is"
        ),
    )


def _load_transforms(
    provider_dir: Path, *, group: str | None = None, _builtin: bool = True
) -> tuple[tuple, tuple, tuple, Any]:
    """Import transforms.py if present, return (pre, post, ir, module).

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
        return (), (), (), None
    prefix = "llm_rosetta.shims.providers" if _builtin else "_llm_rosetta_plugin_shims"
    if group is not None:
        module_name = f"{prefix}.{group}.{provider_dir.name}.transforms"
    else:
        module_name = f"{prefix}.{provider_dir.name}.transforms"
    spec = importlib.util.spec_from_file_location(module_name, tf_path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load %s", tf_path)
        return (), (), (), None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
        mod,
    )


def _load_single_provider(
    provider_dir: Path, *, group: str | None = None, _builtin: bool = True
) -> ProviderShim | None:
    """Load a single provider from *provider_dir* and register it.

    Args:
        provider_dir: Directory containing ``provider.yaml``.
        group: Name of the parent group folder (``None`` for top-level shims).
        _builtin: Whether this is a built-in provider directory.

    Returns:
        The registered :class:`ProviderShim`, or ``None`` on failure.
    """
    yaml_path = provider_dir / "provider.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml_load(f.read())
    if not isinstance(cfg, dict) or "name" not in cfg or "base" not in cfg:
        logger.warning("Skipping %s: missing 'name' or 'base'", yaml_path)
        return None

    pre_t, post_t, ir_t, transforms_mod = _load_transforms(
        provider_dir, group=group, _builtin=_builtin
    )

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

    shim = ProviderShim(
        name=cfg["name"],
        base=cfg["base"],
        default_base_url=cfg.get("default_base_url"),
        default_api_key_env=cfg.get("default_api_key_env"),
        logo=cfg.get("logo"),
        model_id_field=cfg.get("model_id_field"),
        pre_ir_transforms=pre_t,
        post_ir_transforms=post_t,
        ir_transforms=ir_t,
        reasoning=reasoning_cap,
        model_reasoning=model_reasoning,
        response_id_prefix=cfg.get("response_id_prefix", ""),
        supports_custom_tools=cfg.get("supports_custom_tools", False),
        multimodal_tool_result=cfg.get("multimodal_tool_result"),
        tool_search_mode=cfg.get("tool_search_mode", "disabled"),
    )
    register_shim(shim)

    # Register convention-based model_list_transform if the transforms
    # module exports one (e.g. Argo shims that slug-ify display names).
    if transforms_mod is not None:
        mlt = getattr(transforms_mod, "model_list_transform", None)
        if mlt is not None:
            _model_list_transforms[shim.name] = mlt

    logger.debug("Registered provider shim: %s (base=%s)", shim.name, shim.base)
    return shim


def load_providers_from_dir(
    providers_dir: Path, *, group: str | None = None
) -> list[ProviderShim]:
    """Scan *providers_dir* for provider shims and register them.

    This is the public entry point for loading shims from an arbitrary
    directory.  Downstream packages (e.g. argo-proxy) can call this to
    load their own shim directories alongside the built-in ones.

    Plugin transforms use a separate module namespace
    (``_llm_rosetta_plugin_shims.*``) to avoid collisions with built-in
    modules in ``sys.modules``.

    Supports two layouts:

    * **Flat** — a direct child with ``provider.yaml`` (e.g. ``openai/``).
    * **Grouped** — a child WITHOUT ``provider.yaml`` whose own children
      each contain one (e.g. ``argo/anthropic/``, ``argo/openai_chat/``).

    Args:
        providers_dir: Root directory to scan for provider subdirectories.
        group: Optional group name prefix for all shims loaded from this
            directory.  When ``None``, the directory structure determines
            grouping automatically.

    Returns:
        List of registered :class:`ProviderShim` instances.
    """
    builtin = providers_dir == _PROVIDERS_DIR
    shims: list[ProviderShim] = []
    for d in sorted(providers_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        yaml_path = d / "provider.yaml"
        if yaml_path.exists():
            shim = _load_single_provider(d, group=group, _builtin=builtin)
            if shim is not None:
                shims.append(shim)
        else:
            # Potential group folder — scan children.
            child_group = f"{group}.{d.name}" if group else d.name
            for sub in sorted(d.iterdir()):
                if not sub.is_dir() or sub.name.startswith(("_", ".")):
                    continue
                if (sub / "provider.yaml").exists():
                    shim = _load_single_provider(
                        sub, group=child_group, _builtin=builtin
                    )
                    if shim is not None:
                        shims.append(shim)
    return shims


def load_providers() -> list[ProviderShim]:
    """Load built-in provider shims and any plugin shims.

    1. Scans the built-in ``providers/`` directory.
    2. Discovers entry points in the ``llm_rosetta.shim_providers`` group
       and calls each one to let plugins register their own shims.

    Returns:
        Combined list of all registered :class:`ProviderShim` instances
        (built-in + plugin).
    """
    # 1. Built-in shims
    shims = load_providers_from_dir(_PROVIDERS_DIR)

    # 2. Plugin shims via entry points
    shims.extend(_load_plugin_shims())

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

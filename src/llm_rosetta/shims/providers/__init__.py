"""Scan provider directories and register shims from YAML + transforms.py.

Each subdirectory that contains a ``provider.yaml`` is treated as a leaf
provider definition.  An optional ``transforms.py`` alongside the YAML
may export ``to_transforms`` and/or ``from_transforms`` tuples to bridge
schema differences between the provider and its base converter.

**Grouped directories** are also supported: a child directory that does
NOT contain ``provider.yaml`` but DOES contain subdirectories with one is
treated as a *group folder*.  This keeps related shims together (e.g.
``argo/anthropic/`` and ``argo/openai_chat/``) without bloating the
top-level provider list.  Only one level of nesting is supported.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from llm_rosetta._vendor.yaml import load as yaml_load

from ..provider_shim import ProviderShim, ReasoningCapability, register_shim

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent


def _load_transforms(
    provider_dir: Path, *, group: str | None = None
) -> tuple[tuple, tuple]:
    """Import transforms.py if present, return (from_transforms, to_transforms).

    Args:
        provider_dir: Path to the leaf provider directory.
        group: Name of the parent group folder, if this is a grouped shim.
    """
    tf_path = provider_dir / "transforms.py"
    if not tf_path.exists():
        return (), ()
    if group is not None:
        module_name = (
            f"llm_rosetta.shims.providers.{group}.{provider_dir.name}.transforms"
        )
    else:
        module_name = f"llm_rosetta.shims.providers.{provider_dir.name}.transforms"
    spec = importlib.util.spec_from_file_location(module_name, tf_path)
    if spec is None or spec.loader is None:
        logger.warning("Could not load %s", tf_path)
        return (), ()
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (
        getattr(mod, "from_transforms", ()),
        getattr(mod, "to_transforms", ()),
    )


def _load_single_provider(
    provider_dir: Path, *, group: str | None = None
) -> ProviderShim | None:
    """Load a single provider from *provider_dir* and register it.

    Args:
        provider_dir: Directory containing ``provider.yaml``.
        group: Name of the parent group folder (``None`` for top-level shims).

    Returns:
        The registered :class:`ProviderShim`, or ``None`` on failure.
    """
    yaml_path = provider_dir / "provider.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml_load(f.read())
    if not isinstance(cfg, dict) or "name" not in cfg or "base" not in cfg:
        logger.warning("Skipping %s: missing 'name' or 'base'", yaml_path)
        return None

    from_t, to_t = _load_transforms(provider_dir, group=group)

    # Parse optional reasoning capability config from YAML.
    reasoning_cfg = cfg.get("reasoning")
    reasoning_cap: ReasoningCapability | None = None
    if isinstance(reasoning_cfg, dict):
        reasoning_cap = ReasoningCapability(
            disabled=reasoning_cfg.get("disabled", "omit"),
            effort_field=reasoning_cfg.get("effort_field", "reasoning_effort"),
            max_effort=reasoning_cfg.get("max_effort"),
            effort_map=reasoning_cfg.get("effort_map", {}),
        )

    shim = ProviderShim(
        name=cfg["name"],
        base=cfg["base"],
        default_base_url=cfg.get("default_base_url"),
        default_api_key_env=cfg.get("default_api_key_env"),
        logo=cfg.get("logo"),
        model_id_field=cfg.get("model_id_field"),
        from_transforms=from_t,
        to_transforms=to_t,
        reasoning=reasoning_cap,
    )
    register_shim(shim)
    logger.debug("Registered provider shim: %s (base=%s)", shim.name, shim.base)
    return shim


def load_providers() -> list[ProviderShim]:
    """Scan subdirectories for provider shims and register them.

    Supports two layouts:

    * **Flat** — a direct child with ``provider.yaml`` (e.g. ``openai/``).
    * **Grouped** — a child WITHOUT ``provider.yaml`` whose own children
      each contain one (e.g. ``argo/anthropic/``, ``argo/openai_chat/``).

    Returns:
        List of registered :class:`ProviderShim` instances.
    """
    shims: list[ProviderShim] = []
    for d in sorted(_PROVIDERS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        yaml_path = d / "provider.yaml"
        if yaml_path.exists():
            # Leaf shim at top level.
            shim = _load_single_provider(d)
            if shim is not None:
                shims.append(shim)
        else:
            # Potential group folder — scan children.
            for sub in sorted(d.iterdir()):
                if not sub.is_dir() or sub.name.startswith(("_", ".")):
                    continue
                if (sub / "provider.yaml").exists():
                    shim = _load_single_provider(sub, group=d.name)
                    if shim is not None:
                        shims.append(shim)
    return shims

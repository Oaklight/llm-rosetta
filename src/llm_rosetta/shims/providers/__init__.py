"""Scan provider directories and register shims from YAML + transforms.py.

Each subdirectory that contains a ``provider.yaml`` is treated as a
provider definition.  An optional ``transforms.py`` alongside the YAML
may export ``to_transforms`` and/or ``from_transforms`` tuples to bridge
schema differences between the provider and its base converter.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import yaml

from ..provider_shim import ProviderShim, register_shim

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent


def _load_transforms(provider_dir: Path) -> tuple[tuple, tuple]:
    """Import transforms.py if present, return (from_transforms, to_transforms)."""
    tf_path = provider_dir / "transforms.py"
    if not tf_path.exists():
        return (), ()
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


def load_providers() -> list[ProviderShim]:
    """Scan all subdirectories, load provider.yaml + transforms.py, register.

    Returns:
        List of registered :class:`ProviderShim` instances.
    """
    shims: list[ProviderShim] = []
    for d in sorted(_PROVIDERS_DIR.iterdir()):
        yaml_path = d / "provider.yaml"
        if not d.is_dir() or not yaml_path.exists():
            continue
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or "name" not in cfg or "base" not in cfg:
            logger.warning("Skipping %s: missing 'name' or 'base'", yaml_path)
            continue

        from_t, to_t = _load_transforms(d)
        shim = ProviderShim(
            name=cfg["name"],
            base=cfg["base"],
            default_base_url=cfg.get("default_base_url"),
            default_api_key_env=cfg.get("default_api_key_env"),
            from_transforms=from_t,
            to_transforms=to_t,
        )
        register_shim(shim)
        shims.append(shim)
        logger.debug("Registered provider shim: %s (base=%s)", shim.name, shim.base)
    return shims

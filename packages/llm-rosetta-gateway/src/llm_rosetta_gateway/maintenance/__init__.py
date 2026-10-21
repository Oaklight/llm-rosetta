"""Gateway maintenance steps: bring existing state up to what the code expects.

The vocabulary (:class:`~llm_rosetta.observability.maintenance.Policy`,
:class:`~llm_rosetta.observability.maintenance.Registry`,
:class:`~llm_rosetta.observability.maintenance.Step`) lives in
:mod:`llm_rosetta.observability.maintenance`; this package owns only the steps
and the deferred-startup runner.

Each store keeps its own ledger:

- :mod:`.config` — load-time config steps, recorded by ``config_version`` in
  the document itself.
- :mod:`.keystore` — key-store schema steps, recorded in ``keys.db``.
- :mod:`.backfills` — post-boot data repairs, recorded in the deferred
  registry's results.
- :mod:`.startup` — the deferred task that runs the backfills.
- :mod:`.dumps` — error dumps held until their request-log row exists.
- :mod:`.names` — the idempotent provider/shim name correction.
- :mod:`.migrations` — versioned config-schema migrations.

The public names are re-exported lazily: importing a submodule (for example
``gateway.keystore`` importing ``maintenance.keystore``) does not pull in
``startup`` or ``dumps``.  The mapping below is the whole façade; the
submodules stay importable for tests and for callers that want the exact file.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backfills import build_registry as build_deferred_registry
    from .config import (
        STEPS,
        ConfigMigrationError,
        apply,
        build_registry as build_config_registry,
    )
    from .dumps import DeferredDumpStore
    from .keystore import COLUMN_OF, build_registry as build_keystore_registry
    from .migrations import CURRENT_VERSION, migrate
    from .names import correct_legacy_names, legacy_name_error
    from .startup import DeferredStartup, ProviderInitState, ProviderNotReady

#: Public name -> (submodule, attribute).  Kept as data so the façade costs
#: nothing at import time; see the module docstring for why that matters.
_EXPORTS: dict[str, tuple[str, str]] = {
    "COLUMN_OF": (".keystore", "COLUMN_OF"),
    "CURRENT_VERSION": (".migrations", "CURRENT_VERSION"),
    "STEPS": (".config", "STEPS"),
    "ConfigMigrationError": (".config", "ConfigMigrationError"),
    "DeferredDumpStore": (".dumps", "DeferredDumpStore"),
    "DeferredStartup": (".startup", "DeferredStartup"),
    "ProviderInitState": (".startup", "ProviderInitState"),
    "ProviderNotReady": (".startup", "ProviderNotReady"),
    "apply": (".config", "apply"),
    "build_config_registry": (".config", "build_registry"),
    "build_deferred_registry": (".backfills", "build_registry"),
    "build_keystore_registry": (".keystore", "build_registry"),
    "correct_legacy_names": (".names", "correct_legacy_names"),
    "legacy_name_error": (".names", "legacy_name_error"),
    "migrate": (".migrations", "migrate"),
}

__all__ = [
    "COLUMN_OF",
    "CURRENT_VERSION",
    "STEPS",
    "ConfigMigrationError",
    "DeferredDumpStore",
    "DeferredStartup",
    "ProviderInitState",
    "ProviderNotReady",
    "apply",
    "build_config_registry",
    "build_deferred_registry",
    "build_keystore_registry",
    "correct_legacy_names",
    "legacy_name_error",
    "migrate",
]


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

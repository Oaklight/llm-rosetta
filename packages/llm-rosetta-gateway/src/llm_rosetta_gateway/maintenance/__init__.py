"""Gateway maintenance steps: bring existing state up to what the code expects.

The vocabulary (:class:`~llm_rosetta_gateway.observability.maintenance.Policy`,
:class:`~llm_rosetta_gateway.observability.maintenance.Registry`,
:class:`~llm_rosetta_gateway.observability.maintenance.Step`) lives in
:mod:`llm_rosetta_gateway.observability.maintenance`; this package owns only the steps
and the deferred-startup runner.

Each store keeps its own ledger:

- :mod:`.keystore` — key-store schema steps, recorded in ``keys.db``.
- :mod:`.backfills` — post-boot data repairs, recorded in the deferred
  registry's results.
- :mod:`.startup` — the deferred task that runs the backfills.
- :mod:`.dumps` — error dumps held until their request-log row exists.

The config-file migrations (the load-time steps recorded by ``config_version``,
the versioned schema migrations, the retired-name guard and the
``llm-rosetta gateway migrate`` tool) live in :mod:`llm_rosetta_gateway.config.migrate`:
they read the config's own parser, so they belong to the layer above this one.

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
    from .dumps import DeferredDumpStore
    from .keystore import COLUMN_OF, build_registry as build_keystore_registry
    from .startup import DeferredStartup, ProviderInitState, ProviderNotReady

#: Public name -> (submodule, attribute).  Kept as data so the façade costs
#: nothing at import time; see the module docstring for why that matters.
_EXPORTS: dict[str, tuple[str, str]] = {
    "COLUMN_OF": (".keystore", "COLUMN_OF"),
    "DeferredDumpStore": (".dumps", "DeferredDumpStore"),
    "DeferredStartup": (".startup", "DeferredStartup"),
    "ProviderInitState": (".startup", "ProviderInitState"),
    "ProviderNotReady": (".startup", "ProviderNotReady"),
    "build_deferred_registry": (".backfills", "build_registry"),
    "build_keystore_registry": (".keystore", "build_registry"),
}

__all__ = [
    "COLUMN_OF",
    "DeferredDumpStore",
    "DeferredStartup",
    "ProviderInitState",
    "ProviderNotReady",
    "build_deferred_registry",
    "build_keystore_registry",
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

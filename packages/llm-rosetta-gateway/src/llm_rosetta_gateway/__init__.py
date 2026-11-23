"""llm-rosetta Gateway — HTTP proxy/translator between LLM provider formats.

Usage::

    # CLI entry point (after pip install)
    llm-rosetta-gateway --config config.jsonc

    # Module invocation
    python -m llm_rosetta_gateway --config config.jsonc

    # Programmatic usage
    from llm_rosetta_gateway import create_app, GatewayExtensions, load_gateway_config

    config = load_gateway_config("config.jsonc")
    app = await create_app(config)

Module map::

    Core proxy          app, proxy, config, routing/
    Middleware           middleware/ subpackage:
                           auth, ratelimit, request_context, error_format,
                           circuit_breaker, headers, sanitize, affinity
    Pipelines            pipelines/ subpackage:
                           embeddings, rerank, decision
    Model types          model_types/ package (registry + source detectors)
    Observability        observability/ (logging, metrics, request log, dumps)
    State                keystore/ (the API key store), maintenance/
    Quota                quota/ (probe, allowance, signals)
    Admin UI             admin/ (routes, static assets, JS/CSS)
    Transport            transport/ subpackage
    CLI / bootstrap      cli/ package, __main__

    Middleware hook order (registered in create_app):
    request_context → auth → ratelimit → [extension hooks] → handler

The public names are resolved lazily.  Importing one submodule (for example
``llm_rosetta_gateway.keystore``) must not drag the HTTP application in with
it; ``create_app`` pulls that graph when it is actually requested.
"""

# httpserver and httpclient are vendored in _vendor/ — no external deps needed.

from __future__ import annotations

#: Released independently of ``llm_rosetta.__version__`` from 0.15.0 on; they
#: start together.  A literal because the build backend reads it without
#: importing the package.
__version__ = "0.14.0.dev38"

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import GatewayExtensions, create_app
    from .cli import main
    from .config import (
        ConfigIO,
        GatewayConfig,
        JsoncConfigIO,
        discover_config,
        load_config,
        load_gateway_config,
    )
    from .proxy import ProviderMetadataStore
    from .shims.shim_sources import (
        ShimSources,
        apply_shim_section,
        apply_shim_sources,
        parse_shims_section,
        resolve_source_config,
        shims_signature,
    )

#: Public name -> (submodule, attribute).  Data only, so importing the package
#: is free; see the module docstring for why that matters.
_EXPORTS: dict[str, tuple[str, str]] = {
    "ConfigIO": (".config", "ConfigIO"),
    "GatewayConfig": (".config", "GatewayConfig"),
    "GatewayExtensions": (".app", "GatewayExtensions"),
    "JsoncConfigIO": (".config", "JsoncConfigIO"),
    "ProviderMetadataStore": (".proxy", "ProviderMetadataStore"),
    "ShimSources": (".shims.shim_sources", "ShimSources"),
    "apply_shim_section": (".shims.shim_sources", "apply_shim_section"),
    "apply_shim_sources": (".shims.shim_sources", "apply_shim_sources"),
    "create_app": (".app", "create_app"),
    "discover_config": (".config", "discover_config"),
    "load_config": (".config", "load_config"),
    "load_gateway_config": (".config", "load_gateway_config"),
    "main": (".cli", "main"),
    "parse_shims_section": (".shims.shim_sources", "parse_shims_section"),
    "resolve_source_config": (".shims.shim_sources", "resolve_source_config"),
    "shims_signature": (".shims.shim_sources", "shims_signature"),
}

__all__ = [
    "ConfigIO",
    "GatewayConfig",
    "GatewayExtensions",
    "JsoncConfigIO",
    "ProviderMetadataStore",
    "ShimSources",
    "apply_shim_section",
    "apply_shim_sources",
    "create_app",
    "discover_config",
    "load_config",
    "load_gateway_config",
    "main",
    "parse_shims_section",
    "resolve_source_config",
    "shims_signature",
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

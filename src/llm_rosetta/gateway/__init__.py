"""llm-rosetta Gateway — HTTP proxy/translator between LLM provider formats.

Usage::

    # CLI entry point (after pip install)
    llm-rosetta-gateway --config config.jsonc

    # Module invocation
    python -m llm_rosetta.gateway --config config.jsonc

    # Programmatic usage
    from llm_rosetta.gateway import create_app, GatewayConfig, GatewayExtensions, load_config

    raw = load_config("config.jsonc")
    app = create_app(GatewayConfig(raw))

Module map::

    Core proxy          app, proxy, config, providers, routing_strategy
    Middleware           auth, ratelimit, request_context, error_format,
                         circuit_breaker, headers, sanitize, affinity
    Model types          model_types (registry), embeddings, rerank, decision
                         embedding_pipeline, rerank_pipeline
    Observability        (in llm_rosetta.observability, re-exported by admin/)
    Infrastructure       keystore, logging, migrations, deferred_startup
    Admin UI             admin/ (routes, static assets, JS/CSS)
    CLI / bootstrap      cli, banner, __main__

    Middleware hook order (registered in create_app):
    request_context → auth → ratelimit → [extension hooks] → handler
"""

# httpserver and httpclient are vendored in _vendor/ — no external deps needed.

from .app import GatewayExtensions, create_app
from .cli import main
from .config import ConfigIO, GatewayConfig, JsoncConfigIO, discover_config, load_config
from .proxy import ProviderMetadataStore

__all__ = [
    "ConfigIO",
    "GatewayConfig",
    "GatewayExtensions",
    "JsoncConfigIO",
    "ProviderMetadataStore",
    "create_app",
    "discover_config",
    "load_config",
    "main",
]

"""llm-rosetta Gateway — HTTP proxy/translator between LLM provider formats.

Usage::

    # CLI entry point (after pip install)
    llm-rosetta-gateway --config config.jsonc

    # Module invocation
    python -m llm_rosetta.gateway --config config.jsonc

    # Programmatic usage
    from llm_rosetta.gateway import create_app, GatewayConfig, load_config

    raw = load_config("config.jsonc")
    app = create_app(GatewayConfig(raw))
"""

# httpserver and httpclient are vendored in _vendor/ — no external deps needed.

from .app import create_app
from .cli import main
from .config import GatewayConfig, discover_config, load_config
from .proxy import ProviderMetadataStore

__all__ = [
    "ProviderMetadataStore",
    "create_app",
    "main",
    "GatewayConfig",
    "discover_config",
    "load_config",
]

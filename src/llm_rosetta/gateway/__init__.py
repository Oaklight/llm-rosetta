"""llm-rosetta Gateway — HTTP proxy/translator between LLM provider formats.

Requires the ``gateway`` extra to be installed::

    pip install "llm-rosetta[gateway]"

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

_INSTALL_MSG = (
    "llm-rosetta gateway requires extra dependencies.\n"
    "Install them with:  pip install 'llm-rosetta[gateway]'"
)

_missing: list[str] = []
for _pkg in ("starlette", "uvicorn", "httpx"):
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    raise ImportError(f"{_INSTALL_MSG}\nMissing packages: {', '.join(_missing)}")

from .app import create_app  # noqa: E402
from .cli import main  # noqa: E402
from .config import GatewayConfig, discover_config, load_config  # noqa: E402

__all__ = ["create_app", "main", "GatewayConfig", "discover_config", "load_config"]

"""Gateway configuration: JSONC loading, validation, and runtime resolution.

The package facade: ``from llm_rosetta_gateway.config import GatewayConfig``
is the supported import.  Behind it:

============================  ============================================
:mod:`config.io`              JSONC reading, the reader/writer protocol,
                              and the file locks
:mod:`config.loader`          discovery, the load/write facade, and
                              ``load_gateway_config``
:mod:`config.schema`          the value types the parser produces
:mod:`config.core`            :class:`GatewayConfig` itself
:mod:`config.parse`           provider / model / typed-pool parsing
:mod:`config.typed`           the typed (non-LLM) endpoint pools
:mod:`config.settings`        the server / auth / limits / debug sections
:mod:`config.resolve`         selection, provider choice, loop detection
============================  ============================================

Every name the single-module version exported is re-exported here, so no
caller changes.
"""

from __future__ import annotations

from llm_rosetta_gateway._vendor import jsonx
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta_gateway.observability.maintenance import Registry
from llm_rosetta.routing import ResolvedRoute

from ..model_types import ModelTypeDescriptor
from ..providers import BASE_FORMATS, build_provider_info
from ..routing import (
    DEFAULT_STRATEGY,
    AnchorStage,
    QuotaStage,
    ModelRoute,
    ProviderEntry,
    RoutingChain,
    RoutingStage,
    SameFormatStage,
    SelectionContext,
    create_strategy,
)
from ..transport import ProviderInfo

from .core import GatewayConfig
from .io import ConfigIO, JsoncConfigIO, async_config_lock, config_lock, write_config
from .loader import (
    PATHS_TO_TRY,
    discover_config,
    load_config,
    load_config_raw,
    load_gateway_config,
)
from .resolve import detect_routing_loops
from .schema import ProviderIdentity, TypedRoute

__all__ = [
    "BASE_FORMATS",
    "DEFAULT_STRATEGY",
    "PATHS_TO_TRY",
    "AnchorStage",
    "QuotaStage",
    "ConfigIO",
    "GatewayConfig",
    "JsoncConfigIO",
    "ModelRoute",
    "ModelTypeDescriptor",
    "ProviderEntry",
    "ProviderIdentity",
    "ProviderInfo",
    "ProviderType",
    "Registry",
    "ResolvedRoute",
    "RoutingChain",
    "RoutingStage",
    "SameFormatStage",
    "SelectionContext",
    "TypedRoute",
    "async_config_lock",
    "build_provider_info",
    "config_lock",
    "create_strategy",
    "detect_routing_loops",
    "discover_config",
    "load_config",
    "load_config_raw",
    "load_gateway_config",
    "write_config",
]

# ``jsonx`` stays reachable as an attribute of the package: the JSONC reader in
# :mod:`config.io` is what calls it, and a test patches it here.
_ = jsonx

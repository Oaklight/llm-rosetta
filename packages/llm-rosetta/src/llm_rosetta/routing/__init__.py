"""The routing contract: :class:`ResolvedRoute` and the :class:`Router` protocol.

This is the core-layer contract any consumer (gateway, argo-proxy, CLI tools)
can depend on.  Gateway *selection* — stages, strategies and affinity — is a
different concern and lives in :mod:`llm_rosetta_gateway.routing`.

``ProviderType`` and ``SoftErrorPattern`` were reachable from the old flat
module (it had no ``__all__``) and are re-exported here so that surface is
preserved.
"""

from __future__ import annotations

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.shims.provider_shim import SoftErrorPattern

from .contract import ResolvedRoute, Router

__all__ = ["ProviderType", "ResolvedRoute", "Router", "SoftErrorPattern"]

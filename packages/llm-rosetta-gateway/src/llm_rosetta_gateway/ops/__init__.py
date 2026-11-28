"""Unified gateway operations layer.

Every gateway action is an :class:`OpsBase` subclass with automatic
audit recording.  Route handlers construct an Ops object and call
:meth:`~OpsBase.execute` — the base class handles recording.

.. code-block:: python

    result = await OpsClearData(request.app.ops_ctx, table="request_log").execute()
"""

from .base import OpsBase, OpsContext
from .proxy import OpsProxyRequest
from .rejected import OpsRejectedRequest
from .lifecycle import (
    OpsAdminSetup,
    OpsConfigReload,
    OpsGatewayShutdown,
    OpsGatewayStartup,
)
from .provider_validation import (
    OpsProviderValidation,
    OpsStartupValidation,
    startup_validation_entry,
)
from .extensions import (
    OpsExtensionCliInstall,
    OpsExtensionRemove,
    OpsExtensionUpdate,
)
from .keys import OpsKeyCreate, OpsKeyDelete, OpsKeyRotate, OpsKeyUpdate
from .security import (
    OpsPasswordChange,
    OpsSelfUpdate,
    OpsSessionLogoutAll,
    OpsTokenRotate,
)
from .data import (
    OpsClearData,
    OpsClearOpsLog,
    OpsCleanupData,
    OpsDeleteErrorDump,
    OpsPeriodicCleanup,
    OpsPrune,
    OpsRebuildMetrics,
    OpsTrimData,
    OpsVacuum,
)

__all__ = [
    "OpsAdminSetup",
    "OpsBase",
    "OpsCleanupData",
    "OpsConfigReload",
    "OpsClearData",
    "OpsClearOpsLog",
    "OpsContext",
    "OpsDeleteErrorDump",
    "OpsExtensionCliInstall",
    "OpsExtensionRemove",
    "OpsExtensionUpdate",
    "OpsGatewayShutdown",
    "OpsGatewayStartup",
    "OpsKeyCreate",
    "OpsKeyDelete",
    "OpsKeyRotate",
    "OpsKeyUpdate",
    "OpsPasswordChange",
    "OpsPeriodicCleanup",
    "OpsPrune",
    "OpsProxyRequest",
    "OpsProviderValidation",
    "startup_validation_entry",
    "OpsRebuildMetrics",
    "OpsRejectedRequest",
    "OpsSelfUpdate",
    "OpsStartupValidation",
    "OpsTrimData",
    "OpsSessionLogoutAll",
    "OpsTokenRotate",
    "OpsVacuum",
]

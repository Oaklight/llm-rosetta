"""Config CRUD routes, split by resource.

``register_config_routes`` is the one place the config endpoints are wired;
each handler lives in the module for the resource it edits:

============================  ============================================
:mod:`.payload`               the config document the panel reads/writes
:mod:`.providers`             provider create/update/delete/toggle/rename
:mod:`.models`                model create/update/delete/toggle/bulk
:mod:`.model_providers`       model-to-provider links and weights
:mod:`.settings`              server settings, rate limits, log format
:mod:`.shims`                 shim sources, reload, removal advisories
:mod:`.shim_files`            loaded shims: inventory, view, edit, delete
:mod:`.upstream`              upstream model discovery and routing loops
:mod:`.bulk`                  bulk add: merge a provider into a model entry
============================  ============================================
"""

from __future__ import annotations

from typing import Any

from .._shared import _guard
from ..observability import get_provider_key
from .model_providers import (
    remove_model_provider,
    update_model_provider,
)
from .models import (
    _build_model_entry,
    bulk_update_models,
    delete_model,
    put_model,
    toggle_model,
)
from .payload import (
    _base_formats_payload,
    _get_model_type_metadata,
    get_config,
)
from .providers import (
    delete_provider,
    put_provider,
    toggle_provider,
)
from .settings import put_server_settings
from .shim_files import (
    delete_loaded_shim,
    get_loaded_shim,
    get_loaded_shims,
    post_shim_enabled,
    put_loaded_shim,
)
from .shims import (
    _merge_shim_sources_body,
    _shim_removal_advisories,
    _shim_status_payload,
    _validate_shim_sources_body,
    post_shims_reload,
    put_shim_sources,
    reload_config,
)
from .bulk import (
    _add_new_models,
    _append_provider_to_model,
    _merge_provider_into_model,
    bulk_add_models,
)
from .upstream import (
    check_routing_loops,
    fetch_upstream_models,
)

__all__ = [
    "_add_new_models",
    "_append_provider_to_model",
    "_base_formats_payload",
    "_build_model_entry",
    "_get_model_type_metadata",
    "_merge_provider_into_model",
    "_merge_shim_sources_body",
    "_shim_removal_advisories",
    "_shim_status_payload",
    "_validate_shim_sources_body",
    "bulk_add_models",
    "bulk_update_models",
    "check_routing_loops",
    "delete_loaded_shim",
    "delete_model",
    "delete_provider",
    "fetch_upstream_models",
    "get_config",
    "get_loaded_shim",
    "get_loaded_shims",
    "post_shims_reload",
    "post_shim_enabled",
    "put_loaded_shim",
    "put_model",
    "put_provider",
    "put_server_settings",
    "put_shim_sources",
    "register_config_routes",
    "reload_config",
    "remove_model_provider",
    "toggle_model",
    "toggle_provider",
    "update_model_provider",
]


def register_config_routes(app: Any) -> None:
    """Register the config routes: providers, models, settings and shims."""
    app.route("/admin/api/config", methods=["GET"])(get_config)
    app.route("/admin/api/config/providers/<name>", methods=["PUT"])(
        _guard("providers", put_provider)
    )
    app.route("/admin/api/config/providers/<name>", methods=["DELETE"])(
        _guard("providers", delete_provider)
    )
    app.route("/admin/api/config/providers/<name>/toggle", methods=["POST"])(
        _guard("providers", toggle_provider)
    )
    app.route("/admin/api/config/providers/<name>/key", methods=["GET"])(
        get_provider_key
    )
    app.route("/admin/api/config/models/<path:name>", methods=["PUT"])(
        _guard("models", put_model)
    )
    app.route("/admin/api/config/models/<path:name>", methods=["DELETE"])(
        _guard("models", delete_model)
    )
    app.route("/admin/api/config/models/<path:name>/toggle", methods=["POST"])(
        _guard("models", toggle_model)
    )
    app.route("/admin/api/config/models/<path:name>/remove-provider", methods=["POST"])(
        _guard("models", remove_model_provider)
    )
    app.route("/admin/api/config/models/<path:name>/update-provider", methods=["POST"])(
        _guard("models", update_model_provider)
    )
    app.route("/admin/api/config/models/bulk", methods=["POST"])(
        _guard("models", bulk_update_models)
    )
    app.route("/admin/api/config/providers/<name>/models", methods=["GET"])(
        fetch_upstream_models
    )
    app.route("/admin/api/config/models", methods=["POST"])(
        _guard("models", bulk_add_models)
    )
    app.route("/admin/api/config/models/loop-check", methods=["GET"])(
        check_routing_loops
    )
    app.route("/admin/api/config/server", methods=["PUT"])(put_server_settings)
    # Loaded shims: what is registered, and the local ones we may edit.
    app.route("/admin/api/config/shims", methods=["GET"])(
        _guard("providers", get_loaded_shims)
    )
    app.route("/admin/api/config/shims/<name>", methods=["GET"])(
        _guard("providers", get_loaded_shim)
    )
    app.route("/admin/api/config/shims/<name>", methods=["PUT"])(
        _guard("providers", put_loaded_shim)
    )
    app.route("/admin/api/config/shims/<name>", methods=["DELETE"])(
        _guard("providers", delete_loaded_shim)
    )
    app.route("/admin/api/config/shims/<name>/enabled", methods=["POST"])(
        _guard("providers", post_shim_enabled)
    )
    app.route("/admin/api/config/shims", methods=["PUT"])(
        _guard("providers", put_shim_sources)
    )
    app.route("/admin/api/config/shims/reload", methods=["POST"])(
        _guard("providers", post_shims_reload)
    )
    app.route("/admin/api/config/reload", methods=["POST"])(reload_config)

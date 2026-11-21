"""Sidecar that imports every contract module while ``llm_rosetta`` initialises."""

from __future__ import annotations

from typing import Any

import llm_rosetta.contracts.routing
import llm_rosetta.contracts.usage
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.auth import (
    AuthContext,
    AuthExchange,
    AuthExchangeError,
    BaseAuthExchange,
    InteractiveSetupRequired,
)
from llm_rosetta.contracts.quota import QuotaProbe, QuotaWindow
from llm_rosetta.contracts.shim import ProviderShim, load_providers_from_dir
from llm_rosetta.contracts.transform import (
    NamedTransform,
    default_message_field,
    default_tool_description,
    flatten_system_content,
    hoist_late_system_messages,
    rename_field,
    replace_message_field,
    rewrite_harmony_tool_calls,
    strip_fields,
    strip_fields_for_model,
    truncate_images,
    unwind_parallel_tool_calls,
)

IMPORTED = (
    llm_rosetta.contracts.routing.ResolvedRoute,
    llm_rosetta.contracts.usage.UsageReport,
    IR_VERSION,
    SURFACE_VERSION,
    AuthContext,
    AuthExchange,
    AuthExchangeError,
    BaseAuthExchange,
    InteractiveSetupRequired,
    QuotaProbe,
    QuotaWindow,
    ProviderShim,
    load_providers_from_dir,
    default_message_field,
    default_tool_description,
    flatten_system_content,
    rename_field,
    replace_message_field,
    rewrite_harmony_tool_calls,
    strip_fields_for_model,
    truncate_images,
    unwind_parallel_tool_calls,
)


def _keep(body: dict[str, Any]) -> dict[str, Any]:
    return body


pre_ir_transforms = (NamedTransform(_keep, "c1657_keep"), strip_fields("c1657"))
ir_transforms = (hoist_late_system_messages(),)

"""Extensions: installable folders that provide shims and transports.

An extension is one folder (``plugin.yaml`` beside its code) imported as
``llm_rosetta_ext.<id>``.  This package finds the installed ones, reads and
checks their manifests (:mod:`llm_rosetta.contracts.extension`), and registers
their shims through the same vendor-tree reader the built-in shims use.

Where an extension came from, and why one was not loaded, is kept as an
:class:`ExtensionStatus` so a host can show it; a bad extension is skipped and
named, never allowed to stop the others.
"""

from __future__ import annotations

from ._discovery import (
    ENTRY_POINT_GROUP,
    LEGACY_ENTRY_POINT_GROUP,
    NAMESPACE,
    RetiredEntryPoint,
    discover,
    retired_entry_points,
)
from ._dirs import (
    ENV_VAR,
    ExtensionInstallError,
    ExtensionsDirError,
    activate_extensions_dir,
    configure_extensions_dir,
    extensions_dir,
    tool_bin_dir,
    tools_dir,
)
from ._index import (
    DEFAULT_INDEX_URL,
    INDEX_ENV,
    ExtensionIndex,
    IndexEntry,
    UpdatePlan,
    entry_problems,
    install_from_index,
    load_index,
    parse_index,
    plan_updates,
    select_entry,
    version_key,
)
from ._install import (
    ExternalStatus,
    InstalledExtension,
    check_external,
    inspect_wheel,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    missing_dependencies,
    pip_available,
    remove_extension,
)
from ._load import (
    FAILED,
    KINDS,
    LOADED,
    SKIPPED,
    ExtensionStatus,
    KindLoadError,
    extensions_current,
    import_extension_module,
    list_extensions,
    load_extensions,
    register_kind_handler,
)

__all__ = [
    "FAILED",
    "KINDS",
    "LOADED",
    "SKIPPED",
    "extensions_current",
    "DEFAULT_INDEX_URL",
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
    "INDEX_ENV",
    "LEGACY_ENTRY_POINT_GROUP",
    "NAMESPACE",
    "ExtensionIndex",
    "ExtensionInstallError",
    "ExtensionStatus",
    "ExtensionsDirError",
    "ExternalStatus",
    "IndexEntry",
    "InstalledExtension",
    "KindLoadError",
    "RetiredEntryPoint",
    "UpdatePlan",
    "activate_extensions_dir",
    "check_external",
    "configure_extensions_dir",
    "discover",
    "entry_problems",
    "extensions_dir",
    "tool_bin_dir",
    "tools_dir",
    "import_extension_module",
    "inspect_wheel",
    "install_from_index",
    "install_wheel",
    "install_wheel_url",
    "install_with_pip",
    "installed_extensions",
    "list_extensions",
    "load_extensions",
    "load_index",
    "missing_dependencies",
    "parse_index",
    "pip_available",
    "plan_updates",
    "register_kind_handler",
    "remove_extension",
    "retired_entry_points",
    "select_entry",
    "version_key",
]

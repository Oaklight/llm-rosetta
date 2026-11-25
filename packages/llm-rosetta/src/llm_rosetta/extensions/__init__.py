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

from ._commands import (
    COMMANDS_ATTRIBUTE,
    COMMANDS_MODULE,
    ExtensionArea,
    area_verbs,
    extension_areas,
    resolve_extension_command,
)
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
)
from ._index import (
    DEFAULT_INDEX_URL,
    INDEX_ENV,
    ExtensionIndex,
    IndexEntry,
    UpdatePlan,
    install_from_index,
    load_index,
    parse_index,
    plan_updates,
    select_entry,
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
    ExtensionStatus,
    import_extension_module,
    list_extensions,
    load_extensions,
)

__all__ = [
    "COMMANDS_ATTRIBUTE",
    "COMMANDS_MODULE",
    "DEFAULT_INDEX_URL",
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
    "INDEX_ENV",
    "LEGACY_ENTRY_POINT_GROUP",
    "NAMESPACE",
    "ExtensionArea",
    "ExtensionIndex",
    "ExtensionInstallError",
    "ExtensionStatus",
    "ExtensionsDirError",
    "ExternalStatus",
    "IndexEntry",
    "InstalledExtension",
    "RetiredEntryPoint",
    "UpdatePlan",
    "activate_extensions_dir",
    "area_verbs",
    "check_external",
    "configure_extensions_dir",
    "discover",
    "extension_areas",
    "extensions_dir",
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
    "remove_extension",
    "resolve_extension_command",
    "retired_entry_points",
    "select_entry",
]

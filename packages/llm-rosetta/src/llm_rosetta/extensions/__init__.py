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
    discover,
)
from ._dirs import (
    ENV_VAR,
    ExtensionInstallError,
    ExtensionsDirError,
    activate_extensions_dir,
    configure_extensions_dir,
    extensions_dir,
)
from ._install import (
    ExternalStatus,
    InstalledExtension,
    check_external,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    missing_dependencies,
    pip_available,
    remove_extension,
)
from ._load import ExtensionStatus, list_extensions, load_extensions

__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_VAR",
    "LEGACY_ENTRY_POINT_GROUP",
    "NAMESPACE",
    "ExtensionInstallError",
    "ExtensionStatus",
    "ExtensionsDirError",
    "ExternalStatus",
    "InstalledExtension",
    "activate_extensions_dir",
    "check_external",
    "configure_extensions_dir",
    "discover",
    "extensions_dir",
    "install_with_pip",
    "install_wheel",
    "install_wheel_url",
    "installed_extensions",
    "list_extensions",
    "load_extensions",
    "missing_dependencies",
    "pip_available",
    "remove_extension",
]

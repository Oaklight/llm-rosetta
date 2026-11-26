"""Provider shim loading — built-in directory scan + installed extensions.

Shim lifecycle
--------------
**Registration** (startup, once):

1. ``load_providers()`` scans the bundled ``builtin/`` directory via
   ``load_providers_from_dir()``, registering each shim found.
2. It then registers the shims of every installed extension
   (:mod:`llm_rosetta.extensions`), each read through the same vendor-tree
   loader.

**Usage** (per request):

- ``get_shim(name)`` looks up a registered shim by name.
- The gateway / ``convert()`` injects the shim's reasoning config
  and applies transforms around the converter.

Directory layout
----------------
Each subdirectory that contains a ``provider.yaml`` is treated as a leaf
provider definition.  An optional ``transforms.py`` alongside the YAML
may export ``pre_ir_transforms`` and/or ``post_ir_transforms`` tuples
(the 0.13 names ``from_transforms`` / ``to_transforms`` are refused with an
error that names the replacement).

**Grouped directories** are also supported: a child directory that does
NOT contain ``provider.yaml`` but DOES contain subdirectories with one is
treated as a *group folder* (e.g. ``vendor/messages/``, ``vendor/chat_completions/``).

**Model-type directories** name a category, using the endpoint spelling:
``llm/``, ``embeddings/``, ``rerank/`` and ``decisions/``.  Under one, either
the directory itself holds the leaf (``vendor/embeddings/provider.yaml``, the
usual non-chat shape) or it groups format directories
(``vendor/llm/chat_completions/provider.yaml``, because a chat vendor serves
several dialects).  A shim filed under a model-type directory must declare the
matching ``model_type``; a mismatch is refused at load time.

**Vendor manifests** are optional: a ``provider.yaml`` at a vendor root that
declares no ``name`` / ``format`` is a defaults file for every leaf below it.
It may carry any subset of the ``connection:`` block — in practice the
credential declaration, which is a provider fact rather than a per-dialect
one — and each leaf overrides any field it names, so a vendor whose lanes
differ keeps the difference on the leaf.  ``auth.py`` beside the manifest is
the vendor's credential exchange, attached to every leaf.  A vendor that ships
no manifest keeps declaring everything in each leaf, unchanged.

Extension shims
---------------
An extension is a folder with a ``plugin.yaml`` beside its code, imported as
``llm_rosetta_ext.<id>``.  Its ``provider.yaml`` at the folder root is the
vendor manifest and ``llm/<format>/provider.yaml`` holds the lanes, the same
shape a built-in vendor directory has.  Core reads ``plugin.yaml`` first,
checks that this core can run it, and then loads the tree through
``load_providers_from_dir()``; see :mod:`llm_rosetta.extensions`.  The 0.13.0
``llm_rosetta.shim_providers`` entry point group was retired in 0.14.0.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load  # noqa: F401
from llm_rosetta.formats.batch import (  # noqa: F401
    DECLARABLE_DIALECTS,
    batch_spec_for,
)
from llm_rosetta.ir.model.constraints import Range, RequestConstraints  # noqa: F401

from ...contracts import QUOTA_MODULE
from llm_rosetta.contracts.resolver import set_providers_dir_loader
from llm_rosetta.contracts.shim import ProviderShim  # noqa: F401
from ..provider_shim import register_shim  # noqa: F401

#: The built-in providers root, defined beside the vendor data.
#: :func:`load_providers_from_dir` reads this attribute from this module at
#: call time, so rebinding it here redirects the scan.
from ..builtin import _PROVIDERS_DIR

# These names keep the import surface this module had before the loader was
# split into ``_*`` modules: it imported them, so callers could take them from
# here.  Nothing in this module reads them any more.
from ...contracts.auth import FIELD_KEY_RE, FIELD_LIMIT, FIELD_TEXT_LIMIT  # noqa: F401
from llm_rosetta.contracts.shim import (  # noqa: F401
    TYPE_PARAM_KEYS,
    BatchDeclaration,
    ConnectionConfig,
    QuotaDeclaration,
    ReasoningCapability,
    SoftErrorPattern,
    ToolsConfig,
    VendorManifest,
)
from llm_rosetta.ir.primitives.body import Transform  # noqa: F401
from ._constants import (  # noqa: F401
    _CONNECTION_KEYS,
    _REMOVED_LEAF_KEYS,
    _IDENTITY_KEYS,
    _MANIFEST_KEYS,
    _MAX_LANE_DEPTH,
    _TYPE_DIRS,
    logger,
)
from ._fields import (  # noqa: F401
    _FIELD_KEYS,
    _declared_field,
    _declared_fields,
    _declares_identity,
    _merged_fields,
)
from ._leaf import _load_single_provider  # noqa: F401
from ._manifest import (  # noqa: F401
    ManifestLeaf,
    ManifestVocabulary,
    VendorManifestDescription,
    describe_vendor_manifest,
    leaf_connection,
    manifest_vocabulary,
    merge_connection,
    _leaf_connection,
    _leaf_exchange,
    _leaf_quota,
    _merged_connection,
    _vendor_manifest,
)
from ._model_list import (  # noqa: F401
    ModelListTransform,
    _cohere_model_list_transform,
    _jina_model_list_transform,
    _model_list_transforms,
    get_model_list_transform,
    register_model_list_transform,
)
from ._parse import (  # noqa: F401
    _BATCH_KEYS,
    _BATCH_PATH_KEYS,
    _parse_batch,
    _parse_connection_argv,
    _parse_connection_env_unset,
    _parse_connection_meta,
    _check_default_value,
    _expected,
    _parse_connection_token_command,
    _parse_hint_msg,
    _parse_range,
    _parse_reasoning_cap,
    _parse_request_constraints,
    _parse_type_params,
    _resolve_shim_pair,
    _value_matches,
)
from ._sidecars import (  # noqa: F401
    _PLUGIN_NS_PREFIX,
    _clear_plugin_namespaces,
    _declared_transforms,
    _ensure_pkg,
    _load_auth_exchange,
    _load_quota_probes,
    _load_transforms,
    _plugin_namespaces,
    _read_transform_exports,
    _register_plugin_chain,
    _rollback_plugin_load,
    describe_load_failure,
)
from ._walk import (  # noqa: F401
    _lane_holds_shims,
    _load_group_folder,
    _load_provider_tolerant,
    _load_type_folder,
    _path_selected,
    # ``load_providers_from_dir`` stays below: it reads ``_PROVIDERS_DIR`` here.
)


def validate_shim_declaration(
    cfg: dict[str, Any], source: str | Path
) -> tuple[str, str]:
    """Check a parsed ``provider.yaml`` the way the loader does.

    The one place a shim definition's ``(model_type, format)`` is validated, so
    a host that accepts a definition (the admin editor) refuses exactly what
    :func:`load_providers_from_dir` would, with the same message.

    Args:
        cfg: The parsed YAML mapping.
        source: The file it came from, named in the error.

    Returns:
        The ``(format, model_type)`` pair the definition declares.

    Raises:
        ValueError: If the definition declares ``base:``, omits ``model_type`` or
            ``format``, or names a retired or unknown format.
    """
    return _resolve_shim_pair(cfg, Path(source))


def load_providers_from_dir(
    providers_dir: Path,
    *,
    group: str | None = None,
    on_error: str = "raise",
    only_paths: Collection[str] | None = None,
    keep_names: Callable[[str], bool] | None = None,
) -> list[ProviderShim]:
    """Scan *providers_dir* for provider shims and register them.

    This is the public entry point for loading shims from an arbitrary
    directory.  Downstream packages (e.g. argo-proxy) can call this to
    load their own shim directories alongside the built-in ones.

    Plugin transforms use a separate module namespace
    (``_llm_rosetta_plugin_shims.*``) to avoid collisions with built-in
    modules in ``sys.modules``.

    The contract is the model-typed shape stated in
    ``devtools/_shim_template/README_en.md``: ``<vendor>[/<lane>]/<type>`` with a
    ``<format>`` level only when that type holds more than one lane.  Two older
    shapes still load — they are tolerated, not the contract, and
    ``shims check`` reports both:

    * **Flat (legacy)** — a direct child with ``provider.yaml`` (e.g.
      ``openai/``).
    * **Grouped (legacy)** — a child WITHOUT ``provider.yaml`` whose own
      children each contain one (e.g. ``vendor/messages/``,
      ``vendor/chat_completions/``).

    * **Model-typed (the contract)** — a child WITHOUT ``provider.yaml`` whose
      name is a model-type directory (``llm``, ``embeddings``, ``rerank``,
      ``decisions``): the directory itself holds the leaf, or it groups format
      directories, and a lane directory may sit above it.  Every shim below it
      must declare the matching ``model_type``.

    Either of the last two may also carry a **vendor manifest**: a
    ``provider.yaml`` at the vendor root that declares no ``name``/``format``,
    holding the ``connection:`` defaults every leaf inherits (the credential,
    usually) and optionally an ``auth.py`` beside it.  The manifest is
    optional, and a leaf overrides any field it names.

    Args:
        providers_dir: Root directory to scan for provider subdirectories.
        group: Optional group name prefix for all shims loaded from this
            directory.  When ``None``, the directory structure determines
            grouping automatically.
        on_error: ``"raise"`` (default) propagates a bad ``provider.yaml`` or
            a failing ``transforms.py``; ``"warn"`` logs and skips that
            provider so one bad shim cannot abort the whole scan.  External
            sources use ``"warn"`` so a third-party repo cannot take down
            startup; built-ins keep the strict default.
        only_paths: Posix paths, relative to *providers_dir*, that name the
            leaves to load.  A directory that is neither one of them nor on
            the way to one is skipped before its ``provider.yaml`` is read, so
            a source can declare which of its directories the loader may
            parse.  ``None`` scans everything.
        keep_names: Optional predicate over a leaf's ``name``, applied after
            the YAML is parsed and before any module import.  A declined leaf
            is neither imported nor registered; it is not an error.

    Returns:
        List of registered :class:`ProviderShim` instances.
    """
    builtin = providers_dir == _PROVIDERS_DIR
    shims: list[ProviderShim] = []
    for d in sorted(providers_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        if not _path_selected(d, providers_dir, only_paths):
            continue
        if (d / "provider.yaml").exists():
            # A shim (it declares ``name``/``format``) or a vendor manifest
            # (it declares only ``connection:``).  A leaf stands alone; a
            # manifest supplies the defaults for the leaves below it.
            try:
                manifest = _vendor_manifest(d)
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning(
                    "Skipping vendor directory %s: %s", d, describe_load_failure(d, exc)
                )
                continue
            if manifest is None:
                shim = _load_provider_tolerant(
                    d,
                    group,
                    builtin=builtin,
                    root=providers_dir,
                    on_error=on_error,
                    keep_names=keep_names,
                )
                if shim is not None:
                    shims.append(shim)
                continue
        else:
            manifest = None
        # Group folder — load the vendor's shared quota probes once, then scan
        # its leaf providers.
        group_probes: tuple[Any, ...] = ()
        try:
            group_probes = _load_quota_probes(d / QUOTA_MODULE)
        except Exception as exc:
            if on_error == "raise":
                raise
            logger.warning("Skipping quota probes in %s: %s", d, exc)
        shims.extend(
            _load_group_folder(
                d,
                group,
                builtin=builtin,
                root=providers_dir,
                on_error=on_error,
                quota_probes=group_probes,
                manifest=manifest,
                only_paths=only_paths,
                keep_names=keep_names,
            )
        )
    return shims


def load_providers() -> list[ProviderShim]:
    """Load built-in provider shims and any plugin shims.

    1. Scans the bundled ``builtin/`` directory.
    2. Discovers the installed extensions (``llm_rosetta_ext.<id>`` folders and
       the ``llm_rosetta.extensions`` entry points) and registers their shims;
       entry points left in the retired ``llm_rosetta.shim_providers`` group
       are reported, not run.
    3. Registers built-in model list transforms for providers without
       shim directories (Jina, Cohere).

    Returns:
        Combined list of all registered :class:`ProviderShim` instances
        (built-in + plugin).
    """
    # 1. Built-in shims
    shims = load_providers_from_dir(_PROVIDERS_DIR)

    # 2. Extensions: installed folders that provide shims
    _PLUGIN_SOURCES.clear()
    from llm_rosetta.extensions import load_extensions

    shims.extend(load_extensions())

    # 3. Built-in transforms for providers without shim directories
    register_model_list_transform("jina", _jina_model_list_transform)
    register_model_list_transform("cohere", _cohere_model_list_transform)

    return shims


#: Which installed distribution registered each plugin shim, keyed by shim
#: name.  A plugin shim has no ``provider.yaml`` on disk, so nothing can walk to
#: it; this map is how the inventory says where the code came from instead of
#: filing it beside the built-ins.
_PLUGIN_SOURCES: dict[str, str] = {}


def builtin_providers_dir() -> Path:
    """Return the directory the bundled shims are loaded from.

    Returns:
        The bundled ``builtin/`` root, the one :func:`load_providers` scans.
    """
    return _PROVIDERS_DIR


def plugin_source(name: str) -> str | None:
    """Return the distribution that registered plugin shim *name*, if any.

    Args:
        name: A canonical shim name.

    Returns:
        ``"<distribution> <version>"`` for a shim a plugin registered, or
        ``None`` for a shim that came from a directory.
    """
    return _PLUGIN_SOURCES.get(name)


# ``llm_rosetta.contracts.shim.load_providers_from_dir`` hands its work to this
# function; the contract module imports nothing from the shim layer.
set_providers_dir_loader(load_providers_from_dir)

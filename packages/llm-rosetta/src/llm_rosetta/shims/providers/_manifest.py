"""Vendor manifests and the connection/quota/exchange a leaf inherits from one."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load

from ..provider_shim import (
    VendorManifest,
)
from ._fields import _declared_fields, _declares_identity
from ._sidecars import _load_auth_exchange
from ._constants import (
    _CONNECTION_KEYS,
    _IDENTITY_KEYS,
    _MANIFEST_KEYS,
)


def _vendor_manifest(vendor_dir: Path) -> VendorManifest | None:
    """Read *vendor_dir*'s vendor manifest, when the file there is one.

    A ``provider.yaml`` at a vendor root is either a shim (it declares
    ``name`` / ``format``) or a manifest of the defaults its leaves share.  The
    two are told apart by identity rather than by layout, because a flat leaf
    and a vendor root both sit directly under the scan root.

    Args:
        vendor_dir: A directory the caller has not recognised as a leaf.

    Returns:
        The manifest, or ``None`` when the directory has no ``provider.yaml``,
        when that file declares a shim, or when the directory holds neither a
        leaf nor an ``auth.py`` (the leaf loader then reports the missing
        identity).

    Raises:
        ValueError: A manifest declares something a manifest cannot carry.
    """
    path = vendor_dir / "provider.yaml"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        cfg = yaml_load(f.read())
    if cfg is None:
        # A comment-only manifest is a legitimate marker: the vendor's shared
        # facts are its ``auth.py`` beside it, and the comments say why.
        cfg = {}
    if _declares_identity(cfg):
        return None
    if not isinstance(cfg, dict):
        raise ValueError(
            f"{path}: a provider.yaml must be a mapping; a shim declares "
            f"'name' and 'format', a vendor manifest declares 'connection'"
        )
    has_leaf = any(
        child.resolve() != path.resolve() for child in vendor_dir.rglob("provider.yaml")
    )
    if not has_leaf and not (vendor_dir / "auth.py").exists():
        # Nothing below it to inherit: let the leaf loader report the missing
        # identity with its own (skip-and-warn) behaviour, rather than turning
        # one malformed third-party directory into a failed scan.
        return None
    unknown = set(cfg) - set(_MANIFEST_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: a vendor manifest may declare {', '.join(_MANIFEST_KEYS)}; "
            f"found {', '.join(sorted(unknown))}. Shim identity — 'name' "
            f"and 'format' — belongs in a leaf's provider.yaml"
        )
    quota = cfg.get("quota")
    if quota is not None and not isinstance(quota, dict):
        raise ValueError(f"{path}: quota must be a mapping")
    vendor_fields: dict[str, str | None] = {}
    for key in ("logo", "group", "group_label"):
        value = cfg.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string")
        vendor_fields[key] = value
    connection = cfg.get("connection")
    if connection is not None and not isinstance(connection, dict):
        raise ValueError(f"{path}: connection must be a mapping")
    unknown = set(connection or {}) - set(_CONNECTION_KEYS)
    if unknown:
        # A typo here is worse than in a leaf: the value is dropped silently
        # for every lane of the vendor at once, in the file that is supposed
        # to be the single source.
        raise ValueError(
            f"{path}: unknown connection key(s) {', '.join(sorted(unknown))}; "
            f"a vendor manifest may declare {', '.join(_CONNECTION_KEYS)}"
        )
    return VendorManifest(
        vendor=vendor_dir.name,
        path=path,
        connection=dict(connection or {}),
        auth=_load_auth_exchange(vendor_dir / "auth.py"),
        fields=_declared_fields(cfg.get("fields"), path),
        quota=dict(quota or {}),
        **vendor_fields,
    )


def _leaf_quota(
    cfg: dict[str, Any], manifest: VendorManifest | None, path: Path
) -> dict[str, Any] | None:
    """The quota declaration a leaf actually carries.

    A vendor reads one allowance for every lane — the control plane and the
    workspace are the account's, not the format's — so the manifest's block
    applies unless the leaf names a field itself, and per field when it does.

    Raises:
        ValueError: The leaf's ``quota:`` is present but not a mapping.  A
            typo must not be read as "inherits the vendor's", which would put
            another declaration's probe on this lane in place of reporting it.
    """
    declared = cfg.get("quota")
    vendor = dict(manifest.quota) if manifest is not None else {}
    if declared is None:
        return vendor or None
    if not isinstance(declared, dict):
        raise ValueError(f"{path}: quota must be a mapping")
    return {**vendor, **declared}


def _leaf_connection(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a shim's own ``connection:`` values.

    The values live under ``connection:``; the flat top-level spelling was
    removed in 0.14.0 and is refused when the shim is loaded.
    """
    connection = cfg.get("connection")
    return dict(connection) if isinstance(connection, dict) else {}


def _merged_connection(
    leaf: dict[str, Any], manifest: VendorManifest | None
) -> tuple[dict[str, Any], Path | None]:
    """Merge a leaf's connection values over its vendor manifest's.

    The leaf wins per field, and ``headers`` merges per key, so a lane that
    needs one extra header adds it without restating the rest.  A vendor whose
    lanes genuinely differ (ALCF's per-cluster ``base_url``) keeps the
    difference on the leaf.

    Args:
        leaf: The leaf's own ``connection`` values.
        manifest: The vendor manifest, or ``None`` for a standalone shim.

    Returns:
        The merged mapping and the directory a relative ``token_command``
        argument must resolve against — the manifest's when the command came
        from there, else ``None`` so the caller uses the leaf's own directory.
    """
    if manifest is None:
        return leaf, None
    merged = {**manifest.connection, **leaf}
    manifest_headers = manifest.connection.get("headers")
    leaf_headers = leaf.get("headers")
    if isinstance(manifest_headers, dict) or isinstance(leaf_headers, dict):
        merged["headers"] = {**(manifest_headers or {}), **(leaf_headers or {})}
    from_manifest = (
        "token_command" not in leaf and "token_command" in manifest.connection
    )
    return merged, manifest.path.parent if from_manifest else None


def _leaf_exchange(provider_dir: Path, manifest: VendorManifest | None) -> Any | None:
    """The exchange a leaf authenticates with: its own ``auth.py`` or the vendor's.

    A vendor declares one exchange, beside its manifest, and every leaf
    inherits it.  A leaf that ships its own is a second answer to the same
    question, so it is refused rather than allowed to shadow the vendor's.

    Args:
        provider_dir: The leaf directory.
        manifest: The vendor manifest, or ``None`` for a standalone shim.

    Returns:
        The exchange object, or ``None`` when neither file exists.

    Raises:
        ValueError: The leaf and the vendor both declare an exchange.
    """
    exchange = _load_auth_exchange(provider_dir / "auth.py")
    if manifest is None or manifest.auth is None:
        return exchange
    if exchange is not None:
        raise ValueError(
            f"{provider_dir / 'auth.py'}: the vendor already ships its "
            f"exchange in {manifest.path.parent / 'auth.py'}; a vendor "
            f"declares one exchange"
        )
    return manifest.auth


@dataclass(frozen=True)
class ManifestLeaf:
    """One shim a vendor manifest's connection reaches.

    Attributes:
        name: The shim name the leaf declares (empty when it declares none).
        path: The leaf's ``provider.yaml``.
        own: The connection values the leaf declares itself, under their
            ``connection:`` names.
        effective: The connection the loader would use: *own* over the
            manifest's.
    """

    name: str
    path: Path
    own: dict[str, Any]
    effective: dict[str, Any]


@dataclass(frozen=True)
class VendorManifestDescription:
    """What a vendor manifest declares and what its leaves inherit from it.

    Attributes:
        path: The manifest's ``provider.yaml``.
        vendor: The directory name the manifest sits in.
        connection: The ``connection:`` values in force: the manifest's own, or
            the ``connection`` argument of :func:`describe_vendor_manifest`.
        leaves: Every shim below the manifest, in path order.  A nested vendor
            manifest owns the shims under it, so they are not listed here.
        connection_keys: The ``connection:`` keys a manifest may declare.
        identity_keys: The keys that make a ``provider.yaml`` a shim rather
            than a manifest.
        schema: The ``vendorManifest`` and ``connection`` definitions of
            ``provider.schema.json``, as ``{"manifest": ..., "connection": ...}``.
    """

    path: Path
    vendor: str
    connection: dict[str, Any]
    leaves: tuple[ManifestLeaf, ...] = ()
    connection_keys: tuple[str, ...] = _CONNECTION_KEYS
    identity_keys: tuple[str, ...] = _IDENTITY_KEYS
    schema: dict[str, dict[str, Any]] = field(default_factory=dict)


def _read_mapping(path: Path) -> dict[str, Any]:
    """A ``provider.yaml`` as a mapping; a comment-only or unreadable file is empty."""
    try:
        cfg = yaml_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _manifest_schema() -> dict[str, dict[str, Any]]:
    """The manifest and connection definitions of ``provider.schema.json``."""
    text = (Path(__file__).parent / "provider.schema.json").read_text(encoding="utf-8")
    definitions = json.loads(text).get("definitions") or {}
    return {
        "manifest": definitions.get("vendorManifest") or {},
        "connection": definitions.get("connection") or {},
    }


def _leaf_definitions(manifest_path: Path) -> list[Path]:
    """The shim ``provider.yaml`` files below *manifest_path*'s directory.

    A directory whose own ``provider.yaml`` declares no identity is a nested
    vendor manifest; it owns what is below it, so the walk does not enter it.
    """
    root = manifest_path.parent
    found: list[Path] = []
    for directory, subdirs, files in os.walk(root):
        here = Path(directory)
        subdirs.sort()
        if here == root or "provider.yaml" not in files:
            continue
        cfg = _read_mapping(here / "provider.yaml")
        if any(key in cfg for key in _IDENTITY_KEYS):
            found.append(here / "provider.yaml")
        else:
            subdirs[:] = []
    return sorted(found)


def describe_vendor_manifest(
    path: str | os.PathLike[str],
    connection: Mapping[str, Any] | None = None,
) -> VendorManifestDescription:
    """Describe a vendor manifest and what each of its shims inherits from it.

    This is the read-only view an editor needs: the keys a manifest may use,
    the schema, and per shim both the values it declares and the values the
    loader would end up with.  It reads files only and registers nothing.

    Args:
        path: A vendor's ``provider.yaml``, or the vendor directory holding it.
        connection: Describe the leaves as if the manifest declared this
            ``connection:`` instead of what the file holds, for a preview of an
            edit before it is saved.

    Returns:
        The description.

    Raises:
        FileNotFoundError: *path* holds no ``provider.yaml``.
    """
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "provider.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no provider.yaml at {manifest_path}")
    if connection is None:
        declared = _read_mapping(manifest_path).get("connection")
        in_force = dict(declared) if isinstance(declared, dict) else {}
    else:
        in_force = dict(connection)
    manifest = VendorManifest(
        vendor=manifest_path.parent.name, path=manifest_path, connection=in_force
    )

    leaves: list[ManifestLeaf] = []
    for leaf_path in _leaf_definitions(manifest_path):
        cfg = _read_mapping(leaf_path)
        own = _leaf_connection(cfg)
        effective, _resolve = _merged_connection(own, manifest)
        leaves.append(
            ManifestLeaf(
                name=str(cfg.get("name") or ""),
                path=leaf_path,
                own=own,
                effective=dict(effective),
            )
        )
    return VendorManifestDescription(
        path=manifest_path,
        vendor=manifest.vendor,
        connection=in_force,
        leaves=tuple(leaves),
        schema=_manifest_schema(),
    )


@dataclass(frozen=True)
class ManifestVocabulary:
    """What a vendor manifest may say, without reference to any file.

    A host that edits manifest text, or describes a shim that has no file on
    disk, needs the vocabulary and the merge rule but no path.

    Attributes:
        connection_keys: The ``connection:`` keys a manifest may declare.
        identity_keys: The keys that make a ``provider.yaml`` a shim rather
            than a manifest.
        schema: The ``vendorManifest`` and ``connection`` definitions of
            ``provider.schema.json``, as ``{"manifest": ..., "connection": ...}``.
    """

    connection_keys: tuple[str, ...]
    identity_keys: tuple[str, ...]
    schema: dict[str, dict[str, Any]]


def manifest_vocabulary() -> ManifestVocabulary:
    """Return the keys and schema a vendor manifest is read against.

    The values are the loader's own, so a host that validates manifest text
    with them agrees with what the loader accepts.  It reads the schema file
    and registers nothing.
    """
    return ManifestVocabulary(
        connection_keys=_CONNECTION_KEYS,
        identity_keys=_IDENTITY_KEYS,
        schema=_manifest_schema(),
    )


def leaf_connection(cfg: Any) -> dict[str, Any]:
    """Return the connection values one shim declares itself.

    The values come under their ``connection:`` names, from the ``connection:``
    block.  The flat top-level spellings were removed in 0.14.0.

    Args:
        cfg: A shim's parsed ``provider.yaml``.

    Returns:
        The declared values, or an empty mapping when *cfg* is not a mapping.
    """
    return _leaf_connection(cfg) if isinstance(cfg, dict) else {}


def merge_connection(
    manifest_connection: Mapping[str, Any], leaf: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the connection a shim ends up with under a vendor manifest.

    This is the loader's merge: the leaf wins per field, and ``headers``
    merges per key.  Neither argument is modified.

    Args:
        manifest_connection: The ``connection:`` values the manifest declares.
        leaf: The values the shim declares itself, as :func:`leaf_connection`
            returns them.

    Returns:
        The merged mapping.
    """
    manifest = VendorManifest(
        vendor="",
        path=Path("provider.yaml"),
        connection=dict(manifest_connection),
    )
    merged, _resolve = _merged_connection(dict(leaf), manifest)
    return merged

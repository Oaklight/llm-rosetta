"""Vendor manifests and the connection/quota/exchange a leaf inherits from one."""

from __future__ import annotations

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
    _FLAT_CONNECTION_ALIASES,
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
            f"found {', '.join(sorted(unknown))}. Shim identity — 'name', "
            f"'format', 'base' — belongs in a leaf's provider.yaml"
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
    """Return a shim's connection values under their ``connection:`` names.

    A shim declares them under ``connection:`` or, in the pre-rename spelling,
    at the top level.  Both are read into one mapping so a vendor manifest
    merges the same way for either form.
    """
    connection = cfg.get("connection")
    if isinstance(connection, dict):
        return dict(connection)
    flat = {key: cfg[key] for key in _CONNECTION_KEYS if key in cfg}
    for legacy, canonical in _FLAT_CONNECTION_ALIASES.items():
        if legacy in cfg:
            flat[canonical] = cfg[legacy]
    return flat


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

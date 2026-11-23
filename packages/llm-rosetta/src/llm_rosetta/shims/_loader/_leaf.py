"""Building one :class:`ProviderShim` from a leaf directory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_rosetta._vendor.yaml import load as yaml_load

from ...contracts import MODEL_LIST_TRANSFORM, QUOTA_MODULE
from llm_rosetta.contracts.shim import (
    ConnectionConfig,
    ProviderShim,
    QuotaDeclaration,
    ReasoningCapability,
    SoftErrorPattern,
    ToolsConfig,
    VendorManifest,
)
from ..provider_shim import register_shim
from ._fields import _declared_fields, _merged_fields
from ._manifest import _leaf_connection, _leaf_exchange, _leaf_quota, _merged_connection
from ._model_list import _model_list_transforms
from ._parse import (
    _parse_batch,
    _parse_connection_argv,
    _parse_connection_env_unset,
    _parse_connection_meta,
    _parse_connection_token_command,
    _parse_hint_msg,
    _parse_reasoning_cap,
    _parse_request_constraints,
    _parse_type_params,
    _reject_removed_keys,
    _resolve_shim_pair,
)
from ._sidecars import _load_quota_probes, _load_transforms
from ._constants import (
    logger,
)


def _load_single_provider(
    provider_dir: Path,
    *,
    group: str | None = None,
    _builtin: bool = True,
    root: Path | None = None,
    quota_probes: tuple[Any, ...] = (),
    expected_type: str | None = None,
    manifest: VendorManifest | None = None,
    keep_names: Callable[[str], bool] | None = None,
) -> ProviderShim | None:
    """Load a single provider from *provider_dir* and register it.

    Args:
        provider_dir: Directory containing ``provider.yaml``.
        group: Name of the parent group folder (``None`` for top-level shims).
        _builtin: Whether this is a built-in provider directory.
        expected_type: The model type the enclosing model-type directory
            names.  When set, the shim's own ``model_type`` must match it.
        manifest: The vendor manifest whose connection defaults this leaf
            inherits and may override.
        keep_names: Optional predicate over the shim name, applied to the
            parsed leaf before its modules are imported.  A shim it declines
            is neither imported nor registered.

    Returns:
        The registered :class:`ProviderShim`, or ``None`` on failure.
    """
    yaml_path = provider_dir / "provider.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml_load(f.read())
    # A leaf that still declares ``base`` passes this gate on purpose:
    # ``_resolve_shim_pair`` refuses it with the message that names ``format``,
    # where skipping it here would only log a warning.
    if (
        not isinstance(cfg, dict)
        or "name" not in cfg
        or ("format" not in cfg and "base" not in cfg)
    ):
        logger.warning(
            "Skipping %s: missing 'name' or 'format'. A "
            "vendor manifest belongs at the vendor root, not in a leaf "
            "directory",
            yaml_path,
        )
        return None

    # Selection is decided from the parsed leaf, before any module is
    # imported: a declined shim costs neither its ``transforms.py`` import nor
    # its registration, and a broken one cannot fail the load it was left out of.
    if keep_names is not None and not keep_names(str(cfg["name"])):
        return None

    _reject_removed_keys(cfg, yaml_path)
    fmt, model_type = _resolve_shim_pair(cfg, yaml_path)
    if expected_type is not None and model_type != expected_type:
        raise ValueError(
            f"{yaml_path}: directory {provider_dir.name!r} is the "
            f"{expected_type!r} model-type level, but this shim declares "
            f"model_type={model_type!r}; move it under the {model_type!r} "
            f"directory or correct the declaration"
        )

    params = _parse_type_params(cfg, model_type, yaml_path)
    request_constraints = _parse_request_constraints(cfg, yaml_path)
    batch = _parse_batch(cfg, model_type, fmt, yaml_path)

    pre_t, post_t, ir_t, resp_t, transforms_mod = _load_transforms(
        provider_dir, group=group, _builtin=_builtin, root=root
    )

    # A leaf-level quota.py complements the group's: the group module carries
    # a vendor's shared lanes, a leaf module a one-off.  Both are attached to
    # every shim in the leaf so a config can select either by probe id.
    leaf_quota_probes = _load_quota_probes(provider_dir / QUOTA_MODULE)
    all_quota_probes = tuple(quota_probes) + leaf_quota_probes

    # A credential exchange that does not fit in an argv.  It is the module
    # form of the same declaration as ``connection.token_command``, so a shim
    # that ships both leaves the loader with no single answer: reject it here
    # rather than let one silently shadow the other.  A vendor declares its
    # exchange once, beside the manifest, so every lane authenticates alike.
    exchange = _leaf_exchange(provider_dir, manifest)

    # Parse optional reasoning capability config from YAML.
    reasoning_cfg = cfg.get("reasoning")
    reasoning_cap: ReasoningCapability | None = None
    if isinstance(reasoning_cfg, dict):
        reasoning_cap = _parse_reasoning_cap(reasoning_cfg)

    # Parse per-model reasoning overrides (inherit provider defaults).
    model_reasoning: dict[str, ReasoningCapability] | None = None
    if isinstance(reasoning_cfg, dict) and isinstance(
        reasoning_cfg.get("model_overrides"), dict
    ):
        model_reasoning = {}
        for model_id, overrides in reasoning_cfg["model_overrides"].items():
            if not isinstance(overrides, dict):
                continue
            assert reasoning_cap is not None  # model_overrides requires reasoning
            model_reasoning[model_id] = _parse_reasoning_cap(
                overrides, base=reasoning_cap
            )

    # Parse connection config: the vendor manifest's defaults first, then the
    # leaf's own values, which win per field.  A field the leaf omits is the
    # shared one — that is how a vendor keeps one credential declaration for
    # every lane while each lane keeps its own base URL.
    conn_raw, command_dir = _merged_connection(_leaf_connection(cfg), manifest)
    connection = ConnectionConfig(
        base_url=conn_raw.get("base_url"),
        api_key_env=conn_raw.get("api_key_env"),
        models_path=conn_raw.get("models_path"),
        default_path=conn_raw.get("default_path"),
        model_id_field=conn_raw.get("model_id_field"),
        auth_header=conn_raw.get("auth_header"),
        keyless=bool(conn_raw.get("keyless", False)),
        session_header=conn_raw.get("session_header"),
        headers=conn_raw.get("headers"),
        request_header=conn_raw.get("request_header"),
        session_id_format=conn_raw.get("session_id_format"),
        request_id_format=conn_raw.get("request_id_format"),
        stream_required=bool(conn_raw.get("stream_required", False)),
        transport=str(conn_raw.get("transport") or "http"),
        acp_argv=_parse_connection_argv(conn_raw.get("acp_argv"), cfg.get("name", "")),
        acp_env=conn_raw.get("acp_env"),
        acp_env_unset=_parse_connection_env_unset(
            conn_raw.get("acp_env_unset"), cfg.get("name", "")
        ),
        acp_session_meta=_parse_connection_meta(
            conn_raw.get("acp_session_meta"), cfg.get("name", "")
        ),
        token_command=_parse_connection_token_command(
            conn_raw.get("token_command"),
            command_dir or provider_dir,
            cfg.get("name", ""),
        ),
        token_refresh_interval=conn_raw.get("token_refresh_interval"),
        token_command_env=conn_raw.get("token_command_env"),
        manage_usage_url=conn_raw.get("manage_usage_url"),
    )

    if exchange is not None and connection.token_command:
        raise ValueError(
            f"{yaml_path}: declares both auth.py and connection.token_command; "
            f"a shim ships one form of its credential exchange"
        )

    # Parse the quota declaration: which probe reads this lane, and how often
    # the gateway should read it.  Parsers live in the gateway; the shim only
    # names one.
    quota_raw = _leaf_quota(cfg, manifest, yaml_path)
    quota_decl: QuotaDeclaration | None = None
    if isinstance(quota_raw, dict):
        quota_decl = QuotaDeclaration(
            probe=quota_raw.get("probe"),
            url=quota_raw.get("url"),
            interval=int(quota_raw.get("interval", 0) or 0),
            demote_at=float(quota_raw.get("demote_at", 100.0)),
            in_band=bool(quota_raw.get("in_band", False)),
        )

    # Parse tools config.
    tools_raw = cfg.get("tools")
    if isinstance(tools_raw, dict):
        tools = ToolsConfig(
            custom_tools=tools_raw.get("custom_tools", False),
            max_description_length=tools_raw.get("max_description_length"),
            search_mode=tools_raw.get("search_mode", "disabled"),
            multimodal_result=tools_raw.get("multimodal_result"),
            tool_groups=tools_raw.get("tool_groups", False),
            intrinsic_tools=tuple(tools_raw.get("intrinsic_tools") or ()),
        )
    else:
        tools = ToolsConfig()

    # Parse soft-error patterns for 200-but-error detection.
    raw_patterns = cfg.get("soft_error_patterns", [])
    _soft_list: list[SoftErrorPattern] = []
    for p in raw_patterns:
        if not isinstance(p, dict):
            logger.warning("Skipping non-dict soft_error_pattern in %s", yaml_path)
            continue
        sep = SoftErrorPattern(
            pattern=p["pattern"],
            status_code=p["status_code"],
            message=p["message"],
        )
        sep.compiled  # eagerly validate regex at load time
        _soft_list.append(sep)
    soft_errors = tuple(_soft_list)

    shim = ProviderShim(
        name=cfg["name"],
        format=fmt,
        model_type=model_type,
        # The three fields that describe the vendor rather than one lane come
        # from the manifest unless the leaf names its own: a vendor keeps its
        # mark and its grouping once, and a lane that differs still wins.
        logo=cfg.get("logo") or (manifest.logo if manifest else None),
        display_name=cfg.get("display_name"),
        group=cfg.get("group") or (manifest.group if manifest else None),
        group_label=cfg.get("group_label")
        or (manifest.group_label if manifest else None),
        hint_msg=_parse_hint_msg(cfg.get("hint_msg"), yaml_path),
        free_source=cfg.get("free_source", False),
        connection=connection,
        auth=exchange,
        manifest=manifest,
        # The form inputs this shim declares.  A leaf's own block wins per
        # key, and the vendor's fills the rest; the exchange may add more in
        # code, which the gateway merges by key when it renders the form.
        fields=_merged_fields(
            _declared_fields(cfg.get("fields"), yaml_path),
            manifest.fields if manifest is not None else (),
        ),
        quota=quota_decl,
        quota_probes=all_quota_probes,
        batch=batch,
        tools=tools,
        pre_ir_transforms=pre_t,
        post_ir_transforms=post_t,
        ir_transforms=ir_t,
        response_body_transforms=resp_t,
        reasoning=reasoning_cap,
        model_reasoning=model_reasoning,
        response_id_prefix=cfg.get("response_id_prefix", ""),
        preserve_fields=tuple(cfg.get("preserve_fields") or ()),
        request_fields=tuple(cfg.get("request_fields") or ()),
        default_capabilities=tuple(cfg.get("default_capabilities") or ()),
        store_default=cfg.get("store_default"),
        hoist_system_messages=cfg.get("hoist_system_messages", True),
        soft_error_patterns=soft_errors,
        params=params,
        request=request_constraints,
    )
    register_shim(shim)

    # Register convention-based model_list_transform if the transforms
    # module exports one (e.g. shims that slug-ify display names).
    if transforms_mod is not None:
        mlt = getattr(transforms_mod, MODEL_LIST_TRANSFORM, None)
        if mlt is not None:
            _model_list_transforms[shim.name] = mlt

    logger.debug(
        "Registered provider shim: %s (model_type=%s format=%s)",
        shim.name,
        shim.model_type,
        shim.format,
    )
    return shim

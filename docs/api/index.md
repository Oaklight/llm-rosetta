---
title: API Reference
---

# API Reference

## Library API

- [Converters](converters.md) — converter classes for each provider
- [IR Types](types.md) — Intermediate Representation type definitions
- [Tool Ops](tool-ops.md) — tool definition conversion between IR and provider formats
- [Shims](shims.md) — provider and model shim classes and registry
- [Utilities](utils.md) — helper functions and auto-detection

## Gateway API

- [Admin API](admin.md) — gateway admin panel REST API endpoints
- [Observability](observability.md) — the gateway's telemetry stack (metrics, request logging, persistence, profiling) and the core's usage metering

## The core surface {#the-core-surface}

The core states the tier of each public module once, in `llm_rosetta/_tiers.py`. A module that file does not list is internal and may move in any release.

| Tier | Modules | Promise |
| --- | --- | --- |
| User API | `llm_rosetta`, `converters` (with `converters.base`, `converters.registry` and the format packages), `fidelity`, `ir.model`, `pipeline`, `retired_names`, `shims`, `tool_ops`, `types` | The library API: converting requests and responses, the converter classes, the IR types, the pipelines. A breaking change follows the deprecation policy. |
| Extension surface | `llm_rosetta.api`, `llm_rosetta.contracts` and its submodules, `llm_rosetta.testing` | What a host or a shim author builds on. Within a minor release it only gains names. |
| Compat | `llm_rosetta.auto_detect`, `llm_rosetta.gateway`, `llm_rosetta.observability`, `llm_rosetta.shims.provider_shim` (its data-class re-exports), `llm_rosetta.shims.providers`, `llm_rosetta.routing`, `llm_rosetta.capabilities`, `llm_rosetta.converters.base.context`, `llm_rosetta.types.ir` (and `.extensions`) | Deprecated paths kept importable for a release that shipped them. Each names its replacement and its removal version, 0.15.0. `auto_detect` forwards `detect_provider`, `get_converter_for_provider`, `convert`, `convert_response` and `ProviderType`, all importable from `llm_rosetta`. |
| Internal | every other module | May move in any release. |

`tests/test_tiers.py` checks that every module of the first three tiers imports. `scripts/dev/public_api_diff.py` takes its list of compared core modules from the same file, and reads each compared tree's own copy: against a base that ships `_tiers.py` (from 0.14.0 on), the base is measured by the modules it declared, so a module dropped from the table is reported as removed. The 0.13.0 base has no such file and is measured with the script's list; for it, a test keeps a frozen list of the modules v0.13.0 shipped as public and requires each to stay in a tier. The diff compares module-level names and constructor signatures only: an attribute removed from a class, such as the three `ProviderShim` properties of 0.14.0, is outside it and recorded in `REMOVED_ATTRIBUTES` in `tests/test_no_deprecation_bounds.py`.

### Who imports what {#who-imports-what}

- A host, such as the gateway or a router built on core, imports `llm_rosetta.api` and `llm_rosetta.contracts`.
- A shim imports `llm_rosetta.contracts` only. This covers its sidecars (`transforms.py`, `auth.py`, `quota.py`) and the code of an extension.

A shim must not import `llm_rosetta.api`. Extensions and sidecars run while `import llm_rosetta` is still in progress, and `api` needs the finished package. See [What a shim may import](../guide/shims.md#what-a-shim-may-import).

`llm_rosetta.api` has a literal, grouped `__all__` of 82 names: chat conversion, the typed pipelines, the format catalog and model-type vocabulary, provider and shim names, the shim registry (read and write side) and loaders, the manifest vocabulary and leaf merge, the command dispatcher (`run_cli`, `register_command`), the extension loader and installer (`list_extensions`, `install_wheel`, `install_from_index`, `load_index`, `extensions_dir` and related), `run_ext_check`, fidelity and profiling, and the IR types a host inspects. The test helpers that snapshot and restore the registry (`snapshot_registry`, `restore_registry`) are in `llm_rosetta.testing`. Core code never imports `llm_rosetta.api`: the dependency runs from the host to the core.

### The contract modules {#contract-modules}

`llm_rosetta.contracts` holds one module per contract. The package itself holds only constants, so importing it loads no converter and no shim registry.

| Module | Holds | Former path (0.14 development releases) |
| --- | --- | --- |
| `llm_rosetta.contracts` | `SURFACE_VERSION`, `IR_VERSION`, the sidecar file names (`TRANSFORMS_MODULE`, `AUTH_MODULE`, `QUOTA_MODULE`) and the names the shim loader reads from them (`PRE_IR_TRANSFORMS`, `POST_IR_TRANSFORMS`, `IR_TRANSFORMS`, `RESPONSE_BODY_TRANSFORMS`, `MODEL_LIST_TRANSFORM`, `QUOTA_PROBES` = `"PROBES"`, `AUTH_EXCHANGE` = `"EXCHANGE"`) | `SURFACE_VERSION` was defined in `llm_rosetta.api`, which re-exports the same object |
| `llm_rosetta.contracts.shim` | `ProviderShim`, `ConnectionConfig`, `ToolsConfig`, `ReasoningCapability`, `VendorManifest`, `QuotaDeclaration`, `BatchDeclaration`, `SoftErrorPattern`, `TYPE_PARAM_KEYS`, the literal types `ToolSearchMode`, `ReplayMissing`, `UnsignedBlocks`, and `load_providers_from_dir` | `llm_rosetta.shims.provider_shim` (deprecated re-exports, removed in 0.15.0) |
| `llm_rosetta.contracts.extension` | The manifest an extension carries in `plugin.yaml`: `ExtensionManifest`, `parse_extension_manifest`, `validate_extension_manifest` against the shipped `extension_manifest.schema.json`, the compatibility rule `compatibility_problems` (surface equal to `SURFACE_VERSION`, IR not above `IR_VERSION`, platforms), the module names a kind reads (`COMMANDS_MODULE`, `TRANSPORT_MODULE`) and `resolve_transport` | New in 0.14.0 |
| `llm_rosetta.contracts.command` | A subcommand a distribution adds to the `llm-rosetta` dispatcher: `CommandSpec`, `register_command`, `resolve_command`, the entry-point group, and the reserved names `CORE_COMMAND_NAMES` (`ext`, `help`) and `GATEWAY_COMMAND_NAMES` (`gateway`, `serve`) | New in 0.14.0 |
| `llm_rosetta.contracts.resolver` | How the conversion layers read a shim by name without importing the shim layer: `lookup_shim`, `resolve_shim`, `resolve_format`, `match_model_reasoning`, and the hooks the shim layer registers | New in 0.14.0 |
| `llm_rosetta.contracts.transport` | The upstream transport protocol, its response types (`UpstreamResponse`, `UpstreamStream`) and the provider view a lane reads | New in 0.14.0 |
| `llm_rosetta.contracts.cli_login` | The vendor-CLI sign-in driver an `auth.py` sidecar uses (standard library only) | New in 0.14.0 |
| `llm_rosetta.contracts.transform` | The hook types `Transform`, `IRTransform`, `TransformContext`, `NamedTransform`, `NamedIRTransform`; `apply_transforms`, `apply_ir_transforms`; the body primitives (`strip_fields`, `rename_field`, `set_defaults`, `force_fields`, …) and the IR primitives (`auto_cache_breakpoints`, `hoist_late_system_messages`, `truncate_images`, `unwind_parallel_tool_calls`) | `llm_rosetta.transforms` (removed) |
| `llm_rosetta.contracts.auth` | The credential exchange a shim ships in `auth.py`: `AuthExchange`, `BaseAuthExchange`, `AuthContext`, `AuthExchangeError`, `InteractiveSetupRequired`, `credential_for` and the field limits | `llm_rosetta.shims.auth_exchange` (removed) |
| `llm_rosetta.contracts.quota` | The allowance probe a shim ships in `quota.py`: `QuotaProbe`, `QuotaWindow`, `InBandSignal`, `ProbeTarget`, `ProbeMatch`, `resolve_probe`, `probe_fetch` and the helpers | `llm_rosetta.shims.quota_probe` (removed) |
| `llm_rosetta.contracts.routing` | `ProviderType`, `ResolvedRoute`, the `Router` protocol, `SoftErrorPattern` | `llm_rosetta.routing` (now a 0.13.0-compat forwarder, removed in 0.15.0) |
| `llm_rosetta.contracts.usage` | Usage metering semantics and the report: `UsageLedger`, `UsageReport`, `token_total`, `usage_columns`, `STANDARD_SHAPES`, … Import the package, never one of its submodules. | `llm_rosetta.observability.usage` (removed) |

### The gateway rule

The gateway reaches core only through two prefixes: `llm_rosetta.api` and `llm_rosetta.contracts` (for `contracts.usage`, the package only). The rule is strict and has no allowlist: a gateway source file that imports any other core module, or a private (`_`-prefixed) core name, fails `tests/gateway/test_core_surface_boundary.py`. The compat forwarders `llm_rosetta.routing` and `llm_rosetta.capabilities` are refused as well.

Extensions and shims follow the same rule with one prefix. `tests/test_shim_sidecar_boundary.py` fails when a `.py` file under `shims/builtin/<vendor>/`, a scaffold template or the scaffolder's output imports core other than through `llm_rosetta.contracts`, or imports another vendor's directory.

### The version constants {#version-constants}

| Constant | Defined in | Value | Rises when | Read by |
| --- | --- | --- | --- | --- |
| `SURFACE_VERSION` | `llm_rosetta.contracts`; `llm_rosetta.api` re-exports the same object | `1` | A name of the extension surface is removed or changes meaning. An added name does not raise it. | The gateway, at startup; `scripts/dev/public_api_diff.py` |
| `IR_VERSION` | `llm_rosetta.ir.model`; re-exported from `llm_rosetta.contracts` | `1` | A change to the IR types means an IR value written at the old level cannot be read at the new one. An added optional field does not raise it. | No code reads it yet |

Because `SURFACE_VERSION` rises only on a removal or a changed meaning, the surface is additive within a minor release, and a host asserts a minimum. 0.14.0 is the release of the surface. The gateway declares the lowest level it runs on (`llm_rosetta_gateway.surface.REQUIRED_SURFACE_VERSION`) and `create_app` asserts it at startup: an older core stops the gateway with a `RuntimeError` that names both levels, instead of a missing attribute at the first request. A core that predates the constant counts as level 0. `scripts/dev/public_api_diff.py` enforces the rise: a breaking finding in an extension-tier module against the last release, even one acknowledged in the allowlist, fails the check unless the head's `SURFACE_VERSION` is above the release's (a release without the constant counts as 0). The gateway's pin on the core stays `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`.

### Names added with the strict rule

| Name | Used for |
| --- | --- |
| `plugin_source`, `builtin_providers_dir` | the admin panel's shim-file routes |
| `run_cli`, `register_command`, `resolve_command`, `CommandSpec`, `CORE_COMMAND_NAMES` | the `llm-rosetta` dispatcher and the commands a host registers with it |
| `list_extensions`, `ExtensionStatus`, `register_kind_handler`, `KindLoadError`, `import_extension_module` | the extension loader and the per-kind handler registry |
| `install_wheel`, `install_from_index`, `load_index`, `parse_index`, `select_entry`, `remove_extension`, `installed_extensions`, `configure_extensions_dir`, `extensions_dir`, `EXTENSIONS_DIR_ENV` | the installer, the extension index and the extensions directory |
| `run_ext_check` | `ext check` (the check an extension's own test calls) |
| `leaf_connection`, `manifest_vocabulary`, `merge_connection`, `ManifestVocabulary` | the vendor-manifest vocabulary, its type, and the merge of a manifest's connection into a leaf |
| `SURFACE_VERSION` | the surface level described above |

`manifest_vocabulary()` returns a `ManifestVocabulary`, which `api` exports so a host can annotate it. `merge_connection(manifest_connection, leaf)` takes the manifest first, the reverse of the loader's private merge, and returns the merged mapping only: the directory a relative `token_command` resolves against is not returned, because the manifest it is given has no file. Removed names are listed in [Migrating to 0.14](../guide/migrating-to-0.14.md).

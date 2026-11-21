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
| User API | `llm_rosetta`, `auto_detect`, `converters` (with `converters.base`, `converters.registry` and the format packages), `fidelity`, `pipeline`, `retired_names`, `shims`, `tool_ops`, `types`, `types.ir` | The library API: converting requests and responses, the converter classes, the IR types, the pipelines. A breaking change follows the deprecation policy. |
| Extension surface | `llm_rosetta.api`, `llm_rosetta.contracts` and its submodules | What a host or a shim author builds on. Within a minor release it only gains names. |
| Compat | `llm_rosetta.gateway`, `llm_rosetta.observability`, `llm_rosetta.shims.provider_shim` (its data-class re-exports), `llm_rosetta.routing`, `llm_rosetta.capabilities`, `llm_rosetta.types.ir.extensions` | Deprecated paths kept importable for a release that shipped them. Each names its replacement and, where set, its removal version. |
| Internal | every other module | May move in any release. |

`tests/test_tiers.py` checks that every module of the first three tiers imports. `scripts/dev/public_api_diff.py` takes its list of compared core modules from the same file, and reads each compared tree's own copy: against a base that ships `_tiers.py` (from 0.14.0 on), the base is measured by the modules it declared, so a module dropped from the table is reported as removed. The 0.13.0 base has no such file and is measured with the script's list; for it, a test keeps a frozen list of the modules v0.13.0 shipped as public and requires each to stay in a tier. The diff compares module-level names and constructor signatures only: an attribute removed from a class, such as the three `ProviderShim` properties of 0.14.0, is outside it and recorded in `REMOVED_ATTRIBUTES` in `tests/test_no_deprecation_bounds.py`.

### Who imports what {#who-imports-what}

- A host, such as the gateway or a router built on core, imports `llm_rosetta.api` and `llm_rosetta.contracts`.
- A shim imports `llm_rosetta.contracts` only. This covers its sidecars (`transforms.py`, `auth.py`, `quota.py`) and the entry point of a shim package.

A shim must not import `llm_rosetta.api`. Plugin entry points and sidecars run while `import llm_rosetta` is still in progress, and `api` needs the finished package. See [What a shim may import](../guide/shims.md#what-a-shim-may-import).

`llm_rosetta.api` has a literal, grouped `__all__` of 63 names: chat conversion, the typed pipelines, the format catalog and model-type vocabulary, provider and shim names, the shim registry (read and write side) and loaders, the manifest vocabulary and leaf merge, the scaffold entry points, fidelity and profiling, and the IR types a host inspects. Core code never imports `llm_rosetta.api`: the dependency runs from the host to the core.

### The contract modules {#contract-modules}

`llm_rosetta.contracts` holds one module per contract. The package itself holds only constants, so importing it loads no converter and no shim registry.

| Module | Holds | Former path (0.14 development releases) |
| --- | --- | --- |
| `llm_rosetta.contracts` | `SURFACE_VERSION`, `IR_VERSION`, the sidecar file names (`TRANSFORMS_MODULE`, `AUTH_MODULE`, `QUOTA_MODULE`) and the names the shim loader reads from them (`PRE_IR_TRANSFORMS`, `POST_IR_TRANSFORMS`, `IR_TRANSFORMS`, `RESPONSE_BODY_TRANSFORMS`, `MODEL_LIST_TRANSFORM`, `QUOTA_PROBES` = `"PROBES"`, `AUTH_EXCHANGE` = `"EXCHANGE"`) | `SURFACE_VERSION` was defined in `llm_rosetta.api`, which re-exports the same object |
| `llm_rosetta.contracts.shim` | `ProviderShim`, `ConnectionConfig`, `ToolsConfig`, `ReasoningCapability`, `VendorManifest`, `QuotaDeclaration`, `BatchDeclaration`, `SoftErrorPattern`, `TYPE_PARAM_KEYS`, the literal types `ToolSearchMode`, `ReplayMissing`, `UnsignedBlocks`, and `load_providers_from_dir` | `llm_rosetta.shims.provider_shim` (deprecated re-exports, removed in 0.15.0) |
| `llm_rosetta.contracts.shim_manifest` | The manifest a shim source repository publishes (#1660): `validate_manifest` and `parse_manifest` against the shipped `shim_manifest.schema.json`, the records (`SourceManifest`, `VendorRelease`, `ShimRequirement`), and the install rule `select_release` / `requirement_problem` (surface equal to `SURFACE_VERSION`, IR not above `IR_VERSION`) | New in 0.14.0 |
| `llm_rosetta.contracts.transform` | The hook types `Transform`, `IRTransform`, `TransformContext`, `NamedTransform`, `NamedIRTransform`; `apply_transforms`, `apply_ir_transforms`; the body primitives (`strip_fields`, `rename_field`, `set_defaults`, `force_fields`, …) and the IR primitives (`auto_cache_breakpoints`, `hoist_late_system_messages`, `truncate_images`, `unwind_parallel_tool_calls`) | `llm_rosetta.transforms` (removed) |
| `llm_rosetta.contracts.auth` | The credential exchange a shim ships in `auth.py`: `AuthExchange`, `BaseAuthExchange`, `AuthContext`, `AuthExchangeError`, `InteractiveSetupRequired`, `credential_for` and the field limits | `llm_rosetta.shims.auth_exchange` (removed) |
| `llm_rosetta.contracts.quota` | The allowance probe a shim ships in `quota.py`: `QuotaProbe`, `QuotaWindow`, `InBandSignal`, `ProbeTarget`, `ProbeMatch`, `resolve_probe`, `probe_fetch` and the helpers | `llm_rosetta.shims.quota_probe` (removed) |
| `llm_rosetta.contracts.routing` | `ProviderType`, `ResolvedRoute`, the `Router` protocol, `SoftErrorPattern` | `llm_rosetta.routing` (now a 0.13.0-compat forwarder, removed in 0.15.0) |
| `llm_rosetta.contracts.usage` | Usage metering semantics and the report: `UsageLedger`, `UsageReport`, `token_total`, `usage_columns`, `STANDARD_SHAPES`, … Import the package, never one of its submodules. | `llm_rosetta.observability.usage` (removed) |

### The gateway rule

The gateway reaches core only through two prefixes: `llm_rosetta.api` and `llm_rosetta.contracts` (for `contracts.usage`, the package only). The rule is strict and has no allowlist: a gateway source file that imports any other core module, or a private (`_`-prefixed) core name, fails `tests/gateway/test_core_surface_boundary.py`. The compat forwarders `llm_rosetta.routing` and `llm_rosetta.capabilities` are refused as well.

Shims follow the same rule with one prefix. `tests/test_shim_sidecar_boundary.py` fails when a `.py` file under `shims/providers/<vendor>/`, a scaffold template or the scaffolder's output imports core other than through `llm_rosetta.contracts`, or imports another vendor's directory.

### The version constants {#version-constants}

| Constant | Defined in | Value | Rises when | Read by |
| --- | --- | --- | --- | --- |
| `SURFACE_VERSION` | `llm_rosetta.contracts`; `llm_rosetta.api` re-exports the same object | `1` | A name of the extension surface is removed or changes meaning. An added name does not raise it. | The gateway, at startup; `scripts/dev/public_api_diff.py` |
| `IR_VERSION` | `llm_rosetta.types.ir.version`; re-exported from `llm_rosetta.types.ir` and `llm_rosetta.contracts` | `1` | A change to the IR types means an IR value written at the old level cannot be read at the new one. An added optional field does not raise it. | No code reads it yet |

Because `SURFACE_VERSION` rises only on a removal or a changed meaning, the surface is additive within a minor release, and a host asserts a minimum. 0.14.0 is the release of the surface. The gateway declares the lowest level it runs on (`llm_rosetta_gateway.surface.REQUIRED_SURFACE_VERSION`) and `create_app` asserts it at startup: an older core stops the gateway with a `RuntimeError` that names both levels, instead of a missing attribute at the first request. A core that predates the constant counts as level 0. `scripts/dev/public_api_diff.py` enforces the rise: a breaking finding in an extension-tier module against the last release, even one acknowledged in the allowlist, fails the check unless the head's `SURFACE_VERSION` is above the release's (a release without the constant counts as 0). The gateway's pin on the core stays `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`.

### Names added with the strict rule

| Name | Used for |
| --- | --- |
| `plugin_source`, `builtin_providers_dir` | the admin panel's shim-file routes |
| `add_scaffold_arguments`, `run_scaffold`, `run_scaffold_check`, `run_scaffold_migrate` | the gateway's `shims new`, `shims check` and `shims migrate` commands |
| `leaf_connection`, `manifest_vocabulary`, `merge_connection`, `ManifestVocabulary` | the vendor-manifest vocabulary, its type, and the merge of a manifest's connection into a leaf |
| `SURFACE_VERSION` | the surface level described above |

`manifest_vocabulary()` returns a `ManifestVocabulary`, which `api` exports so a host can annotate it. `merge_connection(manifest_connection, leaf)` takes the manifest first, the reverse of the loader's private merge, and returns the merged mapping only: the directory a relative `token_command` resolves against is not returned, because the manifest it is given has no file. Removed names are listed in [Migrating to 0.14](../guide/migrating-to-0.14.md).

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

The gateway and external shim authors build on the core through one flat module of re-exports, `llm_rosetta.api`, and four contract modules. Everything else under `llm_rosetta` is internal and may move in any release. The five allowed import paths are:

- `llm_rosetta.api`
- `llm_rosetta.routing`
- `llm_rosetta.observability.usage` (the package, never its submodules)
- `llm_rosetta.shims.auth_exchange`
- `llm_rosetta.shims.quota_probe`

`llm_rosetta.api` has a literal, grouped `__all__` of 68 names: chat conversion, the typed pipelines, the format catalog and model-type vocabulary, provider and shim names, the shim registry (read and write side) and loaders, the manifest vocabulary and leaf merge, the scaffold entry points, fidelity and profiling, and the IR types a host inspects. Core code never imports `llm_rosetta.api`: the dependency runs from the host to the core.

### The gateway rule

The gateway reaches core only through those five paths. The rule is strict and has no allowlist: a gateway source file that imports any other core module, or a private (`_`-prefixed) core name, fails `tests/gateway/test_core_surface_boundary.py`. The legacy and private allowlists that let the migration land in steps are deleted.

### `SURFACE_VERSION`

`llm_rosetta.api.SURFACE_VERSION` is an integer, now `1`. It rises when a name is removed from the surface or changes meaning, and never when a name is added, so the surface is additive within a minor release. 0.14.0 is the release of the surface. The gateway declares the lowest level it runs on (`llm_rosetta_gateway.surface.REQUIRED_SURFACE_VERSION`) and `create_app` asserts it at startup: an older core stops the gateway with a `RuntimeError` that names both levels, instead of a missing attribute at the first request. A core that predates the constant counts as level 0. The gateway's pin on the core stays `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`.

### Names added with the strict rule

| Name | Used for |
| --- | --- |
| `plugin_source`, `builtin_providers_dir` | the admin panel's shim-file routes |
| `add_scaffold_arguments`, `run_scaffold`, `run_scaffold_check`, `run_scaffold_migrate` | the gateway's `shims new`, `shims check` and `shims migrate` commands |
| `leaf_connection`, `manifest_vocabulary`, `merge_connection` | the vendor-manifest vocabulary and the merge of a manifest's connection into a leaf |
| `SURFACE_VERSION` | the surface level described above |

`manifest_vocabulary()` returns a `ManifestVocabulary`; the type itself is not exported from `api`, because callers read its attributes and nothing annotates it. Removed names are listed in [Migrating to 0.14](../guide/migrating-to-0.14.md).

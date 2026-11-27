---
title: Adding an Extension
---

# Adding an Extension

An extension is one folder that provides shims, transports, commands, or any combination of them. This page covers creating, checking and publishing one. For what an extension is, how it installs and what `plugin.yaml` holds, read [Extensions](../guide/extensions.md) first.

A vendor that belongs in core (the model maker's own API with a static API key, no `auth.py`) is not an extension: it goes in the built-in tree, `packages/llm-rosetta/src/llm_rosetta/shims/builtin/<vendor>/`, scaffolded by `scripts/dev/new_shim.py`. See [Shim Placement and Maturity](../guide/shim-placement.md).

## Create it: `ext new`

```bash
llm-rosetta ext new acme                                  # one chat shim
llm-rosetta ext new acme --provides shims,transports,commands --verb hello --verb status
llm-rosetta ext new acme --format messages --name "Acme AI" --dest ./packages
```

| Argument | Meaning |
|----------|---------|
| `ext_id` | The extension id: lowercase letters, digits and underscores, starting with a letter. It is the folder name, the import name (`llm_rosetta_ext.<id>`) and the vendor key. |
| `--provides KINDS` | Comma-separated kinds to scaffold: `shims`, `transports`, `commands`. Default `shims`. |
| `--format` | Chat format of the first lane, for the shims kind. Default `chat_completions`. Refused when the shims kind is not provided. |
| `--verb VERB` | A command verb for the commands kind; repeat for several. Default `run`. Refused when the commands kind is not provided. |
| `--name` | Display name; the id by default. |
| `--dest DIR` | Directory the folder is created in. Default `packages`. |

The command refuses an existing folder, an unknown kind, an option that names a kind you did not ask for, and a format that is not a chat format in the catalog. Every kind adds its own files to a common base (`plugin.yaml`, `pyproject.toml`, `README.md`, `__init__.py`, `tests/test_extension.py`):

| Kind | Adds |
|------|------|
| shims | `provider.yaml` (the vendor manifest) and `llm/provider.yaml` (the first lane, named `<id>--<format>`) |
| transports | a `transport/` package exposing `TRANSPORTS` |
| commands | `commands.py` exposing `COMMANDS` with one function per verb |

The sample tree is rendered from the template folders in `llm_rosetta/authoring/_ext_template/`, and the shim files come from the same generator that `scripts/dev/new_shim.py` uses for the built-in tree, so the two cannot write different shapes. When `ext new` finishes it prints `next: llm-rosetta ext check <folder>`.

## The tree contract

A vendor directory carries a manifest, every leaf sits under a model-type directory, and the format level appears only when a type holds more than one lane. In an extension the folder is the one vendor:

```
<extension>/                 provider.yaml is the vendor manifest, never a leaf
├── auth.py                  optional, the credential exchange of every lane
├── quota.py                 optional
└── [<lane>/]                optional: named instances of one vendor
    └── <type>/              llm | embeddings | rerank | decisions
        └── [<format>/]      only when that type holds more than one lane
            ├── provider.yaml
            ├── transforms.py
            ├── auth.py
            └── quota.py
```

The rules, in short: a vendor directory is never a leaf; a type directory is always present, even for one lane; a format directory appears only beside a second lane of that type. The file allow-list and the reasons are in `llm_rosetta/authoring/_shim_template/README_en.md`, which is the single source of truth for a shim's shape. Sidecars import `llm_rosetta` only through `llm_rosetta.contracts` ([What a shim may import](../guide/shims.md#what-a-shim-may-import)).

## Check it: `ext check`

```bash
llm-rosetta ext check packages/acme       # one extension
llm-rosetta ext check .                   # every packages/* under a root
llm-rosetta ext check packages/acme --no-build
```

`ext check` applies the manifest schema, the folder rules, the tree contract and the rules of each kind. It never imports the extension. It reports one `problem <id>: <sentence>` line per defect, `skipped <id>: <reason>` for a check that could not run, and `ok <id>` when the folder is sound. It exits `1` when any folder has a problem. Skipped is never counted as passed.

| Area | What it checks |
|------|----------------|
| Folder | The folder name is a valid id and equals `id` in `plugin.yaml`; `README.md`, `__init__.py` and `tests/` exist; `pyproject.toml` parses, and its name, version, package map and `llm_rosetta.extensions` entry point match the manifest. |
| Shims | The tree follows the contract, loads through the real loader (off to the side of the live registry), and registers exactly the names in `provides.shims`. |
| Transports | `transport/` exposes `TRANSPORTS` with exactly the ids in `provides.transports`. |
| Commands | `commands.py` exposes `COMMANDS` with exactly the verbs in `provides.commands`. |
| Wheel | A wheel built from a clean copy holds the folder's files and nothing else. Skipped with `--no-build`, or when pip is not available (for instance inside the binary). |

## Move an older tree: `ext migrate`

An extension written before the tree contract may hold a leaf at the wrong depth. `ext migrate` moves it:

```bash
llm-rosetta ext migrate packages/acme            # print the plan; nothing moves
llm-rosetta ext migrate packages/acme --write    # apply the moves
```

It renames directories, including the second-lane case where the existing leaf moves down into `<type>/<its format>/`. A leaf that cannot be moved without a guess is reported and left. It changes nothing else: not `plugin.yaml`, not the sidecars' contents. A move can leave the package data in `pyproject.toml` naming the old path, and a relative import in `transforms.py` may need one more dot; run `ext check` afterwards (the wheel check finds the first, the extension's own tests the second). A dry run exits `1` when it found work.

## Test and publish

1. Run the extension's own tests: `ext new` writes `tests/test_extension.py`, which imports `run_ext_check` from `llm_rosetta.api`.
2. Build the wheel and install it to try it: `llm-rosetta ext install ./dist/<wheel>.whl`, then `llm-rosetta ext doctor acme`.
3. To publish, an extension repository attaches each wheel to a release and writes `index.json` beside the wheels ([The extension index](../guide/extensions.md#the-extension-index)). A gateway then lists the repository in `shims.sources` and selects the extension in `shims.install`. The two official repositories are `Oaklight/llm-rosetta-ext-contrib` and `Oaklight/llm-rosetta-ext-argonne`.

An extension must run where pip is absent: depend on the standard library and core only.

## Where the layering rules live

The import rules that keep core and the gateway layered are tests, and the tables are in them:

| Table | File |
|-------|------|
| Core package-level edges (`CORE_EDGES`) | `tests/test_core_layering.py` |
| The conversion layers do not import the shim or extension layer | `tests/test_core_layering.py` |
| Gateway layers and edges (`GATEWAY_EDGES`, `LAZY_UPWARD`) | `tests/gateway/test_layers.py` |
| The gateway reaches core only through `llm_rosetta.api` and `llm_rosetta.contracts` | `tests/gateway/test_core_surface_boundary.py` |
| Shim sidecars import only `llm_rosetta.contracts` | `tests/test_shim_sidecar_boundary.py` |
| Core never imports the gateway or `llm_rosetta.api` | `tests/test_core_import_boundary.py` |

A new import that the table does not allow fails with its place; an edge that no longer exists must be deleted from the table. See the [Architecture Guide](architecture.md#layering).

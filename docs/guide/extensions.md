---
title: Extensions
---

# Extensions

An **extension** is the installable unit of everything that does not ship in core: a set of provider shims, a transport that reaches a lane the gateway has no built-in client for, commands for the `llm-rosetta` command line, or any combination. An extension is one folder with a manifest, `plugin.yaml`, beside its code. The folder is imported as `llm_rosetta_ext.<id>`.

Extensions replace the 0.13 plugin mechanisms. The `llm_rosetta.shim_providers` entry-point group, the gateway's local shim directories (`shims.paths`, `shims.yaml`) and the scaffolder's "package shell" are gone; see [Migrating to 0.14](migrating-to-0.14.md#retired-extension-paths).

## The folder

```
acme/                         # the extension id; imported as llm_rosetta_ext.acme
├── plugin.yaml               # the manifest, read before any code is imported
├── pyproject.toml            # packaging; the folder is itself the package
├── README.md
├── __init__.py
├── provider.yaml             # vendor manifest        (shims kind)
├── llm/chat_completions/provider.yaml                  (shims kind)
├── transport/                # TRANSPORTS mapping      (transports kind)
├── commands.py               # COMMANDS mapping        (commands kind)
└── tests/
```

`llm_rosetta_ext` is a PEP 420 namespace: it has no `__init__.py` and no distribution owns it, so any directory on `sys.path` can contribute a folder to it. Discovery reads directories and package metadata and imports no extension code.

## `plugin.yaml`

The manifest is checked against `llm_rosetta/contracts/extension_manifest.schema.json`, and the host reads it before it imports a line of the extension. A core that cannot run the extension skips it with one sentence instead of failing on import.

```yaml
id: acme
name: Acme
version: 0.1.0
maturity: experimental        # experimental | beta | stable
requires:
  surface: 1                  # must equal the core's SURFACE_VERSION
  ir: 1                       # must not exceed the core's IR_VERSION
provides:
  shims: [acme--chat_completions]
  transports: [acme_cli]
  commands:
    hello:
      en: "Say hello"
summary:
  en: "Acme extension for llm-rosetta."
```

| Key | Meaning |
|-----|---------|
| `id`, `name`, `version`, `maturity`, `requires`, `provides` | Required. `provides` must list at least one of `shims`, `transports`, `commands`. |
| `provides.shims` | Shim names: the id itself, `<id>--<name>`, or `<id>_<lane>--<name>`. |
| `provides.transports` | Transport ids the extension implements. A shim's `connection.transport` resolves to its own extension's transport first, then a host built-in, then a transport of an extension listed in `requires.extensions`. |
| `provides.commands` | Verbs the extension adds, each with a one-line summary per language (`en` required). `llm-rosetta <id> <verb>` runs one; underscores in the id are written as hyphens. |
| `requires.extensions` | Ids of other extensions whose transports this one names. |
| `external` | Programs the extension drives and does not ship (`name`, `check`, `minimum`, `install_hint`, and `install`, the exact-version install spec that [`ext install-cli`](#external-programs) follows); `ext doctor` reports them. |
| `platforms` | A subset of `linux`, `darwin`, `windows`. |
| `tags`, `homepage`, `summary` | Listing metadata. |

Compatibility is two comparisons: `requires.surface` equals `SURFACE_VERSION`, and `requires.ir` does not exceed `IR_VERSION`, plus `platforms`. See [The version constants](../api/index.md#version-constants).

## What each kind means

| Kind | The loader reads | Registered by |
|------|------------------|---------------|
| `shims` | The shim tree: `provider.yaml` files and the `auth.py` / `quota.py` / `transforms.py` sidecars. The tree contract is in [Provider Shims](shims.md#declarative-provider-directory). | The shim layer |
| `transports` | `llm_rosetta_ext.<id>.transport`, which exposes `TRANSPORTS`, a mapping of transport id to factory. | The gateway |
| `commands` | `llm_rosetta_ext.<id>.commands`, which exposes `COMMANDS`, a mapping of verb to callable. | The command line |

Core manages packages only. What a `provides` kind means belongs to the layer that owns it, and that layer registers a handler for the kind (`register_kind_handler`). A handler registered after the first load is applied to the extensions already loaded. A bad extension is skipped and named, and never stops the others.

## Where extensions live

The extensions directory is chosen in this order:

1. a host that owns a data directory (the gateway uses `<data dir>/extensions`);
2. the `LLM_ROSETTA_EXTENSIONS_DIR` environment variable;
3. a per-user default: `$XDG_DATA_HOME/llm-rosetta/extensions` or `~/.local/share/llm-rosetta/extensions` on Linux, `~/Library/Application Support/llm-rosetta/extensions` on macOS, `%LOCALAPPDATA%\llm-rosetta\extensions` on Windows.

The default never falls back to a path under `/`: a container uid with no passwd entry gets `HOME=/`, and core refuses with a message that names `LLM_ROSETTA_EXTENSIONS_DIR`. The two Docker entrypoints set `LLM_ROSETTA_EXTENSIONS_DIR` to `$CONFIG_DIR/extensions` (`/config/extensions`) unless it is already set, so extensions survive the container. The directory must be writable by the container's uid.

Extensions are found in the extensions directory and in installed distributions that declare an entry point in the `llm_rosetta.extensions` group (name: the id; value: `llm_rosetta_ext.<id>`). The same extension reached by both routes loads once. A distribution that still declares an entry point in the retired `llm_rosetta.shim_providers` group is reported and not run.

## Installing

Installing needs no pip: a wheel is a zip whose members are `llm_rosetta_ext/<id>/...` and the `.dist-info`, and the installer unpacks exactly that. Every refusal (member names, size limits, manifest, compatibility, SHA-256) happens before a byte is written; the wheel is extracted into a temporary directory and swapped in. Member names with a drive or stream separator, or names Windows cannot store, are refused.

```bash
llm-rosetta ext install kilo_free              # an id, from the extension index
llm-rosetta ext install ./acme-0.1.0-py3-none-any.whl
llm-rosetta ext install https://example.org/acme.whl --sha256 <hex>
llm-rosetta ext update                         # newest compatible version of each
llm-rosetta ext doctor                         # what is wrong, and what is missing
```

### The extension index

An extension repository publishes each wheel as a release asset and writes `index.json` beside them. The file is generated from the extensions' own files. Version 1:

```json
{
  "index_version": 1,
  "source": "contrib",
  "generated_at": "2026-11-26T08:00:00Z",
  "extensions": {
    "kilo_free": [
      {
        "version": "0.4.1",
        "url": "llm_rosetta_ext_kilo_free-0.4.1-py3-none-any.whl",
        "sha256": "<64 hex digits>",
        "surface": 1,
        "ir": 1,
        "requires_python": ">=3.10",
        "requires_extensions": [],
        "name": "Kilo (Free)",
        "maturity": "beta",
        "tags": ["tier:free"],
        "shims": ["kilo_free--chat_completions"],
        "logo": "https://…/kilocode.svg",
        "commit": "<git sha>"
      }
    ]
  }
}
```

`version`, `url` (absolute, or relative to the index) and `sha256` are what an install needs. `surface`, `ir`, `requires_python` and `requires_extensions` say whether this host can run the wheel. `name`, `maturity`, `tags`, `shims` and `logo` are what a listing shows for an extension that is not installed yet. `index_version` is the integer `1`. The installer picks the newest version this interpreter and core accept, installs a shared requirement once, and checks the digest before it writes anything.

The location is `--index`, else `$LLM_ROSETTA_EXTENSION_INDEX`, else the contrib repository's `releases/latest/download/index.json`. It is an `https` URL or a local file path. An index that names a source other than the catalogue id it was fetched as is refused by the gateway; the `source` header is the index's own claim and proves no publisher.

!!! warning "Trust"
    An installed extension is code that runs on load. An index is code execution at the trust level of the repository that publishes it. The digest proves the wheel is the one the index lists, not that either is safe.

### The `ext` verbs

| Verb | Does |
|------|------|
| `list [--dir D] [--json]` | Extensions installed or found, with their state (`loaded`, `skipped`, `failed`, `not loadable`). |
| `show <id> [--json]` | One extension's manifest and state. |
| `install <wheel \| URL \| id> [--sha256 H] [--version V] [--index I] [--pip SPEC] [--dry-run]` | Install. `--pip` installs with pip, only where pip is available. |
| `remove <id> [--dry-run]` | Remove an installed extension. |
| `update [<id>] [--index I] [--dry-run]` | Install the newest compatible version found in the index. |
| `install-cli <id> [--program P] [--dry-run] [--json]` | Install the external programs the extension declares (`external[].install`) at their pinned version; see [External programs](#external-programs). |
| `doctor [<id>] [--no-import] [--json]` | Compatibility, missing required extensions, external programs (including the ones `install-cli` put in the tools directory), and an import check in a child process. |
| `new`, `check`, `migrate` | Authoring; see [Adding an extension](../contributing/adding-an-extension.md). |

`list`, `show`, `doctor` and `install-cli` take `--json`; `install`, `remove`, `update` and `install-cli` take `--dry-run`, which says what would change and changes nothing. Exit codes: `0` success, `1` a failure or a problem found, `2` a usage error.

### External programs { #external-programs }

An extension that drives an agent CLI (`claude`, `agy`) does not ship it. It declares the program under `external:` and, so that the operator does not have to guess a version, how to install it at **one exact version**:

```yaml
external:
  - name: claude
    check: ["claude", "--version"]
    minimum: "2.1.299"
    install:
      npm: "@anthropic-ai/claude-code"
      version: "2.1.299"
  - name: agy
    check: ["agy", "--version"]
    install:
      version: "1.5.15"
      release:
        url: "https://github.com/google-antigravity/antigravity-cli/releases/download/{version}/agy_cli_{platform}.tar.gz"
        sha256: {linux_x64_musl: "a12514fb…", linux_arm64_musl: "455d20cc…"}
        binary: antigravity   # the file inside the archive
        link: agy             # the name `check` runs
```

`version` is exact (`2.1.299`, optionally `-pre`): a range, `latest` or a partial version is refused, and `ext check` reports it. Exactly one of `npm` and `release` is given. A release's URL is https, `{platform}` is `linux_x64_musl`, `linux_arm64_musl`, `linux_x64_gnu`, `linux_arm64_gnu`, `darwin_x64` or `darwin_arm64`, and every platform it is available for carries the archive's SHA-256. `check` must run the name the entry link has (`link`, else the npm `bin`, else the program's name).

```bash
llm-rosetta ext install-cli claude          # as the user the gateway runs as, not root
llm-rosetta ext install-cli claude --dry-run
llm-rosetta ext doctor claude               # external claude: ok 2.1.299
```

`install-cli` puts the program under `<tools>/<id>/`, where `<tools>` is `$LLM_ROSETTA_TOOLS_DIR`, else `tools` beside the extensions directory (`/config/tools` in the image):

| Path | What |
|------|------|
| `<tools>/<id>/npm-<version>/` or `release-<version>/` | The install |
| `<tools>/<id>/bin/<name>` | The entry link `check` and `ext doctor` find |
| `<tools>/bin/<name>` | One directory for `PATH`; the image's entrypoint puts it first |
| `<tools>/node/<version>/` | The Node.js (with its npm) that npm programs are installed with and run on |
| `<tools>/<id>/.home/`, `<tools>/<id>/.npm-cache/` | npm's `HOME` and cache; nothing is written to the user's `$HOME` |

A run at the pinned version changes nothing. A new pinned version replaces the old one. A wrong checksum, a failed download or npm run, an archive that holds a path outside its directory, or a program whose check does not pass leaves nothing installed and exits `1`. An npm package is checked by npm against the registry's integrity record; a release archive is checked against the SHA-256 in the manifest before it is unpacked. A program that comes from npm needs Node.js, and `install-cli` installs that too: a pinned Node.js (24.22.3, with the npm it bundles) goes into `<tools>/node/<version>/` first, and npm runs from there, so the image needs neither `node` nor `npm`. glibc and macOS builds come from nodejs.org; musl builds come from `unofficial-builds.nodejs.org`, which publishes a checksum list but no signature, so the SHA-256 of every build is pinned in the source and checked before unpacking. On Alpine, `node` needs the `libstdc++` package, which the image carries. A program that no longer starts after the image changed (for example its `node` cannot load) is `broken` in `ext doctor` and in the panel's Extensions view; running `install-cli` again repairs it. The panel's Extensions view has an **Install** button for the same thing: administrators only, it installs only the version the extension's manifest pins (the request carries just the program's name), one install per extension at a time, and each attempt is an `extension_cli_install` record in the ops log.

Pinned versions the shipped extensions declare:

| Extension | Program | Source | Version | Platforms |
|-----------|---------|--------|---------|-----------|
| `claude` | `claude` | npm `@anthropic-ai/claude-code` | `2.1.299` | any npm supports (the musl build is chosen on Alpine) |
| `antigravity` | `agy` | release archive `agy_cli_<platform>.tar.gz` | `1.5.15` | `linux_x64_musl`, `linux_arm64_musl` |

## The gateway and extensions

The gateway's `shims` section selects what to install from extension indexes and what to switch off; see [External shim sources](../gateway/configuration.md#external-shim-sources). An extension that is installed loads whether or not `shims.install` still selects it. A shim name that core already registers wins: it is not replaced, and the extension's status says which owner holds the name. A vendor that moved out of core is installed by itself when a config names its shim ([Migrating to 0.14](migrating-to-0.14.md#moved-to-contrib)).

An extension can add commands: `llm-rosetta <id> <verb>` (for example `llm-rosetta-gateway argo tunnel` reaches the same area through the gateway entry). The names `ext`, `help`, `gateway` and `serve` are reserved against extension ids.

## See also

- [Adding an extension](../contributing/adding-an-extension.md): `ext new`, `ext check`, `ext migrate`, the tree contract.
- [Provider Shims](shims.md): the shim kind in detail.
- [Shim Placement and Maturity](shim-placement.md): what lives in core and what in an extension repository.

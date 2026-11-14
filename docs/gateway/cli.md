---
title: CLI Reference
---

# CLI Reference

## Usage

```
llm-rosetta-gateway [OPTIONS] [COMMAND]
```

## Options

| Flag | Description |
|------|-------------|
| `--config`, `-c` PATH | Config file path (auto-discovered if omitted) |
| `--version`, `-V` | Show version and exit |
| `--no-banner` | Suppress the startup banner |
| `--edit`, `-e` | Open config file in `$EDITOR` for editing |
| `--host` HOST | Override server host |
| `--port` PORT | Override server port |
| `--proxy` URL | HTTP/SOCKS proxy URL for all upstream requests |
| `--verbose`, `-v` | Enable verbose (DEBUG) logging; overrides config and `--log-level` |
| `--log-level` LEVEL | Log level: `debug`, `info`, `warning`, `error` (default: `info`) |

## Commands

### `init`

Create a template `config.jsonc` at `~/.config/llm-rosetta-gateway/`:

```bash
llm-rosetta-gateway init
```

### `add provider <name>`

Add a provider entry to the config file:

```bash
llm-rosetta-gateway add provider chat_completions
llm-rosetta-gateway add provider anthropic --api-key "${ANTHROPIC_API_KEY}"
```

| Flag | Description |
|------|-------------|
| `--api-key` KEY | API key or `${ENV_VAR}` placeholder |
| `--base-url` URL | Provider base URL (auto-filled for known providers) |

### `add model <name>`

Add a model routing entry to the config file:

```bash
llm-rosetta-gateway add model gpt-4o --provider my-openai
llm-rosetta-gateway add model claude-sonnet-4-20250514 --provider my-anthropic
llm-rosetta-gateway add model gemini-2.0-flash --provider my-google
```

| Flag | Description |
|------|-------------|
| `--provider` NAME | Target provider name |

### `shims new [vendor]`

Create the files for a new provider shim, then load them through the same loader
a gateway uses:

```bash
llm-rosetta-gateway shims new                  # asks for every field
llm-rosetta-gateway shims new acme --no-input \
  --base-url https://api.acme.example/v1 --check
```

Run with no flags on a terminal it is a questionnaire — the scan root first, then
vendor, model type, wire format, layout, base URL, credential variable, optional
row label / logo / group title, and whether to add a transforms stub — and
pressing Enter takes the value in brackets.  The root is asked first because
everything after it follows what the package already does (see below).
**Interactive is the default**; a value a flag already gave is not asked for
again.  `--no-input` (implied when stdin is not a
terminal) never prompts, so the same command works in a script.

| Flag | Description |
|------|-------------|
| `--root DIR` | The directory a gateway scans; default `providers` |
| `--model-type` | `llm` (default), `embeddings`, `rerank`, `decisions` |
| `--format` | Wire format; the prompt and `--format`'s help list every format the model type registers, and an unregistered one is refused before anything is written |
| `--layout` | `typed` (`<vendor>/llm/<format>`) or `flat` (`<vendor>/<format>`); the default follows the scan root's existing shape, else `typed` |
| `--base-url` | Upstream base URL; a placeholder is written when omitted |
| `--probe` | Check that `--base-url` answers, and print what it answered |
| `--models-path` | Model-list URL or path, when it is not the default |
| `--api-key-env` | Credential variable; default `<VENDOR>_API_KEY` |
| `--display-name` | Row label inside the vendor group |
| `--logo`, `--group`, `--group-label` | Vendor identity, written to the vendor manifest |
| `--auto-logo` | Take the mark from the lobehub icon set on jsDelivr, found by vendor name; fails when there is no icon under that name |
| `--localize-logo` | Save `--logo` into the package's `assets/` and reference its jsDelivr URL |
| `--transform` | Also write a `transforms.py` stub |
| `--vendor-manifest` | Write `<vendor>/provider.yaml` for the identity and credential every lane shares; the default is yes unless the scan root is a bundle of single-lane vendors |
| `--no-vendor-manifest` | State every field on the leaf instead |
| `--no-registry-manifest` | Do not touch the package's `shims.yaml` |
| `--check` | Load the new leaf and report what it resolves to |

When the vendor already has a `provider.yaml`, the credential and the vendor
identity stay there and the leaf carries only its own fields.  When the scan root
ships a `shims.yaml`, the new leaf is appended to it — that file is authoritative
for a gateway, so a leaf it does not list never loads.

**The scan root decides the shape.**  A root whose vendors are single leaf
directories (`llm-rosetta-shim-free`: `kilo/provider.yaml` declaring a name) gets
a flat leaf and no vendor manifest, because that package models each provider as
its own top-level vendor.  A root whose vendors are manifests over `llm/` or
`<type>/` directories (`google`: two chat dialects plus `embeddings/`) gets the
typed layout.  A root that mixes the two has no convention to copy, so `typed`
stays the default and the prompt asks instead of guessing.

Three targets are refused before anything is written:

- a vendor whose own `provider.yaml` is a shim — the loader reads that directory
  as that one shim and never looks below it, so a second lane there would load
  nowhere;
- a shim name another file already declares, with the file named;
- a vendor name the loader skips or misreads — an `_`/`.` prefix, or one of
  `llm`, `embeddings`, `rerank`, `decisions`.

A vendor that already holds lanes in the *other* layout is only noted: both
shapes load under one vendor, the package just carries two.

A pasted base URL is checked before it is written.  A trailing `/models` (what
vendor consoles copy) is dropped, then `<base>/models` is requested without a
credential: `200`, `401` or `403` prove the host answers on that path, and any
other answer is reported with the base URL's own status so a wrong suffix shows
up now rather than as an empty model list later.  The interactive mode always
checks; in flag mode pass `--probe`.  A negative answer is a warning, not a
refusal — the host may not be reachable from this machine.

One <kbd>Ctrl</kbd>+<kbd>C</kbd> re-asks the current question; a second one
cancels the run with exit code 130 and says whether anything was written.
<kbd>Ctrl</kbd>+<kbd>D</kbd> cancels at once — there is no value to take from an
ended stream, and every later prompt would read it ended too.  There is no
traceback in any of the three cases.

A mark lives in the package, not on someone else's host: when a logo URL is
given, the interactive mode asks whether to download it into the package's
`assets/` and point `logo:` at the jsDelivr copy of that file.  The package is
identified from the scan root's git remote, so the URL is the one this repository
will actually serve; a scan root outside a git repository keeps the URL as typed
and says why.  `--localize-logo` does the same without the question.

Before it asks, the vendor name is looked up in the lobehub icon set
(`@lobehub/icons-static-svg` on jsDelivr) and a hit becomes the default answer —
Enter takes it.  The whole name is tried first, then the leading segment and the
name without a pool marker, so `opencode_zen_free` still finds `opencode`; each
try is one `HEAD`, and an unreachable CDN ends the search rather than costing a
timeout per name.  A URL that is already on jsDelivr is kept as typed, because
there is nothing to copy: the mark is served from the CDN either way.
`--auto-logo` does the lookup without the questionnaire and fails when there is
no icon.

### `credential <provider>`

Run a shim's interactive sign-in on this machine and emit the credential
record it produces:

```bash
llm-rosetta-gateway credential chatgpt --out record.json
```

A provider whose shim ships `auth.py` has no key to type in — the exchange
mints its credential through an OAuth sign-in that needs a browser.  When the
gateway runs somewhere else, the sign-in happens where the browser is, and
this command is that half: it runs the exchange's `login()`, and what comes
back is the record the admin API's `PUT
/admin/api/config/providers/<name>/credential` accepts.  The gateway is not
contacted — hand the record over in the panel.

| Flag | Description |
|------|-------------|
| `--out` FILE | Write the record to `FILE` with mode `0600` instead of stdout |

The record holds live tokens (access, refresh and ID token).  Paste it into
the panel promptly: the ID token is verified when it arrives, so a record left
lying around expires into a refusal and the sign-in has to be repeated.  The
sign-in also stores a copy in this machine's record directory
(`<data_dir>/shim-auth/<provider>`); when the gateway is elsewhere, delete it
there once the record has been handed over.  `--out` writes through a
randomly named temporary file that is created `0600` exclusively and fsynced
before the rename, so a file or symlink planted at a predictable name in a
shared directory cannot capture the record.  The
command refuses a provider whose shim ships no `auth.py` (`mints no credential
of its own`) and one whose exchange has no browser flow (`does not implement
login()`).

## Config Auto-Discovery

When `--config` is not specified, the gateway searches these paths in order:

1. `./config.jsonc` — current working directory
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG standard location
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile convention

## Programmatic Usage

The gateway can also be used as a library:

```python
from llm_rosetta_gateway import create_app, GatewayConfig, load_config

# Load config and create ASGI app
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = create_app(config)

# Mount in your own ASGI application, or run with any ASGI server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## Reading usage from the shell

`llm-rosetta-usage` is a separate command for the token ledger the gateway
keeps — the same figures the admin panel's dashboard and API Keys tab show,
without starting anything:

```console
$ llm-rosetta-usage --db /var/lib/llm-rosetta --window 24h --by subject,model
```

It reads the store and, when it is beside it, the keys database, so it reports
against a running gateway's files.  See
[Observability — The `llm-rosetta-usage` command](../api/observability.md#the-llm-rosetta-usage-command).

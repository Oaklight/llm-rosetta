---
title: CLI Reference
---

# CLI Reference

## The command surface

The `llm-rosetta` command is a dispatcher. Core owns two names, `ext` (manage extensions) and `help`; every other command is contributed by an installed distribution. The gateway distribution adds `gateway` and `serve`. Running `llm-rosetta` with no arguments prints the list and never starts a server.

| Command | Does |
|---------|------|
| `llm-rosetta gateway <verb>` | The gateway's verbs: `serve`, `init`, `add`, `credential`, `password`, `db`, `usage`, `migrate` |
| `llm-rosetta serve` | Shortcut for `llm-rosetta gateway serve` |
| `llm-rosetta ext <verb>` | Extensions: `list`, `show`, `install`, `remove`, `update`, `doctor`, `new`, `check`, `migrate`; see [Extensions](../guide/extensions.md) |
| `llm-rosetta <id> <verb>` | A command an installed extension adds |
| `llm-rosetta-gateway [COMMAND]` | The gateway's own entry point; same verbs, and no command starts the server |

`llm-rosetta-gateway` is the only console script of the gateway distribution. The console scripts `llm-rosetta-usage` and `llm-rosetta-migrate` that earlier 0.14 development releases shipped are removed, and so is `python -m llm_rosetta_gateway`:

| Earlier 0.14 development releases | Now |
|--------|-----|
| `llm-rosetta-usage ...` | `llm-rosetta gateway usage ...` |
| `llm-rosetta-migrate ...` | `llm-rosetta gateway migrate ...` |
| `llm-rosetta-gateway shims new / check / migrate` | `llm-rosetta ext new / check / migrate` (see [Adding an extension](../contributing/adding-an-extension.md)) |
| `llm-rosetta-gateway set-password` (0.13.0) | `llm-rosetta gateway password` (`set-password` is still accepted as an alias) |
| `python -m llm_rosetta_gateway` | `llm-rosetta serve` or `llm-rosetta-gateway` |

Every verb is also reachable as `llm-rosetta-gateway <verb>`, so the examples below work with either spelling. `llm-rosetta-gateway` also forwards `ext`, `help` and the commands of installed extensions to the dispatcher: the compiled binary and the Docker image have no separate `llm-rosetta` script, and this is how an operator without pip installs an extension (`llm-rosetta-gateway ext install <id>`). Options placed before such a command are refused.

Exit codes: `0` success, `1` the command failed, `2` a usage error; a command that documents another code keeps it (`gateway migrate --check` returns `3` for pending work). An unexpected exception prints one line, `llm-rosetta <command>: <message>`, and exits `1`; set `LLM_ROSETTA_DEBUG` to any value to get the traceback.

## Usage

```
llm-rosetta serve [OPTIONS]
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
| `--socket`, `-S` PATH | Listen on a Unix domain socket instead of TCP |
| `--tls-cert` PEM, `--tls-key` PEM | Serve TLS; each requires the other |
| `--data-dir` DIR | Directory for `gateway.db` and other persistent data (overrides config) |

## Commands

### `init`

Create a template `config.jsonc` at `~/.config/llm-rosetta-gateway/`:

```bash
llm-rosetta gateway init
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

### `password [config]`

Set or change the admin password (recovery when the password is forgotten):

```bash
llm-rosetta gateway password --config config.jsonc
```

### `db <action>`

Database maintenance. `--data-dir` (or `--config`) selects the directory that holds `gateway.db`.

| Action | Does |
|--------|------|
| `cleanup [--max-age-days N]` | Delete all records older than `N` days (default 90) and vacuum |
| `cleanup-logs [--max-age-days N]` | Delete request logs older than `N` days |
| `cleanup-errors [--max-age-days N]` | Delete error dumps older than `N` days |
| `export-errors [--start D] [--end D] [-o FILE]` | Export error dumps to a `tar.gz` (default `error-dumps.tar.gz`) |

### `usage`

See [Reading usage from the shell](#reading-usage-from-the-shell).

### `migrate`

Rewrite what the 0.14.0 removals break: a config file, the persisted tags in a gateway database, and source code. A dry run is the default.

```bash
llm-rosetta gateway migrate --config config.jsonc            # dry run: diff and report
llm-rosetta gateway migrate --config config.jsonc --write    # .bak copy, then rewrite
llm-rosetta gateway migrate --db /var/lib/llm-rosetta --write
llm-rosetta gateway migrate --scan src/ --json
```

See [Migrating to 0.14](../guide/migrating-to-0.14.md#llm-rosetta-migrate).

### `ext`

`llm-rosetta ext` is not a gateway verb, but the gateway binary forwards it. See [Extensions](../guide/extensions.md#the-ext-verbs).

### `credential <provider>`

Run a shim's interactive sign-in on this machine and emit the credential
record it produces:

```bash
llm-rosetta gateway credential chatgpt --out record.json
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
import asyncio

import uvicorn

from llm_rosetta_gateway import create_app, GatewayConfig, load_config

# Load config and create ASGI app.  create_app is async since 0.14.0.
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = asyncio.run(create_app(config))

# Mount in your own ASGI application, or run with any ASGI server
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## Reading usage from the shell

`llm-rosetta gateway usage` reports the token ledger the gateway keeps, the same figures the admin panel's dashboard and API Keys tab show, without starting anything:

```console
$ llm-rosetta gateway usage --db /var/lib/llm-rosetta --window 24h --by subject,model
```

It reads the store and, when it is beside it, the keys database, so it reports against a running gateway's files. See [Observability: The usage command](../api/observability.md#the-llm-rosetta-usage-command).

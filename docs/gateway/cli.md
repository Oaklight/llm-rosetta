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

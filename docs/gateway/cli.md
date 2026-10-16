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

## Config Auto-Discovery

When `--config` is not specified, the gateway searches these paths in order:

1. `./config.jsonc` — current working directory
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG standard location
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile convention

## Programmatic Usage

The gateway can also be used as a library:

```python
from llm_rosetta.gateway import create_app, GatewayConfig, load_config

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

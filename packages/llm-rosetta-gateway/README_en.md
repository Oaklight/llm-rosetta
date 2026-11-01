<p align="center">
  <img src="https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/design/logo/out/rosetta-icon-github.svg" alt="LLM-Rosetta" width="96">
</p>

<h1 align="center">llm-rosetta-gateway</h1>

<p align="center">
  <a href="https://pypi.org/project/llm-rosetta-gateway/"><img src="https://img.shields.io/pypi/v/llm-rosetta-gateway?color=green" alt="PyPI"></a>
  <a href="https://pypi.org/project/llm-rosetta-gateway/"><img src="https://img.shields.io/pypi/pyversions/llm-rosetta-gateway" alt="Python"></a>
  <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
</p>

<p align="center">
  <a href="https://llm-rosetta.readthedocs.io/en/latest/gateway/"><img src="https://img.shields.io/badge/docs-English-2f6fdd" alt="Docs (English)"></a>
  <a href="https://llm-rosetta.readthedocs.io/zh-cn/latest/gateway/"><img src="https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-2f6fdd" alt="Docs (中文)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
</p>

<p align="center">
  English | <a href="https://github.com/Oaklight/llm-rosetta/blob/master/packages/llm-rosetta-gateway/README_zh.md">中文</a>
</p>

A self-hosted HTTP proxy that accepts requests in any supported LLM API format
and forwards them to any configured upstream provider, converting on the fly
(including streaming for chat formats). It adds key rotation, routing, failover
and an admin console.

This is the **gateway distribution**. It depends on `llm-rosetta` for the
conversion core and adds no third-party runtime dependencies of its own. On
Python 3.10, the core package also installs `typing_extensions`. The HTTP server
and client are vendored, stdlib-only modules.

The point is control: the configuration for every upstream, credential and
route remains under your control. LLM-Rosetta itself does not add a hosted aggregator to the path.

## The Top Level

The core library gives you a toolbox, raw converters and a pipeline. The gateway
adds the HTTP-proxy level: a proxy that owns the provider call, routing, keys and
admin console. The same IR and converters run underneath, so the translation
does not change when you move up.

## Install

```bash
pip install llm-rosetta-gateway
```

`pip install "llm-rosetta[gateway]"` also works as a transitional alias that
installs both distributions.

## Run

```bash
llm-rosetta-gateway init     # writes a template config
llm-rosetta-gateway          # serves http://localhost:8765
```

The config is auto-discovered at `./config.jsonc`,
`~/.config/llm-rosetta-gateway/config.jsonc` or
`~/.llm-rosetta-gateway/config.jsonc`; pass `--config` to point elsewhere. The
admin console is at `http://localhost:8765/admin/`.

## Configure

List your upstreams under `providers`, and map each model name clients send to a
provider under `models`. The gateway routes by model name and converts between
the source and upstream formats:

```jsonc
{
  "providers": {
    "my-openai":    { "type": "chat_completions", "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "my-anthropic": { "type": "messages",         "api_key": "${ANTHROPIC_API_KEY}", "base_url": "https://api.anthropic.com" }
  },
  "models": {
    "gpt-4o": "my-openai",
    "claude-sonnet-4-5": "my-anthropic"
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    // Local demo only. For a shared deployment, bind 0.0.0.0 behind your
    // own network controls, create a gateway key in /admin, and send it
    // in the format's native auth header.
    "open_on_no_keys": true
  }
}
```

## Send a Request

Send an OpenAI-format request to an Anthropic model — the gateway translates
both directions, including streaming:

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello!"}]}'
```

With a gateway key configured, send it in the format's native header —
`Authorization: Bearer <key>` for OpenAI-style routes, `x-api-key: <key>` for
Messages, `x-goog-api-key: <key>` (or `?key=<key>`) for Google. A plain
`Authorization: Bearer` also works on every route; an OpenAI-style route rejects
`x-api-key`.

The gateway serves the OpenAI Chat Completions, Open Responses, Anthropic
Messages, Google generateContent and Google Interactions routes, plus the
embedding, rerank and decision endpoints.

## Admin Console

The built-in console at `http://localhost:8765/admin/` manages providers,
models, API keys and observability data:

![Admin console](https://llm-rosetta.readthedocs.io/en/latest/gateway/images/admin-dashboard.png)

## Why Not LiteLLM or OpenRouter?

**OpenRouter is primarily a hosted service.** In its standard setup, OpenRouter
manages provider access and routing, and your traffic passes through its
service; it also offers bring-your-own-key options. That is its strength: you
reach models without signing up with each vendor. LLM-Rosetta assumes the opposite — you own every
upstream credential and run the gateway yourself.

**LiteLLM is self-hostable and offers broad provider coverage.** It also
provides team keys, budgets and spend tracking. Its core interface is primarily
OpenAI-shaped; other formats are mapped onto that centre. LLM-Rosetta's IR
belongs to no vendor: every supported format can sit on either side, so the
client's dialect and the model source stay independent, and your application can
keep its payload in IR.

Coverage on both sides changes quickly; check their current docs.

- **Choose OpenRouter** if you do not want to hold provider accounts at all.
- **Choose LiteLLM** if you need broad provider coverage or team-level cost
  controls.
- **Choose LLM-Rosetta** if you own your model sources and need to serve clients
  in different formats from them.

## Docker

```bash
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

The image reads `/config/config.jsonc`.

## CLI Integrations

Use the gateway behind Claude Code, Codex CLI, Aider, Antigravity CLI, OpenCode
and Kilo Code, and in front of Ollama — see
[CLI Integrations](https://llm-rosetta.readthedocs.io/en/latest/gateway/cli-integrations/).

## Documentation

- [Full README](https://github.com/Oaklight/llm-rosetta#readme)
- [Gateway Quick Start](https://llm-rosetta.readthedocs.io/en/latest/getting-started/gateway-quickstart/)
- [Configuration](https://llm-rosetta.readthedocs.io/en/latest/gateway/configuration/)
- [Admin Console](https://llm-rosetta.readthedocs.io/en/latest/gateway/admin-panel/)
- [Core distribution](https://pypi.org/project/llm-rosetta/)

## License

MIT — see the [LICENSE](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE)
file for details.

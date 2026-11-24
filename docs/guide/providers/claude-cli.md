---
title: Claude (official CLI)
---

# Claude (official CLI)

The `claude--messages` shim serves a Claude subscription or Console account by
running the official **Claude Code CLI** as the model client. The CLI builds
and signs the upstream request itself, so the gateway never holds a Claude
credential: the CLI's own login store is the account.

The shim ships in the contrib source rather than in core. Install the `claude`
vendor and select the Messages lane:

```jsonc
{
  "shims": { "sources": ["contrib"], "install": ["claude"] },
  "providers": {
    "claude-cli": {
      "type": "claude--messages",
      "models": ["claude-haiku-4-5", "claude-sonnet-5"]
    }
  }
}
```

The vendor also installs `claude--acp`, the ACP lane; `type` selects the one a
provider uses.

!!! warning "Read this before you point an account at it"
    Anthropic's terms decide what may drive the subscription, and an account
    used outside Anthropic's own clients can be suspended. This is the same
    exposure as the ACP lane. Use an account you can afford to lose.

## How the turn works

1. The gateway converts the client's request to Messages and writes the
   conversation to the CLI as NDJSON replay frames.
2. It starts `claude -p --input-format stream-json --output-format stream-json`
   with `ANTHROPIC_BASE_URL` pointing at a per-turn loopback relay.
3. The CLI sends its one upstream request; the relay forwards it with the
   CLI's own headers and captures the stream for the gateway.
4. The gateway maps the CLI's stream events back to Messages events. A client
   that asked for a non-streaming reply gets the same events folded into one
   body.

The relay is also the single-request gate: the CLI's own recovery attempts are
refused locally, except one retry after an upstream 401 so the CLI can refresh
its login.

## Prerequisites

- The official `claude` CLI on the gateway host's `PATH`, or an absolute path
  in `cli_argv`. The lane was measured on 2.1.299.
- A signed-in CLI:

  ```bash
  claude auth login --claudeai   # Pro / Max subscription
  claude auth login --console    # Console account, billed per token
  ```

- Permission for the gateway process to spawn subprocesses. A container needs
  the CLI inside the image, or a mounted copy.

## Configuration

The shim declares the vendor facts; a provider entry may override them:

| Key | Meaning |
|:---|:---|
| `cli_argv` | The command to run. Replaces the shim's `["claude"]`. |
| `cli_args` | Extra flags, appended last. Replaces the shim's list. |
| `cli_env` | Environment overrides for the child; the provider wins per key. |
| `cli_env_unset` | Names removed before `cli_env` applies. The shim's names stay removed. |
| `cli_upstream` | Where the relay forwards. Defaults to Anthropic's API; `https://` only, except a loopback host. |
| `cli_processes` | How many CLI processes this provider may run at once. Default `1`; requests over the cap queue. |

The shim sets `--setting-sources user` because a host may supply the credential
through an `apiKeyHelper`; without it the CLI starts with no credential and the
upstream answers 401. It also removes `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
the model overrides and the `CLAUDE_CODE_USE_*` backend selectors, so an
exported API key cannot outrank the stored login. The relay pins the child's
base URL through a private `--settings` file, because a user settings file wins
over the process environment.

```jsonc
"claude-cli": {
  "type": "claude--messages",
  "cli_argv": ["/opt/claude/bin/claude"],
  "cli_env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"},
  "cli_processes": 2
}
```

`max_tokens` stays the CLI's own: it asks for 32,000 output tokens because its
adaptive thinking shares that budget. Set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` in
`cli_env` for a hard ceiling.

## Capabilities

`text`, `vision` and `reasoning` are enabled. Images ride through the replay
unchanged, and thinking blocks arrive from the CLI's stream events.

`tools` is **off**. The CLI builds the upstream body and the transport does not
inject the host's tool definitions into it: a live request that declared a tool
degraded to text instead of a `tool_use` block. Tool *history* (`tool_use` /
`tool_result`, ids and nested content) replays unchanged, so the gap is only
the outbound definition. Do not enable `tools` on this lane until that path is
fixed and measured.

## Operations

- One process per request, roughly 250 MB. `cli_processes` queues a turn over
  the cap instead of failing it.
- A turn fails when the CLI stays silent past the upstream timeout; closing
  the stream stops the whole process tree (`killpg` on POSIX, `taskkill /T` on
  Windows).
- The request-log profile records what the lane alone knows: the CLI's native
  `usage`, `total_cost_usd` (list-price equivalent), `modelUsage[*]`
  (including `contextWindow`), and the relay's `restored` flag, upstream status
  and `anthropic-ratelimit-*` headers.
- Model ids reach `--model` verbatim. A `[1m]` suffix stays the CLI's own 1M
  opt-in, and an unknown id falls back to the CLI's 200K default.
- The CLI appends an environment/date annotation after every replayed user
  message. The relay drops those annotations and moves the cache breakpoint to
  the turn's last block, which is what keeps the prefix cacheable.

## Limitations

- **Experimental and version-coupled.** The replay frames, acknowledgments,
  settings injection and session ids were measured on CLI 2.1.299. Re-run the
  live checks on a version bump.
- **Tools are not sent upstream.** See [Capabilities](#capabilities).
- **Error mapping is partial.** A refused retry repeats the real upstream
  status; plan/session 429 mapping with `anthropic-ratelimit-unified-reset` as
  `resets_at`, and the 401/402/529/503 mapping, are not complete.
- **No proxy support yet.** `https_proxy` / `HTTP_PROXY` are not wired for the
  relay.

## See also

- [Provider Shims](../shims.md)
- [ACP lane (Antigravity)](antigravity.md) — the other way to serve a vendor
  CLI, where the agent owns the loop.

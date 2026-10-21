---
title: OpenCode
---

# OpenCode

OpenCode's paid offering is two lanes behind one credential, differing only by
base URL:

| Lane | Base URL | Billing | `x-opencode-session` |
|:---|:---|:---|:---|
| **Go** | `https://opencode.ai/zen/go/v1` | monthly subscription | required |
| **Zen** | `https://opencode.ai/zen/v1` | account balance | not required |

Both serve three API standards from the same base, so LLM-Rosetta exposes three
shims and defaults to the Go lane. Use Zen by overriding `base_url` on the
provider entry.

## Shims

| Shim Name | Base Type | Default Base URL | Credentials |
|:---|:---|:---|:---|
| `opencode--chat_completions` | `chat_completions` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |
| `opencode--messages` | `messages` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |
| `opencode--open_responses` | `open_responses` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` |

Each resolves to its format's path under the base: `/chat/completions`,
`/messages` (the key is sent as `x-api-key`), and `/responses`. Not every model
is on every endpoint — the [Go docs](https://opencode.ai/docs/go/) list which
model is served where.

## Session header

The Go lane rejects an inference request that carries no `x-opencode-session`
with `400 MissingSessionID`; the upstream uses it to route the request and to
keep a prompt-cache-affine backend for the conversation. The shims declare
`connection.session_header: x-opencode-session`, and the gateway fills it:

- by default with a value **derived per conversation** — a hash of the same
  affinity identity used for key selection (client key plus the message prefix),
  so every turn of a conversation reuses one session and a different
  conversation gets a different one;
- or with the client's own value, forwarded verbatim, when the request already
  carries that header — handy for clients such as Codex that send a native
  session id.

Zen ignores the header.

## Gateway configuration

Go lane (default):

```jsonc
{
  "providers": {
    "opencode": { "shim": "opencode--chat_completions" }
  },
  "models": {
    "glm-5.3": "opencode"
  }
}
```

Zen lane — the same shim with a `base_url` override:

```jsonc
{
  "providers": {
    "opencode-zen": {
      "shim": "opencode--chat_completions",
      "base_url": "https://opencode.ai/zen/v1"
    }
  },
  "models": {
    "glm-5.3": "opencode-zen"
  }
}
```

Set `OPENCODE_API_KEY` in the gateway's environment — the credential is the same
for both lanes.

## Quota

The Go subscription's windows (5-hour, weekly, monthly) are readable per key,
shown on the provider card and used to skip a spent credential. See [Provider
Quota](../quota.md).

## Public free lane

The keyless `/zen/v1` pool is a separate shim,
`opencode_zen_free--chat_completions`. The upstream refuses a request that does
not look like its own desktop client, so the shim reproduces that shape in its
`connection` block and `transforms.py`:

- a `User-Agent` of `opencode/<version>` and the `x-opencode-client` /
  `x-opencode-project` markers;
- an `x-opencode-session` whose value decodes back into a recent timestamp
  (`session_id_format: opencode`), plus a per-request `x-opencode-request`;
- one tool declared under the exact name `read` — injected only when the client
  did not declare one, and calls to an injected tool are removed rather than
  forwarded;
- `stream: true` (`stream_required`), with the non-streaming reply reassembled
  by the gateway.

This is an anti-abuse gate rather than a public pool: it can change or close
without notice, its quota is per source address and session, and prompts may be
logged upstream. The provider appears under **Free Resource** with the panel's
third-party disclosure. See [issue #884](https://github.com/Oaklight/llm-rosetta/issues/884)
for the measurements.

```jsonc
{
  "providers": {
    "zen-free": { "type": "opencode_zen_free--chat_completions" }
  },
  "models": {
    "beam-free": { "provider": "zen-free" }
  }
}
```

No key is required. The Go subscription also grants the two *Preview Free* models
through the paid lane.

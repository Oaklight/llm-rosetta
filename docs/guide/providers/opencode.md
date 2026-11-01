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
`opencode_zen_free--chat_completions`, and it ships as a standalone package
rather than in core: [llm-rosetta-shim-opencode-free](https://github.com/Oaklight/llm-rosetta-shim-opencode-free).
Declare the package before using the name:

```jsonc
{
  "shims": { "presets": ["opencode_zen_free"] },
  "providers": {
    "zen-free": { "type": "opencode_zen_free--chat_completions" }
  },
  "models": {
    "beam-free": { "provider": "zen-free" }
  }
}
```

No API key is required: the lane's credential is the literal word `public`,
which the shim sends as `Authorization: Bearer public`. The upstream refuses a
request that does not look like its own client, so the shim reproduces that
shape in its `connection` block and `transforms.py`:

- a `User-Agent` of `opencode/<version>` and the `x-opencode-client` /
  `x-opencode-project` markers;
- an `x-opencode-session` whose value decodes back into a recent timestamp
  (`session_id_format: opencode`), plus a per-request `x-opencode-request`;
- one tool declared under the exact name `read` — injected only when the client
  did not declare one. The declaration describes itself as a transport
  placeholder and tells the model not to call it, so the client's own tools stay
  the ones worth calling; a call to it is removed rather than forwarded, and a
  turn whose every call was removed ends with `finish_reason: stop` instead of
  leaving the client waiting for a result that will never arrive;
- `stream: true` (`stream_required`), with the non-streaming reply reassembled
  by the gateway.

Its model listing is filtered to the zero-cost slice: the lane's `/models`
answers with the whole Zen catalogue, but the keyless lane serves only the
models whose declared input and output cost is zero, so the panel offers those
and nothing else. A listing that states no cost at all falls back to the
`-free` suffix.

!!! warning "Measured 2026-11-01: the gate now refuses every non-OpenCode client"
    The lane answers the OpenCode client itself and refuses everything else with
    `429 FreeUsageLimitError`. A request that reproduces the real client's body
    and headers byte for byte — captured through a local MITM proxy — is still
    refused when sent with curl, Node `https`, Node `fetch`, Bun `fetch`, or
    browser-impersonating TLS, on a host where the genuine client answers in the
    same minute. The gate therefore keys on the client's transport (TLS/HTTP
    stack), which a shim cannot set.

    `429 FreeUsageLimitError` is also what genuine quota exhaustion returns, so
    the two cannot be told apart from the response alone. Treat this lane as
    experimental: it is kept because the upstream may relax the gate, and its
    request-shape work (model filter, placeholder tool, finish-reason rewrite)
    stays useful either way. See [issue
    #884](https://github.com/Oaklight/llm-rosetta/issues/884) for the measurement
    history.

The provider appears under **Free Resource** with the panel's third-party
disclosure: it is an anti-abuse gate rather than a public pool, it can change or
close without notice, and prompts may be logged upstream. The Go subscription
also grants the two *Preview Free* models through the paid lane.

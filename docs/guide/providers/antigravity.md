---
title: Google (Antigravity)
---

# Google (Antigravity)

The `antigravity--acp` shim serves a Google **Antigravity** plan through the
vendor's own `agy` CLI. It does not call Google's Code Assist API: the gateway
starts `agy acp` on the host and speaks the
[Agent Client Protocol](https://agentclientprotocol.com/) to it over stdio, so
the Google sign-in, the access token and the project stay with `agy`, exactly
as they do when you run `agy` yourself.

!!! warning "Read this before you point an account at it"
    Google's terms decide what may use the plan, and Google may suspend an
    account it sees used outside Antigravity's own client. Driving `agy` is the
    least indirect way to reach the plan from a gateway — it *is* the vendor's
    client — but it is still a plan you can lose. Use an account you can afford
    to lose, and do not ship this lane to users you cannot support.

## Why `agy` and not the API

Calling Code Assist directly with Antigravity's own OAuth client is the route
Google suspends accounts for: it sees a request that is not coming from
Antigravity's own client. Driving the vendor's CLI keeps every request inside
the client Google expects. `agy` therefore owns the credential, and this shim
declares none.

| Route | Who owns the sign-in | Status |
|:---|:---|:---|
| Code Assist API with Antigravity's OAuth client | the gateway | not offered — accounts get suspended |
| `agy acp` (this shim) | `agy` | the route this repository supports |

## Shims

| Shim Name | Base Type | Transport | Credentials |
|:---|:---|:---|:---|
| `antigravity--acp` | `acp` | `acp` (local process, stdio) | none — `agy` keeps the token |

`acp` is an **upstream-only** format: no client sends it to the gateway. A
provider of this type is a target for the formats you already serve, so a Chat
Completions, Messages, Responses or generateContent request reaches it after
conversion like any other provider.

## Prerequisites

- `agy` installed **on the gateway host**, on `PATH` (or name another command
  in `connection.acp_argv`), and signed in:

  ```bash
  agy            # first run: completes the Google sign-in
  agy acp        # what the gateway starts; Ctrl-D to leave
  ```

- A Google account that already has the Antigravity plan you intend to spend.
- Permission for the gateway process to spawn subprocesses. A container needs
  `agy` inside the image, or a mounted copy.

## Setup

1. Add a provider whose type is `antigravity--acp`. No API key.
2. Name it in a model's `providers` list, asking for a model `agy` serves (see
   [Models](#models)):

```jsonc
{
  "providers": {
    "antigravity": {
      "type": "antigravity--acp"
      // connection comes from the shim: transport "acp", argv ["agy", "acp"]
      // override either here:
      // "transport": "acp",
      // "acp_argv": ["/usr/local/bin/agy", "acp"]
    }
  },
  "models": {
    "gemini-3.1-pro": {
      "providers": [{"name": "antigravity"}]
    }
  }
}
```

3. Send a request. Streaming and non-streaming clients are both served: the
   gateway reads the agent's updates as they arrive and writes the client's own
   dialect.

## Models

The model name in the request is handed to the agent's own `model` option for
the session. The names are the agent's, not this repository's — fetch them from
the agent (`agy` lists its models; the ACP session also advertises a `model`
option with its choices). A name the agent refuses is logged and the turn
proceeds on the session's own default, rather than failing the request.

## What travels

| Request side | Treatment |
|:---|:---|
| System instruction | sent as one string, when the agent accepts one |
| Conversation | rendered as labelled text blocks (`[User]`, `[Assistant]`, `[Tool result]`), in order |
| Images | sent as inline image blocks beside their message's text |
| Tool declarations, `tool_choice`, `tool_config` | **dropped, and reported** — there is no tool bridge yet |
| `generation`, `response_format`, `reasoning`, cache settings | dropped, and reported |
| File and audio parts | dropped, and reported |

| Response side | Treatment |
|:---|:---|
| Message chunks | streamed as text deltas |
| Thought chunks | streamed as reasoning deltas |
| Usage update | carried as a context-occupancy figure, not a per-turn token bill |
| Stop reason | mapped to the IR finish reason (`end_turn` → `stop`, `max_tokens` → `length`, …) |
| Tool calls | not surfaced yet (phase 2) |

Every dropped field is a conversion warning, so a request log says what this
lane could not carry.

## Limitations

- **No tool calling yet.** A request that declares tools still succeeds, but
  the model cannot call them. This is the first thing to arrive in the next
  phase, over an MCP bridge to the agent.
- **One session per conversation, when the client names it.** A request that
  declares a conversation id (`x-claude-code-session-id`, `x-opencode-session`,
  any `x-<vendor>-session-id` with the shape of an id, or the body's
  `metadata.user_id.session_id` / `prompt_cache_key`) keeps its agent session
  across turns: the ledger knows which prompt blocks the agent has already
  seen, so the next turn sends only the new tail. A history that no longer
  extends that record — a rewritten turn, a compaction — closes the session and
  replays everything into a fresh one. A request that declares nothing still
  gets one session per request, because a guessed identity could hand one
  conversation's transcript to another.
- **A restart replays every conversation once.** The ledger is volatile: it
  describes live agent sessions, so it starts empty and the first turn of each
  conversation after a restart sends the whole history again.
- **Processes and sessions are bounded.** `acp.processes` (default 1) sets how
  many agent processes the provider runs — one process serves its sessions one
  turn at a time, so this is what lets two conversations run at once;
  `acp.max_sessions` (default 32) and `acp.session_ttl` (seconds, default
  1800) bound what those processes hold, and an idle session is given back by a
  periodic sweep rather than only when the next request arrives.
- **The agent's own tools are its business.** `agy` may act on the host it
  runs on. Give it a working directory you are willing to let a model write in
  — the gateway points it at a scratch directory under the system temp dir
  (`llm-rosetta-acp/<provider>`) unless the request supplies `cwd`.
- **Plan windows and limits are Google's**, and are not visible here.
- **`base_url` is a placeholder.** `http://agy.local` exists because the
  transport contract is shared with the HTTP lanes; nothing dials it.

## Errors

| Symptom | Meaning | Fix |
|:---|:---|:---|
| `503` “could not open a session … Sign the agent CLI in” | the agent refused `session/new`, usually because nobody is signed in | sign in with `agy` on the gateway host, then retry |
| `503` “could not start the ACP agent ['agy', 'acp']” | the command is not on `PATH`, or exits at startup | install `agy` for the gateway process's user, or set `connection.acp_argv` |
| `502` with the agent's own words | a non-sign-in refusal (a model the agent rejected, an upstream error) | read the message; it is the agent's |
| An error event mid-stream (`finish_reason: error`) | the agent failed after the turn started | the request log carries the reason |
| Every turn is slow | the whole history is replayed per request | expected in this phase (see Limitations) |

## Security

- The gateway never holds, reads or logs a Google token: `agy` stores it where
  it stores it for its own use.
- The agent is started once per provider and reused across requests; it is
  stopped when the gateway shuts down. One provider is one process, not one
  process per request.
- The process inherits the gateway's environment. If `agy` reads anything
  sensitive from it, use `acp_argv` with a wrapper that clears it.

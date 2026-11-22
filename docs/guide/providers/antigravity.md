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
| Tool declarations | served to the agent as a loopback MCP server — the model can call them |
| `tool_choice`, `tool_config` | dropped, and reported |
| `generation`, `response_format`, `reasoning`, cache settings | dropped, and reported |
| File and audio parts | dropped, and reported |

| Response side | Treatment |
|:---|:---|
| Message chunks | streamed as text deltas |
| Thought chunks | streamed as reasoning deltas |
| Usage update | carried as a context-occupancy figure, not a per-turn token bill |
| Stop reason | mapped to the IR finish reason (`end_turn` → `stop`, `max_tokens` → `length`, …) |
| Tool calls | surfaced as tool-call events; the turn stops with `tool_calls` until the next request carries the result |

Every dropped field is a conversion warning, so a request log says what this
lane could not carry.

Client tools keep the client in the loop. A request declares a conversation id
and one or more tools; the agent calls one through the gateway's loopback MCP
endpoint, the gateway ends that response with the call and
`stop_reason: "tool_calls"`, and the client sends the result on its next
request. The gateway answers the pending MCP call and the agent continues the
same turn. A request that declares tools but no conversation id is refused
with `400`, because the result could not find its way back. Verified against
`agy` 1.5.14; 1.5.13 routes the same call through its `call_mcp_tool`
meta-tool.

## Limitations

- **Client tools need a conversation id.** The client's result is matched to
  the suspended turn by the conversation, so a one-off request that declares
  tools is refused with `400` rather than losing the call. Declare a session
  id (see the next bullet) when the request carries tools.
- **The agent's own tools stay off.** `agy` runs in a gateway-owned profile: a
  custom agent with no tools, a `PreToolUse` hook that allows only the
  gateway's loopback MCP server, `toolPermission: strict`, and an ACP client
  that refuses every built-in tool. A turn in which a built-in tool starts
  fails with `502` rather than running on the host.
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
- **The working directory is still the host's.** The agent's built-in tools
  cannot run, but the session's `cwd` is real. The gateway points it at a
  scratch directory under the system temp dir (`llm-rosetta-acp/<provider>`)
  unless the request supplies `cwd`.
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
| Every turn is slow | the request declares no conversation id, so each turn opens a session and replays the history | declare a session id (see Limitations) |

## Security

- The gateway never holds, reads or logs a Google token: `agy` stores it where
  it stores it for its own use.
- The agent runs in a profile the gateway owns and re-writes before every
  session: its built-in tools cannot run on the host, and client tools reach
  it only through the gateway's loopback MCP endpoint.
- The agent is started once per provider and reused across requests; it is
  stopped when the gateway shuts down. One provider is one process, not one
  process per request.
- The process inherits the gateway's environment. If `agy` reads anything
  sensitive from it, use `acp_argv` with a wrapper that clears it.

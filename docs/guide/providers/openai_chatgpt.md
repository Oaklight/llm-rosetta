---
title: OpenAI (ChatGPT plan)
---

# OpenAI (ChatGPT plan)

The `openai_chatgpt--open_responses` shim serves the ChatGPT **plan** route:
your Go/Plus/Pro allowance, not Platform API billing. It speaks the same
[Open Responses](../api-standards.md) wire format as `openai--open_responses`, so a
request converts exactly as any other Responses request does. What differs is
where the credential comes from, which models are visible, and what the route
refuses.

!!! warning "Not an API-key replacement"
    The plan route is the integration OpenAI offers for local and open-source
    applications. Commercial or remotely hosted deployments need to be on
    OpenAI's waitlist. Nothing on this page changes what OpenAI bills you.

## Shims

| Shim Name | Base Type | Default Base URL | Credentials |
|:---|:---|:---|:---|
| `openai_chatgpt--open_responses` | `open_responses` | `https://api.openai.com/v1` | none — signed in |

There is deliberately no `api_key_env`: the credential is minted by the shim's
own `auth.py`. An `api_key` beside an exchange that does not consume one is
rejected at config load, because the two are alternative sources for one
credential.

## Prerequisites

- A plan that includes the route (Go, Plus or Pro), on the account you sign in
  with.
- A gateway host and a **browser** for the first sign-in. The sign-in does not
  have to happen on the gateway host — see [Signing in](#signing-in).
- A persistent data directory: the credential record lives in
  `<data_dir>/shim-auth/<provider>` and must survive restarts.

## Setup

1. Add a provider whose type is `openai_chatgpt--open_responses`. No API key.
2. Sign in (below). The provider reports “needs an interactive sign-in” until
   it has a record, and does not retry on every request while it is missing.
3. Fetch the model list — the route answers `{models: [{slug, …}]}` rather than
   `{data: [...]}`, and the shim keeps the entries it marks `visibility: "list"`.
4. Send a streaming request.

```jsonc
{
  "providers": {
    "chatgpt": {
      "type": "openai_chatgpt--open_responses",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

## Signing in

The sign-in needs a browser and a loopback listener, so it happens where the
browser is — not necessarily on the gateway host:

```bash
llm-rosetta-gateway credential chatgpt --out record.json
```

That runs the shim's interactive login and writes the credential record with
mode `0600`. Hand it to the gateway promptly — the ID token is verified on
arrival, so a record left overnight is refused and the sign-in has to be
repeated:

- **Admin panel** → Providers → the provider → **Credential** → **Install a
  record**. The panel also shows who is signed in, when the access token
  expires, a **Manage usage** link where the shim declares one, and **Sign
  out**.
- **API**: `PUT /admin/api/config/providers/<name>/credential` with the record
  as the JSON body.

A successful install re-seeds the live provider, so the next request uses it
without a restart. Repeating the sign-in is safe: a rotating refresh token is
replaced, and the account identity (the verified `sub`) does not change.

## What the gateway sends

The route accepts far less of the Responses surface than the format allows. The
shim's request transform removes what it refuses rather than letting the request
fail, and forces what it requires:

| Field | Treatment |
|:---|:---|
| `store` | forced to `false` (the route requires it, sent or not) |
| `previous_response_id` | stripped — the route keeps no server state, so send the history in `input` |
| `temperature`, `top_p`, `max_output_tokens`, `top_logprobs` | stripped |
| `background`, `conversation`, `max_tool_calls`, `metadata`, `moderation`, `multi_agent`, `prompt`, `prompt_cache_retention`, `safety_identifier`, `truncation`, `user` | stripped |

A stripped field is visible in the request log's applied-transform list
(`drop_refused_fields(...)`); the request itself still succeeds. `instructions`
carries system guidance, `input` carries the full history, and function tools
are sent with the namespacing the route expects.

**Streaming is required.** The route answers SSE only, so the gateway forces
`stream: true` upstream even when the client asked for one response. A
non-streaming client is refused with a `400` naming `stream=true` before any
upstream call: reassembling a Responses stream into one JSON body is not
implemented yet ([#1061](https://github.com/Oaklight/llm-rosetta/issues/1061)).
Most agent clients stream, so this affects only simple callers.

## Limitations

- **No hosted tools** except web search. Code interpreter, file search, image
  generation, computer use, hosted MCP and `tool_search` are not declared,
  because the route refuses them; declaring them would advertise what it
  cannot serve. The `web_search` tool is available.
- **No output-length control** — a stripped `max_output_tokens` cannot be
  enforced by the gateway.
- **Plan windows**: Plus shares a 5-hour window across apps, and each app has a
  weekly cap. The cap lives in your ChatGPT settings, not behind an API, which
  is why the panel only links to it.
- **Non-streaming clients** are refused (above).

## Errors

A plan failure that arrives *after* the stream has started keeps its
machine-readable code — the gateway terminates the client's stream with it
rather than a generic “stream failed”, because the recovery differs:

| Code | Status | What to do |
|:---|:---|:---|
| `subscription_sharing_usage_limit_exceeded` | 429 | pause plan requests; check your usage page |
| `subscription_sharing_user_not_eligible` | 403 | the account is not on the route — do not retry or re-run the sign-in |
| `subscription_sharing_usage_unavailable` | 503 | bounded backoff; keep the credential |
| `subscription_sharing_unsupported_capability` | 400 | inspect `error.param` and remove that field or tool |
| `subscription_sharing_route_not_supported` | 403 | wrong endpoint or method |
| `subscription_sharing_invalid_user` | 401 | credential failure — sign in again |
| `subscription_sharing_user_unavailable` | 503 | bounded backoff |
| `chatpass_v2_scope_not_authorized`, `chatpass_v2_invalid_authorization_context` | 403 | a grant or configuration problem — do not retry |

A request refused *before* the Responses layer answers with a plain
`{"detail": "…"}` body; the gateway preserves that status and body, and the
request id, rather than rewriting them.

## Security

- The record is written `0600` inside a `0700` directory, atomically, under an
  exclusive lock. Every read-modify-write takes that lock because a refresh
  token is spent when it is used: two concurrent refreshes would rotate it
  twice and lose the account.
- Tokens are never rendered by the admin panel, written to the request log or
  included in a credential listing; the panel shows the account and expiry the
  exchange's summary allows.
- **Sign out** revokes the grant upstream where the vendor allows it and takes
  the token out of the gateway's key ring immediately. Deleting the record
  directory by hand leaves the gateway believing it is connected until the next
  refresh fails — use the panel or `DELETE …/credential`.

## Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| “needs an interactive sign-in” at startup | no record at `<data_dir>/shim-auth/<provider>` | run the sign-in and install the record |
| The record is refused when pasted | the ID token has expired, or it belongs to another client | sign in again and paste promptly |
| `subscription_sharing_user_not_eligible` | the account is not on the route | use an eligible account; this is not a transient error |
| `usage_limit_exceeded` after a few requests | the plan's window or the app's weekly cap | pause; link to your usage page |
| Requests still 401 after a sign-in | the record was installed for a different provider name | two providers are two accounts; install per provider |

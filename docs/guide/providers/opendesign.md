---
title: Open Design
---

# Open Design

Open Design Cloud serves its models on one HTTPS gateway. Three shims expose the
three chat API standards that gateway accepts; they share a base URL and differ
only in the request body and the credential header.

## Shims

| Shim Name | Base Type | Default Base URL | Credentials |
|:---|:---|:---|:---|
| `opendesign--chat_completions` | `chat_completions` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |
| `opendesign--open_responses` | `open_responses` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |
| `opendesign--messages` | `messages` | `https://amr-link.open-design.ai/v1` | `OPENDESIGN_API_KEY` |

The model listing is `https://amr-link.open-design.ai/v1/models`.

Open Design serves every model on all three APIs, so pick the format the client
speaks. The shim signs the request the way the endpoint expects:

- `/chat/completions` and `/responses` send `Authorization: Bearer sk-amr-…`.
- `/messages` sends `x-api-key: sk-amr-…` and `anthropic-version: 2023-06-01`.

## Standalone package

The shims ship in
[`llm-rosetta-shim-opendesign`](https://github.com/Oaklight/llm-rosetta-shim-opendesign),
not in core. Declare the vendor before using the names:

```jsonc
{
  "shims": { "install": ["contrib:opendesign"] }
}
```

## Gateway configuration

```jsonc
{
  "shims": { "install": ["contrib:opendesign"] },
  "providers": {
    "opendesign": {
      "type": "opendesign--chat_completions",
      "api_key": "${OPENDESIGN_API_KEY}"
    }
  },
  "models": {
    "deepseek-v4.1-flash": { "provider": "opendesign" }
  }
}
```

The vendor manifest supplies the base URL, the `OPENDESIGN_API_KEY` environment
default, and the model-list URL; a provider entry may override any of them.

## Quota

The Coding Plan allowance is read on the **control plane**
(`https://amr-api.open-design.ai`), which accepts only the control key
(`vela_ctrl_…`) — never the runtime key. Both keys come from the same device
sign-in, and the shim's `quota.py` declares `credential_role: control`, so the
gateway asks the sign-in record for the second credential and files the reading
as **advisory**: the panel shows the 5-hour and 7-day windows and the plan
tier, and the reading can weight the lane's share, but it never demotes,
excludes or dates a model credential.

Without a control key the provider's quota reads as unsupported and no billing
request is made: there is no fallback to the runtime key, which the control
plane refuses with `401`.

## Status

The endpoint, the format-specific headers, and the model-list path follow the
documented Open Design cloud surface. Live behavior has not been exercised in
this project's tests.

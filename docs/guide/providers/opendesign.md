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
not in core. Declare the preset before using the names:

```jsonc
{
  "shims": { "presets": ["opendesign"] }
}
```

## Gateway configuration

```jsonc
{
  "shims": { "presets": ["opendesign"] },
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

This shim does not read the Coding Plan allowance. Open Design reports that
allowance on a separate control plane that does not accept the runtime API key,
and the control key is issued by a device-authorization flow rather than shown
in the web console. The gateway reports the provider's quota as unsupported. Tracking issue: [#1499](https://github.com/Oaklight/llm-rosetta/issues/1499).

## Status

The endpoint, the format-specific headers, and the model-list path follow the
documented Open Design cloud surface. Live behavior has not been exercised in
this project's tests.

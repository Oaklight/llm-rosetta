---
title: Keyless free endpoints
---

# Keyless free endpoints

Four providers publish a free pool that needs no account. None of them carries
a contract: the roster, the rate limit and the endpoint itself can change or
close without notice. They therefore ship as a standalone package,
[llm-rosetta-shim-free](https://github.com/Oaklight/llm-rosetta-shim-free) —
so a pool that disappears can be withdrawn without a core release.

| Shim Name | Vendor | Default Base URL | Free slice |
|:---|:---|:---|:---|
| `kilo_free--chat_completions` | Kilo Code | `https://api.kilo.ai/api/gateway` | Models marked `isFree` |
| `llm7--chat_completions` | LLM7.io | `https://api.llm7.io/v1` | The `turbo` tier |
| `opencode_zen_free--chat_completions` | OpenCode (Zen) | `https://opencode.ai/zen/v1` | Zero-cost models |
| `ovhcloud--chat_completions` | OVHcloud AI Endpoints | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` | The whole catalogue |

All four are OpenAI Chat Completions lanes, all are `keyless`, and all carry
`free_source: true`, so the admin UI groups them under **Free Resource**.

!!! warning "Free is not private, and not unlimited"
    Every one of these upstreams states that free-tier prompts may be logged.
    Never send secrets, credentials, or sensitive data through a free resource.

## Declaring the package

```jsonc
{
  "shims": { "presets": ["free"] },
  "providers": {
    "kilo-free": { "type": "kilo_free--chat_completions" },
    "llm7-free": { "type": "llm7--chat_completions" },
    "ovh-free": { "type": "ovhcloud--chat_completions" }
  }
}
```

`free` replaces the earlier `opencode_zen_free` preset. The OpenCode shim name
is unchanged, so a config written against that package only needs its preset
declaration switched.

## Rate limits

| Vendor | Anonymous limit | With a key |
|:---|:---|:---|
| Kilo | 200 requests/hour per IP | The key switches the same shim to Bearer auth |
| LLM7 | The `turbo` tier | A free token raises the ceiling to 60 RPM / 250 requests/hour |
| OpenCode Zen | Gated to the OpenCode client's request shape | — |
| OVHcloud | 2 requests/minute per IP per model | A Public Cloud key raises it to 400 RPM per project per model |

## What the shims repair

- **Reasoning field.** Kilo, LLM7 and OVHcloud serve reasoning text on
  `message.reasoning` / `delta.reasoning`, while the converter reads
  `reasoning_content`. Each ships a rename, a no-op when no reasoning is
  present.
- **Model roster.** Kilo and OpenCode list their paid models beside the free
  ones, so their `model_list_transform` keeps only the free slice. LLM7's
  catalogue is typed and tiered. OVHcloud's carries no type field and is
  offered whole.
- **Request shape.** OpenCode's lane refuses a request that does not declare a
  tool named exactly `read`; the shim injects a placeholder declaration and
  removes any call to it from the response.

## Kilo's paid lane

Kilo is the one vendor on both sides of the split. The paid gateway is a
built-in shim, `kilo--chat_completions`, which always sends `KILO_API_KEY` and
offers the whole catalogue; see [Kilo](kilo.md).

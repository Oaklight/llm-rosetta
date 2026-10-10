---
title: Kilo
---

# Kilo

Kilo's public gateway serves a free model pool. LLM-Rosetta exposes it as an
ordinary provider that needs no credential, shown in the admin UI as
**Kilo (Free)**.

!!! warning "Free is not private, and not unlimited"
    Free pools state that prompts may be logged by the upstream provider. Never
    send secrets, credentials, or sensitive data through a free resource. The
    model roster and the quota policy belong to the upstream and can change or
    disappear at any time.

## Shim

| Shim Name | Base Type | Default Base URL | Credentials |
|:---|:---|:---|:---|
| `kilo--openai_chat` | `openai_chat` | `https://api.kilo.ai/api/gateway` | none (keyless) |

The shim declares `display_name: Kilo`, `free_source: true`, a logo, and
`connection.keyless: true`:

- **Keyless** — with no `api_key` configured, requests carry **no** auth header,
  and the entry is titled **Kilo (Free)** under the *Free Resource* section.
- **Key optional** — supplying an `api_key` (or `api_key_env`) switches the same
  shim to `Authorization: Bearer <key>`. That entry is an ordinary provider: it
  is titled after the name you give it and appears under *Providers*, not *Free
  Resource*.

## Model roster

The upstream `GET /models` advertises its whole catalogue and marks the free
slice with `isFree: true`. The shim's `model_list_transform` exposes **only** that
slice.

The roster is discovered, not hardcoded. In the admin UI, **Fetch Models** pulls
it, and the **Refresh** button on the provider card re-pulls it, reports the diff
(`{total}` free models, `{new}` new), and offers anything new for one-click apply.
Refresh never adds or removes models silently.

There is **no server-side re-check** that a routed model is still free — the
routing table is operator-controlled. If the upstream withdraws a model, requests
for it fail upstream; remove it from the table.

## Gateway configuration

```jsonc
{
  "providers": {
    "kilo-free": {
      "type": "kilo--openai_chat"
      // no api_key — the free pool needs none.
      // add "api_key": "${KILO_API_KEY}" to use your own Kilo account instead.
    }
  }
}
```

Or use the **+ Kilo (Free)** preset in the admin UI, which creates this entry and
collapses the optional API-key field behind a "Use my own key" link.

## Privacy

Free-model traffic goes to Kilo. The admin UI shows a neutral disclosure —
*"Served by an external provider; prompts may be logged by that provider."*

## Aggregating multiple free sources

Each free source is its own provider entry (`kilo-free`, and later others such as
an OpenCode entry), all grouped under the **Free Resource** section — which only
appears once at least one such provider is configured. When the same model is
available from two free sources, route it as one model with both providers and a
strategy:

```jsonc
{
  "models": {
    "some-free-model": {
      "providers": ["kilo-free", "opencode-free"],
      "strategy": "affinity_round_robin"
    }
  }
}
```

The gateway's multi-provider routing supports `weighted_round_robin` and
`affinity_round_robin` strategies.

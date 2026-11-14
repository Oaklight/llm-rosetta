---
title: Kilo
---

# Kilo

Kilo Code runs a public model gateway. Two shims serve it, and the split is
deliberate: the paid lane sends a credential and offers the whole catalogue,
the free pool sends none and offers only the models the upstream marks free.

| Shim Name | Base Type | Default Base URL | Credentials | Ships in |
|:---|:---|:---|:---|:---|
| `kilo--chat_completions` | `chat_completions` | `https://api.kilo.ai/api/gateway` | `KILO_API_KEY` | core |
| `kilo_free--chat_completions` | `chat_completions` | `https://api.kilo.ai/api/gateway` | none (keyless) | [llm-rosetta-shim-free](https://github.com/Oaklight/llm-rosetta-shim-free) |

## Paid lane

`kilo--chat_completions` always authenticates. A provider on it with no key is
refused at startup rather than sent upstream unauthenticated, and its model
list is the whole catalogue the account can reach. Both shims declare
`group_label: Kilo`, so the picker files them under one vendor.

```jsonc
{
  "providers": {
    "kilo": {
      "type": "kilo--chat_completions",
      "api_key": "${KILO_API_KEY}"
    }
  }
}
```

## Free pool

`kilo_free--chat_completions` is keyless, carries `free_source: true`, and is
published outside core because the pool belongs to the upstream and can change
or close without notice. Declare the package before using the name:

```jsonc
{
  "shims": { "presets": ["free"] },
  "providers": {
    "kilo-free": { "type": "kilo_free--chat_completions" }
  }
}
```

!!! warning "Free is not private, and not unlimited"
    Free pools state that prompts may be logged by the upstream provider. Never
    send secrets, credentials, or sensitive data through a free resource. The
    model roster and the quota policy belong to the upstream and can change or
    disappear at any time.

### Model roster

The upstream `GET /models` advertises its whole catalogue and marks the free
slice with `isFree: true`. The free shim's `model_list_transform` exposes
**only** that slice; the paid shim has no filter.

In the admin UI, **Fetch Models** pulls the roster, and **Refresh** on the
provider card re-pulls it, reports the diff, and offers anything new for
one-click apply. Refresh never adds or removes models silently.

There is **no server-side re-check** that a routed model is still free — the
routing table is operator-controlled. If the upstream withdraws a model,
requests for it fail upstream; remove it from the table.

### Rate limits

The free pool allows 200 requests per hour per IP. Supplying an `api_key` to
the free shim switches it to `Authorization: Bearer <key>` and is the supported
way to use your own Kilo account on that lane — but the paid shim is the one
that offers the full catalogue.

## Aggregating multiple free sources

Each free source is its own provider entry (`kilo-free`, and later others such
as an OVHcloud or LLM7 entry), all grouped under the **Free Resource** section
— which only appears once at least one such provider is configured. When the
same model is available from two free sources, route it as one model with both
providers and a strategy:

```jsonc
{
  "models": {
    "some-free-model": {
      "providers": ["kilo-free", "ovh-free"],
      "strategy": "affinity_round_robin"
    }
  }
}
```

The gateway's multi-provider routing supports `weighted_round_robin` and
`affinity_round_robin` strategies.

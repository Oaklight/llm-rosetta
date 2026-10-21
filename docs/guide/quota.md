---
title: Provider Quota
---

# Provider Quota

A subscription lane — OpenCode Go, a ChatGPT plan, a coding plan — meters a
credential in windows rather than in dollars per request. The gateway can read
those windows, show them in the provider view, and stop routing to a credential
whose window is spent.

## What reads what

The layer is split so that each fact lives where it is maintained:

| Piece | Holds | Declared in |
|:---|:---|:---|
| **Shim** | Where this vendor's allowance is read and how it decodes | `provider.yaml` → `quota:`, plus the probe in `quota.py` |
| **Provider** | Which credentials exist, and any override of the shim | `config.jsonc` → `quota:` |
| **Quota state** | The last reading and any demotion, per credential | Runtime only |

The gateway package (`llm_rosetta.gateway.quota`) owns the **runtime**: the
per-credential state, probe execution and the admin surface. The **contract and
resolution** — the `QuotaProbe` interface, the window and in-band signal types,
`resolve_probe` — live in `llm_rosetta.shims.quota_probe`, because a shim's
`quota.py` implements it and the translation core must never import the
gateway. It contains no vendor code. A shim **ships its probe** as `quota.py` beside `provider.yaml`
(group-level for a vendor spanning several formats, leaf-level for a one-off)
and names it with `quota.probe`. The endpoint, the auth header, the body
decoder and the error wording are vendor facts and live with the vendor, so a
vendor that changes its answer is a change to that one shim module.

## State is per credential

A reading belongs to an account, not to a provider entry. The state key is
`(probe, sha256(key)[:16])`:

- two providers configured with the same key share one reading;
- the same key on two lanes (Go and Zen) has two readings, because the probe
  differs;
- no raw key is stored, logged or returned — the admin payload carries only the
  fingerprint.

The state is per process. A deployment running several gateway workers probes
and demotes independently in each; the upstream allowance is still shared, so
the readings converge, but a demotion learned in one worker is not visible to
another until its own probe or 429.

## Rotation

`KeyRing` keeps its round-robin and rendezvous-affinity behaviour and gains one
predicate: a credential is skipped while the state has demoted it. A provider
whose keys are all demoted falls back to normal rotation rather than failing —
a demotion is advisory and the upstream is the final authority.

An exhausted window demotes a credential until its `resetsAt`. A provider that
names the exhausted window in a 429 body is demoted in-band, without waiting for
the next probe. A failed *probe*, by contrast, is not evidence of anything: it
never demotes.

## Supported probes

| Probe | Endpoint | Credential | Shipped in |
|:---|:---|:---|:---|
| `opencode_go` | `{base_url}/usage` on `/zen/go/v1` | the Go key, as `Authorization: Bearer` | `providers/opencode/quota.py` |
| `openai_codex` | `https://chatgpt.com/backend-api/wham/usage` | the plan's access token, plus `chatgpt-account-id` from its JWT claim | `providers/openai/quota.py` |

The Zen lane (`/zen/v1`) has no keyed usage endpoint; asking for its quota
reports that instead of probing a URL that 404s. A pay-as-you-go balance is a
console-session reading and is out of scope.

## Configuration

The OpenCode shims declare the probe and a five-minute background refresh:

```yaml
quota:
  probe: opencode_go
  interval: 300
```

`opencode_go` is implemented in `providers/opencode/quota.py`, which the shim
loader executes once for the group and attaches to every OpenCode leaf shim;
the declaration only names it. A provider that needs its own vendor forces a
probe by name and URL:

```jsonc
{
  "providers": {
    "opencode-go": { "type": "opencode--chat_completions", "api_key": "${OPENCODE_API_KEY}" }
  }
}
```

| Field | Meaning |
|:---|:---|
| `probe` | `opencode_go` or `openai_codex` |
| `url` | Explicit usage URL, when it is not derivable from `base_url` |
| `interval` | Background refresh in seconds; `0` reads on demand only |
| `demote_at` | Used share that counts as spent; `100` keeps the vendor's rule |

## Reading it

The provider card in the admin panel has a **Quota** button. It calls
`GET /admin/api/config/providers/<name>/quota` (add `?refresh=1` to bypass the
one-minute cache) and shows one row per credential and window, plus any
demotion and its reason.

```json
{
  "provider": "opencode-go",
  "supported": true,
  "probe": "opencode_go",
  "credentials": [
    {
      "credential": "e8c2967da349b907",
      "reading": {
        "status": "ok",
        "windows": [
          {"name": "5 hours", "used_percent": 0, "resets_at": "2026-10-19T16:09:07Z", "exhausted": false},
          {"name": "Month", "used_percent": 47, "resets_at": "2026-11-11T22:26:38Z", "exhausted": false}
        ]
      },
      "demotion": null
    }
  ]
}
```

`used_percent` is the **used** share, which is how both vendors report it. A
window whose status is not `ok` (OpenCode's `rate-limited`) is exhausted even
when its share is below 100.

## Keyless lanes

The [OpenCode Zen public lane](providers/opencode.md#public-free-lane) has no
credential, so it has no per-key allowance. A 429 there puts the provider on a
short cooldown instead of demoting a key.

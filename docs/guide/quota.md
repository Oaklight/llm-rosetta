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

The gateway package (`llm_rosetta_gateway.quota`) owns the **runtime**: the
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

## How a reading reaches the router

A reading used to matter only at its extremes: a spent window demoted the
credential (or blocked the provider), and until then the quota layer was
invisible to routing. Two providers both "ready" split the traffic evenly even
when one of them was at 96%.

Now every selection weighs the candidates by their remaining allowance. The
selector takes the **smallest remaining share** across a provider's credentials
and scales its configured weight:

```
weight × (0.25 + 0.75 × headroom)
```

So an untouched provider keeps its weight, a nearly spent one serves about a
quarter of its share, and **nothing is scaled to zero** — a provider is removed
by the quota state (a spent window, a live block), never by this factor. A
provider nothing has measured keeps its configured weight: being unmeasured is
not the same as being spent.

Two properties worth knowing:

- The **configured weight is still the operator's statement**. The ratio only
  scales it, so a provider configured at `weight: 3` remains three times as
  attractive as its peer at the same headroom.
- Client affinity still hashes first. The headroom only affects the choice
  among providers, never whether a conversation stays where its cache is.

A provider is **removed** rather than down-weighted when a *fresh* reading says
every one of its credentials is spent. The rule is deliberately narrow, and
all-or-nothing across the key ring, because an exclusion cannot correct itself:
nothing routes to a provider that is out, so a credential the probe never read
would stay unread, and the provider would remain out for a whole freshness
window. Down-weighting has no such trap — traffic keeps flowing, the credential
gets read, and the weight recovers. Exclusion therefore requires complete
evidence: every credential in the ring has a reading, that reading is fresh, it
decoded a window, and every window it decoded is exhausted. A reading older
than fifteen minutes — or older than the provider's own `quota.interval`, when
that is longer, so a slow poll never makes every reading stale on arrival —
says nothing about now, and a lane with no probe at all is never excluded for
being unmeasured.

The exclusion expires at the vendor's own reset as well as with the reading's
age, and it applies before the strategy counts the candidates, so the surviving
providers split the traffic rather than one of them absorbing the turns the
excluded provider would have taken. When readings leave a model with no
provider at all, the client gets a `503` that says the allowance is spent — with
the reset in `Retry-After` when the vendor named one — rather than the
"initializing" message, which would send it looking for a startup problem that
does not exist.

The stage is called `quota` in the selection chain and runs before the anchor,
so it informs the choice rather than making it — the strategies stay what they
are.

The provider card shows both conclusions — the share, and whether the provider
is skipped with the reason — from the same functions the router calls, so
"why is this provider not serving anything" does not have to be inferred from
percentages.

## Keyless lanes

The [OpenCode Zen public lane](providers/opencode.md#public-free-lane) has no
credential, so it has no per-key allowance. A 429 there puts the provider on a
short cooldown instead of demoting a key.

## Lanes with no usage endpoint

Some lanes have an allowance the vendor never exposes to the application: the
[ChatGPT plan route](providers/openai_chatgpt.md) is metered in a 5-hour window
and a per-app weekly cap that live in the account's own settings. The only
signal the route gives is a refusal naming the exhausted window —
`subscription_sharing_usage_limit_exceeded`, on the response or mid-stream as
`response.failed` — so there is nothing to poll.

Such a shim declares its probe **in-band only**:

```yaml
quota:
  probe: openai_codex
  in_band: true
```

The probe is kept for its `inband()` classifier and never polled. When the
refusal arrives, the credential is demoted for a short bounded cooldown — 60
seconds — rather than until a reset time the vendor did not give: the same rule
as everywhere else, that a failed reading never demotes, and an error never
invents a window. The provider card reports **no readable quota** with that
reason rather than an error, and its **Manage usage** link points at the
vendor's settings page, which is the only place the cap is visible.

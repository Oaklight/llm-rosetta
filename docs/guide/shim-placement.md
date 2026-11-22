---
title: Shim Placement and Maturity
---

# Shim Placement and Maturity

A provider shim lives in one of two places: in **core** (it ships inside the
`llm-rosetta` distribution) or in a **source repository** that publishes a
versioned manifest and that the gateway installs from
([Manifest sources](shims.md#manifest-sources)). This page states which shims
belong where, how a shim moves between the two, and what a gateway does when
both provide the same name. The rules come from epic
[#1660](https://github.com/Oaklight/llm-rosetta/issues/1660).

## The placement rule { #placement-rule }

A shim belongs in core only when both of these hold:

1. **It is the model maker's own API**, not an aggregator, a reseller or a
   mirror.
2. **It authenticates with a static API key only.** The vendor directory has no
   `auth.py`. A `quota.py` is allowed when its probe uses only that key.

Everything else lives in a source repository permanently:

- OAuth and subscription lanes (OpenCode, OpenDesign, Codex /
  `openai_chatgpt`, Gemini and Claude plans);
- aggregators and free endpoints (Kilo, OpenRouter, Groq, NVIDIA, SiliconFlow,
  ModelScope, the free tier);
- Antigravity.

Keeping them apart keeps fast-changing login flows and terms-of-service risk
out of core releases.

**Eligibility is necessary, not sufficient.** A new shim, or one that is
rarely used, starts in `contrib` and is promoted when it is mature. This
applies to the existing core vendors too.

## The maturity ladder { #maturity-ladder }

| Stage | Where | Meaning |
|-------|-------|---------|
| `experimental` | contrib | New, or the upstream API is still changing; no stability promise |
| `beta` | contrib | Has tests and one live verification; the shim may still change |
| `stable` | contrib | Meets the promotion criteria; a candidate for core |
| core | core | Ships with core, under core's compatibility rules |

The source records the stage per vendor in its manifest (`maturity`, `since`,
and the last `conformance` result). The admin panel shows it on every vendor
card (**Shims → Available**).

## Promotion into core { #promotion }

A vendor is promoted only when **all** of these hold:

1. It is eligible under the [placement rule](#placement-rule).
2. It has been `stable` in contrib for at least one minor release cycle.
3. A live `tools/conformance` run passed for each of its formats within the
   last 30 days.
4. No shim-caused bug fix in the last 4 weeks, and no open bug against it.
5. Contrib has registration and transform tests for it, and a docs entry.
6. There is evidence of real use on any gateway. This one is advisory, not
   required.
7. The promotion is a reviewed PR.

When the vendor reaches core, its manifest record says
`promoted_to: core@X.Y.Z`. A gateway running core `X.Y.Z` or later stops
installing the vendor from the source (see [Precedence](#precedence)).

## Demotion back to contrib { #demotion }

A core vendor moves back to contrib when **any one** of these holds:

- The vendor changes its authentication so it is no longer a plain API key.
- A conformance failure stays unfixed for one minor cycle.
- There is no evidence of use and no maintainer for two minor cycles.

The shim name does not change, and the move keeps a notice for one minor
release.

## Names do not depend on location { #names }

A shim name — `type: <vendor>--<format>` in a provider entry — is the same in
core and in a source. Promotion and demotion change only where the shim is
installed from, never a user's config. A gateway that selects the vendor from
a source keeps working after a promotion (core provides the name) and after a
demotion (the source provides it, once `shims.install` selects it).

## Precedence between core and a source { #precedence }

When core and a source provide the same shim name:

- **Core wins, per shim.** The gateway does not load that name from the source
  and logs the source, the vendor and the ignored names. The vendor's other
  shims still install. A vendor whose shims are all core names is skipped, and
  its card says *core provides …*.
- **`promoted_to` is read from the release that would be installed.** It holds
  only when the running core is at or after the named version (a dev or
  pre-release of that version is before it) **and** core registers the
  record's shims. A claim that is due but not borne out is logged, and the
  vendor installs from the source as before. A later release without the claim
  un-promotes the vendor.
- **Between two sources, the first in `shims.sources` order wins** for a vendor
  both publish; the other copy is logged and ignored.
- **Explicit sources override manifest installs.** A manifest install loads
  before `shims.paths`.

## Where today's vendors sit { #placement-today }

The placement decided on 2026-11-21:

| Where | Vendors |
|-------|---------|
| Core | `openai`, `anthropic`, `google`, `deepseek`; the format references `cohere`, `jina`, `voyage` (embeddings and rerank) and `typesafe` (decisions), kept because core's tests and conformance need one real vendor per format |
| Contrib, `stable`, eligible for promotion | `minimax`, `zhipu` / `zai`, `xai`, `qwen`, `moonshot`, `volcengine` |
| Contrib, permanently | `opencode`, `openai_chatgpt` (with the `openai_codex` probe), `antigravity`, `kilo`, `openrouter`, `groq`, `nvidia`, `siliconflow`, `modelscope`, the free-tier vendors, `opendesign` |
| Contrib, new, starting at `experimental` | `mistral` |
| Argonne | `alcf`, `argo`, `asksage` |

The moves out of core are tracked in #1660 (S7). Until a vendor has moved, core
still provides it, and core wins over the source copy.

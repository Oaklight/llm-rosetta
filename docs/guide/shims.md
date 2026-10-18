---
title: Provider Shims
---

# Provider Shims

LLM-Rosetta uses only four **converters** — one per API standard (OpenAI Chat, OpenAI Responses, Anthropic, Google). But the LLM ecosystem has many more *providers* (DeepSeek, xAI, Qwen, Moonshot, …) that follow one of those standards with minor variations.

The **shim layer** bridges this gap. A shim is a lightweight identity card that declares which converter a provider uses, along with connection defaults and optional **transforms** that adapt request/response fields to match the provider's dialect.

## Architecture

```text
ProviderShim ("deepseek--chat_completions")
├── name: "deepseek--chat_completions"
├── model_type: "llm"                     → the capability category
├── format: "chat_completions"            → selects the converter
├── default_base_url: "https://api.deepseek.com"
├── default_api_key_env: "DEEPSEEK_API_KEY"
├── logo: "https://cdn.jsdelivr.net/..."
├── post_ir_transforms: (strip_fields("n", "logit_bias", "seed"),)
└── pre_ir_transforms: ()
```

- **ProviderShim** — provider identity: name, model type, wire format, default URL, default API key env var, logo URL, and optional transforms.
- **Transforms** — pure `dict → dict` functions applied around converters. `post_ir_transforms` adapt outgoing requests to the provider's dialect; `pre_ir_transforms` normalize incoming responses.

!!! note "Backward-compatible aliases"
    The old field names `to_transforms` and `from_transforms` are still accepted as aliases — both in `ProviderShim` constructor kwargs and in `transforms.py` exports.

### Declarative Provider Directory

Built-in shims are defined as a directory structure under `shims/providers/`:

```text
src/llm_rosetta/shims/providers/
├── __init__.py              # Auto-discovery: scans subdirectories
├── openai/
│   └── provider.yaml        # Provider identity (YAML)
├── deepseek/
│   ├── provider.yaml        # Provider identity
│   └── transforms.py        # Field-level transforms
├── volcengine/
│   ├── provider.yaml
│   └── transforms.py
└── ...
```

Each provider subdirectory contains:

- **`provider.yaml`** (required) — declares `name`, `model_type`, `format`, connection settings (`connection.base_url`, `connection.api_key_env`, `connection.auth_header`, `connection.keyless`, `connection.models_path`), plus optional `logo`, `display_name`, and `free_source`.  `base` is the deprecated alias for `format`; the loader validates the `(model_type, format)` pair against the converter registry.
- **`transforms.py`** (optional) — exports `post_ir_transforms` and/or `pre_ir_transforms` tuples (the old names `to_transforms` / `from_transforms` also work)

`connection.keyless` marks an upstream that accepts requests with no credential: the
provider sends no auth header while no `api_key` is configured, and falls back to the
base type's auth when one is supplied. `free_source: true` marks a shim whose gateway
serves a free model pool — the admin UI groups its keyless entry under a "Free
Resource" section and titles it `<display_name> (Free)`.

**How the admin panel groups shims.** The provider-type picker has two levels: a
top level (the *group*) and one row per shim under it. Both strings can be
declared, because they are not always derivable from the name:

| Field | Meaning | Default |
|-------|---------|---------|
| `display_name` | This shim's own label, shown in the second column — for a shim that also declares `group` or `group_label` | the name segment after `--` |
| `group` | The top-level key this shim is filed under | the name segment before `--` |
| `group_label` | Title of that top level | the group key |

The defaults are the old behaviour — the shim name split on its first `--` — so a
shim that declares none of these renders exactly as it always has, and neither
third-party nor external shims need to change. Declaring `group` is what lets one
vendor own several clusters × several formats: ALCF's three clusters each declare
`group: alcf` and `display_name: metis` / `minerva` / `sophia`, and the picker shows
one `alcf` top level whose rows read `metis · chat_completions`.

A row that shows `label · base` is the picker saying the shim's label is not its
base format — an affordance any shim can use, not a special case for one family.

`display_name` did not always mean this. Before `group` existed it named the
shim's **top level** — the vendor — and a shim written against that older schema
still says so. Which reading applies is decided by whether the shim declares a
grouping field: declaring neither `group` nor `group_label` keeps the older
meaning, so such a shim renders exactly as it did before (`display_name` titles
the group, and the row label still comes from the name). Declaring either makes
`display_name` this shim's own row label. Without that split, an older shim's
vendor name would be printed twice — once as the group title and again on its
only row. The related consequence: the group title is also the brand shown on
free-source buttons (`<Brand> (Free)`), so a grouped vendor that wants more than
its capitalized group key there declares `group_label`.

Example `provider.yaml`:

```yaml
name: deepseek
model_type: llm
format: chat_completions
default_base_url: https://api.deepseek.com
default_api_key_env: DEEPSEEK_API_KEY
logo: https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/deepseek.svg
```

Example `transforms.py`:

```python
from llm_rosetta.transforms import strip_fields

# DeepSeek does not support n, logit_bias, or seed
post_ir_transforms = (strip_fields("n", "logit_bias", "seed"),)
pre_ir_transforms = ()
```

At import time, `shims/__init__.py` scans all provider directories and registers them automatically, then discovers any plugin shims via entry points.

## Shim Lifecycle

```mermaid
flowchart LR
    subgraph Registration["Registration (startup)"]
        direction TB
        A["load_providers()"] --> B["load_providers_from_dir()
        scan built-in YAML dirs"]
        A --> C["_load_plugin_shims()
        scan entry points"]
        B --> D["register_shim()
        _SHIM_REGISTRY[name] = shim"]
        C --> D
    end

    subgraph Usage["Usage (per request)"]
        direction TB
        E["get_shim(name)"] --> F["_SHIM_REGISTRY.get(name)"]
        F --> G["inject reasoning_cap
        into ConversionContext"]
        F --> H["apply post_ir_transforms
        / pre_ir_transforms"]
    end

    Registration --> Usage
```

- **Registration** happens once at import time. `load_providers()` scans the built-in `providers/` directory, then discovers `llm_rosetta.shim_providers` entry points to load plugin shims.
- **Usage** happens per request. `get_shim(name)` looks up the registry; the gateway injects reasoning config and applies transforms around the converter.

## Plugin Shims

Downstream packages can register their own shims without modifying llm-rosetta. Two approaches:

### Entry Point (recommended)

Declare an entry point in your `pyproject.toml`:

```toml
[project.entry-points."llm_rosetta.shim_providers"]
my_provider = "my_package.shims:register_shims"
```

The callable scans a local YAML directory and returns the registered shims:

```python
# my_package/shims/__init__.py
from pathlib import Path
from llm_rosetta.shims import load_providers_from_dir

def register_shims():
    return load_providers_from_dir(Path(__file__).parent / "providers")
```

### Conditional Registration

For advanced use cases (environment-specific shims, dynamic config), call `register_shim()` directly:

```python
import os
from llm_rosetta.shims import register_shim, ProviderShim

def register_shims():
    if os.getenv("MY_INTERNAL_PROVIDER"):
        register_shim(ProviderShim(
            name="my-internal",
            format="chat_completions",
            default_base_url="http://internal:8080/v1",
        ))
```

!!! note
    When a plugin registers a shim with the same name as a built-in, the built-in is silently overridden (INFO log emitted). This is intentional — it allows plugins to customize built-in provider behavior.

### Config-declared directories and remotes

A shim set can also be consumed **without installing a plugin**. The gateway's
top-level `shims` config section scans extra directories and fetches remote
archives into a local cache — useful when the official image is used as-is (it
ships no `pip` and no `git`) and the shims live in their own repository:

```jsonc
"shims": {
  "paths": ["/config/shims"],                     // dirs holding <name>/provider.yaml
  "remotes": [
    {
      "url": "https://github.com/Oaklight/llm-rosetta-shim-alcf",
      "ref": "v0.1.0",                             // pin for reproducibility; omit => HEAD, pinned on first fetch
      "path": "src/llm_rosetta_shim_alcf/providers",
      "token_env": "GITHUB_TOKEN"                  // private repos
    }
  ]
}
```

A remote `path` is the directory inside the archive that holds the provider
directories (for the `llm-rosetta-shim-*` packages that is
`src/<package>/providers`). See
[Gateway → External Shim Sources](../gateway/configuration.md#external-shim-sources)
for the full field list, caching and refresh behaviour, precedence, and the
security note. Programmatically, use `llm_rosetta.shims.rebuild_shims()` (no
config file needed) or `llm_rosetta.gateway.apply_shim_sources(raw, config_path)`.

!!! warning
    A declared source may ship a `transforms.py` that is executed on load — the
    same trust level as installing a plugin package. Pin a ref and only add
    sources you trust.

## Built-in Shims

LLM-Rosetta ships with 34 built-in provider shims, one directory per shim under
`src/llm_rosetta/shims/providers/`. `list_shims()` prints the live set; this is
that set. A name marked **†** is scheduled for removal — see
[Shims scheduled for removal](#shims-scheduled-for-removal).

| Name | Base | Default Base URL | API Key Env | Transforms |
|------|------|-----------------|-------------|------------|
| † `alcf_metis--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1` | `ALCF_API_KEY` | strips `logprobs`, `top_logprobs`, `parallel_tool_calls`; developer → system; null content/description defaults |
| † `alcf_minerva--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1` | `ALCF_API_KEY` | strips `logprobs`, `top_logprobs`; developer → system; null content/description defaults |
| † `alcf_sophia--chat_completions` | `chat_completions` | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | `ALCF_API_KEY` | strips `logprobs`, `top_logprobs`; developer → system; null content/description defaults; renames `reasoning_content` → `reasoning` |
| `anthropic--messages` | `messages` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | — |
| † `argo--chat_completions` | `chat_completions` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` | `max_tokens` → `max_completion_tokens` (gemini); developer → system; per-model temperature stripping; gemini system flattening; image truncation; parallel-tool-call unwinding |
| † `argo--messages` | `messages` | `https://apps.inside.anl.gov/argoapi` | `ARGO_API_KEY` | OpenAI → Anthropic response normalization; per-model temperature stripping |
| † `argo--open_responses` | `open_responses` | `https://apps.inside.anl.gov/argoapi/v1` | `ARGO_API_KEY` | per-model temperature stripping |
| † `asksage--chat_completions` | `chat_completions` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` | `max_tokens` → `max_completion_tokens` |
| † `asksage--google_generate` | `google_generate` | `https://api.asksage.ai/server/google/v1beta` | `ASKSAGE_API_KEY` | — |
| † `asksage--messages` | `messages` | `https://api.asksage.ai/server/anthropic` | `ASKSAGE_API_KEY` | — |
| † `asksage--open_responses` | `open_responses` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` | `max_tokens` → `max_completion_tokens` |
| `deepseek--chat_completions` | `chat_completions` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | strips `n`, `logit_bias`, `seed`, penalties; backfills `reasoning_content` |
| `deepseek--open_responses` | `open_responses` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | strips `n`, `logit_bias`, `seed` |
| `google--google_generate` | `google_generate` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | — |
| `google--google_interactions` | `google_interactions` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | — |
| `kilo--chat_completions` | `chat_completions` | `https://api.kilo.ai/api/gateway` | *(keyless)* | — |
| `minimax--chat_completions` | `chat_completions` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` | strips `logprobs`, `top_logprobs`, `seed`, `stop`; injects `reasoning_split`; parses `think` tags |
| `minimax--messages` | `messages` | `https://api.minimaxi.com/anthropic` | `MINIMAX_API_KEY` | — |
| `moonshot` | `chat_completions` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | strips `logprobs`, `top_logprobs`, `logit_bias`, `seed` |
| `openai--chat_completions` | `chat_completions` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `openai--open_responses` | `open_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `openai_decisions` | `decision` | `https://api.openai.com` | `OPENAI_API_KEY` | — |
| `opencode--chat_completions` | `chat_completions` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` | — |
| `opencode--messages` | `messages` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` | — |
| `opencode--open_responses` | `open_responses` | `https://opencode.ai/zen/go/v1` | `OPENCODE_API_KEY` | — |
| `openrouter--chat_completions` | `chat_completions` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | renames `reasoning` → `reasoning_content` |
| `openrouter--messages` | `messages` | `https://openrouter.ai/api` | `OPENROUTER_API_KEY` | — |
| `openrouter--open_responses` | `open_responses` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | — |
| `qwen` | `chat_completions` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | strips `frequency_penalty`, `logit_bias` |
| `typesafe` | `decision` | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` | — |
| `volcengine--chat_completions` | `chat_completions` | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` | strips `logprobs`, `top_logprobs` |
| `volcengine--open_responses` | `open_responses` | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` | — |
| `xai` | `chat_completions` | `https://api.x.ai/v1` | `XAI_API_KEY` | strips `logit_bias` |
| `zhipu` | `chat_completions` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | strips `n`, penalties, `logprobs`, `logit_bias`, `seed` |

The column lists field-level transforms. `hoist_late_system_messages()`, which
rewrites mid-conversation system/developer messages so the prompt-cache prefix
stays stable, is declared by most `chat_completions` shims and is not repeated
here; every shim's full declaration is in its own `transforms.py`.

### Shims scheduled for removal

Ten of the built-ins are moving into their own repositories. The ALCF, Argo and
AskSage shims are tied to deployments whose upstreams change on their own
schedule, so they are maintained as standalone shim packages that a gateway
declares as an [external source](../gateway/configuration.md#external-shim-sources)
— and consumed without a new core release.

They **still ship, and still work**: nothing is removed in 0.14 or 0.15. What is
new is that the removal is declared, both to you and to the code.

| Shim | Removed in | Replacement source |
|------|-----------|--------------------|
| † `alcf_metis--chat_completions` | 0.16.0 | [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf) |
| † `alcf_minerva--chat_completions` | 0.16.0 | [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf) |
| † `alcf_sophia--chat_completions` | 0.16.0 | [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf) |
| † `argo--chat_completions` | 0.16.0 | [llm-rosetta-shim-argo](https://github.com/Oaklight/llm-rosetta-shim-argo) |
| † `argo--messages` | 0.16.0 | [llm-rosetta-shim-argo](https://github.com/Oaklight/llm-rosetta-shim-argo) |
| † `argo--open_responses` | 0.16.0 | [llm-rosetta-shim-argo](https://github.com/Oaklight/llm-rosetta-shim-argo) |
| † `asksage--chat_completions` | 0.16.0 | [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage) |
| † `asksage--google_generate` | 0.16.0 | [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage) |
| † `asksage--messages` | 0.16.0 | [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage) |
| † `asksage--open_responses` | 0.16.0 | [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage) |

**What to do about it.** Add the source; the provider's `shim` value does not
change, because the replacement shims keep the names.

```jsonc
"shims": {
  "remotes": [
    {
      "url": "https://github.com/Oaklight/llm-rosetta-shim-alcf",
      "path": "src/llm_rosetta_shim_alcf/providers",
      "token_env": "GITHUB_TOKEN"          // the three repositories are private
    }
  ]
}
```

The three replacement repositories are private, so `token_env` is required; the
per-repository `path` is the one thing that differs between the three. No `ref`
is pinned in the preset, because none of the repositories is tagged yet: a remote
without a `ref` resolves `HEAD` to a commit once and records it in the cache
stamp, so the same commit is reused on later runs. Once they are tagged, pin the
tag — see [Ref resolution](../gateway/configuration.md#external-shim-sources).

The admin panel does this for you: **Settings → Shim Sources** lists the
advisories for the shims your config actually uses, and adding a source from the
preset catalog appends exactly the entry above to the remotes list, for you to
review before saving. The gateway also logs one warning per affected provider at
startup, naming the shim, the release that removes it and the source that
replaces it — and it stops warning as soon as a declared source supplies that
shim, so a quiet log is the confirmation that the migration is done.

Aliases keep older configs working in the meantime: `alcf--metis`,
`alcf--minerva`, `alcf--sophia`, `argo--openai_chat`, `argo--anthropic`,
`argo--openai_responses`, `asksage--openai_chat`, `asksage--anthropic` and
`asksage--openai_responses` resolve to their new names and are rewritten in the
config file at startup, with a `.pre-migration.bak` backup. Those aliases go away
with the shims they stand for, in 0.16.0.

A CI check holds the schedule to the code, so the removal cannot happen early or
late: `scripts/check_shim_removals.py` (`make check-shim-removals`) reads the same
manifest and fails if a shim disappears before its `removed_in` release, and once
the version reaches that release it fails while the shim is still registered, or
still referenced by an example, a Docker file, the docs, or the source — a
leftover alias entry in `provider_names.py` counts, and so does the shim's own
`provider.yaml`. The boundary is the version, not a date — nothing has to be
remembered.

## Argo Shims

`argo--chat_completions`, `argo--messages` and `argo--open_responses` target the **Argo gateway** — a proxy layer used at certain institutions (such as Argonne National Laboratory) that fronts multiple upstream LLM providers behind a single endpoint.

All three shims use `model_id_field: internal_id` — the model identifier is sent as `internal_id` instead of `model`.

### `argo--chat_completions`

OpenAI-compatible shim with one transform: `max_tokens` → `max_completion_tokens` (newer OpenAI models reject the deprecated name).

### `argo--messages`

- **Model-level `thinking_type` override**: `thinking_type: enabled` by default, with `model_overrides` for `claudeopus47: thinking_type: adaptive` (Vertex AI backend). Handled declaratively via reasoning config.
- **`unsigned_reasoning_blocks: preserve`**: Prior thinking blocks without valid signatures are preserved in metadata instead of being forwarded (avoids Argo 400 errors).
- **`pre_ir_transforms` — OpenAI response normalization**: Argo may return OpenAI Chat format from `/v1/messages`. The transform converts it to Anthropic format before the converter sees it.

### `argo--open_responses`

The Responses-format lane, with per-model temperature stripping for the GPT/o-series models that reject it.

### Configuration

Override the `default_base_url` in your gateway config:

```jsonc
{
  "providers": {
    "argo": {
      "shim": "argo--messages",
      "base_url": "https://your-argo-instance.example.com/",
      "api_key": "${ARGO_API_KEY}"
    }
  }
}
```

!!! note
    The default URL (`https://apps.inside.anl.gov/argoapi/`) is only reachable from within the ANL network. All three Argo shims are [scheduled for removal](#shims-scheduled-for-removal) in 0.16.0, when they move to the standalone `llm-rosetta-shim-argo` package — declare it as an external source and the names in your config keep working.

## Reasoning Configuration

Since v0.6.8, provider shims can declare how they handle reasoning effort and disabled state via the `reasoning` section in `provider.yaml`. This replaces the previously hardcoded effort mapping branches in each converter.

### `ReasoningCapability` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disabled` | `"omit"` \| `"thinking_disabled"` | `"omit"` | How to serialize `mode: "disabled"` — omit the field entirely, or emit a provider-specific disabled marker |
| `effort_field` | `"reasoning_effort"` \| `"output_config.effort"` \| ... | `"reasoning_effort"` | Where the provider expects the effort value in the request body |
| `thinking_type` | `"enabled"` \| `"adaptive"` \| `null` | `null` | Force `thinking.type` to this value; `null` means no override |
| `max_effort` | effort level or `null` | `null` | Highest effort level this shim should emit; higher values are clamped |
| `effort_map` | `{IR_level: provider_string}` | identity | Mapping from IR effort levels to provider-specific effort strings |

### Example: Declaring Reasoning in `provider.yaml`

```yaml
name: anthropic--messages
model_type: llm
format: messages
default_base_url: https://api.anthropic.com
default_api_key_env: ANTHROPIC_API_KEY
reasoning:
  disabled: thinking_disabled
  effort_field: output_config.effort
  effort_map:
    minimal: low
    low: low
    medium: medium
    high: high
    xhigh: xhigh
    max: max
```

```yaml
name: openai--chat_completions
model_type: llm
format: chat_completions
default_base_url: https://api.openai.com/v1
default_api_key_env: OPENAI_API_KEY
reasoning:
  disabled: omit
  effort_field: reasoning_effort
  max_effort: high
  effort_map:
    minimal: low
    low: low
    medium: medium
    high: high
    xhigh: high
    max: high
```

### Per-Model Overrides (`model_overrides`)

When models under the same provider have different reasoning capabilities, use `model_overrides` to declare per-model config. Each override inherits provider-level defaults for unset fields, keyed by **upstream model ID** (post-alias):

```yaml
name: argo--messages
model_type: llm
format: messages
reasoning:
  thinking_type: enabled       # default for most models
  effort_map: { ... }
  model_overrides:
    claudeopus47:
      thinking_type: adaptive  # Vertex AI requires adaptive
```

The gateway resolves the upstream model ID after alias resolution (e.g. `argo:claude-opus-4.7` → `claudeopus47`) and applies the matching override if found.

### How It Works

1. The gateway injects the shim's `ReasoningCapability` into `ConversionContext` before conversion. If the upstream model has a `model_overrides` entry, that override takes precedence
2. All four converters call the shared `apply_reasoning_config()` helper, which:
    - Looks up the IR effort in the shim's `effort_map`
    - Clamps to `max_effort` if set
    - Serializes `mode: "disabled"` according to the `disabled` strategy
    - Places the effort value in the correct field via `effort_field`
    - Applies `thinking_type` override if set (e.g. forces `enabled` → `adaptive` for Vertex AI models)
3. Input normalization (`normalize_reasoning_input()`) converts provider-native values like `"none"`, `"xhigh"`, `"max"` to IR-canonical form before conversion begins

!!! note "Safety: `thinking_type: enabled` without `budget_tokens`"
    Anthropic requires `budget_tokens` when `thinking.type = "enabled"`. If a `thinking_type: enabled` override is applied but the request has no `budget_tokens`, the helper automatically falls back to `"adaptive"` to avoid an invalid payload.

If a shim does not declare a `reasoning` section, default behavior is used (effort passed through as-is, disabled → omitted).

For full details on the IR effort ladder and per-provider mapping tables, see [Reasoning / Thinking Parameters](reasoning.md).

## Capability Flags

Beyond reasoning, shims can declare provider-level capability flags that control converter behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `supports_custom_tools` | `bool` | `False` | Whether the provider's API natively accepts `{type: "custom"}` tool definitions. When `False`, custom tools are downgraded to function wrappers with a synthesized JSON schema. |
| `hoist_system_messages` | `bool` | `True` | Rewrite mid-conversation system/developer messages as user-role `[System: ...]` envelopes to preserve prompt cache prefix stability. |
| `multimodal_tool_result` | `bool \| None` | `None` | Whether the provider supports multimodal content (images, files) in tool results natively. `None` defers to the converter's class-level default. `True` forces native pass-through; `False` forces dual-encoding (text fallback + synthetic user message). |

Example in `provider.yaml`:

```yaml
name: my-provider
model_type: llm
format: chat_completions
supports_custom_tools: true
hoist_system_messages: true
multimodal_tool_result: false
```

These flags are injected into `ConversionContext.options` at conversion time, so converters can query them without knowing which shim is active.

## Transforms

Transforms are pure `dict → dict` functions that bridge the gap between a provider's actual API dialect and the "ideal" standard that the corresponding base converter expects. They handle field-level quirks (strip unsupported fields, rename parameters, inject defaults) — **not** semantic API-standard translation, which is the converter's job.

### Built-in Transform Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `strip_fields(*keys)` | Remove unsupported fields from the body | `strip_fields("logprobs", "top_logprobs")` |
| `rename_field(old, new)` | Rename a top-level field | `rename_field("max_tokens", "max_length")` |
| `set_defaults(**kv)` | Set fields only when absent (idempotent) | `set_defaults(temperature=0.7)` |
| `default_tool_description(text)` | Inject a placeholder description for tools with empty descriptions | `default_tool_description("No description provided")` |

### Response Body Transforms

In addition to request transforms (`post_ir_transforms` / `pre_ir_transforms`), shims can declare `response_body_transforms` — pure `dict → dict` functions applied to the raw upstream response body before any IR conversion. This is useful for fixing malformed responses from non-standard endpoints.

```yaml
# provider.yaml — declare that response transforms exist
response_body_transforms: true
```

```python
# transforms.py
from llm_rosetta.transforms import harmony_tool_call_safeguard

response_body_transforms = (harmony_tool_call_safeguard(),)
```

Built-in response transform: `harmony_tool_call_safeguard()` fixes malformed tool call JSON from Harmony/vLLM endpoints.

### How Transforms Apply

Transforms are applied at two levels:

**1. `convert()` public API** — automatically via `resolve_transforms()`:

```python
from llm_rosetta import convert

# Transforms are applied automatically when source/target is a shim name
result = convert(request_body, source="chat_completions", target="volcengine--chat_completions")
# → logprobs and top_logprobs stripped from the output
```

**2. Gateway proxy pipeline** — applied around the converter:

```mermaid
flowchart LR
    subgraph Request
        direction LR
        R1[Client body] --> R2["source.from_provider()"] --> R3[IR] --> R4["target.to_provider()"] --> R5["post_ir_transforms"] --> R6[Upstream API]
    end
    subgraph Response
        direction LR
        S1[Upstream] --> S2["pre_ir_transforms"] --> S3["target.response_from_provider()"] --> S4[IR] --> S5["source.response_to_provider()"] --> S6[Client]
    end
    subgraph Stream
        direction LR
        T1[Chunk] --> T2["pre_ir_transforms"] --> T3["target.stream_from_provider()"] --> T4[IR] --> T5["source.stream_to_provider()"] --> T6[Client]
    end
```

### Design Principles

- **Idempotent**: applying the same transform twice is harmless
- **Non-overlapping**: transforms should operate on different fields by convention
- **Composable**: multiple transforms are applied sequentially via `apply_transforms()`

## Using Shims

### Resolving a Converter by Shim Name

`get_converter_for_provider()` accepts both base converter type strings and shim names:

```python
from llm_rosetta import get_converter_for_provider

# Base type — works as before
converter = get_converter_for_provider("chat_completions")

# Shim name — resolved to "chat_completions" via the registry
converter = get_converter_for_provider("deepseek--chat_completions")
```

### Resolving the Format and the Model Type

Use `resolve_format()` to map a shim name to its wire format, and
`resolve_model_type()` to get its capability category:

```python
from llm_rosetta import resolve_format, resolve_model_type

resolve_format("deepseek--chat_completions")  # → "chat_completions"
resolve_format("chat_completions")            # → "chat_completions" (pass-through)
resolve_format("unknown")                     # → "unknown" (pass-through)

resolve_model_type("deepseek--chat_completions")  # → "llm"
resolve_model_type("typesafe")                    # → "decision"
resolve_model_type("openai")                      # → None (ambiguous: vendor and embedding format)
```

`resolve_base()` is the deprecated alias for `resolve_format()`.

## Registering Custom Shims

### Programmatic Registration

Register a custom provider shim for any OpenAI-compatible service:

```python
from llm_rosetta import ProviderShim, register_shim
from llm_rosetta.transforms import strip_fields

my_shim = ProviderShim(
    name="my-provider",
    format="chat_completions",
    default_base_url="https://api.my-provider.com/v1",
    default_api_key_env="MY_PROVIDER_API_KEY",
    post_ir_transforms=(strip_fields("logprobs", "seed"),),
)
register_shim(my_shim)
```

After registration the shim name works everywhere — `get_converter_for_provider()`, `resolve_format()`, `convert()`, and gateway config.

### Adding a YAML-based Provider

To add a new provider to the built-in registry:

1. Create a directory under `src/llm_rosetta/shims/providers/<name>/`
2. Add a `provider.yaml` with required fields:

    ```yaml
    name: my-provider
    model_type: llm
    format: chat_completions
    default_base_url: https://api.my-provider.com/v1
    default_api_key_env: MY_PROVIDER_API_KEY
    logo: https://example.com/logo.svg
    ```

3. Optionally add a `transforms.py` if the provider has field-level quirks:

    ```python
    from llm_rosetta.transforms import strip_fields

    post_ir_transforms = (strip_fields("unsupported_field"),)
    pre_ir_transforms = ()
    ```

The provider is automatically discovered and registered at import time.

### Listing and Removing Shims

```python
from llm_rosetta import list_shims, unregister_shim

# List all registered shims
for shim in list_shims():
    print(f"{shim.name} → {shim.format}")

# Remove a shim
unregister_shim("my-provider")
```

## Gateway Integration

In a gateway configuration file, use the `"shim"` field to reference a registered shim instead of specifying `"type"` directly:

```jsonc
{
  "providers": {
    "my-deepseek": {
      "shim": "deepseek--chat_completions",
      "api_key": "${DEEPSEEK_API_KEY}"
      // base_url defaults to shim's default_base_url
    }
  },
  "models": {
    "deepseek-chat": "my-deepseek"
  }
}
```

Resolution order for provider type:

1. `"shim"` field — resolved via the shim registry to a base converter type
2. `"type"` field — used directly as the converter type
3. Provider config key name — used as fallback

When a shim is found:

- `default_base_url` and `default_api_key_env` serve as fallbacks if not set in config
- `post_ir_transforms` are applied to outgoing requests before sending to the upstream provider
- `pre_ir_transforms` are applied to incoming responses/stream chunks before conversion

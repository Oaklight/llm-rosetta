---
title: Provider Shims
---

# Provider Shims

LLM-Rosetta uses only four **converters** — one per API standard (OpenAI Chat, OpenAI Responses, Anthropic, Google). But the LLM ecosystem has many more *providers* (DeepSeek, xAI, Qwen, Moonshot, …) that follow one of those standards with minor variations.

The **shim layer** bridges this gap. A shim is a lightweight identity card that declares which converter a provider uses, along with connection defaults and optional **transforms** that adapt request/response fields to match the provider's dialect.

## Architecture

```text
ProviderShim ("deepseek")
├── name: "deepseek"
├── base: "openai_chat"              → selects the converter
├── default_base_url: "https://api.deepseek.com"
├── default_api_key_env: "DEEPSEEK_API_KEY"
├── logo: "https://cdn.jsdelivr.net/..."
├── to_transforms: (strip_fields("n", "logit_bias", "seed"),)
└── from_transforms: ()
```

- **ProviderShim** — provider identity: name, base converter type, default URL, default API key env var, logo URL, and optional transforms.
- **Transforms** — pure `dict → dict` functions applied around converters. `to_transforms` adapt outgoing requests to the provider's dialect; `from_transforms` normalize incoming responses.

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

- **`provider.yaml`** (required) — declares `name`, `base`, `default_base_url`, `default_api_key_env`, and `logo`
- **`transforms.py`** (optional) — exports `to_transforms` and/or `from_transforms` tuples

Example `provider.yaml`:

```yaml
name: deepseek
base: openai_chat
default_base_url: https://api.deepseek.com
default_api_key_env: DEEPSEEK_API_KEY
logo: https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/deepseek.svg
```

Example `transforms.py`:

```python
from llm_rosetta.shims.transforms import strip_fields

# DeepSeek does not support n, logit_bias, or seed
to_transforms = (strip_fields("n", "logit_bias", "seed"),)
from_transforms = ()
```

At import time, `shims/__init__.py` scans all provider directories and registers them automatically.

## Built-in Shims

LLM-Rosetta ships with 13 built-in provider shims:

| Name | Base | Default Base URL | API Key Env | Transforms |
|------|------|-----------------|-------------|------------|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | — |
| `google` | `google` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | — |
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | strips `n`, `logit_bias`, `seed` |
| `volcengine` | `openai_chat` | — | `VOLCENGINE_API_KEY` | strips `logprobs`, `top_logprobs` |
| `xai` | `openai_chat` | `https://api.x.ai/v1` | `XAI_API_KEY` | strips `logit_bias` |
| `qwen` | `openai_chat` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | strips `frequency_penalty`, `logit_bias` |
| `moonshot` | `openai_chat` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | strips `logprobs`, `top_logprobs`, `logit_bias`, `seed` |
| `minimax` | `openai_chat` | `https://api.minimax.chat/v1` | `MINIMAX_API_KEY` | strips `logprobs`, `top_logprobs`, `seed`, `stop` |
| `zhipu` | `openai_chat` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | strips `n`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `seed` |
| `openrouter` | `openai_chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | — |

## Transforms

Transforms are pure `dict → dict` functions that bridge the gap between a provider's actual API dialect and the "ideal" standard that the corresponding base converter expects. They handle field-level quirks (strip unsupported fields, rename parameters, inject defaults) — **not** semantic API-standard translation, which is the converter's job.

### Built-in Transform Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| `strip_fields(*keys)` | Remove unsupported fields from the body | `strip_fields("logprobs", "top_logprobs")` |
| `rename_field(old, new)` | Rename a top-level field | `rename_field("max_tokens", "max_length")` |
| `set_defaults(**kv)` | Set fields only when absent (idempotent) | `set_defaults(temperature=0.7)` |

### How Transforms Apply

Transforms are applied at two levels:

**1. `convert()` public API** — automatically via `resolve_transforms()`:

```python
from llm_rosetta import convert

# Transforms are applied automatically when source/target is a shim name
result = convert(request_body, source="openai_chat", target="volcengine")
# → logprobs and top_logprobs stripped from the output
```

**2. Gateway proxy pipeline** — applied around the converter:

```text
Request:  client body → source.from_provider() → IR → target.to_provider()
          → [to_transforms] → upstream API

Response: upstream → [from_transforms] → target.response_from_provider()
          → IR → source.response_to_provider() → client

Stream:   chunk → [from_transforms] → target.stream_from_provider()
          → IR → source.stream_to_provider() → client
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
converter = get_converter_for_provider("openai_chat")

# Shim name — resolved to "openai_chat" via the registry
converter = get_converter_for_provider("deepseek")
```

### Resolving a Base Type

Use `resolve_base()` to map a shim name to its base converter type:

```python
from llm_rosetta import resolve_base

resolve_base("deepseek")       # → "openai_chat"
resolve_base("openai_chat")    # → "openai_chat" (pass-through)
resolve_base("unknown")        # → "unknown" (pass-through)
```

## Registering Custom Shims

### Programmatic Registration

Register a custom provider shim for any OpenAI-compatible service:

```python
from llm_rosetta import ProviderShim, register_shim
from llm_rosetta.shims.transforms import strip_fields

my_shim = ProviderShim(
    name="my-provider",
    base="openai_chat",
    default_base_url="https://api.my-provider.com/v1",
    default_api_key_env="MY_PROVIDER_API_KEY",
    to_transforms=(strip_fields("logprobs", "seed"),),
)
register_shim(my_shim)
```

After registration the shim name works everywhere — `get_converter_for_provider()`, `resolve_base()`, `convert()`, and gateway config.

### Adding a YAML-based Provider

To add a new provider to the built-in registry:

1. Create a directory under `src/llm_rosetta/shims/providers/<name>/`
2. Add a `provider.yaml` with required fields:

    ```yaml
    name: my-provider
    base: openai_chat
    default_base_url: https://api.my-provider.com/v1
    default_api_key_env: MY_PROVIDER_API_KEY
    logo: https://example.com/logo.svg
    ```

3. Optionally add a `transforms.py` if the provider has field-level quirks:

    ```python
    from llm_rosetta.shims.transforms import strip_fields

    to_transforms = (strip_fields("unsupported_field"),)
    from_transforms = ()
    ```

The provider is automatically discovered and registered at import time.

### Listing and Removing Shims

```python
from llm_rosetta import list_shims, unregister_shim

# List all registered shims
for shim in list_shims():
    print(f"{shim.name} → {shim.base}")

# Remove a shim
unregister_shim("my-provider")
```

## Gateway Integration

In a gateway configuration file, use the `"shim"` field to reference a registered shim instead of specifying `"type"` directly:

```jsonc
{
  "providers": {
    "my-deepseek": {
      "shim": "deepseek",
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
- `to_transforms` are applied to outgoing requests before sending to the upstream provider
- `from_transforms` are applied to incoming responses/stream chunks before conversion

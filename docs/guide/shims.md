---
title: Provider Shims
---

# Provider Shims

LLM-Rosetta uses only four **converters** — one per API standard (OpenAI Chat, OpenAI Responses, Anthropic, Google). But the LLM ecosystem has many more *providers* (DeepSeek, Volcengine, Together, …) that follow one of those standards with minor variations.

The **shim layer** bridges this gap. A shim is a lightweight identity card that declares which converter a provider uses, along with connection defaults and model-level metadata.

## Architecture

```text
ProviderShim ("deepseek")
├── base: "openai_chat"              → selects the converter
├── default_base_url: "https://api.deepseek.com"
├── default_api_key_env: "DEEPSEEK_API_KEY"
└── models:
    └── ModelShim("deepseek-*", capabilities={"reasoning", "tools"})
```

- **ProviderShim** — provider identity: name, base converter type, default URL, default API key env var, and nested model shims.
- **ModelShim** — model-level metadata nested within a provider. Uses glob patterns (`fnmatch`) to match model names.

A `ModelShim` always belongs to exactly one `ProviderShim`. This nested design enables atomic registration and easy copy/fork workflows.

## Built-in Shims

LLM-Rosetta ships with the following built-in shims, auto-registered at import time:

| Name | Base | Default Base URL | Models |
|------|------|-----------------|--------|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `o1-*`, `o3-*`, `o4-*`, `gpt-*` |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | (same as openai) |
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `claude-*` |
| `google` | `google` | `https://generativelanguage.googleapis.com` | `gemini-2.5-*`, `gemini-*` |
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `deepseek-*` |
| `volcengine` | `openai_chat` | — | — |

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

### Querying Model Capabilities

```python
from llm_rosetta import get_shim

shim = get_shim("openai")
model_shim = shim.get_model_shim("o3-mini")
print(model_shim.capabilities)  # frozenset({'reasoning', 'tools', 'vision'})
```

## Registering Custom Shims

Register a custom provider shim for any OpenAI-compatible service:

```python
from llm_rosetta import ProviderShim, ModelShim, register_shim

my_shim = ProviderShim(
    name="my-provider",
    base="openai_chat",
    default_base_url="https://api.my-provider.com/v1",
    default_api_key_env="MY_PROVIDER_API_KEY",
    models=(
        ModelShim("my-model-*", frozenset({"tools", "vision"})),
    ),
)
register_shim(my_shim)
```

After registration the shim name works everywhere — `get_converter_for_provider()`, `resolve_base()`, and gateway config.

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

When a shim is found, its `default_base_url` and `default_api_key_env` serve as fallbacks if the provider config does not specify them explicitly.

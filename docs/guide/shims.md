---
title: 提供商 Shim
---

# 提供商 Shim

LLM-Rosetta 仅使用四个**转换器** —— 每种 API 标准一个（OpenAI Chat、OpenAI Responses、Anthropic、Google）。但 LLM 生态中有更多*提供商*（DeepSeek、火山引擎、Together 等）遵循其中某一标准，只有细微差异。

**Shim 层**弥合了这一差距。Shim 是一张轻量级的身份卡，声明提供商使用哪个转换器，同时携带连接默认值和模型级元数据。

## 架构

```text
ProviderShim ("deepseek")
├── base: "openai_chat"              → 选择转换器
├── default_base_url: "https://api.deepseek.com"
├── default_api_key_env: "DEEPSEEK_API_KEY"
└── models:
    └── ModelShim("deepseek-*", capabilities={"reasoning", "tools"})
```

- **ProviderShim** —— 提供商身份：名称、基础转换器类型、默认 URL、默认 API 密钥环境变量，以及嵌套的模型 shim。
- **ModelShim** —— 嵌套在提供商内的模型级元数据。使用 glob 模式（`fnmatch`）匹配模型名称。

`ModelShim` 始终属于且仅属于一个 `ProviderShim`。这种嵌套设计支持原子注册和便捷的复制/派生工作流。

## 内置 Shim

LLM-Rosetta 内置以下 shim，导入时自动注册：

| 名称 | 基础类型 | 默认 Base URL | 模型 |
|------|---------|--------------|------|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `o1-*`、`o3-*`、`o4-*`、`gpt-*` |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | （同 openai） |
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `claude-*` |
| `google` | `google` | `https://generativelanguage.googleapis.com` | `gemini-2.5-*`、`gemini-*` |
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `deepseek-*` |
| `volcengine` | `openai_chat` | — | — |

## 使用 Shim

### 通过 Shim 名称解析转换器

`get_converter_for_provider()` 同时接受基础转换器类型字符串和 shim 名称：

```python
from llm_rosetta import get_converter_for_provider

# 基础类型 —— 与之前一样
converter = get_converter_for_provider("openai_chat")

# Shim 名称 —— 通过注册表解析为 "openai_chat"
converter = get_converter_for_provider("deepseek")
```

### 解析基础类型

使用 `resolve_base()` 将 shim 名称映射到基础转换器类型：

```python
from llm_rosetta import resolve_base

resolve_base("deepseek")       # → "openai_chat"
resolve_base("openai_chat")    # → "openai_chat"（直接透传）
resolve_base("unknown")        # → "unknown"（直接透传）
```

### 查询模型能力

```python
from llm_rosetta import get_shim

shim = get_shim("openai")
model_shim = shim.get_model_shim("o3-mini")
print(model_shim.capabilities)  # frozenset({'reasoning', 'tools', 'vision'})
```

## 注册自定义 Shim

为任何 OpenAI 兼容服务注册自定义提供商 shim：

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

注册后，shim 名称可在所有地方使用 —— `get_converter_for_provider()`、`resolve_base()` 和网关配置。

### 列出和移除 Shim

```python
from llm_rosetta import list_shims, unregister_shim

# 列出所有已注册的 shim
for shim in list_shims():
    print(f"{shim.name} → {shim.base}")

# 移除 shim
unregister_shim("my-provider")
```

## 网关集成

在网关配置文件中，使用 `"shim"` 字段引用已注册的 shim，而非直接指定 `"type"`：

```jsonc
{
  "providers": {
    "my-deepseek": {
      "shim": "deepseek",
      "api_key": "${DEEPSEEK_API_KEY}"
      // base_url 默认使用 shim 的 default_base_url
    }
  },
  "models": {
    "deepseek-chat": "my-deepseek"
  }
}
```

提供商类型的解析顺序：

1. `"shim"` 字段 —— 通过 shim 注册表解析为基础转换器类型
2. `"type"` 字段 —— 直接用作转换器类型
3. 提供商配置键名 —— 作为后备

当找到 shim 时，其 `default_base_url` 和 `default_api_key_env` 在提供商配置未明确指定时作为后备值使用。

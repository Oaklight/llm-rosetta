---
title: 提供商 Shim
---

# 提供商 Shim

LLM-Rosetta 仅使用四个**转换器** —— 每种 API 标准一个（OpenAI Chat、OpenAI Responses、Anthropic、Google）。但 LLM 生态中有更多*提供商*（DeepSeek、xAI、Qwen、Moonshot 等）遵循其中某一标准，只有细微差异。

**Shim 层**弥合了这一差距。Shim 是一张轻量级的身份卡，声明提供商使用哪个转换器，同时携带连接默认值和可选的**转换规则（transforms）**，用于适配提供商特有的请求/响应字段差异。

## 架构

```text
ProviderShim ("deepseek")
├── name: "deepseek"
├── base: "openai_chat"              → 选择转换器
├── default_base_url: "https://api.deepseek.com"
├── default_api_key_env: "DEEPSEEK_API_KEY"
├── logo: "https://cdn.jsdelivr.net/..."
├── to_transforms: (strip_fields("n", "logit_bias", "seed"),)
└── from_transforms: ()
```

- **ProviderShim** —— 提供商身份：名称、基础转换器类型、默认 URL、默认 API 密钥环境变量、Logo URL，以及可选的转换规则。
- **Transforms** —— 纯 `dict → dict` 函数，围绕转换器应用。`to_transforms` 将输出请求适配为提供商方言；`from_transforms` 标准化输入响应。

### 声明式提供商目录

内置 shim 以目录结构定义在 `shims/providers/` 下：

```text
src/llm_rosetta/shims/providers/
├── __init__.py              # 自动发现：扫描子目录
├── openai/
│   └── provider.yaml        # 提供商身份（YAML）
├── deepseek/
│   ├── provider.yaml        # 提供商身份
│   └── transforms.py        # 字段级转换规则
├── volcengine/
│   ├── provider.yaml
│   └── transforms.py
└── ...
```

每个提供商子目录包含：

- **`provider.yaml`**（必需）—— 声明 `name`、`base`、`default_base_url`、`default_api_key_env` 和 `logo`
- **`transforms.py`**（可选）—— 导出 `to_transforms` 和/或 `from_transforms` 元组

`provider.yaml` 示例：

```yaml
name: deepseek
base: openai_chat
default_base_url: https://api.deepseek.com
default_api_key_env: DEEPSEEK_API_KEY
logo: https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/deepseek.svg
```

`transforms.py` 示例：

```python
from llm_rosetta.shims.transforms import strip_fields

# DeepSeek 不支持 n、logit_bias 和 seed
to_transforms = (strip_fields("n", "logit_bias", "seed"),)
from_transforms = ()
```

导入时，`shims/__init__.py` 自动扫描所有提供商目录并注册。

## 内置 Shim

LLM-Rosetta 内置 14 个提供商 shim：

| 名称 | 基础类型 | 默认 Base URL | API Key 环境变量 | 转换规则 |
|------|---------|--------------|-----------------|---------|
| `openai` | `openai_chat` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `openai_responses` | `openai_responses` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | — |
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | — |
| `google` | `google` | `https://generativelanguage.googleapis.com` | `GOOGLE_API_KEY` | — |
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | 剥离 `n`、`logit_bias`、`seed` |
| `volcengine` | `openai_chat` | — | `VOLCENGINE_API_KEY` | 剥离 `logprobs`、`top_logprobs` |
| `xai` | `openai_chat` | `https://api.x.ai/v1` | `XAI_API_KEY` | 剥离 `logit_bias` |
| `qwen` | `openai_chat` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | 剥离 `frequency_penalty`、`logit_bias` |
| `moonshot` | `openai_chat` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` | 剥离 `logprobs`、`top_logprobs`、`logit_bias`、`seed` |
| `minimax` | `openai_chat` | `https://api.minimax.chat/v1` | `MINIMAX_API_KEY` | 剥离 `logprobs`、`top_logprobs`、`seed`、`stop` |
| `zhipu` | `openai_chat` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` | 剥离 `n`、`presence_penalty`、`frequency_penalty`、`logprobs`、`top_logprobs`、`logit_bias`、`seed` |
| `openrouter` | `openai_chat` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | — |
| `argo_openai_chat` | `openai_chat` | `https://apps.inside.anl.gov/argoapi/` | — | `model_id_field: internal_id` |
| `argo_anthropic` | `anthropic` | `https://apps.inside.anl.gov/argoapi` | — | OpenAI 响应格式归一化 |

## 推理配置

从 v0.6.8 起，shim 可以通过 `provider.yaml` 中的 `reasoning` 段声明推理能力配置。运行时由 `apply_reasoning_config()` 读取该配置，替代了之前硬编码在各转换器中的 effort 降级逻辑。

### ReasoningCapability 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `disabled` | `"omit"` \| `"thinking_disabled"` \| `"thinking_budget_zero"` | `mode: disabled` 时的序列化策略：`omit` 不发送任何参数，`thinking_disabled` 发送 `{"thinking": {"type": "disabled"}}`，`thinking_budget_zero` 发送 `{"thinking_config": {"thinking_budget": 0}}` |
| `effort_field` | `"reasoning_effort"` \| `"reasoning.effort"` \| `"output_config.effort"` \| `"thinking_level"` \| `"none"` | effort 值在提供商请求中的序列化位置；`"none"` 表示提供商不支持 effort |
| `max_effort` | `EffortLevel` \| `null` | 最高允许的归一化 effort 级别；超过此值的 effort 会被截断到该级别 |
| `thinking_type` | `"enabled"` \| `"adaptive"` \| `null` | 强制 `thinking.type` 为该值；`null` 表示不覆盖 |
| `effort_map` | `dict[str, str]` | 从归一化 IR effort 到提供商特定 effort 字符串的映射 |

### 配置示例

=== "Anthropic"

    ```yaml
    # provider.yaml
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

=== "OpenAI"

    ```yaml
    # provider.yaml
    reasoning:
      disabled: omit
      effort_field: reasoning_effort
      max_effort: high
      effort_map:
        minimal: minimal
        low: low
        medium: medium
        high: high
        xhigh: high
        max: high
    ```

### 按模型覆盖（`model_overrides`）

当同一提供商下不同模型有不同推理能力时，使用 `model_overrides` 声明按模型的配置。每个覆盖项继承提供商级默认值，按**上游模型 ID**（别名解析后）为键：

```yaml
name: argo_anthropic
base: anthropic
reasoning:
  thinking_type: enabled       # 大多数模型的默认值
  effort_map: { ... }
  model_overrides:
    claudeopus47:
      thinking_type: adaptive  # Vertex AI 要求 adaptive
```

网关在别名解析后确定上游模型 ID（如 `argo:claude-opus-4.7` → `claudeopus47`），并应用匹配的覆盖配置。

### 运行机制

当网关代理处理请求时：

1. `_inject_shim_reasoning()` 从目标提供商的 shim 中提取 `ReasoningCapability`，注入转换上下文的 `ctx.options["reasoning_cap"]`。如果上游模型有 `model_overrides` 条目，则使用该覆盖配置
2. 各转换器的 `request_to_provider` 将 `reasoning_cap` 传给 `ir_reasoning_config_to_p`
3. `ir_reasoning_config_to_p` 委托给 `apply_reasoning_config()`，按 shim 配置执行：
    - 输入归一化（`none` → disabled）
    - disabled 序列化（按 `cap.disabled` 策略）
    - effort 截断（按 `cap.max_effort`）
    - effort 映射（按 `cap.effort_map`）
    - 结构化传透（mode、budget_tokens 等提供商特定字段）

没有声明 `reasoning` 段的 shim 使用基础转换器类型的内置默认配置。

## Argo Shim

`argo_openai_chat` 和 `argo_anthropic` 面向 **Argo 网关** —— 这是某些机构（如 Argonne 国家实验室）使用的代理层，将多个上游 LLM 提供商统一暴露在单一端点之后。

这两个 shim 有一个共同特点：模型标识符通过请求体中的 `internal_id` 字段传递，而非标准的 `model` 字段。这一行为由 shim 声明中的 `model_id_field` 透明处理，无需手动干预。

### `argo_openai_chat`

一个直接的 OpenAI 兼容 shim，唯一的非标准行为是将模型字段替换为 `internal_id`，不需要其他 transform。

### `argo_anthropic`

该 shim 额外附带两个 transform，用于处理 Argo 的特殊行为：

- **模型级 `thinking_type` 覆盖**：Argo 的 Vertex AI 后端模型（如 `claudeopus47`）要求 `thinking.type = "adaptive"`，而旧模型要求 `"enabled"`。此差异通过推理配置中的 `model_overrides` 声明式处理，无需 transform。网关在别名解析后自动应用正确的 `thinking_type`。

- **`from_transforms` —— OpenAI 响应格式归一化**：Argo 可能从其 `/v1/messages` 端点返回 OpenAI Chat Completions 格式的响应体。该 transform 会检测这种情况，并在 `anthropic` 转换器处理之前将响应转换为 Anthropic Messages 格式，从而保证后续管线正常运行。

### 配置

`default_base_url` 因机构而异，建议在网关配置中覆盖：

```jsonc
{
  "providers": {
    "argo": {
      "shim": "argo_anthropic",
      "base_url": "https://your-argo-instance.example.com/",
      "api_key": "${ARGO_API_KEY}"
    }
  }
}
```

!!! note
    如果未设置 `base_url`，shim 会回退到 `https://apps.inside.anl.gov/argoapi/`，该地址仅在 ANL 内网可达。

## 转换规则（Transforms）

转换规则是纯 `dict → dict` 函数，用于弥合提供商实际 API 方言与对应基础转换器所期望的"标准"格式之间的差异。它们处理字段级差异（剥离不支持的字段、重命名参数、注入默认值）—— **不**处理语义级 API 标准转换，那是转换器的职责。

### 内置转换原语

| 原语 | 描述 | 示例 |
|------|------|------|
| `strip_fields(*keys)` | 从请求体中移除不支持的字段 | `strip_fields("logprobs", "top_logprobs")` |
| `rename_field(old, new)` | 重命名顶层字段 | `rename_field("max_tokens", "max_length")` |
| `set_defaults(**kv)` | 仅在字段不存在时设置（幂等） | `set_defaults(temperature=0.7)` |

### 应用方式

转换规则在两个层面应用：

**1. `convert()` 公共 API** —— 通过 `resolve_transforms()` 自动应用：

```python
from llm_rosetta import convert

# 当 source/target 是 shim 名称时，转换规则自动应用
result = convert(request_body, source="openai_chat", target="volcengine")
# → logprobs 和 top_logprobs 从输出中剥离
```

**2. 网关代理管线** —— 围绕转换器应用：

```text
请求:  客户端请求体 → source.from_provider() → IR → target.to_provider()
       → [to_transforms] → 上游 API

响应:  上游响应 → [from_transforms] → target.response_from_provider()
       → IR → source.response_to_provider() → 客户端

流式:  chunk → [from_transforms] → target.stream_from_provider()
       → IR → source.stream_to_provider() → 客户端
```

### 设计原则

- **幂等**：重复应用同一转换规则无副作用
- **不重叠**：按约定，不同转换规则应操作不同字段
- **可组合**：多个转换规则通过 `apply_transforms()` 顺序应用

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

## 注册自定义 Shim

### 编程式注册

为任何 OpenAI 兼容服务注册自定义提供商 shim：

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

注册后，shim 名称可在所有地方使用 —— `get_converter_for_provider()`、`resolve_base()`、`convert()` 和网关配置。

### 添加 YAML 提供商

要向内置注册表添加新提供商：

1. 在 `src/llm_rosetta/shims/providers/<name>/` 下创建目录
2. 添加 `provider.yaml`，包含必填字段：

    ```yaml
    name: my-provider
    base: openai_chat
    default_base_url: https://api.my-provider.com/v1
    default_api_key_env: MY_PROVIDER_API_KEY
    logo: https://example.com/logo.svg
    ```

3. 如果提供商有字段级差异，可选添加 `transforms.py`：

    ```python
    from llm_rosetta.shims.transforms import strip_fields

    to_transforms = (strip_fields("unsupported_field"),)
    from_transforms = ()
    ```

提供商在导入时自动发现并注册。

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

当找到 shim 时：

- `default_base_url` 和 `default_api_key_env` 在配置未明确指定时作为后备值使用
- `to_transforms` 应用于发送给上游提供商的请求
- `from_transforms` 应用于接收到的响应/流式 chunk，在转换之前执行

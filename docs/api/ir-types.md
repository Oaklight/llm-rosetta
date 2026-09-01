---
title: IR 推理类型
---

# IR 推理类型

IR 推理词汇表定义了 LLM-Rosetta 内部使用的固定值集合。Provider 侧的值是自由字符串，在各 provider 的 shim 配置中定义。

这些类型位于 [`llm_rosetta.types.ir.reasoning`][llm_rosetta.types.ir.reasoning]。

## IRMode

控制模型是否执行显式推理。

| 值 | 描述 |
|----|------|
| `"auto"` | 模型自行决定何时以及思考多少 |
| `"enabled"` | 显式思考，部分 provider 需要 `budget_tokens` |
| `"disabled"` | 不思考 |

Provider 映射通过 [`thinking_modes`][llm_rosetta.shims.provider_shim.ReasoningCapability] 按 shim 配置。

## IREffort

规范化的 effort 阶梯，从低到高排列：

```
minimal < low < medium < high < xhigh < max
```

每个 shim 通过 `effort_range` 声明支持的范围。超出范围的值会被夹紧到最近的边界。

## IRVisibility

控制思考输出如何返回给调用者。

| 值 | 描述 |
|----|------|
| `"auto"` | Provider 决定可见性 |
| `"concise"` | 推理的简要摘要 |
| `"detailed"` | 完整的推理输出 |
| `"none"` | 从响应中省略推理 |

Provider 映射通过 `visibility_modes` 按 shim 配置。

## ReasoningConfig

IR 推理配置 TypedDict，用于 `ir_request["reasoning"]`。

::: llm_rosetta.types.ir.reasoning.ReasoningConfig
    options:
      show_source: false
      heading_level: 3

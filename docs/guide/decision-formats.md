---
title: Decision API 格式
---

# Decision API 格式

Decision 模型对 state（上下文）执行类型化的 questions，返回结构化的概率分布答案——不涉及文本生成。这是与 chat completions、embedding、rerank 并列的独立模型范式。

LLM-Rosetta 目前支持 **2 个格式族**，预计随着范式成熟会有更多 provider 加入。

## 概览

| 格式族 | Provider | Endpoint | Converter 类 |
|-------|----------|----------|--------------|
| TypeSafe System One | TypeSafe AI (Jev) | `POST /v1/systemone` | `TypeSafeDecisionConverter` |
| OpenAI Decisions | OpenAI | `POST /v1/decisions` | `OpenAIDecisionsConverter` |

## 核心概念

### 三种问题原语

Decision API 使用类型化的 questions，每种对应固定的答案空间：

| IR 类型 | TypeSafe Wire 名 | 命题类型 | 输入 | 输出 |
|---------|------------------|----------|------|------|
| `assertion` | `noul` | 命题 | 是/否断言 + 可选 criteria | P(true) ∈ [0, 1] |
| `choice` | `choice` | 分类命题 | 选项及描述 | 选中选项 + 概率分布 |
| `score` | `score` | 有序命题 | 有序等级（≥ 2 级） | 加权分数 + 概率分布 |

!!! note "命名"
    三种 IR 原语按所评估的命题类型命名：`assertion`（**命题**）、`choice`（**分类命题**）、`score`（**有序命题**）。IR 名称 `assertion` 由 `TypeSafeDecisionConverter` 转换为 TypeSafe wire 名称 `noul`。

### Entry

每个问题都是一组 **entry** 的列表，答案是在它们之上的概率分布：

| 类型 | `criteria` | 顺序 | 答案 |
|------|-----------|------|------|
| `assertion` | 0 或 2 个 entry，label 为 `False` / `True` | 无序 | 标量 `probability` = P(True) |
| `choice` | N 个 entry | 无序 | `choice` + 分布 |
| `score` | N 个 entry | **有序**（序号 = 位置） | `score` + 分布 |

`DecisionEntry` 为 `{label: str | bool, description?: Description}`：`label` 是 entry 的标识（也是答案 `probabilities` 的 key）；`description` 是可选的丰富判据（对应 TypeSafe 的 `string | object | array`）。

`choice` / `score` / `confidence` 均为可选（provider 未返回时由 `converters.decision.derived` 推导）；`abstained` 始终由 `unknown_probability` 推导。

### State

评估的输入上下文。可以是字符串、JSON 对象或数组：

```python
# 字符串 state
state = "帮帮我！我的付款已经失败三天了。"

# 结构化 state
state = {
    "message": "我要退款",
    "order_id": "ORD-12345",
    "history": ["之前的消息1", "之前的消息2"]
}
```

### 多模态（`images[]`）

System One 家族（Cloudflare Clef、classifier.dev）在请求上扩展了一个顶层
`images` 数组，元素为内联 base64 data URL。`TypeSafeDecisionConverter` 把它
映射到 IR 的 `state` content parts —— 文本 part 拼进 `state`，图片 part 变成
`images`（入方向反向）：

```python
state = [
    {"type": "text", "text": "检查这张照片中的商品。"},
    {"type": "image", "image_data": {"media_type": "image/png", "data": "..."}},
]
```

嵌在**结构化** `state` dict 里的图片不会携带（没有位置 key 可回填）——请把图片
放在顶层 part 列表里。

## IR 类型

```python
from llm_rosetta.types.ir.decision import (
    # 共享 entry 与值别名
    DecisionEntry,       # {label, description?} —— 一个选项 / 层级 / 断言侧
    Description,         # str | dict | list（丰富文本槽）
    DecisionInputPart,   # TextPart | ImagePart（多模态证据）

    # 问题类型 —— 三者都携带 `criteria: list[DecisionEntry]`
    AssertionQuestion,   # 命题 → P(true)
    ChoiceQuestion,      # 分类命题 → 类别分布
    ScoreQuestion,       # 有序命题 → 有序分布

    # 答案类型
    AssertionAnswer,     # {type, probability}
    ChoiceAnswer,        # {type, choice, probabilities, confidence?}
    ScoreAnswer,         # {type, score, probabilities, confidence?}
    RefusalAnswer,       # {type, refusal, reason?}

    # 请求/响应
    IRDecisionRequest,
    IRDecisionResponse,
    DecisionUsageInfo,
)
```

## 请求格式

### TypeSafe System One

```json
{
  "model": "jev-latest",
  "state": "帮帮我！我的付款已经失败三天了。",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "是否表达了紧急性？"
    },
    "department": {
      "type": "choice",
      "instructions": "哪个团队应该处理？",
      "criteria": {
        "billing": "付款、发票、退款",
        "technical": "Bug、故障、集成"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "客户有多沮丧？",
      "criteria": ["平静", "沮丧", "非常愤怒"]
    }
  }
}
```

### IR 等价形式

```python
request: IRDecisionRequest = {
    "model": "jev-latest",
    "state": "帮帮我！我的付款已经失败三天了。",
    "questions": {
        "is_urgent": AssertionQuestion(
            type="assertion",
            instructions="是否表达了紧急性？",
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="哪个团队应该处理？",
            criteria=[
                {"label": "billing", "description": "付款、发票、退款"},
                {"label": "technical", "description": "Bug、故障、集成"},
            ],
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="客户有多沮丧？",
            criteria=[
                {"label": "平静"},
                {"label": "沮丧"},
                {"label": "非常愤怒"},
            ],
        ),
    },
}
```

### OpenAI Decisions

OpenAI Decisions API 接收 `input`（字符串或 user messages）与 `questions`
**数组**；每个问题带 `name`，类型为 `predicate` / `choice` / `score`。

```json
{
  "model": "gpt-6-luna",
  "input": "帮帮我！我的付款已经失败三天了。",
  "questions": [
    {"type": "predicate", "name": "is_urgent",
     "instructions": "是否表达了紧急性？"},
    {"type": "choice", "name": "department",
     "instructions": "哪个团队应该处理？",
     "choices": [{"value": "billing", "description": "付款、发票、退款"},
                 {"value": "technical", "description": "Bug、故障、集成"}]},
    {"type": "score", "name": "frustration",
     "instructions": "客户有多沮丧？",
     "levels": [{"label": "平静"}, {"label": "沮丧"}, {"label": "非常愤怒"}]}
  ]
}
```

其 IR 请求与上方 TypeSafe 的 IR 完全一致；`OpenAIDecisionsConverter` 把
`questions` 映射为数组（按 `name` 作 key），并把断言的 entries 折进
`instructions` 里一个带标记、可逆的 JSON 信封。

## 响应格式

### TypeSafe System One

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.92
    },
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
      "confidence": 0.82
    },
    "frustration": {
      "type": "score",
      "score": 1.6,
      "legend": {"0": "平静", "1": "沮丧", "2": "非常愤怒"},
      "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
      "confidence": 0.78
    }
  },
  "usage": {"input_tokens": 588, "output_tokens": 212}
}
```

### IR 等价形式

Converter 添加 `object: "decision"`，将 `assertion` 原语与 wire 名称 `noul` 互相转换，并把 score 的概率从序号位置重映射到层级 label：

```python
response: IRDecisionResponse = {
    "object": "decision",
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": AssertionAnswer(type="assertion", probability=0.92),
        "department": ChoiceAnswer(
            type="choice", choice="technical",
            probabilities={"billing": 0.08, "technical": 0.85, "sales": 0.07},
            confidence=0.82,
        ),
        "frustration": ScoreAnswer(
            type="score", score=1.6,
            probabilities={"平静": 0.05, "沮丧": 0.3, "非常愤怒": 0.65},
            confidence=0.78,
        ),
    },
    "usage": {"input_tokens": 588, "output_tokens": 212},
}
```

### OpenAI Decisions

```json
{
  "model": "gpt-6-luna",
  "answers": [
    {"type": "predicate", "name": "is_urgent", "probability": 0.92},
    {"type": "choice", "name": "department", "choice": "billing",
     "probabilities": [{"value": "billing", "probability": 0.95},
                       {"value": "technical", "probability": 0.02}],
     "confidence": 0.93},
    {"type": "score", "name": "frustration", "score": 1.1,
     "probabilities": [{"value": 0, "label": "平静", "probability": 0.1},
                       {"value": 1, "label": "沮丧", "probability": 0.6},
                       {"value": 2, "label": "非常愤怒", "probability": 0.3}],
     "confidence": 0.55}
  ]
}
```

`answers` 为数组、按 `name` 作 key；也可能是 `{"type": "refusal", "name": ...}`。
`OpenAIDecisionsConverter` 把 `probabilities` 数组重映射回 IR 的分布字典。

## 使用 Converter

```python
from llm_rosetta.converters.decision import TypeSafeDecisionConverter

converter = TypeSafeDecisionConverter()

# TypeSafe wire → IR
ir_request = converter.request_from_provider(typesafe_request)
ir_response = converter.response_from_provider(typesafe_response)

# IR → TypeSafe wire
wire_request, warnings = converter.request_to_provider(ir_request)
wire_response = converter.response_to_provider(ir_response)
```

## 网关路由

网关注册了三个 decision 路由：

- `POST /v1/decision` — 标准路由
- `POST /v1/decisions` — OpenAI 兼容别名
- `POST /v1/systemone` — TypeSafe 兼容别名

!!! warning "Phase 1 限制"
    网关的 decision provider 路由（`GatewayConfig` 中的 `resolve_decision` / `decision_models`）尚未实现。两个路由目前返回 501。

## 与 Chat 结构化输出的对比

| 维度 | Decision 模型 (Jev) | LLM + 结构化输出 |
|------|---------------------|-----------------|
| 输出 | 校准的概率分布 | 受 schema 约束的生成 token |
| 延迟 | 70–500ms | 秒级 |
| 输出成本 | $0 | 按 token 计费 |
| 流式 | 不支持 | 支持 |
| 置信度 | 原生支持（从概率推导） | 不可用 |
| 灵活性 | 固定原语（assertion/choice/score） | 任意 JSON schema |

## 相关链接

- [TypeSafe API 文档](https://docs.typesafe.ai/api)
- [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) — LLM-backed decision 的参考实现

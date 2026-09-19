---
title: Decision API 格式
---

# Decision API 格式

Decision 模型对 state（上下文）执行类型化的 questions，返回结构化的概率分布答案——不涉及文本生成。这是与 chat completions、embedding、rerank 并列的独立模型范式。

LLM-Rosetta 目前支持 **1 个格式族**，预计随着范式成熟会有更多 provider 加入。

## 概览

| 格式族 | Provider | Endpoint | Converter 类 |
|-------|----------|----------|--------------|
| TypeSafe System One | TypeSafe AI (Jev) | `POST /v1/systemone` | `TypeSafeDecisionConverter` |

## 核心概念

### 三种问题原语

Decision API 使用类型化的 questions，每种对应固定的答案空间：

| IR 类型 | 输入 | 输出 |
|---------|------|------|
| `noul` | 是/否命题 + 可选 criteria | P(true) ∈ [0, 1] |
| `choice` | 选项及描述 | 选中选项 + 概率分布 |
| `score` | 有序等级（≥ 2 级） | 加权分数 + 概率分布 |

!!! note "Noul 词源"
    "Noul" 取自 "ber-**noul**-li" 的中间四个字母——bool 的概率化版本。IR 使用与 TypeSafe wire format 相同的名称。

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

## IR 类型

```python
from llm_rosetta.types.ir.decision import (
    # 问题类型
    NoulQuestion,        # 是/否 → P(true)
    ChoiceQuestion,      # 选一个 → 类别分布
    ScoreQuestion,       # 打分 → 有序分布

    # 答案类型
    NoulAnswer,          # {type, noul}
    ChoiceAnswer,        # {type, choice, probabilities, confidence}
    ScoreAnswer,         # {type, score, legend, probabilities, confidence}

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
        "is_urgent": NoulQuestion(
            type="noul",
            instructions="是否表达了紧急性？",
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="哪个团队应该处理？",
            criteria={"billing": "付款、发票、退款",
                      "technical": "Bug、故障、集成"},
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="客户有多沮丧？",
            criteria=["平静", "沮丧", "非常愤怒"],
        ),
    },
}
```

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

Converter 添加 `object: "decision"` 并直接透传 questions/answers：

```python
response: IRDecisionResponse = {
    "object": "decision",
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": NoulAnswer(type="noul", noul=0.92),
        "department": ChoiceAnswer(
            type="choice", choice="technical",
            probabilities={"billing": 0.08, "technical": 0.85, "sales": 0.07},
            confidence=0.82,
        ),
        # ...
    },
    "usage": {"input_tokens": 588, "output_tokens": 212},
}
```

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

网关注册了两个 decision 路由：

- `POST /v1/decision` — 标准路由
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
| 灵活性 | 固定原语（noul/choice/score） | 任意 JSON schema |

## 相关链接

- [TypeSafe API 文档](https://docs.typesafe.ai/api)
- [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) — LLM-backed decision 的参考实现

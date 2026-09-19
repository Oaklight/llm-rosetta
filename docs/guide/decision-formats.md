---
title: Decision API Formats
---

# Decision API Formats

Decision models evaluate state against typed questions and return structured probabilistic answers — no text generation involved. This is a distinct model paradigm alongside chat completions, embedding, and rerank.

LLM-Rosetta currently supports **1 format family** for decision APIs, with more expected as the paradigm matures.

## Overview

| Format Family | Provider | Endpoint | Converter Class |
|--------------|----------|----------|-----------------|
| TypeSafe System One | TypeSafe AI (Jev) | `POST /v1/systemone` | `TypeSafeDecisionConverter` |

## Key Concepts

### Three Question Primitives

Decision APIs use typed questions with constrained answer spaces:

| IR Type | TypeSafe Wire Name | Input | Output |
|---------|-------------------|-------|--------|
| `noul` | `noul` | Yes/no proposition + optional criteria | P(true) ∈ [0, 1] |
| `choice` | `choice` | Options with descriptions | Selected option + probability distribution |
| `score` | `score` | Ordered rubric levels (≥ 2) | Weighted score + probability distribution |

!!! note "Noul etymology"
    "Noul" comes from the middle of "ber-**noul**-li" — a probabilistic counterpart to bool. The IR uses the same name as the TypeSafe wire format.

### State

The input context to evaluate. Can be a string, JSON object, or array:

```python
# String state
state = "Help! My payouts have been failing for 3 days."

# Structured state
state = {
    "message": "I need a refund",
    "order_id": "ORD-12345",
    "history": ["previous message 1", "previous message 2"]
}
```

## IR Types

```python
from llm_rosetta.types.ir.decision import (
    # Questions
    NoulQuestion,   # yes/no → P(true)
    ChoiceQuestion,      # pick one → categorical distribution
    ScoreQuestion,       # rate on scale → ordinal distribution

    # Answers
    NoulAnswer,     # {type, value}
    ChoiceAnswer,        # {type, choice, probabilities, confidence}
    ScoreAnswer,         # {type, score, legend, probabilities, confidence}

    # Request/Response
    IRDecisionRequest,
    IRDecisionResponse,
    DecisionUsageInfo,
)
```

## Request Format

### TypeSafe System One

```json
{
  "model": "jev-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?"
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

### IR Equivalent

```python
request: IRDecisionRequest = {
    "model": "jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": NoulQuestion(
            type="noul",
            instructions="Does this convey urgency?",
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team should handle this?",
            criteria={"billing": "Payments, invoicing, refunds",
                      "technical": "Bugs, outages, integrations"},
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
}
```

## Response Format

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
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
      "confidence": 0.78
    }
  },
  "usage": {"input_tokens": 588, "output_tokens": 212}
}
```

### IR Equivalent

The converter adds `object: "decision"` and passes questions/answers through:

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

## Using the Converter

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

## Gateway

The gateway registers two routes for decision requests:

- `POST /v1/decision` — canonical route
- `POST /v1/systemone` — TypeSafe-compatible alias

!!! warning "Phase 1 limitation"
    Gateway decision routing (`resolve_decision` / `decision_models` in config) is not yet implemented. Both routes currently return 501. This will be added when provider routing for the decision paradigm is wired into `GatewayConfig`.

## Comparison with Chat Structured Output

Decision models and LLM structured output both return typed data, but they differ fundamentally:

| Aspect | Decision Model (Jev) | LLM + Structured Output |
|--------|---------------------|------------------------|
| Output | Calibrated probability distributions | Generated tokens constrained to a schema |
| Latency | 70–500ms | Seconds |
| Output cost | $0 | Per token |
| Streaming | No | Yes |
| Confidence | Native (derived from probabilities) | Not available |
| Flexibility | Fixed primitives (noul/choice/score) | Arbitrary JSON schema |

## Related

- [TypeSafe API docs](https://docs.typesafe.ai/api)
- [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) — reference for LLM-backed decision evaluation

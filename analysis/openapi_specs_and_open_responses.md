# OpenAPI Specs & Open Responses Research

Date: 2026-03-20

## 1. OpenAPI Spec Availability by Provider

### OpenAI (Chat Completions + Responses API)

- **Official**: Yes, complete
- **Format**: OpenAPI 3.0.0 YAML (~990KB)
- **Repo**: https://github.com/openai/openai-openapi (branch: `manual_spec`)
- **Raw URL**: `https://raw.githubusercontent.com/openai/openai-openapi/manual_spec/openapi.yaml`
- **License**: MIT
- **Version**: 2.3.0
- **Notes**: Single spec covers ALL OpenAI APIs (Chat Completions, Responses, Assistants, Audio, Images, etc.). Both `POST /v1/chat/completions` and `POST /v1/responses` are defined here. The `master` branch 404s for `openapi.yaml` — use `manual_spec` branch.

### Anthropic (Messages API)

- **Official**: Semi-official (Stainless-generated, used to build their SDKs)
- **Format**: OpenAPI 3.1.0 YAML
- **URL**: `https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic%2Fanthropic-dd2dcd00a757075370a7e4a7f469a1e2d067c2118684c3b70d7906a8f5cf518b.yml`
- **How to find current URL**: Check `openapi_spec_url` in https://github.com/anthropics/anthropic-sdk-python/blob/main/.stats.yml (hash rotates with each API update)
- **License**: Not explicitly published; extracted from SDK build pipeline
- **Notes**:
  - Anthropic does NOT officially publish an OpenAPI spec (see https://github.com/anthropics/anthropic-sdk-typescript/issues/252)
  - The Stainless-hosted spec is the authoritative source used to generate their Python/TypeScript/Java/Go SDKs
  - Community alternative: https://github.com/laszukdawid/anthropic-openapi-spec (downgraded to OpenAPI 3.0 for wider tooling compatibility)

### Google GenAI / Gemini API

- **Official OpenAPI**: NO — Google uses their own **Discovery Document** format
- **Discovery URL**: `https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta`
- **Format**: Google REST Discovery JSON (proprietary schema, not OpenAPI)
- **Protobuf source**: https://github.com/googleapis/googleapis/tree/master/google/ai/generativelanguage/v1beta
- **Notes**:
  - Discovery Document is large JSON (~hundreds of KB) describing all endpoints, schemas, and methods
  - Can be converted to OpenAPI using tools like `google/gnostic` but conversion is lossy
  - The `v1beta` version is the recommended one (covers latest Gemini models)
  - `v1` is also available: `https://generativelanguage.googleapis.com/$discovery/rest?version=v1`
  - No community-maintained OpenAPI spec found for Gemini specifically

### Open Responses

- **Official**: Yes, complete
- **Format**: OpenAPI 3.1.0 JSON (~93KB)
- **Repo**: https://github.com/openresponses/openresponses
- **Path in repo**: `public/openapi/openapi.json`
- **Version**: 2.3.0 (titled "OpenAI API" — inherits from OpenAI's spec)
- **Notes**: Used for TypeScript code generation via `kubb.config.ts` in the repo

---

## 2. Open Responses — Full Analysis

### What Is It?

Open Responses is an **open-source specification** (not just an implementation) initiated by **OpenAI** in January 2026. It turns the proprietary OpenAI Responses API into a **vendor-neutral standard** with formal extensibility rules.

- **Spec repo**: https://github.com/openresponses/openresponses (~951 stars, Apache 2.0)
- **Website**: https://www.openresponses.org/
- **Governance**: Technical Steering Committee (TSC), RFC 2119/8174 language
- **Separate implementation** (different project): https://github.com/open-responses/open-responses by Julep AI (~209 stars) — a self-hosted Go server

### Relationship to OpenAI Responses API

Open Responses is a **proper superset** of OpenAI's Responses API. From the spec:

> "Any implementer can be considered to have an Open Responses-compliant API if they have an API that implements this spec directly or is a proper superset of Open Responses."

The core request/response shape, item model, streaming event types, and tool invocation patterns are **structurally identical**. A client already talking to OpenAI's Responses API can talk to an Open Responses endpoint with **minimal changes** (primarily adding an `OpenResponses-Version` header).

### Schema Differences from OpenAI Responses

| Aspect | OpenAI Responses API | Open Responses Spec |
|---|---|---|
| **Reasoning `content` field** | Only `summary` + `encrypted_content` | Adds raw `content` field for open-weight model reasoning traces |
| **Provider-specific types** | Built-in types like `web_search_call`, `file_search_call` are first-class | Moved to namespaced extensions: `openai:web_search_call`. Core spec only defines generic items |
| **Extensibility** | No formal extension mechanism | Formal slug-prefixed extension system for items, tools, and streaming events |
| **`allowed_tools`** | Not in OpenAI spec | New field for cache-preserving tool restriction |
| **Provider/Router routing** | N/A (single vendor) | Formal provider specification and routing semantics |
| **Hosted tools** | OpenAI-specific (file_search, web_search, code_interpreter, etc.) | Generic pattern: any implementor can expose hosted tools via `implementor_slug:tool_name` |
| **Versioning header** | None | `OpenResponses-Version` header |
| **Streaming events** | ~40+ event types including provider-specific ones | Reduced to ~23 core semantic event types; provider-specific events use slug prefixes |
| **`logit_bias`** | Supported | **Not included** in the spec |
| **Model enumeration** | Hardcoded model IDs in schema | Model field is an open string |
| **`store` parameter** | Default `true` (server-side storage) | Stateless by default |
| **Input/output content asymmetry** | Implicit | Explicitly formalized as separate `UserContent` and `ModelContent` discriminated unions |
| **Error types** | Similar | Formally specified with `server_error`, `invalid_request`, `not_found`, `model_error`, `too_many_requests` |

### Key Additions Over OpenAI Responses

1. **Expanded Reasoning Visibility**: raw `content` field for reasoning traces from open-weight models (in addition to `summary` and `encrypted_content`)
2. **Extensibility via Implementor Slugs**: all provider-specific items, tools, and streaming events MUST be prefixed with a canonical slug (e.g., `openai:web_search_call`, `acme:telemetry_chunk`) — prevents type-name collisions
3. **`allowed_tools` Field**: cache-preserving control surface — keep full `tools` list intact but restrict which tools the model may invoke
4. **Provider/Router Distinction**: formal separation of "Model Providers" (inference) from "Routers" (intermediaries like OpenRouter)
5. **Compliance Testing Suite**: CLI + web UI (`bin/compliance-test.ts` and `/compliance` on website) for providers to validate spec compliance
6. **`OpenResponses-Version` Header**: versioning mechanism for the spec itself

### Adoption Status (as of March 2026)

**Launch partners (January 2026):**
- OpenRouter — standardizing on Open Responses (broad model coverage since it proxies to nearly every provider)
- Hugging Face — blog post, demo application, Inference Providers integration
- Vercel — supporting the spec
- LM Studio — blog post about Open Responses with local models
- Ollama — early adopter for local inference
- vLLM — early adopter for production serving

**Notable absences:**
- Anthropic — not a launch partner
- Google DeepMind — not a launch partner

---

## 3. Recommendation for llm-rosetta

### OpenAPI Spec Strategy

For type/schema validation and code generation:
- **OpenAI**: Download from `manual_spec` branch — covers both Chat and Responses
- **Anthropic**: Fetch dynamically from Stainless URL in `.stats.yml`, or pin a known-good hash
- **Google**: Use Discovery Document directly or convert with `gnostic`; alternatively use protobuf definitions from `googleapis/googleapis`
- **Open Responses**: Use `public/openapi/openapi.json` from the spec repo

### Open Responses Support Strategy

Rather than building a completely separate Open Responses converter, **extend the existing `openai_responses` converter** with optional Open Responses support. The delta is small:

1. **Reasoning `content` field**: add support in IR's `ReasoningPart` for raw content (not just summary/encrypted)
2. **Slug-prefixed extension items**: passthrough or strip `implementor:type_name` items gracefully
3. **`allowed_tools` field**: map to/from IR (new field on `IRRequest.generation_config` or similar)
4. **`OpenResponses-Version` header**: handle in gateway header management
5. **Stateless default**: already compatible (llm-rosetta doesn't assume server-side state)

This could be exposed as:
- A flag: `output_format="open_responses"` on the converter
- Or a thin subclass: `OpenResponsesConverter(OpenAIResponsesConverter)`
- Gateway: detect via `OpenResponses-Version` header and route accordingly

The core request shape (`POST /v1/responses` with `model`, `input`, `instructions`, `tools`, etc.) is identical — ~95% of the existing OpenAI Responses converter code applies directly.

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

---

## 4. Ollama API

### Overview

Ollama is a local LLM runner (Go-based). It exposes **three API surfaces**:

1. **Native Ollama API** — custom endpoints under `/api/`
2. **OpenAI-compatible API** — at `/v1/` prefix (Chat Completions, Completions, Embeddings, Models)
3. **Anthropic-compatible API** — at `/v1/messages` (since v0.14.0)

### OpenAPI Spec

- **Official**: Yes, OpenAPI 3.1.0 at `docs/openapi.yaml` in the repo
- **Repo**: https://github.com/ollama/ollama
- **Raw URL**: `https://raw.githubusercontent.com/ollama/ollama/main/docs/openapi.yaml`
- **Coverage**: Native Ollama API endpoints (`/api/*`); the OpenAI-compatible `/v1/*` endpoints follow the OpenAI spec
- **Notes**: Community issue requesting kept-up-to-date spec: https://github.com/ollama/ollama/issues/3383

### Native API Endpoints (`/api/`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/generate` | POST | Generate text (non-chat, raw completion) |
| `/api/chat` | POST | Chat completion (Ollama-native format) |
| `/api/embed` | POST | Generate embeddings |
| `/api/tags` | GET | List local models |
| `/api/ps` | GET | List running/loaded models |
| `/api/show` | POST | Show model info |
| `/api/pull` | POST | Pull a model |
| `/api/push` | POST | Push a model |
| `/api/create` | POST | Create a model from Modelfile |
| `/api/copy` | POST | Copy a model |
| `/api/delete` | DELETE | Delete a model |
| `/api/blobs/:digest` | HEAD/POST | Check/create blobs |
| `/api/version` | GET | Get Ollama version |

### OpenAI-Compatible Endpoints (`/v1/`)

| Endpoint | Method | Since |
|---|---|---|
| `/v1/chat/completions` | POST | Early versions |
| `/v1/completions` | POST | Early versions |
| `/v1/models` | GET | Early versions |
| `/v1/models/:model` | GET | Early versions |
| `/v1/embeddings` | POST | Early versions |
| `/v1/responses` | POST | **v0.13.3** |
| `/v1/messages` | POST | **v0.14.0** (Anthropic format) |

### Responses API Support

**Yes, since Ollama v0.13.3.** Ollama implements the non-stateful flavor of the OpenAI Responses API at `/v1/responses`.

Supported: streaming, tools/function calling, reasoning summaries.
Not supported: `previous_response_id`, `conversation`, `truncation`.

Ollama is also listed as an **Open Responses early adopter** — the `/v1/responses` endpoint is compatible with the Open Responses standard.

### Key Differences from Standard OpenAI Chat API

- `api_key` required in SDK clients but ignored (no auth by default)
- No support for `logprobs`, `logit_bias`, `n > 1`, image URLs (base64 only)
- Uses Ollama model names (e.g., `llama3.2`, `qwen3:8b`)
- Native API durations in nanoseconds
- Defaults `stream: true` in native API
- Supports `think` parameter and `reasoning_effort` for reasoning models

### Relevance to llm-rosetta

Ollama's OpenAI-compatible endpoints work with llm-rosetta's `openai_chat` converter, and its `/v1/responses` works with the `openai_responses` converter. The `/v1/messages` endpoint works with the `anthropic` converter. **All three of llm-rosetta's major converters are already compatible with Ollama out of the box.** A dedicated Ollama native converter would only be needed for the `/api/` endpoints.

---

## 5. Hugging Face Text Generation Inference (TGI) API

### Overview

TGI is HF's open-source inference server (Rust + Python). It exposes:

1. **Native TGI API** — custom endpoints (`/generate`, `/generate_stream`, `/info`, `/health`)
2. **OpenAI-compatible Messages API** — at `/v1/chat/completions` (since TGI v1.4.0)

HF also operates **Inference Providers**, a cloud routing layer at `https://router.huggingface.co/v1` that supports additional API formats.

### OpenAPI Spec

- **Official**: Yes, at `docs/openapi.json` in the repo
- **Repo**: https://github.com/huggingface/text-generation-inference
- **Raw URL**: `https://raw.githubusercontent.com/huggingface/text-generation-inference/main/docs/openapi.json`
- **Published**: https://huggingface.github.io/text-generation-inference/openapi.json
- **Interactive docs**: Any running TGI instance serves Swagger UI at `/docs`
- **Notes**: Auto-generated from Rust code, covers both native and OpenAI-compatible endpoints

### TGI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | Generate (legacy) |
| `/generate` | POST | Generate text (non-streaming) |
| `/generate_stream` | POST | Stream text generation (SSE) |
| `/v1/chat/completions` | POST | OpenAI Chat Completions compatible |
| `/v1/completions` | POST | OpenAI Completions compatible |
| `/v1/models` | GET | List model(s) |
| `/chat_tokenize` | POST | Tokenize a chat request |
| `/tokenize` | POST | Tokenize raw input |
| `/health` | GET | Health check |
| `/info` | GET | Model and server info |
| `/metrics` | GET | Prometheus metrics |
| `/invocations` | POST | AWS SageMaker compatibility |

### HF Inference Providers Router

| Endpoint | Base URL |
|---|---|
| Chat Completions | `https://router.huggingface.co/v1/chat/completions` |
| Completions | `https://router.huggingface.co/v1/completions` |
| **Responses API** | `https://router.huggingface.co/v1/responses` |
| Models | `https://router.huggingface.co/v1/models` |

The router aggregates 15+ backend providers (Groq, Nebius, Together AI, etc.) with routing policies: `:fastest`, `:cheapest`, `:preferred`, or specific provider selection.

### Responses API / Open Responses Support

**Yes, both supported via HF Inference Providers.**

The Responses API is available (beta) at `https://router.huggingface.co/v1/responses`. Features:
- Plain text and multimodal inputs, multi-turn conversations
- Event-based streaming (`response.created`, `output_text.delta`, `response.completed`)
- Tool calling, structured outputs, remote MCP execution
- Reasoning effort controls (`low`, `medium`, `high`)

HF is a **primary backer** of the Open Responses specification. Blog: https://huggingface.co/blog/open-responses

Note: TGI itself does NOT implement Responses API — only the cloud Inference Providers router does.

### Key Differences from Standard OpenAI Chat API

- TGI serves a single model per instance; model param often ignored
- Additional params: `best_of`, `decoder_input_details`, `top_n_tokens`, `typical_p`, `watermark`, `repetition_penalty`, `grammar`
- No embeddings endpoint in TGI (separate Text Embeddings Inference server)
- HF auth via `hf_XXX` Bearer token
- Inference Providers uses HF Hub model IDs (e.g., `Qwen/Qwen2.5-VL-7B-Instruct`)

### Relevance to llm-rosetta

TGI's OpenAI-compatible endpoints work with llm-rosetta's `openai_chat` converter. The Inference Providers router's Responses API endpoint works with the `openai_responses` converter. Native TGI format (`/generate`) is simpler (just `inputs` string + `parameters` dict) and would only need a converter if targeting the native API directly.

---

## 6. Competing / Same-Name Projects

### mathisxy/llmir

- **URL**: https://github.com/mathisxy/llmir
- **What it does**: LLM Intermediate Representation — defines canonical Pydantic IR types (`AIMessage`, `AIChunkText`, `AIChunkFile`, `AITool`, etc.) with per-provider adapters (serializers)
- **Language**: Python (Pydantic v2, Rich)
- **Direction**: **Unidirectional** — IR → provider format only. No parser from provider format back to IR
- **Adapters**: OpenAI, Ollama (extensible via `BaseAdapter`)
- **Stars**: 0, **Forks**: 0
- **Created**: 2026-01-09, **Last commit**: 2026-03-18 (actively maintained)
- **CI/CD**: Yes (type checking, docs deployment, PyPI publishing as `llmir`)
- **Tests**: None visible
- **License**: MIT
- **Key difference**: Users must construct messages in the llmir IR type system, then serialize out. Unlike llm-rosetta which converts between native provider formats bidirectionally

### bitlab-tech/llm-rosetta

- **URL**: https://github.com/bitlab-tech/llm-rosetta
- **What it does**: Strategy-pattern based translation from OpenAI format to other providers
- **Language**: TypeScript (uses `@huggingface/transformers` for chat templates)
- **Direction**: **Unidirectional** — OpenAI → provider format only
- **Providers**: Anthropic/Bedrock, Hugging Face custom models, Lingshu (medical), Gemma (WIP)
- **Stars**: 0, **Forks**: 0
- **Created**: 2025-10-08, **Last commit**: 2025-10-20 (**dormant — 5 months inactive**)
- **Tests**: None (`"test": "echo \"Error: no test specified\" && exit 1"`)
- **Published on npm**: `llm-rosetta` (name collision in the npm ecosystem)
- **License**: MIT
- **Key difference**: TypeScript (different ecosystem entirely), unidirectional, appears abandoned

### Comparison Summary

| Dimension | Oaklight/llm-rosetta | mathisxy/llmir | bitlab-tech/llm-rosetta |
|---|---|---|---|
| **Language** | Python | Python | TypeScript |
| **Direction** | Bidirectional (any ↔ any via IR) | IR → provider only | OpenAI → provider only |
| **Providers** | 4 (OpenAI Chat, Responses, Anthropic, Google) | 2 (OpenAI, Ollama) | 4 (Bedrock, HF, Lingshu, Gemma) |
| **Streaming** | Full support | No | Partial |
| **Tests** | Comprehensive | None | None |
| **Gateway** | Yes | No | No |
| **Activity** | Very active | Active | Dormant |
| **Package** | `llm-rosetta` on PyPI | `llmir` on PyPI | `llm-rosetta` on **npm** |

Neither poses a competitive threat. Oaklight/llm-rosetta is the only project with bidirectional conversion, streaming, a gateway, and comprehensive tests.

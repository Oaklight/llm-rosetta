---
title: Gateway
---

# Gateway

The LLM-Rosetta Gateway is an HTTP proxy that translates between LLM provider API formats in real time. Send requests in any supported format — the gateway converts and forwards them to the configured upstream provider.

```mermaid
graph LR
    C1["Client<br/><small>OpenAI format</small>"] --> GW["Gateway"]
    C2["Client<br/><small>Anthropic format</small>"] --> GW
    C3["Client<br/><small>Google format</small>"] --> GW
    GW --> U1["Anthropic API"]
    GW --> U2["OpenAI API"]
    GW --> U3["Any provider"]

    style GW fill:#f9a825,stroke:#f57f17,color:#000
```

New here? Start with the [Gateway Quick Start](../getting-started/gateway-quickstart.md).

## Endpoints

| Path | Source Format | Description |
|------|-------------|-------------|
| `POST /v1/chat/completions` | OpenAI Chat | Drop-in for OpenAI SDK |
| `POST /v1/messages` | Anthropic | Drop-in for Anthropic SDK |
| `POST /v1/responses` | OpenAI Responses | Drop-in for OpenAI Responses SDK |
| `POST /v1/responses/compact` | OpenAI Responses | Context compaction, relayed unchanged; needs an `open_responses` upstream (`501` otherwise) |
| `POST /v1/messages/count_tokens` | Anthropic | Exact input-token count, delegated to the target's own endpoint; needs a `messages` upstream (`501` otherwise) |
| `POST /v1/embeddings` | OpenAI Embeddings | Cross-format conversion via IR (OpenAI, Cohere, Jina, Voyage) |
| `POST /v1/rerank` | Rerank (Jina default) | Cross-format conversion via IR (Jina, Cohere, Voyage) |
| `POST /v2/rerank` | Rerank (Cohere) | Auto-detected Cohere format; same handler as `/v1/rerank` |
| `POST /v1beta/models/{model}:generateContent` | Google GenAI | Drop-in for Google REST API |
| `POST /v1beta/models/{model}:streamGenerateContent` | Google GenAI (streaming) |
| `POST /v1beta/interactions` | Google Interactions | Drop-in for Google streaming |
| `GET /v1/models` | OpenAI / Anthropic | List configured models with `api_standard` and `capabilities` |
| `GET /v1beta/models` | Google GenAI | List configured models (Google SDK format) |
| `GET /health` | — | Health check |
| `GET /admin/` | — | [Admin panel](admin-panel.md) (web UI) |

The endpoint path determines the source format — no auto-detection needed.

## Streaming

Streaming is supported for all provider combinations. Request streaming the same way you would with the native API:

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

The gateway converts SSE chunks in real time between provider formats.

## Token counting

`POST /v1/messages/count_tokens` answers the Anthropic counting endpoint for
clients that count before sending a request (Claude Code is one). The gateway
does not estimate: it forwards the body — the `model` name rewritten to the
route's upstream model, nothing else touched — to the target provider's own
`/count_tokens` endpoint and relays the answer verbatim
(`x-rosetta-conversion: passthrough`).

The count is therefore exact whenever the target implements the Messages
counting endpoint, and identical to what that provider's own SDK reports. Two
limits follow from delegating rather than estimating:

- The body must be a valid counting request. Anthropic rejects generation-only
  fields there (`max_tokens`, `stream`), and the gateway relays that `400`
  unchanged instead of stripping the fields for you.
- A model routed to a provider that does not speak Messages cannot be counted
  this way, and the route answers `501` in the Messages error envelope. OpenAI
  Chat and OpenAI Responses targets have no counting endpoint at all; a Google
  target has `:countTokens`, but it consumes a Google-shaped request and
  reports `totalTokens`, so the gateway does not use it for a Messages-shaped
  count in this release.

The route is recorded in the request log and metrics like any other, with no
token usage — a count spends nothing.

## Authentication

API keys are managed exclusively through the **admin panel** (SQLite keystore).
Generate keys in the admin UI at `/admin`, then provide them in the format
native to each API standard (Bearer token, `x-api-key` header, etc.).

See [Gateway API Keys](configuration.md#gateway-api-keys) for details.

## How It Works

The gateway uses LLM-Rosetta's converter pipeline:

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Upstream

    Client->>Gateway: Request (source format)
    Gateway->>Gateway: source_converter.request_from_provider() → IR
    Gateway->>Gateway: Look up model → target provider
    Gateway->>Gateway: target_converter.request_to_provider() → target format
    Gateway->>Upstream: Forward request
    Upstream-->>Gateway: Response (target format)
    Gateway->>Gateway: target_converter.response_from_provider() → IR
    Gateway->>Gateway: source_converter.response_to_provider() → source format
    Gateway-->>Client: Response (source format)
```

For streaming, the same pipeline operates at the SSE chunk level using `stream_response_from_provider()` and `stream_response_to_provider()` with `StreamContext` for stateful conversion.

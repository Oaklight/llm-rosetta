---
title: Compliance Testing
---

# Compliance Testing with llm-comply

[llm-comply](https://github.com/Oaklight/llm-comply) is a companion tool that validates LLM API endpoints against official specs. It performs schema validation (via bundled OpenAPI specs) and semantic checks (lifecycle ordering, streaming events, field presence) for each supported format.

**[Try it online →](https://llm-comply.service.oaklight.top)**

## Supported Formats

| Format | Flag | Tests |
|--------|------|:-----:|
| OpenAI Chat Completions | `openai-chat` | 8 |
| Open Responses | `open-responses` | 12 |
| Anthropic Messages | `anthropic` | 8 |
| Google GenAI | `google-genai` | 9 |

## Installation

```bash
pip install llm-comply

# For colored terminal output:
pip install llm-comply[rich]
```

## CLI Usage

### Basic Usage

```bash
# Test with Open Responses format (default)
llm-comply -u https://your-gateway/v1 -k $API_KEY -m your-model

# Test a specific format
llm-comply -u https://your-gateway/v1 -k $API_KEY -m your-model --format openai-chat
```

### Testing Through LLM-Rosetta Gateway

When the gateway is running (e.g. via `llm-rosetta-gateway` or [argo-proxy](https://github.com/Oaklight/argo-proxy)):

```bash
# OpenAI Chat format
llm-comply --format openai-chat \
  -u http://localhost:44497/v1 \
  -k $API_KEY -m gpt-4o-mini

# Anthropic format (requires custom auth header)
llm-comply --format anthropic \
  -u http://localhost:44497/v1 \
  -k $API_KEY -m claude-haiku-4-5 \
  --auth-header x-api-key --no-bearer \
  -H anthropic-version:2023-06-01

# Google GenAI format
llm-comply --format google-genai \
  -u http://localhost:44497 \
  -k $API_KEY -m gemini-2.5-flash \
  --auth-header x-goog-api-key --no-bearer
```

### Common Options

```
-u, --base-url URL     API base URL (required)
-k, --api-key KEY      API key
-m, --model MODEL      Model name (default: gpt-4o-mini)
--format FORMAT        API format: open-responses, openai-chat, anthropic, google-genai
-f, --filter IDS       Comma-separated test IDs to run
-i, --ignore PATTERNS  Ignore errors matching substrings
-H, --header K:V       Extra headers
--auth-header NAME     Auth header name (default: Authorization)
--no-bearer            Don't prepend "Bearer " to API key
-v, --verbose          Show request/response on failure
--json                 Output results as JSON
--list                 List available tests without running
```

## Web UI

llm-comply includes a browser-based interface for interactive testing:

```bash
llm-comply run --web --host 0.0.0.0 --port 8080
```

A hosted instance is available at [llm-comply.service.oaklight.top](https://llm-comply.service.oaklight.top).

## CI Integration

The LLM-Rosetta repository includes an on-demand [Compliance workflow](https://github.com/Oaklight/llm-rosetta/actions/workflows/compliance.yml) that runs llm-comply against a gateway endpoint.

### Triggering via CLI

```bash
gh workflow run Compliance \
  -f base_url=https://rosetta-dev.service.oaklight.top/v1 \
  -f api_key=your-key \
  -f model=deepseek-v4-flash \
  -f formats="open-responses openai-chat" \
  -f timeout=60
```

### Triggering via GitHub UI

Navigate to **Actions → Compliance → Run workflow**, fill in the inputs, and click **Run workflow**. Results appear as a summary table in the workflow run page, and JSON artifacts are retained for 30 days.

### Available Workflow Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `base_url` | `https://rosetta-dev.service.oaklight.top/v1` | API base URL to test |
| `api_key` | — | API key (required) |
| `model` | `deepseek-v4-flash` | Model to test with |
| `formats` | `open-responses openai-chat` | Space-separated formats to test |
| `timeout` | `60` | Request timeout in seconds |

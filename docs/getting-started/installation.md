---
title: Installation
---

# Installation

## Library

```bash
pip install llm-rosetta
```

The core library has minimal dependencies (`typing_extensions>=4.0.0`).

### Provider SDKs (Optional)

Install provider SDKs if you need to make direct API calls:

```bash
# Individual providers
pip install openai
pip install anthropic
pip install google-genai

# All providers at once
pip install "llm-rosetta[openai,anthropic,google]"
```

!!! note

    Provider SDKs are only needed for making API calls. LLM-Rosetta's conversion functions work with plain dictionaries and don't require the SDKs.

## Gateway

```bash
pip install "llm-rosetta[gateway]"
```

The gateway has **zero external runtime dependencies** — it uses vendored stdlib-only modules for HTTP server and client.

### Docker

Pre-built images are available on DockerHub:

```bash
docker pull oaklight/llm-rosetta-gateway:latest
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

Or use Docker Compose — see `docker/docker-compose.yaml` in the repository.

## Development

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[all]"
```

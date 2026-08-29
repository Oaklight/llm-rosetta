---
title: Installation
---

# Installation

!!! info "Requirements"
    Python **≥ 3.10** is required. The core library has minimal dependencies (`typing_extensions>=4.0.0`).

## Library

```bash
pip install llm-rosetta
```

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

### pip

```bash
pip install "llm-rosetta[gateway]"
```

The gateway has **zero external runtime dependencies** — it uses vendored stdlib-only modules for HTTP server and client.

### Standalone Binaries

Pre-compiled single-file executables are available on [GitHub Releases](https://github.com/Oaklight/llm-rosetta/releases). No Python runtime required.

| Platform | File |
|----------|------|
| Linux x86_64 (glibc) | `llm-rosetta-gateway-<ver>-linux-x86_64` |
| Linux x86_64 (musl) | `llm-rosetta-gateway-<ver>-linux-x86_64-musl` |
| Linux arm64 (glibc) | `llm-rosetta-gateway-<ver>-linux-arm64` |
| Linux arm64 (musl) | `llm-rosetta-gateway-<ver>-linux-arm64-musl` |
| macOS arm64 | `llm-rosetta-gateway-<ver>-macos-arm64` |
| Windows x86_64 | `llm-rosetta-gateway-<ver>-windows-x86_64.exe` |

```bash
# Download and run (Linux/macOS)
chmod +x llm-rosetta-gateway-*
./llm-rosetta-gateway-<ver>-linux-x86_64 --help
```

!!! tip

    Use **musl** binaries for Alpine-based environments and Docker. Use **glibc** binaries for Ubuntu, Debian, and most other Linux distributions.

### Docker

Three image variants are available on [DockerHub](https://hub.docker.com/r/oaklight/llm-rosetta-gateway):

| Tag | Base | Size | Use case |
|-----|------|------|----------|
| `:<ver>` / `latest` | Alpine + binary | ~21 MB | Default, smallest |
| `:<ver>-glibc` | busybox:glibc + binary | ~25 MB | glibc-only environments |
| `:<ver>-python` | python:alpine + pip | ~80 MB | Need pip extensions |

```bash
# Default (Alpine, recommended)
docker pull oaklight/llm-rosetta-gateway:latest

# Run with config volume
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway

# Custom UID/GID mapping
docker run --user $(id -u):$(id -g) -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway
```

Or use Docker Compose — see `docker/docker-compose.yaml` in the repository.

## Development

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[all]"
```

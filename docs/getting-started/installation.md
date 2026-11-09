---
title: Installation
---

# Installation

!!! info "Requirements"
    Python **≥ 3.10** is required. On Python 3.11+ the core has no third-party runtime dependency; Python 3.10 adds only `typing_extensions`, the standard library's `typing` backport.

## Library

```bash
pip install llm-rosetta
```

## Gateway

### pip

```bash
pip install llm-rosetta-gateway
```

The gateway is published as its own distribution and depends on a compatible
`llm-rosetta` core. `pip install "llm-rosetta[gateway]"` still works as a
transitional alias that installs both.

The gateway has **zero external runtime dependencies** — it uses vendored stdlib-only modules for HTTP server and client.

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

#### Building from source

The images above install released wheels. To run the gateway from a checkout
instead, the repository ships a source build that installs both distributions
(`packages/llm-rosetta` and `packages/llm-rosetta-gateway`) straight from the
working tree:

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
echo "ADMIN_PASSWORD=change-me" > .env   # compose reads .env
docker compose up -d --build             # builds docker/Dockerfile.source
```

Without Compose:

```bash
docker build -f docker/Dockerfile.source -t llm-rosetta-gateway:src .
docker run -p 8765:8765 -v "$PWD/config:/config" llm-rosetta-gateway:src
```

The root `docker-compose.yaml` mounts `./config` as the container's `/config`
and reads `.env` for variables your `config.jsonc` references (for example
`"${ADMIN_PASSWORD}"`). The release path is unchanged: `make build-docker`
still builds `docker/Dockerfile`, which installs wheels from `dist/` or PyPI.

## Development

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e "packages/llm-rosetta[conformance]" -e packages/llm-rosetta-gateway
```

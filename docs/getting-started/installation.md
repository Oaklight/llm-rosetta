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
instead, `docker/docker-compose.source.yaml` builds `docker/Dockerfile.source`,
which installs both distributions (`packages/llm-rosetta` and
`packages/llm-rosetta-gateway`) straight from the working tree:

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
mkdir -p config   # the container runs as uid 1000 and cannot create it
docker compose -f docker/docker-compose.source.yaml up -d --build
```

Without Compose:

```bash
docker build -f docker/Dockerfile.source -t llm-rosetta-gateway:src .
mkdir -p config
docker run -p 8765:8765 -v "$PWD/config:/config" llm-rosetta-gateway:src
```

The source file mounts the repository's `config/` and takes `PUID`, `PGID` and
any `environment:` entries the same way the image file does; the two files are
separate entry points — `docker/docker-compose.yaml` runs the published image,
`docker/docker-compose.source.yaml` builds from the checkout — and both publish
port 8765, so run one at a time. Create `config/` before the first start: a
missing bind-mount directory is created by compose as `root`, and the gateway
runs as uid 1000, so its first-run `config.jsonc` write would fail. An empty
directory is enough, but it must be writable by uid 1000: if your user id is
not 1000, run `sudo chown 1000:1000 config`. The image's `PUID`/`PGID`
variables are not read by the gateway today (issue
[#1481](https://github.com/Oaklight/llm-rosetta/issues/1481)), so passing them
does not change this.

The release path is unchanged: `make build-docker` still builds
`docker/Dockerfile`, which installs wheels from `dist/` or PyPI.

## Development

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e "packages/llm-rosetta[conformance]" -e packages/llm-rosetta-gateway
```

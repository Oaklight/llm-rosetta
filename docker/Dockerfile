# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG REGISTRY_MIRROR
FROM ${REGISTRY_MIRROR:-docker.io}/python:${PYTHON_VERSION}-alpine AS base

ARG PYPI_MIRROR
# Dev builds only (e.g. TestPyPI) — do not set for production images
ARG EXTRA_INDEX_URL
ARG LOCAL_WHEEL
ARG PACKAGE_VERSION

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser

# `llm-rosetta ext install-cli` installs Node.js and the agent CLIs into
# /config/tools, so they follow the volume and not the image.  The musl build of
# Node.js is linked against libstdc++ (and libgcc), which this base lacks.
RUN apk add --no-cache libstdc++

RUN mkdir -p /tmp/dist/
COPY dist/ /tmp/dist/

# NB: $MIRROR_IDX / $EXTRA_IDX are intentionally unquoted — empty vars must
# word-split to nothing so pip sees no extra flags.
RUN set -e; \
    EXTRA_IDX=""; \
    if [ -n "$EXTRA_INDEX_URL" ]; then \
        EXTRA_IDX="--extra-index-url $EXTRA_INDEX_URL"; \
    fi; \
    MIRROR_IDX=""; \
    if [ -n "$PYPI_MIRROR" ]; then \
        MIRROR_IDX="-i $PYPI_MIRROR"; \
    fi; \
    if [ -n "$LOCAL_WHEEL" ] && [ -f "/tmp/dist/$LOCAL_WHEEL" ]; then \
        echo "Installing from local wheel: $LOCAL_WHEEL"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir /tmp/dist/llm_rosetta-*.whl /tmp/dist/llm_rosetta_gateway-*.whl; \
    elif [ -n "$PACKAGE_VERSION" ]; then \
        echo "Installing llm-rosetta version: $PACKAGE_VERSION"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir "llm-rosetta==$PACKAGE_VERSION" "llm-rosetta-gateway==$PACKAGE_VERSION"; \
    else \
        echo "Installing latest llm-rosetta from PyPI"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir llm-rosetta llm-rosetta-gateway; \
    fi; \
    pip uninstall -y pip setuptools 2>/dev/null || true; \
    rm -rf /tmp/dist/

RUN mkdir -p /config && chown appuser:appgroup /config
VOLUME /config

# The extensions and the programs they drive live on the volume.  The gateway and
# `docker exec <container> llm-rosetta ext ...` read these variables, so both meet
# in one place (without them the command reads the per-user directory).
ENV LLM_ROSETTA_EXTENSIONS_DIR=/config/extensions \
    LLM_ROSETTA_TOOLS_DIR=/config/tools

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

USER appuser
EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]
CMD ["llm-rosetta-gateway", "--config", "/config/config.jsonc", "--host", "0.0.0.0"]

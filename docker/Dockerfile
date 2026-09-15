# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG REGISTRY_MIRROR
FROM ${REGISTRY_MIRROR:-docker.io}/python:${PYTHON_VERSION}-alpine AS base

ARG PYPI_MIRROR
ARG EXTRA_INDEX_URL
ARG LOCAL_WHEEL
ARG PACKAGE_VERSION

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser

RUN mkdir -p /tmp/dist/
COPY dist/ /tmp/dist/

RUN EXTRA_IDX=""; \
    if [ -n "$EXTRA_INDEX_URL" ]; then \
        EXTRA_IDX="--extra-index-url $EXTRA_INDEX_URL"; \
    fi; \
    MIRROR_IDX=""; \
    if [ -n "$PYPI_MIRROR" ]; then \
        MIRROR_IDX="-i $PYPI_MIRROR"; \
    fi; \
    if [ -n "$LOCAL_WHEEL" ] && [ -f "/tmp/dist/$LOCAL_WHEEL" ]; then \
        echo "Installing from local wheel: $LOCAL_WHEEL"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir "/tmp/dist/${LOCAL_WHEEL}[gateway,profiling]"; \
    elif [ -n "$PACKAGE_VERSION" ]; then \
        echo "Installing llm-rosetta version: $PACKAGE_VERSION"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir "llm-rosetta[gateway,profiling]==$PACKAGE_VERSION"; \
    else \
        echo "Installing latest llm-rosetta from PyPI"; \
        pip install $MIRROR_IDX $EXTRA_IDX --no-cache-dir "llm-rosetta[gateway,profiling]"; \
    fi && \
    pip uninstall -y pip setuptools 2>/dev/null || true && \
    rm -rf /tmp/dist/

RUN mkdir -p /config && chown appuser:appgroup /config
VOLUME /config

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

USER appuser
EXPOSE 8765

ENTRYPOINT ["/entrypoint.sh"]
CMD ["llm-rosetta-gateway", "--config", "/config/config.jsonc", "--host", "0.0.0.0"]

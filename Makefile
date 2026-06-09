# Makefile for llm-rosetta package

# Variables
PACKAGE_NAME := llm-rosetta
DOCKER_IMAGE := oaklight/llm-rosetta-gateway
DIST_DIR := dist
VERSION := $(shell grep -oE '__version__[[:space:]]*=[[:space:]]*"[^"]+"' src/llm_rosetta/__init__.py | grep -oE '"[^"]+"' | tr -d '"' || echo "0.1.0")

# Optional variables
V ?= $(VERSION)
PYPI_MIRROR ?=
REGISTRY_MIRROR ?=

# Default target
all: lint test build

# ──────────────────────────────────────────────
# Linting & Formatting
# ──────────────────────────────────────────────

# Run ruff linter
lint:
	@echo "Running ruff check..."
	ruff check src/ tests/
	@echo "Running ruff format check..."
	ruff format --check src/ tests/
	@echo "Lint complete."

# Auto-fix lint issues
lint-fix:
	@echo "Auto-fixing lint issues..."
	ruff check --fix src/ tests/
	ruff format src/ tests/
	@echo "Lint fix complete."

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ --ignore=tests/integration -v --tb=short
	@echo "Tests completed."

# Run integration tests (requires API keys; uses proxychains if available)
test-integration:
	@echo "Running integration tests..."
	@if command -v proxychains >/dev/null 2>&1; then \
		echo "(using proxychains)"; \
		proxychains -q pytest tests/integration/ -v --tb=short; \
	else \
		pytest tests/integration/ -v --tb=short; \
	fi
	@echo "Integration tests completed."

# Run gateway integration tests (all SDKs × all models via llm_api_simple_tests)
test-gateway:
	@echo "Running gateway integration tests..."
	@./scripts/run_gateway_integration.sh
	@echo "Gateway integration tests completed."

# ──────────────────────────────────────────────
# Package targets
# ──────────────────────────────────────────────

# Build the Python package
build-package: clean-package
	@echo "Building $(PACKAGE_NAME) package..."
	python -m build
	@echo "Build complete. Distribution files are in $(DIST_DIR)/"

# Push the package to PyPI
push-package:
	@echo "Pushing $(PACKAGE_NAME) to PyPI..."
	twine upload $(DIST_DIR)/*
	@echo "Package pushed to PyPI."

# Clean up build and distribution files
clean-package:
	@echo "Cleaning up build and distribution files..."
	rm -rf $(DIST_DIR) *.egg-info build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleanup complete."

# Aliases
build: build-package
push: push-package
clean: clean-package

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

build-docker:
	@echo "Building Docker image $(DOCKER_IMAGE):$(V)..."
	@BUILD_ARGS=""; \
	if [ -n "$(REGISTRY_MIRROR)" ]; then \
		echo "Using registry mirror: $(REGISTRY_MIRROR)"; \
		BUILD_ARGS="$$BUILD_ARGS --build-arg REGISTRY_MIRROR=$(REGISTRY_MIRROR)"; \
	fi; \
	LOCAL_WHEEL=""; \
	if [ -d "dist" ] && [ -n "$$(ls -A dist/*$(V)*.whl 2>/dev/null)" ]; then \
		LOCAL_WHEEL=$$(ls dist/*$(V)*.whl | head -n 1 | xargs basename); \
		echo "Found local wheel: $$LOCAL_WHEEL"; \
		BUILD_ARGS="$$BUILD_ARGS --build-arg LOCAL_WHEEL=$$LOCAL_WHEEL"; \
	elif echo "$(V)" | grep -qE '^[0-9]+\.[0-9]+'; then \
		echo "Using version from PyPI: $(V)"; \
		BUILD_ARGS="$$BUILD_ARGS --build-arg PACKAGE_VERSION=$(V)"; \
	elif [ -d "dist" ] && [ -n "$$(ls -A dist/*.whl 2>/dev/null)" ]; then \
		LOCAL_WHEEL=$$(ls dist/*.whl | head -n 1 | xargs basename); \
		echo "Non-version tag '$(V)', using local wheel: $$LOCAL_WHEEL"; \
		BUILD_ARGS="$$BUILD_ARGS --build-arg LOCAL_WHEEL=$$LOCAL_WHEEL"; \
	else \
		echo "No local wheel found, will install latest from PyPI"; \
	fi; \
	if [ -n "$(PYPI_MIRROR)" ]; then \
		echo "Using PyPI mirror: $(PYPI_MIRROR)"; \
		BUILD_ARGS="$$BUILD_ARGS --build-arg PYPI_MIRROR=$(PYPI_MIRROR)"; \
	fi; \
	cd docker && docker build -f Dockerfile $$BUILD_ARGS -t $(DOCKER_IMAGE):$(V) -t $(DOCKER_IMAGE):latest ..
	@echo "Docker image built successfully."

push-docker:
	@echo "Pushing Docker images $(DOCKER_IMAGE):$(V) and $(DOCKER_IMAGE):latest..."
	docker push $(DOCKER_IMAGE):$(V)
	docker push $(DOCKER_IMAGE):latest
	@echo "Docker images pushed successfully."

clean-docker:
	@echo "Cleaning Docker images..."
	docker rmi $(DOCKER_IMAGE):latest 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):$(V) 2>/dev/null || true

# ──────────────────────────────────────────────
# Dev-test deployment
# ──────────────────────────────────────────────

SSH_TARGET ?=
COMPOSE_SERVICE ?= llm-rosetta-gateway-devtest

deploy-dev:
ifndef SSH_TARGET
	$(error SSH_TARGET is required. Usage: make deploy-dev SSH_TARGET=cloud.usa2)
endif
	@COMMIT=$$(git rev-parse --short HEAD); \
	DEV_VER="$(VERSION).dev0+g$$COMMIT"; \
	echo "==> Building dev wheel $$DEV_VER..."; \
	sed -i "s/__version__ = \"$(VERSION)\"/__version__ = \"$$DEV_VER\"/" src/llm_rosetta/__init__.py; \
	rm -rf dist build; \
	conda run -n llm-rosetta python -m build --wheel -q; \
	git checkout src/llm_rosetta/__init__.py; \
	WHEEL=$$(ls dist/*.whl | head -1 | xargs basename); \
	echo "==> Building Docker image from $$WHEEL..."; \
	docker build -f docker/Dockerfile --build-arg LOCAL_WHEEL=$$WHEEL -t $(DOCKER_IMAGE):dev-test -q .; \
	echo "==> Deploying to $(SSH_TARGET) via zstd..."; \
	docker save $(DOCKER_IMAGE):dev-test | zstd -3 | ssh $(SSH_TARGET) \
		'zstd -d | docker load && \
		 cd /dockervol/dockge/stacks/llm-rosetta && \
		 docker compose up -d --force-recreate $(COMPOSE_SERVICE) && \
		 sleep 3 && \
		 curl -sS http://127.0.0.1:54982/health && echo && \
		 docker exec llm-rosetta-$(COMPOSE_SERVICE)-1 python -c "import llm_rosetta; print(llm_rosetta.__version__)"'; \
	echo "==> Dev-test deployed successfully."

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  lint           - Run ruff linter and format check"
	@echo "  lint-fix       - Auto-fix lint and formatting issues"
	@echo "  test               - Run unit tests with pytest"
	@echo "  test-integration   - Run integration tests via proxychains"
	@echo "  test-gateway       - Run gateway integration tests (all SDKs × all models)"
	@echo ""
	@echo "Package:"
	@echo "  build-package  - Build the Python package"
	@echo "  push-package   - Push the package to PyPI"
	@echo "  clean-package  - Clean up build and distribution files"
	@echo ""
	@echo "Docker:"
	@echo "  build-docker   - Build Docker image (local x64)"
	@echo "  push-docker    - Push Docker image to registry"
	@echo "  clean-docker   - Clean Docker images"
	@echo ""
	@echo "Aliases:"
	@echo "  build          - Alias for build-package"
	@echo "  push           - Alias for push-package"
	@echo "  clean          - Alias for clean-package"
	@echo ""
	@echo "Composite targets:"
	@echo "  all            - Run lint, test, and build (default)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make build-docker                  # build from local wheel or PyPI, tag=VERSION"
	@echo "  make build-docker V=0.5.0          # install 0.5.0 from PyPI, tag=0.5.0"
	@echo "  make build-docker V=dev-test       # use local wheel in dist/, tag=dev-test"
	@echo "  make build-docker PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
	@echo "  make build-docker REGISTRY_MIRROR=docker.1ms.run"
	@echo ""
	@echo "Variables:"
	@echo "  V=<version|tag>          - Docker image tag (default: auto-detected from __init__.py)"
	@echo "                             Semver values also set the PyPI install version"
	@echo "                             Non-semver values (e.g. dev-test) use local wheel in dist/"
	@echo "  PYPI_MIRROR=<url>        - PyPI mirror URL"
	@echo "  REGISTRY_MIRROR=<host>   - Docker registry mirror"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev     - Build dev image and deploy to remote dev-test gateway"
	@echo ""
	@echo "Usage examples:"
	@echo "  make deploy-dev SSH_TARGET=cloud.usa2"
	@echo ""
	@echo "Detected version: $(VERSION)"

.PHONY: all lint lint-fix test test-integration test-gateway build-package push-package clean-package build push clean build-docker push-docker clean-docker deploy-dev help

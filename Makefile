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
# Nuitka binary builds
# ──────────────────────────────────────────────

# Detect platform
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
UNAME_M := $(shell uname -m 2>/dev/null || echo x86_64)
ifeq ($(UNAME_S),Linux)
  BINARY_OS := linux
else ifeq ($(UNAME_S),Darwin)
  BINARY_OS := macos
else
  BINARY_OS := windows
endif
ifeq ($(UNAME_M),aarch64)
  BINARY_ARCH := arm64
else ifeq ($(UNAME_M),arm64)
  BINARY_ARCH := arm64
else
  BINARY_ARCH := x86_64
endif

BINARY_NAME = llm-rosetta-gateway-$(VERSION)-$(BINARY_OS)-$(BINARY_ARCH)
BINARY_NAME_MUSL = llm-rosetta-gateway-$(VERSION)-linux-$(BINARY_ARCH)-musl
BINARY_DIR := build
NUITKA_ENTRY := _nuitka_entry.py
NUITKA_JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

NUITKA_FLAGS = \
	--standalone \
	--onefile \
	--jobs=$(NUITKA_JOBS) \
	--output-dir=$(BINARY_DIR) \
	--include-package=llm_rosetta \
	--include-package=pyinstrument \
	--include-data-files=src/llm_rosetta/gateway/admin/admin.html=llm_rosetta/gateway/admin/admin.html \
	--include-data-dir=src/llm_rosetta/gateway/admin/css=llm_rosetta/gateway/admin/css \
	--include-data-dir=src/llm_rosetta/gateway/admin/js=llm_rosetta/gateway/admin/js \
	--include-data-dir=src/llm_rosetta/shims/providers=llm_rosetta/shims/providers \
	--nofollow-import-to=pytest \
	--nofollow-import-to=setuptools \
	--nofollow-import-to=pip \
	--nofollow-import-to=_pytest \
	--assume-yes-for-downloads

# Build native binary (glibc on Linux, system libc on macOS, MSVC on Windows)
build-binary:
	@echo "Building native binary: $(BINARY_NAME)..."
	@printf 'from llm_rosetta.gateway import main\nmain()\n' > $(NUITKA_ENTRY)
	python -m nuitka $(NUITKA_FLAGS) \
		--output-filename=$(BINARY_NAME)$(if $(filter windows,$(BINARY_OS)),.exe,) \
		$(NUITKA_ENTRY); \
	ret=$$?; rm -f $(NUITKA_ENTRY); exit $$ret
	@ls -lh $(BINARY_DIR)/$(BINARY_NAME)*
	@echo "Binary build complete."

# Build musl-linked binary via Alpine Docker container (Linux only)
build-binary-musl:
	@echo "Building musl binary: $(BINARY_NAME_MUSL)..."
	@mkdir -p $(BINARY_DIR)
	docker run --rm \
		-v $(CURDIR):/workspace:ro \
		-v $(CURDIR)/$(BINARY_DIR):/output \
		$(REGISTRY_MIRROR:%=%/)python:3.12-alpine \
		/bin/sh -c '\
			mkdir -p /tmp/build && tar -cf - -C /workspace --exclude=.git --exclude=__pycache__ . | tar -xf - -C /tmp/build && cd /tmp/build && \
			apk add --no-cache gcc musl-dev python3-dev git >/dev/null && \
			pip install --break-system-packages patchelf -q && \
			pip install --break-system-packages -e ".[profiling]" -q && \
			pip install --break-system-packages "nuitka[onefile]" ordered-set -q && \
			printf "from llm_rosetta.gateway import main\nmain()\n" > /tmp/_entry.py && \
			python -m nuitka \
				--standalone --onefile \
				--jobs=$$(nproc) \
				--output-dir=/output \
				--output-filename=$(BINARY_NAME_MUSL) \
				--include-package=llm_rosetta \
				--include-package=pyinstrument \
				--include-data-files=src/llm_rosetta/gateway/admin/admin.html=llm_rosetta/gateway/admin/admin.html \
				--include-data-dir=src/llm_rosetta/gateway/admin/css=llm_rosetta/gateway/admin/css \
				--include-data-dir=src/llm_rosetta/gateway/admin/js=llm_rosetta/gateway/admin/js \
				--include-data-dir=src/llm_rosetta/shims/providers=llm_rosetta/shims/providers \
				--nofollow-import-to=pytest \
				--nofollow-import-to=setuptools \
				--nofollow-import-to=pip \
				--nofollow-import-to=_pytest \
				--assume-yes-for-downloads \
				/tmp/_entry.py && \
			rm -rf /output/_entry.* '
	@ls -lh $(BINARY_DIR)/$(BINARY_NAME_MUSL)
	@echo "Musl binary build complete."

clean-binary:
	@echo "Cleaning binary build artifacts..."
	rm -rf $(BINARY_DIR)/_nuitka_entry.* $(BINARY_DIR)/_entry.* $(NUITKA_ENTRY)
	@echo "Clean complete. Binaries in $(BINARY_DIR)/ preserved."

clean-binary-all:
	@echo "Cleaning all binary artifacts..."
	rm -rf $(BINARY_DIR)
	rm -f $(NUITKA_ENTRY)
	@echo "Clean complete."

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

# Build Alpine Docker image with musl binary
build-docker-alpine:
	@BINARY=$(BINARY_DIR)/$(BINARY_NAME_MUSL); \
	if [ ! -f "$$BINARY" ]; then \
		echo "::error::Musl binary not found: $$BINARY"; \
		echo "Run 'make build-binary-musl' first."; \
		exit 1; \
	fi; \
	echo "Building Alpine Docker image $(DOCKER_IMAGE):$(V)-alpine..."; \
	docker build -f docker/Dockerfile.binary \
		--build-arg BASE_IMAGE=$(REGISTRY_MIRROR:%=%/)alpine \
		--build-arg BINARY=$$BINARY \
		-t $(DOCKER_IMAGE):$(V)-alpine \
		-t $(DOCKER_IMAGE):$(V) \
		-t $(DOCKER_IMAGE):latest .
	@echo "Alpine Docker image built successfully."

# Build glibc Docker image with native binary
build-docker-glibc:
	@BINARY=$(BINARY_DIR)/$(BINARY_NAME); \
	if [ ! -f "$$BINARY" ]; then \
		echo "::error::Native binary not found: $$BINARY"; \
		echo "Run 'make build-binary' first."; \
		exit 1; \
	fi; \
	echo "Building glibc Docker image $(DOCKER_IMAGE):$(V)-glibc..."; \
	docker build -f docker/Dockerfile.binary \
		--build-arg BASE_IMAGE=$(REGISTRY_MIRROR:%=%/)busybox:glibc \
		--build-arg BINARY=$$BINARY \
		-t $(DOCKER_IMAGE):$(V)-glibc .
	@echo "Glibc Docker image built successfully."

# Build Python-based Docker image (existing, with pip install)
build-docker-python:
	@echo "Building Python Docker image $(DOCKER_IMAGE):$(V)-python..."
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
	cd docker && docker build -f Dockerfile $$BUILD_ARGS -t $(DOCKER_IMAGE):$(V)-python ..
	@echo "Python Docker image built successfully."

# Legacy alias
build-docker: build-docker-python

push-docker:
	@echo "Pushing Docker images..."
	@for tag in $(V)-alpine $(V) latest $(V)-glibc $(V)-python; do \
		if docker image inspect $(DOCKER_IMAGE):$$tag >/dev/null 2>&1; then \
			echo "  Pushing $(DOCKER_IMAGE):$$tag"; \
			docker push $(DOCKER_IMAGE):$$tag; \
		fi; \
	done
	@echo "Docker images pushed successfully."

clean-docker:
	@echo "Cleaning Docker images..."
	@for tag in $(V)-alpine $(V) latest $(V)-glibc $(V)-python; do \
		docker rmi $(DOCKER_IMAGE):$$tag 2>/dev/null || true; \
	done

# ──────────────────────────────────────────────
# Dev-test deployment
# ──────────────────────────────────────────────

SSH_TARGET ?=
DEVTEST_STACK ?= /dockervol/dockge/stacks/llm-rosetta-devtest
DEVTEST_CONTAINER ?= llm-rosetta-devtest-llm-rosetta-gateway-devtest-1

deploy-dev:
ifndef SSH_TARGET
	$(error SSH_TARGET is required. Usage: make deploy-dev SSH_TARGET=cloud.usa2)
endif
	@set -e; \
	COMMIT=$$(git rev-parse --short HEAD); \
	ORIG_VER=$$(python -c 'import re; print(re.search(r"__version__ = \"([^\"]+)\"", open("src/llm_rosetta/__init__.py").read()).group(1))'); \
	DEV_VER="$$ORIG_VER.dev0+g$$COMMIT"; \
	echo "==> Building dev wheel $$DEV_VER..."; \
	python -c 'from pathlib import Path; p=Path("src/llm_rosetta/__init__.py"); s=p.read_text(); p.write_text(s.replace("__version__ = \"'"$$ORIG_VER"'\"", "__version__ = \"'"$$DEV_VER"'\""))'; \
	rm -rf dist build; \
	conda run -n llm-rosetta python -m build --wheel -q; \
	python -c 'from pathlib import Path; p=Path("src/llm_rosetta/__init__.py"); s=p.read_text(); p.write_text(s.replace("__version__ = \"'"$$DEV_VER"'\"", "__version__ = \"'"$$ORIG_VER"'\""))'; \
	WHEEL=$$(ls dist/*.whl | head -1 | xargs basename); \
	echo "==> Building Docker image from $$WHEEL..."; \
	docker build -f docker/Dockerfile --build-arg LOCAL_WHEEL=$$WHEEL -t $(DOCKER_IMAGE):dev-test -q .; \
	echo "==> Deploying to $(SSH_TARGET) via zstd..."; \
	docker save $(DOCKER_IMAGE):dev-test | zstd -3 | ssh $(SSH_TARGET) \
		'zstd -d | docker load && \
		 cd $(DEVTEST_STACK) && \
		 docker compose up -d --force-recreate && \
		 for i in 1 2 3 4 5 6 7 8 9 10; do \
		   curl -sS http://127.0.0.1:54982/health && break; \
		   echo "  waiting for gateway ($$i/10)..."; sleep 3; \
		 done && echo && \
		 docker exec $(DEVTEST_CONTAINER) python -c "import llm_rosetta; print(llm_rosetta.__version__)"'; \
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
	@echo "Binary:"
	@echo "  build-binary       - Build native Nuitka binary for current platform"
	@echo "  build-binary-musl  - Build musl-linked binary via Alpine Docker"
	@echo "  clean-binary       - Clean build artifacts (keep binaries)"
	@echo "  clean-binary-all   - Clean all binary artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  build-docker-alpine  - Build Alpine Docker image (musl binary)"
	@echo "  build-docker-glibc   - Build glibc Docker image (native binary)"
	@echo "  build-docker-python  - Build Python Docker image (pip install)"
	@echo "  build-docker         - Alias for build-docker-python"
	@echo "  push-docker          - Push all Docker images"
	@echo "  clean-docker         - Clean Docker images"
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
	@echo "  SSH_TARGET=<host>        - SSH target for deploy-dev (required)"
	@echo "  DEVTEST_STACK=<path>     - Remote compose stack path (default: /dockervol/dockge/stacks/llm-rosetta-devtest)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make deploy-dev SSH_TARGET=cloud.usa2"
	@echo ""
	@echo "Detected version: $(VERSION)"

.PHONY: all lint lint-fix test test-integration test-gateway build-package push-package clean-package build push clean build-binary build-binary-musl clean-binary clean-binary-all build-docker-alpine build-docker-glibc build-docker-python build-docker push-docker clean-docker deploy-dev help

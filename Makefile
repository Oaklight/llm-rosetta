# Makefile for llm-rosetta package

# Variables
PACKAGE_NAME := llm-rosetta
DIST_DIR := dist
VERSION := $(shell grep -oE '__version__[[:space:]]*=[[:space:]]*"[^"]+"' src/llm_rosetta/__init__.py | grep -oE '"[^"]+"' | tr -d '"' || echo "0.1.0")

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
	pytest tests/ -v --tb=short
	@echo "Tests completed."

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

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  lint           - Run ruff linter and format check"
	@echo "  lint-fix       - Auto-fix lint and formatting issues"
	@echo "  test           - Run tests with pytest"
	@echo ""
	@echo "Package targets:"
	@echo "  build-package  - Build the Python package"
	@echo "  push-package   - Push the package to PyPI"
	@echo "  clean-package  - Clean up build and distribution files"
	@echo ""
	@echo "Aliases:"
	@echo "  build          - Alias for build-package"
	@echo "  push           - Alias for push-package"
	@echo "  clean          - Alias for clean-package"
	@echo ""
	@echo "Composite targets:"
	@echo "  all            - Run lint, test, and build (default)"
	@echo ""
	@echo "Detected version: $(VERSION)"

.PHONY: all lint lint-fix test build-package push-package clean-package build push clean help

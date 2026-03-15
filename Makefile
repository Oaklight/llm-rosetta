# Makefile for cicada pip package

# Variables
PACKAGE_NAME := llm_rosetta
DIST_DIR := dist

# Default target
all: build push clean

# Build the package
build: clean
	@echo "Building $(PACKAGE_NAME) version ..."
	python -m build
	@echo "Build complete. Distribution files are in $(DIST_DIR)/"

# Push the package to PyPI
push:
	@echo "Pushing $(PACKAGE_NAME) version to PyPI..."
	twine upload dist/*
	@echo "Package pushed to PyPI."

# Clean up build and distribution files
clean:
	@echo "Cleaning up build and distribution files..."
	rm -rf $(DIST_DIR) *.egg-info
	@echo "Cleanup complete."

# Help target
help:
	@echo "Available targets:"
	@echo "  build   - Build the pip package"
	@echo "  push    - Push the package to PyPI"
	@echo "  clean   - Clean up build and distribution files"
	@echo "  help    - Show this help message"

.PHONY: all build push clean help

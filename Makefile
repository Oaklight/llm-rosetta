# Zensical Documentation Build Script for LLMIR
#

# You can set these variables from the command line.
ZENSICAL      ?= zensical
SOURCEDIR     = docs
BUILDDIR      = site

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Available targets:"
	@echo "  clean  - Remove build artifacts"
	@echo "  html   - Build HTML documentation"
	@echo "  serve  - Build and locally serve documentation"
	@echo "  live   - Live reload server for development"

.PHONY: help clean html serve live

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)

# Build HTML documentation
html: clean
	@echo "Building HTML documentation..."
	@$(ZENSICAL) build

# Build and serve documentation locally
serve: html
	@echo "Serving documentation at http://localhost:8000"
	@cd $(BUILDDIR) && python -m http.server 8000

# Live reload documentation during development
live:
	@echo "Starting live documentation server..."
	@$(ZENSICAL) serve

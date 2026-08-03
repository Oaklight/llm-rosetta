---
title: Style Guide
---

# Style Guide

## Python Code

- Code follows `ruff` defaults — run `make lint` or let pre-commit catch issues on commit
- Type hints are encouraged on all public APIs
- Do not edit files under `src/llm_rosetta/_vendor/` — those are managed externally

## Docstrings

- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- All public functions and classes should have docstrings
- Write docstrings and comments in English

## Naming Conventions

See [Naming Conventions](naming-conventions.md) for detailed rules on variable, function, class, and file naming across Python, Go, and Shell.

## Tooling

| Tool | Purpose | Config |
|------|---------|--------|
| `ruff` | Linting + formatting | `pyproject.toml` |
| `ty` | Type checking | `pyproject.toml` |
| `pre-commit` | Git hooks (runs ruff, ty on commit) | `.pre-commit-config.yaml` |
| `complexipy` | Complexity analysis | `pyproject.toml` |

### Pre-commit Workflow

Pre-commit hooks run automatically on `git commit`. If a hook modifies files (e.g. `ruff format`), re-stage the changes and commit again. If a hook reports errors (e.g. `ty check`), fix manually and retry.

!!! note
    The project uses `language: system` hooks, which depend on tools being available in your current shell `PATH`. Make sure the project's conda environment is activated before committing.

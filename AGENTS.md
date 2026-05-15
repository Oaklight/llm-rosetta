# AGENTS.md — LLM-Rosetta

> Context file for AI coding assistants. Symlinked as `CLAUDE.md`.

## What this project is

LLM-Rosetta is a **Python library for converting between different LLM provider
API formats** using a hub-and-spoke architecture with a central IR (Intermediate
Representation). It solves the N² provider conversion problem — each provider
only needs a single converter to/from IR, and any-to-any conversion is
automatically supported.

The project also ships an **API gateway** (`llm-rosetta-gateway`) that proxies
requests between providers with live format conversion, streaming, and key
rotation.

Zero required dependencies at its core; provider SDKs are optional extras.

## Architecture

### Conversion pipeline

```
Provider A ──→ IR ──→ Provider B
```

Four converters, one per API standard:

| Converter | API Standard | Module |
|-----------|-------------|--------|
| `openai_chat` | OpenAI Chat Completions | `converters/openai_chat/` |
| `openai_responses` | OpenAI Responses API | `converters/openai_responses/` |
| `anthropic` | Anthropic Messages API | `converters/anthropic/` |
| `google` | Google GenAI API | `converters/google_genai/` |

Each converter implements bidirectional conversion (request/response) and
streaming. Converters are provider-agnostic — provider-specific quirks are
handled by the **shim layer**.

### Shim layer

A **ProviderShim** is a lightweight identity card that declares which base
converter a provider uses, plus connection defaults and field-level transforms.
This avoids per-provider conversion code — converters stay generic, shims
declare the differences.

| Component | File | Purpose |
|-----------|------|---------|
| `ProviderShim` / `ModelShim` | `shims/provider_shim.py` | Data classes + registry |
| `Transform` primitives | `shims/transforms.py` | `strip_fields`, `rename_field`, `set_defaults` |
| Built-in shims | `shims/builtins.py` | OpenAI, Anthropic, Google, DeepSeek, Volcengine |

### Gateway

The gateway is a standalone HTTP proxy server that accepts requests in any
supported format and forwards them to any configured upstream provider.

| Module | Purpose |
|--------|---------|
| `gateway/app.py` | Route handlers, app factory |
| `gateway/proxy.py` | Core proxy engine (non-streaming, streaming, SSE, transforms) |
| `gateway/config.py` | JSONC config loading, env-var substitution |
| `gateway/providers.py` | Provider info, auth headers, key rotation |
| `gateway/admin/` | Admin UI, metrics, request logging, persistence |

### IR type system

Typed dataclasses for the intermediate representation live in `types/ir/`.
Provider-specific types (for documentation, not runtime) in
`types/anthropic/`, `types/google/`.

## Repository layout

```
src/llm_rosetta/
├── __init__.py              # Public API: convert(), get_converter_for_provider()
├── auto_detect.py           # Provider auto-detection from request body
├── tool_ops.py              # Cross-provider tool call utilities
├── converters/              # 4 bidirectional converters
│   ├── base/                # Abstract base + ConversionContext
│   ├── openai_chat/
│   ├── openai_responses/
│   ├── anthropic/
│   └── google_genai/
├── shims/                   # Provider/model identity cards + transforms
│   ├── provider_shim.py
│   ├── transforms.py
│   └── builtins.py
├── gateway/                 # HTTP proxy gateway
│   ├── app.py, proxy.py, config.py, providers.py
│   ├── auth.py, logging.py, cli.py, banner.py
│   └── admin/               # Admin panel (metrics, request log, persistence)
├── types/                   # Typed IR and provider-specific types
│   └── ir/                  # IR dataclasses (messages, parts, tools, stream)
└── _vendor/                 # Vendored dependencies (DO NOT EDIT)

tests/
├── converters/              # Per-converter test suites
├── gateway/                 # Gateway unit tests
├── integration/             # E2E tests (require API keys)
├── test_types/              # IR type system tests
├── test_shims.py
├── test_transforms.py
├── test_auto_detect.py
└── test_tool_ops.py

docs_en/, docs_zh/           # Documentation (git worktrees, orphan branches)
docker/                      # Dockerfile for gateway image
examples/                    # Usage examples
```

## Setup and commands

```bash
conda activate llm-rosetta
pip install -e ".[all]"
```

Run `make help` for all targets. Key ones:

```bash
make lint          # ruff check + ruff format --check
make lint-fix      # ruff check --fix + ruff format
make test          # pytest tests/ --ignore=tests/integration -v
make build         # python -m build
make push          # twine upload
make build-docker  # Build gateway Docker image
```

Tooling config (ruff, ty, complexipy) lives in `pyproject.toml`.

## Definition of done

1. `make lint` and `make test` exit 0
2. `ruff check --fix && ruff format` applied to changed files
3. New code has tests in `tests/`
4. Google-style docstrings on public APIs; comments in English
5. No manual edits to `_vendor/` — update upstream, re-vendor
6. Python ≥ 3.10 compatibility

## Workflow

- **Branch from master**, open a PR, require CI green before merge.
- **Merge strategy: rebase** — keep commits atomic and well-messaged.
- Branch naming: `feature/...`, `fix/...`, `refactor/...`, `test/...`, `docs/...`
- Never force-push to `master`.

## Documentation

User-facing docs live on **orphan branches** (`docs-en`, `docs-zh`), mounted
as git worktrees at `./docs_en/` and `./docs_zh/`. MkDocs + mkdocs-material,
deployed to ReadTheDocs.

## Escalation

- Converter output mismatch → check IR types in `types/ir/`, then the
  converter's `*_ops.py` modules
- Shim/transform issue → check `shims/builtins.py` and `shims/transforms.py`
- Gateway config issue → check `gateway/config.py` resolution order
  (shim → type → name fallback)
- `_vendor/` issues → never fix in-place; update upstream, re-vendor
- Integration test failure → likely missing API keys or network; these are
  excluded from `make test` by default
- Test failure after 3 attempts → stop, report full output
- Never: delete files to fix errors, skip tests, modify `_vendor/` directly

## Files to never edit

- `src/llm_rosetta/_vendor/**` — vendored dependencies, managed externally
- `docs_en/`, `docs_zh/` — separate git branches, edit inside the worktree only

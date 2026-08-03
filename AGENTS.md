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

## Contributing guidelines

Read `CONTRIBUTING.md` before making changes — it links to the full
contributing guide, style guide, and architecture guide on ReadTheDocs.

## Definition of done

1. `make lint` and `make test` exit 0
2. `ruff check --fix && ruff format` applied to changed files
3. New code has tests in `tests/`
4. Google-style docstrings on public APIs; comments in English
5. No manual edits to `_vendor/` — update upstream, re-vendor
6. Python ≥ 3.10 compatibility
7. Converter changes must pass round-trip tests (A→IR→A, A→IR→B,
   A→IR→B→IR→A) — see the
   [Architecture Guide](https://llm-rosetta.readthedocs.io/en/latest/contributing/architecture/)

## Integration testing with agentabi

After gateway or converter changes that affect cross-format conversion,
run the `agentabi` test matrix to verify end-to-end behavior with real
agent CLIs. This requires a running proxy (e.g. argo-proxy or
llm-rosetta-gateway) with upstream access.

### Setup

```bash
conda activate agentabi
# Ensure the proxy is running (e.g. on :44497)
```

### Test matrix

```python
from agentabi import run_sync

PROXY_ENV = {
    "OPENAI_BASE_URL": "http://127.0.0.1:44497/v1",
    "OPENAI_API_KEY": "<token>",
}

tests = [
    # (agent, model, prompt) — covers same-format and cross-format paths
    ("codex", "gpt-5-nano", "What is 2+2? Reply with just the number."),
    ("codex", "claude-haiku-4-5", "What is 3+3? Reply with just the number."),
    ("codex", "claude-opus-4-6", "What is 4+4? Reply with just the number."),
    ("opencode", "gpt-5-nano", "What is 5+5? Reply with just the number."),
]

for agent, model, prompt in tests:
    result = run_sync(prompt, agent=agent, model=model, env=PROXY_ENV,
                      max_turns=1, timeout=60)
    print(f"{agent}/{model}: {result.get('status')} → {result.get('result_text')}")
```

### Multi-turn tool-use test

For deeper validation (file reads, writes, multi-step reasoning):

```python
result = run_sync(
    "Read src/llm_rosetta/shims/provider_shim.py and write a summary to /tmp/test-output.md",
    agent="codex", model="claude-haiku-4-5",
    env=PROXY_ENV, max_turns=10, timeout=120,
    working_dir="/path/to/llm-rosetta",
)
# Check /tmp/test-output.md was created with meaningful content
```

### What to verify

- Same-format pass-through (OpenAI → OpenAI)
- Cross-format conversion (OpenAI Responses → Anthropic, via shim)
- Streaming SSE round-trip
- Multi-turn tool calls (file read/write)
- Reasoning/thinking fields preserved across formats

## Workflow

- **Branch from master**, open a PR, require CI green before merge.
- **Merge strategy: rebase** — keep commits atomic and well-messaged.
- Branch naming: `feature/...`, `fix/...`, `refactor/...`, `test/...`, `docs/...`
- Never force-push to `master`.
- **No AI co-author tags in commits.** Do not add `Co-authored-by` lines for AI
  tools in git commit messages. Disclose AI usage in PR descriptions instead.

## Release process

Releases are triggered manually via GitHub Actions, not by tag push.

### Steps

1. **Bump version** in `src/llm_rosetta/__init__.py`
2. **Update changelogs** — move `[Unreleased]` entries into a versioned section
   (`## vX.Y.Z — YYYY-MM-DD`) in both `docs_en/docs/changelog.md` and
   `docs_zh/docs/changelog.md`. Commit and push both doc worktrees.
3. **Commit and tag**:
   ```bash
   git add src/llm_rosetta/__init__.py
   git commit -m "release: vX.Y.Z"
   git tag vX.Y.Z
   git push origin master vX.Y.Z
   ```
4. **Trigger the Release workflow** (builds wheel → publishes to PyPI →
   triggers Docker build automatically):
   ```bash
   gh workflow run "Release" --repo Oaklight/llm-rosetta -f version=X.Y.Z
   ```
5. **Create GitHub Release** (optional — the Release workflow also creates one,
   but you can create it manually first for custom release notes):
   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."
   ```

### Dev deployment (pre-release testing)

```bash
make deploy-dev SSH_TARGET=cloud.usa2
```

Builds a dev wheel from the current working tree, packages it into a Docker
image tagged `dev-test`, and deploys to the remote host via `docker save |
zstd | ssh`. The dev image version includes the git commit hash (e.g.
`0.6.11.dev0+gaa73770`).

**Important**: `make deploy-dev` builds from the **working tree**, not from
the committed state. Always `git pull` and verify no dirty files in `src/`
before deploying — otherwise the image may not contain the expected code.

## Documentation

User-facing docs live on **orphan branches** (`docs-en`, `docs-zh`), mounted
as git worktrees at `./docs_en/` and `./docs_zh/`. Built with zensical,
deployed to ReadTheDocs.

### When to update docs worktrees

Update `docs_en/` and `docs_zh/` whenever any of the following happens:

- **New public API added or signature changed**: update the relevant API
  reference pages in both languages.
- **Behavior change or bug fix affecting documented functionality**: update
  affected guide/reference pages.
- **Changelog-worthy change merged to main branch**: update
  `docs_en/docs/changelog.md` and `docs_zh/docs/changelog.md` under the
  `[Unreleased]` section. Follow the [Keep a Changelog](https://keepachangelog.com/)
  format. Entries should cover: features, enhancements, bug fixes,
  breaking changes, and infrastructure.
- **Release published**: move `[Unreleased]` entries into a new versioned
  section in both changelogs.

### Cross-language consistency (enforced)

**Both language versions must be updated in the same task/agent run.** Never
update only one language and leave the other for later. The workflow is:

1. Make changes to `docs_en/` first (English is the source of truth).
2. In the same task, apply equivalent changes to `docs_zh/` before committing.
3. Commit and push both worktrees before the task is considered done.

Splitting the two languages across separate agents or separate sessions is not
allowed — it leads to drift and missed pages.

Commits in doc worktrees use `PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit` since
those branches have no `.pre-commit-config.yaml`.

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

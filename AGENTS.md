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

Five chat converters, one per API standard:

| Converter | API Standard | Module |
|-----------|-------------|--------|
| `chat_completions` | OpenAI Chat Completions | `converters/llm/chat_completions/` |
| `open_responses` | Open Responses spec (vendor-neutral) | `converters/llm/open_responses/` |
| `messages` | Anthropic Messages API | `converters/llm/messages/` |
| `google_generate` | Google generateContent API | `converters/llm/google_generate/` |
| `google_interactions` | Google Interactions API | `converters/llm/google_interactions/` |

`OpenResponsesConverter` implements the vendor-neutral Open Responses spec (the
base of the two Responses classes); `OpenAIResponsesConverter` derives from it as
the OpenAI profile (adds the `resp_` id prefix, `store: true`, and OpenAI-only
echo fields). The `OpenAIResponses*` ops classes are aliases of the
`OpenResponses*` ones.

Base formats are vendor-neutral: a bare `type: <base>` provider (e.g.
`type: open_responses`) carries no implied host or credential env var and must
set `base_url` explicitly. Vendors ship a `<vendor>--<base>` shim (e.g.
`openrouter--open_responses`, `openai--open_responses`) whose
`connection.base_url` / `connection.api_key_env` supply the defaults; the
vendor URL for a bare format is offered in the admin UI as a *placeholder*
hint only.

`ChatCompletionsConverter` (module `converters/llm/chat_completions/`) is the
canonical name of the Chat Completions converter; the `OpenAIChat*` classes and
the `converters/openai_chat/` module are retained as deprecated aliases.

Each chat converter implements bidirectional conversion (request/response) and
streaming. Converters are provider-agnostic — provider-specific quirks are
handled by the **shim layer**.

### Format catalog

`formats/catalog.py` holds one `FormatSpec` per `(model_type, format)` — the
single home for per-format facts: wire family, auth builder key, URL
templates, SSE formatter key, usage shape, chat route path, default upstream
path, models-listing path, picker order and detection order.  The model
vocabulary, the gateway's provider tables, the SSE / error-envelope /
usage-shape maps, `auto_detect.detect_provider` and the model-type descriptor
format lists and chat routes are all derived from it.  Adding a format means
adding a record, its converter binding (`converters/registry.py`) and its body
detector (`formats/detect.py`) — nothing else.

### Decision converters

Decision is a separate model paradigm (like embedding and rerank) for
probabilistic structured decisions — state + typed questions → answers
with calibrated probabilities.

| Converter | API Standard | Module |
|-----------|-------------|--------|
| `systemone` | System One (Jev / TypeSafe) | `converters/decision/` |
| `openai_decisions` | OpenAI Decisions API | `converters/decision/` |

IR question types: `assertion` (a proposition → P(true) ∈ [0,1]),
`choice` (a categorical proposition), `score` (an ordinal proposition).

### Embedding and rerank converters

Embedding and rerank each have one converter per wire dialect, all under one
module per family (their dialects differ in parameters, not in message
structure):

| Model type | Formats | Module |
|-----------|---------|--------|
| `embedding` | `openai`, `cohere`, `jina`, `voyage`, `google_generate` | `converters/embedding/` |
| `rerank` | `jina`, `cohere`, `voyage` | `converters/rerank/` |

They are reached through `EmbeddingConversionPipeline` /
`RerankConversionPipeline` (or `get_embedding_converter(format)`), not through
`auto_detect.detect_provider()`: the gateway's model-type descriptors own
source detection for embedding, rerank and decision routes
(`gateway/source_formats.py`), and an embedding body cannot be attributed to
a vendor from its model name.  The decision detector also reads the route
(`/v1/decisions` vs `/v1/systemone`), so a provider's default path follows
the format it declares.

### Shim layer

A **ProviderShim** is a lightweight identity card that declares which base
converter a provider uses, plus connection defaults and field-level transforms.
This avoids per-provider conversion code — converters stay generic, shims
declare the differences.

| Component | File | Purpose |
|-----------|------|---------|
| `ProviderShim` / `ModelShim` | `shims/provider_shim.py` | Data classes + registry |
| `Transform` primitives | `shims/transforms.py` | `strip_fields`, `rename_field`, `set_defaults` |
| Built-in shims | `shims/providers/<name>/provider.yaml` | One shim per vendor/format pair (33 built in) |
| External shim sources | `llm_rosetta_gateway/shims/` | Directory/archive loaders, standalone catalog, scheduled removals |

### Gateway

The gateway is a standalone HTTP proxy server that accepts requests in any
supported format and forwards them to any configured upstream provider.

| Module | Purpose |
|--------|---------|
| `gateway/app/` | Route handlers (chat/models/health/typed), app factory, tasks |
| `gateway/proxy/` | Core proxy engine: errors, metadata, fidelity, preflight, engine, streaming |
| `gateway/config/` | Config package: JSONC loading, locking, parsing, resolution |
| `gateway/providers.py` | Provider info, auth headers, key rotation |
| `gateway/admin/` | Admin UI, metrics, request logging, persistence |
| `gateway/observability/` | Telemetry: metrics, request log, SQLite persistence, ops log, error dumps, capture, maintenance, retention, sanitizing, profiler state |
| `gateway/ops/` | Unified operations audit layer (OpsBase + typed operation subclasses) |
| `gateway/routing/` | Candidate selection: the route container plus the pluggable strategies (weighted round-robin, affinity) |
| `gateway/middleware/` | Auth, rate limiting, error format, hop limit, headers, etc. |

### IR type system

Typed dataclasses for the intermediate representation live in `types/ir/`.
Four paradigms have their own IR types: chat (`request.py`, `response.py`),
embedding (`embedding.py`), rerank (`rerank.py`), and decision (`decision.py`).
Provider-specific types (for documentation, not runtime) in
`types/anthropic/`, `types/google/`.

The logic that operates on IR values lives in `ir/`:

| Sub-package | Holds |
|-------------|-------|
| `ir/primitives/` | Body/IR transforms plus their stateless helpers |
| `ir/tools/` | Stateless tool helpers (intrinsic kinds, call ids, error markers) |
| `ir/passes/` | Capability passes (vision, reasoning, custom tools, names) |

`ir/primitives` and `ir/tools` must not import `converters`; a pass may use the
converter *base* contracts and the shim contract. `llm_rosetta.transforms` and
`llm_rosetta.capabilities` are facades, and the old
`converters/base/{helpers,tools}/*` submodule paths re-export from `ir/`.

## Repository layout

Two distributions live side by side in one repository (epic #1127):

```
packages/
├── llm-rosetta/             # core distribution (import root `llm_rosetta`)
│   ├── pyproject.toml
│   └── src/llm_rosetta/
│       ├── __init__.py              # Public API: convert(), get_converter_for_provider()
│       ├── auto_detect.py           # Provider auto-detection (delegates to formats.detect)
│       ├── tool_ops.py              # Cross-provider tool call utilities
│       ├── formats/                 # Format catalog: one FormatSpec per (model_type, format)
│       │   ├── spec.py              # The record type
│       │   ├── catalog.py           # The records + BY_PAIR
│       │   └── detect.py            # Priority-ordered body detectors
│       ├── converters/              # one directory per (model_type, format)
│       │   ├── base/                # Abstract base + ConversionContext
│       │   ├── llm/                 # chat: one package per wire format
│       │   │   ├── chat_completions/
│       │   │   ├── open_responses/
│       │   │   ├── messages/
│       │   │   ├── google_generate/
│       │   │   └── google_interactions/
│       │   ├── embedding/
│       │   ├── rerank/
│       │   └── decision/            # Decision paradigm (TypeSafe System One)
│       ├── shims/                   # Provider/model identity cards + transforms
│       │   ├── provider_shim.py
│       │   ├── transforms.py
│       │   └── providers/           # One provider.yaml per vendor/format pair
│       ├── ir/                      # IR logic: primitives/, tools/, passes/
│       ├── types/                   # Typed IR and provider-specific types
│       │   └── ir/                  # IR dataclasses (messages, parts, tools, stream)
│       ├── transforms/              # Facade over ir.primitives (compat)
│       ├── capabilities/            # Facade over ir.passes (compat)
│       ├── observability/usage/     # The meaning of a usage report (no store)
│       ├── gateway/                 # Deprecated shim: aliases llm_rosetta_gateway,
│       │                            #   removed in 0.16.0 (#919)
│       └── _vendor/                 # validate, yaml, profiler (DO NOT EDIT)
└── llm-rosetta-gateway/     # gateway distribution (import root `llm_rosetta_gateway`)
    ├── pyproject.toml
    └── src/llm_rosetta_gateway/
        ├── app/                 # package: factory, routes_llm/models/health/
        │                        #   typed, tasks, _state
        ├── proxy/               # package: errors, metadata, fidelity, preflight,
        │                        #   engine, streaming, stream_merge
        ├── config/              # config package: io, loader, schema, core,
        │                        #   parse, typed, settings, resolve
        ├── providers.py
        ├── shims/               # External shim sources, catalog, scheduled removals
        ├── shim_sources.py      # Config `shims` section → shim registry
        ├── model_types.py, keystore.py, logging.py
        ├── routing/, maintenance/  # Selection strategies; config/key/backfill steps
        ├── cli.py, banner.py, __main__.py
        ├── middleware/          # Auth, rate limiting, error format, headers, etc.
        ├── pipelines/           # typed.py shell + embeddings/rerank/decision specs
        ├── transport/           # Upstream HTTP transport
        ├── ops/                 # Unified operations audit layer
        │   ├── base.py          # OpsContext, OpsBase ABC
        │   ├── proxy.py         # OpsProxyRequest (hot path)
        │   ├── data.py          # Data mutation ops
        │   ├── security.py      # Auth/session ops
        │   └── keys.py          # API key management ops
        ├── observability/       # Telemetry: metrics, request log, persistence,
        │                        #   ops log, error dumps, capture, maintenance,
        │                        #   retention, sanitize, profiling
        │   └── usage/cli.py     # The llm-rosetta-usage console script
        ├── admin/               # Admin panel over the observability layer
        │   └── routes/          # one module per resource; routes/config/ is
        │                        #   split into payload/providers/models/
        │                        #   model_providers/settings/shims/upstream/bulk
        └── _vendor/             # httpserver, httpclient, sse, ratelimit,
                                 #   aiosqlite, jsonx (DO NOT EDIT)

tests/
├── converters/              # Per-converter test suites
├── gateway/                 # Gateway unit tests
├── integration/             # E2E tests (require API keys)
├── test_types/              # IR type system tests
├── test_core_import_boundary.py   # core must not import the gateway
├── test_package_split.py    # the deprecated llm_rosetta.gateway alias
├── test_shims.py
├── test_transforms.py
├── test_auto_detect.py
└── test_tool_ops.py

scripts/
├── bench_pipeline.py             # Conversion pipeline benchmarks
├── rosetta-test-*.sh             # Agent integration test runners
├── run_gateway_integration.sh    # Gateway integration test runner
├── dev/                          # Developer tools (not user-facing)
│   ├── extract_google_types.py
│   ├── extract_tool_choice_types.py
│   ├── scan_sdk_types.py
│   ├── merge-pr.sh
│   ├── public_api_diff.py        # public-API break check (knows both layouts)
│   └── probe/                    # Upstream API probing
│       └── probe_argo.py         # Unified ARGO model probe
└── integration/                  # Integration test scripts
    ├── test_roundtrip_inflation.py
    └── test_roundtrip_live.py

docs_en/, docs_zh/           # Documentation (git worktrees, orphan branches)
docker/                      # Dockerfile for gateway image
examples/                    # Usage examples
```

## Setup and commands

```bash
conda activate llm-rosetta
pip install -e "packages/llm-rosetta[all]" -e packages/llm-rosetta-gateway
```

Run `make help` for all targets. Key ones:

```bash
make lint          # ruff check + ruff format --check
make lint-fix      # ruff check --fix + ruff format
make test          # pytest tests/ --ignore=tests/integration -v
make build         # python -m build
make push          # twine upload
make build-docker  # Build gateway Docker image
make bench         # Run conversion pipeline benchmarks
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
    "Read packages/llm-rosetta/src/llm_rosetta/shims/provider_shim.py and write a summary to /tmp/test-output.md",
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

1. **Bump version** in `packages/llm-rosetta/src/llm_rosetta/__init__.py`
   (and `packages/llm-rosetta-gateway/src/llm_rosetta_gateway/__init__.py`; the
   release workflow verifies both against the tag). On a new **minor**, also
   raise the gateway's dependency range in
   `packages/llm-rosetta-gateway/pyproject.toml` (`llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`).
   The `.dev0` lower bound is what lets a dev pair on TestPyPI resolve; when
   the **final** `X.Y.0` ships, drop it (`llm-rosetta>=X.Y.0,<X.(Y+1)`), or a
   released gateway can pull a development core from PyPI.
2. **Update changelogs** — move `[Unreleased]` entries into a versioned section
   (`## vX.Y.Z — YYYY-MM-DD`) in both `docs_en/docs/changelog.md` and
   `docs_zh/docs/changelog.md`. Commit and push both doc worktrees.
3. **Commit and tag**:
   ```bash
   git add packages/llm-rosetta/src/llm_rosetta/__init__.py \
           packages/llm-rosetta-gateway/src/llm_rosetta_gateway/__init__.py
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

## Adding a new converter (checklist)

When adding a converter for a new API standard, **every** item below
must be completed before the work is considered done.  Use this as a
gate — missing any item means the converter is not fully integrated.

1. **Converter module** — `converters/llm/<format>/` (chat) or the
   model-type package (embedding/rerank/decision), with `__init__.py`,
   `converter.py`, `content_ops.py`, `tool_ops.py`, `config_ops.py`,
   `message_ops.py`.  Converter class extends `BaseConverter` and
   implements all abstract hooks.
2. **Unit tests** — `tests/converters/<name>/` mirroring the ops modules.
   Must include round-trip tests (provider→IR→provider).
3. **Public API exports** — add the converter class to both
   `converters/__init__.py` and the top-level `__init__.py` `__all__`.
4. **Catalog record** — add one `FormatSpec` to
   `formats/catalog.py` (model_type, format, wire_family, detect +
   detect_priority, picker_order, auth, url_template, sse, usage_shape,
   route_path / default_path, models_path) and register its detector in
   `formats/detect.py::DETECTORS`.  Bind the converter class in
   `converters/registry.py::_CONVERTER_CLASSES`.  Everything else derives
   from the catalog: `FORMATS_BY_TYPE`, the provider tables, the SSE,
   error-envelope and usage-shape maps, the detection order, and the
   model-type descriptor format lists and chat routes.  The registry's
   import-time check refuses a catalog entry with no converter binding.
   Add the name to the `ProviderType` literal in `auto_detect.py`.
5. **Provider shim** — create `shims/providers/<name>/provider.yaml` with
   `model_type: llm` and `format: <format>`.  The loader validates the
   `(model_type, format)` pair against the converter registry; `base:` is the
   deprecated alias for `format:`.
6. **Gateway route** — register a route handler in
   `llm_rosetta_gateway/app/routes_llm.py`.
   Update `llm_rosetta_gateway/proxy/` for error responses, stream detection,
   stream flag injection, preflight body, and usage extraction.
7. **CLAUDE.md** — update the converter table and repository layout
   in this file.
8. **Documentation** — update `docs_en/` and `docs_zh/` (both in the
   same task): changelog, converters guide, API reference.  Push both
   worktrees.
9. **CI smoke test** — if applicable, add an import check in
   `.github/workflows/ci.yml`.
10. **`test_public_api.py`** — add the new class to `EXPECTED_EXPORTS`.
11. **`test_shim_loader.py`** — add to the expected shim set and the
    `(model_type, format)` mapping.

## Escalation

- Converter output mismatch → check IR types in `types/ir/`, then the
  converter's `*_ops.py` modules
- Shim/transform issue → check `shims/providers/` and `shims/transforms.py`
- Gateway config issue → check `gateway/config/` resolution order
  (`loader` → `core` → `parse`/`typed`/`settings`, then `resolve`)
  (shim → type → name fallback)
- Ops audit issue → check `gateway/ops/base.py` execute flow, verify
  OpsContext wiring in `gateway/admin/__init__.py` (`setup_admin`)
- `_vendor/` issues → never fix in-place; update upstream, re-vendor
- Integration test failure → likely missing API keys or network; these are
  excluded from `make test` by default
- Test failure after 3 attempts → stop, report full output
- Never: delete files to fix errors, skip tests, modify `_vendor/` directly

## Files to never edit

- `packages/*/src/*/_vendor/**` — vendored dependencies, managed externally
  (`zerodep`; each distribution has its own `[tool.zerodep]` section)
- `docs_en/`, `docs_zh/` — separate git branches, edit inside the worktree only

## Adding a new model type (checklist)

When adding a new model type (like `decision`), **every** item below
must be completed.  The model type registry makes most of this
declarative — steps 1–3 are the core work, the rest is wiring.

1. **Handler** — describe the type with a `TypedHandler` in
   `gateway/pipelines/<name>.py` and serve it through
   `pipelines/typed.py::handle_typed_request`; a type supplies its
   pipeline, its resolver, its detector and any request/response
   hooks, and nothing else.  `handle_<name>(request, config)` stays the
   public entry point (a three-line wrapper).
2. **Register descriptor** — in `gateway/model_types.py`, add a
   `_<name>_pipeline()` lazy import wrapper and a
   `_register_<name>()` function that calls `register_model_type()`
   with a `ModelTypeDescriptor`.  Required fields: `name`, `routes`,
   `formats`, `pipeline`, `badge_class`, `config_format_key`,
   `config_path_key`, `default_format`, `default_path`, `icon_svg`,
   `color`.  Call it at module level after `_register_builtins()`.
3. **Config pool** — nothing to do: `GatewayConfig.__init__` walks the
   registry and builds `<name>_providers` / `<name>_provider_infos` /
   `<name>_models` plus `default_<name>_format` from the descriptor's
   config keys, reading both the unified `providers:` entries and the
   legacy `<name>_providers` section through one parser.
   `_distribute_typed_models()` is registry-driven too.
4. **Frontend test spec** — in `admin/js/test.js`, add a
   `registerTestType('<name>', { endpoint, payloads, renderResult })`
   call.  This is the **only** JS change needed — seg-controls,
   badges, provider modal, and fetch-models are all data-driven
   from the registry metadata.
5. **i18n (optional)** — add `label.<name>`, `label.<name>Endpoint`,
   `label.<name>Path` to both language dictionaries in
   `admin/js/i18n.js`.  If omitted, the `t()` fallback auto-
   capitalizes the type name (e.g. `"sentiment"` → `"Sentiment"`).
6. **Unit tests** — add tests for the handler module and config
   pool resolution.
7. **CLAUDE.md** — update the converter/model-type tables in this
   file.
8. **Documentation** — update `docs_en/` and `docs_zh/` (both in
   the same task): changelog, model types guide.  Push both
   worktrees.

### What you do NOT need to change

Thanks to the model type registry and data-driven admin UI:

- `llm_rosetta_gateway/app/` — routes auto-registered from descriptor
- `admin/admin.html` — no HTML changes (checkboxes, endpoint
  sections, seg-controls, fetch-models radios all generated by JS)
- `admin/css/components.css` — badge/dot colors injected
  dynamically from the `color` field in the descriptor
- `admin/js/providers.js` — capability detection, badge rendering,
  save logic all loop over registry metadata
- `admin/js/models.js` — type filter, badges, test menus all
  data-driven
- `admin/js/state.js` — icons populated from `icon_svg` metadata
- `admin/js/fetch-models.js` — type radios generated from metadata

## Adding a new Ops subclass (checklist)

When adding a new operation to the gateway ops layer, **every** item
below must be completed.

1. **Subclass `OpsBase`** in the appropriate module under
   `gateway/ops/` (`data.py`, `security.py`, `keys.py`, or a new
   file).
2. **Set class variables**: `event_type`, `severity`, `source`.
3. **Implement abstract methods**: `_run()`, `_message(result)`,
   `_details(result)`.  Override `_record()` only if the op should
   not write to `ops_log` (e.g. `OpsProxyRequest` writes to
   `request_log` + `metrics` instead).
4. **Add event constant** — add an `EVENT_*` constant to
   `llm_rosetta_gateway/observability/ops_log.py` and include it in
   `ALL_EVENT_TYPES`.
5. **Wire into route handler** — construct the Ops subclass with
   `OpsContext` and `await op.execute()`.
6. **Export** — add the class to `gateway/ops/__init__.py` `__all__`.
7. **Unit tests** in `tests/gateway/ops/`.
8. **Update CLAUDE.md** ops module description if a new file was
   added.

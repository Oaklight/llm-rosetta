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

One chat converter per API standard:

| Converter | API Standard | Module |
|-----------|-------------|--------|
| `chat_completions` | OpenAI Chat Completions | `converters/llm/chat_completions/` |
| `open_responses` | Open Responses spec (vendor-neutral) | `converters/llm/open_responses/` |
| `messages` | Anthropic Messages API | `converters/llm/messages/` |
| `google_generate` | Google generateContent API | `converters/llm/google_generate/` |
| `google_interactions` | Google Interactions API | `converters/llm/google_interactions/` |
| `acp` | Agent Client Protocol (a local agent CLI over stdio) | `converters/llm/acp/` |

`acp` is an **upstream-only** lane: an agent process is the model endpoint, so
the converter writes a turn body (system string, ``[Role]``-labelled prompt
blocks, model) and assembles replies from the agent's ``session/update``
notifications.  It implements no source direction — no client speaks ACP to
the gateway.  A request's tools travel over MCP, not the prompt: the transport
serves them from a per-session loopback endpoint (named
``llm_rosetta_tools``) given to the agent in ``session/new``, turns a
``tools/call`` into an IR tool call, and resumes the pending turn when the next
request carries the result.  The agent's own built-in tools are refused.  The
matching transport is `transport/acp.py`, chosen per provider by
`connection.transport`; its MCP bridge is `transport/acp_mcp.py`, its session
ledger is `transport/acp_ledger.py`, and the vendored protocol client is
`_vendor/acp.py`.

`OpenResponsesConverter` implements the vendor-neutral Open Responses spec (the
base of both Responses classes); `OpenAIResponsesConverter` derives from it as
the OpenAI profile (adds the `resp_` id prefix, `store: true`, and OpenAI-only
echo fields). The ops classes and the stream context carry the
`OpenResponses*` names only; the `OpenAIResponses*` ops aliases were removed in
0.14.0.

Base formats are vendor-neutral: a bare `type: <base>` provider (e.g.
`type: open_responses`) carries no implied host or credential env var and must
set `base_url` explicitly. Vendors ship a `<vendor>--<base>` shim (e.g.
`deepseek--open_responses`, `openai--open_responses`) whose
`connection.base_url` / `connection.api_key_env` supply the defaults; the
vendor URL for a bare format is offered in the admin UI as a *placeholder*
hint only.

`ChatCompletionsConverter` (module `converters/llm/chat_completions/`) is the
name of the Chat Completions converter.  The 0.13.0 names (`OpenAIChat*`,
`AnthropicConverter`, `GoogleGenAIConverter`, `GoogleConverter`) and the compat
packages (`converters/openai_chat/`, `converters/anthropic/`,
`converters/openai_responses/`, `converters/google_genai/`,
`converters/google_generate/`, `converters/google_interactions/`,
`converters/embedding/`) were removed in 0.14.0; do not reintroduce them.

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
| `systemone` | System One (Jev / TypeSafe) | `converters/decisions/` |
| `openai_decisions` | OpenAI Decisions API | `converters/decisions/` |

IR question types: `assertion` (a proposition → P(true) ∈ [0,1]),
`choice` (a categorical proposition), `score` (an ordinal proposition).

### Embedding and rerank converters

Embedding and rerank each have one converter per wire dialect, all under one
module per family (their dialects differ in parameters, not in message
structure).  The model-type vocabulary is `llm`, `embeddings`, `rerank`,
`decisions`; the category follows its endpoint's spelling.  `embedding` and
`decision` were retired in 0.14.0: every boundary that reads a model type
rejects them with the replacement and the `llm-rosetta gateway migrate` command.

| Model type | Formats | Module |
|-----------|---------|--------|
| `embeddings` | `openai`, `cohere`, `jina`, `voyage`, `google_generate`, `minimax` | `converters/embeddings/` |
| `rerank` | `jina`, `cohere`, `voyage` | `converters/rerank/` |

They are reached through `EmbeddingConversionPipeline` /
`RerankConversionPipeline` (or `get_embedding_converter(format)`), not through
`auto_detect.detect_provider()`: the gateway's model-type descriptors own
source detection for embedding, rerank and decision routes
(`model_types/source_formats.py`), and an embedding body cannot be attributed to
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
| `ProviderShim` and the declaration data classes | `contracts/shim.py` | What a shim declares (plus `load_providers_from_dir`, the plugin registration call) |
| Registry | `shims/provider_shim.py` | `register_shim`, `get_shim`, ...; re-exports the data classes until 0.15.0 |
| `AuthExchange` | `contracts/auth.py` | Credential-exchange contract a shim ships in `auth.py` |
| `QuotaProbe` | `contracts/quota.py` | Allowance-probe contract a shim ships in `quota.py` |
| `Transform` types and primitives | `contracts/transform.py` | `NamedTransform`, `strip_fields`, `hoist_late_system_messages`, ... (implementations in `ir/primitives/`) |
| Loader | `shims/_loader/` | Reads a shim tree: directory walk, `provider.yaml` parsing, sidecars, vendor manifests, `provider.schema.json` (internal) |
| Built-in shims | `shims/builtin/<vendor>/<type>/provider.yaml` | One shim per vendor/format pair, grouped by model type |
| Vendor manifest | `shims/builtin/<vendor>/provider.yaml` | Optional per-vendor defaults every leaf inherits: `connection:` (the credential) plus the vendor's own `logo` / `group` / `group_label`; `auth.py` sits beside it |
| Tree contract | `devtools/_shim_template/` | The path rules (`<vendor>[/<lane>]/<type>[/<format>]/provider.yaml`), the file allow-list, and the skeleton `scripts/dev/new_shim.py` renders; `llm-rosetta ext check` reports an extension folder (its tree included) that departs from it |
| External shim sources | `llm_rosetta_gateway/shims/` | Extension-index installs (`manifest_sources/`: plan, selection, freshness, install, status), source catalogue |

### Gateway

The gateway is a standalone HTTP proxy server that accepts requests in any
supported format and forwards them to any configured upstream provider.

| Module | Purpose |
|--------|---------|
| `app/` | Route handlers (`routes_llm.py`, `routes_models.py`, `routes_health.py`, `routes_typed.py`, plus the relayed `routes_compact.py` and `routes_count_tokens.py`), app factory, tasks |
| `proxy/` | Core proxy engine: the request stages both paths share (`prepare.py`), the non-streaming path (`engine.py`) and streaming, errors, failover, metadata, fidelity (and the same-format check in `fidelity_check.py`), preflight, and the instrumented request lifecycle every proxy route runs through (`lifecycle.py`) |
| `config/` | Config package: JSONC loading, locking, parsing, resolution; `providers.py` is the provider registry and factory that resolve shim config into runtime `ProviderInfo` (key rotation, auth headers) |
| `pipelines/` | Typed request handlers (embeddings, rerank, decision) behind `handle_typed_request` |
| `model_types/` | The model-type registry (`__init__.py`): descriptors, routes, formats, config keys; `source_formats.py` holds the source detectors its typed-route descriptors carry (embedding, rerank, decision) |
| `shims/` | External shim sources (`shim_sources.py`), the `manifest_sources/` package (plan, selection, freshness, install, status; a release is a wheel that core's `install_wheel` places in `<data dir>/extensions`), manifest-source catalogue, the admin's extension listing (`extension_view.py`); the `shims` config section |
| `routing/` | Candidate selection: the route container plus the pluggable strategies (weighted round-robin, affinity) |
| `jobs/` | Background jobs: the envelope (`model.py`), store, queue, worker, handlers |
| `quota/` | Provider quota probing and quota-aware credential rotation: `service.py` runs the probes (callers import every quota function from it), `allowance.py` answers what routing and admin read, `signals.py` records what live requests report |
| `maintenance/` | Maintenance steps: backfills, config, dumps, keystore, migrations, startup, the retired-name guard (`names.py`); `migrate_0140.py` is the `migrate` command (`llm-rosetta gateway migrate`) |
| `middleware/` | Auth, rate limiting, error format, hop limit, headers |
| `transport/` | Upstream transports: `http` (HTTP/SSE), `acp` (a local agent CLI over stdio JSON-RPC; its body/event codec is `acp_chunks.py`); a provider picks one with `connection.transport`. An extension adds transports scoped to itself (the `claude` extension ships `claude_cli`): its manifest declares them and `llm_rosetta_ext.<id>.transport` exposes `TRANSPORTS`; `registry.py` resolves a shim's name from its own extension, then the built-ins, then the extensions it requires, and builds each once at app creation |
| `ops/` | Unified operations audit layer (OpsBase + typed operation subclasses) |
| `admin/` | Admin panel and its routes, over the observability layer; reaches the gateway through `admin_api.py` (see *Admin boundary*) |
| `admin_api.py` | The `AdminContext` protocols admin depends on; `create_app` exposes the adapter (`app/admin_context.py`) as `app.admin` |
| `observability/` | Telemetry: the gateway logger and its formatters (`logging.py`), metrics, request log, SQLite persistence, ops log, error dumps, capture, retention, sanitizing, profiler state; `observability/usage/cli.py` is the `usage` command (`llm-rosetta gateway usage`) |
| `cli/` | The gateway's two commands as `CommandSpec`s that `main()` registers and `pyproject.toml` declares under `llm_rosetta.commands`: `gateway` (`llm-rosetta gateway <verb>`, verbs `serve`, `init`, `add`, `credential`, `password`, `db`, `usage`, `migrate`, held in `_VERBS`) and the `serve` shortcut (`gateway`, `serve` are reserved extension names, `GATEWAY_COMMAND_NAMES`), the `llm-rosetta-gateway` console script (`llm-rosetta-gateway [args]` is `llm-rosetta serve [args]` when the arguments are empty or start with an option) (`cli/__init__.py`; `python -m llm_rosetta serve` replaces `python -m llm_rosetta_gateway`, which was removed), and the startup banner |
| `keystore/` | The SQLite-backed API key store (`__init__.py`); its import path is unchanged, so argo-proxy's `KeyContext` import still resolves |
| `updates.py` | Release check and self-update; stays at the root because it reads the changelog the release workflow writes beside the package root |
| `surface.py` | The core surface level the gateway asserts at startup (see *Host surface*) |
| `_vendor/` | Vendored modules (DO NOT EDIT) |

The packages form layers, bottom to top: foundation (observability, transport, shims, model_types and the root leaves), state (quota, routing, maintenance, keystore, jobs), policy (config, middleware), operations (ops), engine (proxy, pipelines), serving (app, admin, `admin_api.py`), entry (cli and the package entry points). `tests/gateway/test_layers.py` holds the table and fails on a module-level import into a higher layer, a new function-local one (keyed by file, target and the function it sits in), an unplaced package, or a new root module. Place a new package in a layer there; keep the root for entry points and boundaries.

### Host surface

`llm_rosetta.api` is the small public surface that the gateway imports (#1630): re-exports only, a literal `__all__` grouped by capability. The extension contracts live in `llm_rosetta.contracts` (#1657): `shim`, `transform`, `auth`, `quota`, `routing`, `usage`, plus `SURFACE_VERSION`, `IR_VERSION` and the sidecar hook names in its `__init__`. A shim imports only `llm_rosetta.contracts…` (its sidecars run during `import llm_rosetta`, before `api` can load); `llm_rosetta/_tiers.py` states each public module's tier. Within one minor it only gains names; a rename or removal keeps a deprecated alias for one minor. `SURFACE_VERSION` (an int) rises on a removal or a changed meaning, never on an addition; the gateway asserts a minimum of it at startup (`llm_rosetta_gateway/surface.py`). 0.14.0 is the release of the surface: the gateway pin stays `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)` and no bigger version bump is needed. Core must never import `llm_rosetta.api` (`tests/test_core_import_boundary.py`).

The gateway reaches core only through two prefixes: `llm_rosetta.api` and
`llm_rosetta.contracts` (the `contracts.usage` package, never its
submodules).  `tests/gateway/test_core_surface_boundary.py` enforces this
over every gateway file, `admin/` included.  The rule is strict: there is
no allowlist for the gateway source, and a private core name is never allowed.
A missing capability becomes an additive name on `llm_rosetta.api` with a test.
Gateway tests may import non-private internal modules but not private names;
`_TESTS_PRIVATE_ALLOWLIST` in the same file can only shrink.  See #1630.

### Admin boundary

`admin/**` imports from the gateway only `admin`, `admin_api` and
`_vendor.httpserver`.  New needs become a member of `AdminContext` in
`admin_api.py` plus its implementation in `app/admin_context.py`.  Imports that
predate the rule sit on `_LEGACY_ALLOWLIST` in
`tests/gateway/test_admin_import_boundary.py`; the list may only shrink
(regenerate with `python tests/gateway/test_admin_import_boundary.py`).  Only
`app/factory.py` imports `admin` (`setup_admin`, `register_admin_routes`).
Admin never imports `app._state`.

### IR type system

The IR is one package, `ir/`. Its data lives in `ir/model/`: the typed
dataclasses, type guards, validation, helpers and `IR_VERSION`. Each paradigm
has its own IR types: chat (`request.py`, `response.py`), embedding
(`embedding.py`), rerank (`rerank.py`), and decision (`decision.py`).
Provider shapes for reference (documentation, not runtime) live in
`types/messages/`, `types/openai/`, and `types/google/`.
`llm_rosetta.types.ir` shipped in 0.13.0 and stays as a deprecated forwarder to
`ir.model` until 0.15.0.

The logic that operates on IR values lives beside the model in `ir/`:

| Sub-package | Holds |
|-------------|-------|
| `ir/model/` | The data: IR dataclasses, guards, validation, helpers; imports only the standard library and `_vendor` |
| `ir/primitives/` | Body/IR transforms plus their stateless helpers |
| `ir/tools/` | Stateless tool helpers (intrinsic kinds, call ids, error markers) |
| `ir/passes/` | Passes over IR that need no shim (intrinsic history, tool names, description limits, request constraints) |

The direction is `ir/model <- ir/primitives, ir/tools <- ir/passes <- converters`.
Nothing in `ir/` may import `converters` (enforced by
`tests/test_ir_import_boundary.py`, which also holds the model and the
primitives to their layer); `ConversionContext` / `StreamContext` live
in `ir/context.py` and `converters/base/context.py` re-exports them. No module
in `ir/` imports `contracts` (`CORE_EDGES` in `tests/test_core_layering.py`): a
pass does not use the shim contract, and the passes that read a shim's
declarations (reasoning, vision, custom and intrinsic tools) are
`pipeline/enforcement.py`. `llm_rosetta.contracts.transform` re-exports the
transform types and primitives shims use, and the package-level
`converters.base.{helpers,tools}` names re-export from `ir/`.

## Repository layout

The repository holds the core distribution and the gateway distribution
(epic #1127).

| Distribution | Import root | Owns |
|--------------|-------------|------|
| `packages/llm-rosetta/` | `llm_rosetta` | Format catalog, converters, IR, shims — how a request is converted |
| `packages/llm-rosetta-gateway/` | `llm_rosetta_gateway` | Routes, config, proxy engine, admin — how a request is served |

Read the tree with `ls` when a path is needed. This section records
module-level ownership and the conventions that the filesystem cannot show.
Do not record counts here — a count goes stale as soon as a file is added.
Count from the tree when a number is needed.

Core modules (`packages/llm-rosetta/src/llm_rosetta/`):

| Module | Holds |
|--------|-------|
| `__init__.py` | Public API: `convert()`, `get_converter_for_provider()` |
| `auto_detect.py` | Provider auto-detection (delegates to `formats/detect.py`) |
| `tool_ops.py` | Cross-provider tool-call utilities |
| `model_vocabulary.py` | The model-type and format vocabulary |
| `retired_names.py` | The table of renamed provider types, shims and model types, and the error each one raises (an error table, never an alias) |
| `conversion_warnings.py` | The warning sink for lossy conversions |
| `formats/` | `spec.py` (record type), `catalog.py` (records and `BY_PAIR`), `detect.py` (body detectors) |
| `converters/` | One package per `(model_type, format)`; `base/` is the abstract contract; `registry.py` binds formats to classes |
| `ir/` | The IR: `model/` (data), `primitives/`, `tools/`, `passes/` (logic) |
| `types/` | Provider shapes for reference in `types/messages/`, `types/openai/`, `types/google/`; `types/ir/` is the deprecated 0.13.0 forwarder to `ir/model/` |
| `contracts/` | The extension contracts: `shim.py`, `transform.py`, `auth.py`, `quota.py`, `routing.py`, `command.py` (a subcommand a distribution adds to `llm-rosetta`), `usage/` |
| `cli.py` | The `llm-rosetta` dispatcher (`python -m llm_rosetta`): it owns `ext` (`list`, `show`, `install`, `remove`, `update`, `doctor`, `new`, `check`; `extensions/_cli.py`) and `help`, runs every other command from the `llm_rosetta.commands` entry point group or `register_command`, and runs an extension's command as `llm-rosetta <id> <verb>` (the id with hyphens; the manifest's `provides.commands` maps each verb to its summary and `llm_rosetta_ext.<id>.commands` exposes `COMMANDS`; `extensions/_commands.py` reads the areas, verbs and summaries from manifests without importing the extension, so help works in a build that strips docstrings); an unhandled exception is one line and exit 1 (`LLM_ROSETTA_DEBUG` raises it); it never starts a server |
| `extensions/` | Discovery, the loader, the stdlib wheel installer (`_install.py`: `install_wheel`, `inspect_wheel`, `fetch_bytes`; it refuses a wheel whose `Requires-Python` this interpreter fails), the extension index (`_index.py`: `ext install <id>` and `ext update` read an `index.json` of release wheels with their `sha256`, from `--index`, `$LLM_ROSETTA_EXTENSION_INDEX` or the official repository), the child-process import probe `ext doctor` runs (`_probe.py`; in process where no Python interpreter can be spawned), and the `ext` commands (`_cli.py`: `--json` on `list`/`show`/`doctor`, `--dry-run` on `install`/`remove`/`update`) |
| `shims/` | `provider_shim.py` (the registry), `_loader/` (the code that reads a shim tree), `builtin/` (the bundled vendor data: one manifest per vendor/model-type/format), and `providers/` (the 0.13.0 loader path: forwards its names, deprecated, until 0.15.0) |
| `pipeline/` | Chat and typed conversion pipelines |
| `fidelity/` | Fidelity baseline and checker |
| `profiling/` | The profiler |
| `observability/` | Deprecated 0.13 package: alias finder for the telemetry that moved to the gateway |
| `_tiers.py` | The tier of each public module (user, extension, compat, removed) |
| `gateway/` | Deprecated forward shim to `llm_rosetta_gateway`; it warns on import and will be removed in 0.15.0 |
| `_vendor/` | Vendored modules (DO NOT EDIT) |

Other top-level paths:

| Path | Holds |
|------|-------|
| `tests/` | Unit tests; `tests/integration/` needs live API keys and is excluded from `make test` |
| `tools/conformance/` | Developer-only probe harness, never shipped in a wheel |
| `scripts/` | Benchmarks, agent integration runners, and developer tools (`dev/new_shim.py` scaffolds a shim; `dev/check_shim_catalog.py` compares the shim sources with the published repos) |
| `docker/`, `examples/` | Gateway image; usage examples |
| `docs_en/`, `docs_zh/` | Documentation worktrees on orphan branches |

Conventions the filesystem does not show:

- `llm_rosetta/gateway/` forwards to `llm_rosetta_gateway`; it is deprecated
  and will be removed in 0.15.0. Do not add code to it.
- Provider-type, shim and model-type names that were renamed have no alias.
  A retired spelling raises, naming its replacement and the
  `llm-rosetta gateway migrate` command (`retired_names.py`); to retire another name,
  add a row there. Do not add a translating table.
- The 0.13.0 converter compat packages and class aliases are removed.
  The packages are `converters/{embedding,openai_chat,openai_responses,anthropic,
  google_genai,google_generate,google_interactions}/`; the canonical homes are
  `converters/embeddings/` and `converters/llm/<format>/`.
- Core must not import the gateway distribution
  (`tests/test_core_import_boundary.py`).
- Core vendors follow the placement rule (#1660): the model maker's own API
  with a static API key — no `auth.py`, and a `quota.py` probe that its own
  shims select (`tests/test_shim_placement.py`).  Every other vendor lives in
  `Oaklight/llm-rosetta-ext-contrib`; `retired_names.MOVED_VENDORS` lists the
  ones that left core, and the gateway installs a listed vendor by itself when
  a config names its shim (`llm_rosetta_gateway/shims/moved.py`).
  `tests/fixtures/moved_vendors/` holds frozen copies for core's own tests
  (load them with the `moved_vendors` fixture); nothing ships them.
- `packages/*/src/*/_vendor/**` is vendored through `zerodep`; never edit it
  in place.
- `docs_en/`, `docs_zh/` are separate branches; edit inside the worktree only.

## Setup and commands

```bash
conda activate llm-rosetta
pip install -e "packages/llm-rosetta[all]" -e packages/llm-rosetta-gateway
```

The provider SDKs (`openai`, `anthropic`, `google-genai`) are not runtime
dependencies; only `tools/conformance` reads their types. Install them with
`-e "packages/llm-rosetta[conformance]"` when you run that harness (the
version they are pinned to lives in `tools/conformance/sdk-pins.txt`).

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

1. `make lint` and `make test` exit 0. The `ty` pre-commit hook (`scripts/dev/run_ty.py`) must pass without `SKIP=ty`; it matches CI from any worktree, so a failure is a real diagnostic
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

Reviews are tiered by risk, and a PR is checked against in-flight PRs before it
opens (`python scripts/dev/pr_overlap.py`). Keep one concern per PR, request
review only when CI is fully green, and fill in
`.github/pull_request_template.md`. The rules are in the "Pull requests"
section of `CONTRIBUTING.md`.

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

Docs are release-bound: write or update `docs_en/` and `docs_zh/` only when the
change is ready to ship and a version is being cut. While a change is
unfinished or unverified, do not edit the docs worktrees; record the pending
doc items in the PR or commit instead.

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

**Both language versions must land together whenever the docs are written.**
Never update only one language and leave the other for later. Docs are written
at release time, so both languages ship with the release; the code task itself
is done without them. The workflow is:

1. Make changes to `docs_en/` first (English is the source of truth).
2. In the same task, apply equivalent changes to `docs_zh/` before committing.
3. Commit and push both worktrees together.

Splitting the two languages across separate agents or separate sessions is not
allowed — it leads to drift and missed pages.

Commits in doc worktrees use `PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit` since
those branches have no `.pre-commit-config.yaml`.

## Adding a new converter (checklist)

When adding a converter for a new API standard, **every** item below
must be completed before the work is considered done.  Use this as a
gate — missing any item means the converter is not fully integrated.

1. **Converter module** — `converters/llm/<format>/` (chat) or the
   model-type package (embeddings/rerank/decisions), with `__init__.py`,
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
   Add the name to the `ProviderType` literal in `contracts/routing.py` (`auto_detect.py` imports it).
5. **Provider shim** — create `shims/builtin/<vendor>/llm/<format>/provider.yaml`
   (chat) or `shims/builtin/<vendor>/<type>/provider.yaml` (embeddings, rerank,
   decision) with `model_type: <type>` and `format: <format>`.  The loader
   validates the `(model_type, format)` pair against the converter registry and
   the directory name against `model_type`; `base:` was removed in 0.14.0 and the
   loader refuses it with a message that names `format:`.
6. **Gateway route** — register a route handler in
   `llm_rosetta_gateway/app/routes_llm.py`.
   Update `llm_rosetta_gateway/proxy/` for error responses, stream detection,
   stream flag injection, preflight body, and usage extraction.
7. **CLAUDE.md** — update the converter table in this file.
8. **Documentation** — update `docs_en/` and `docs_zh/` (both in the
   same task) when the release that ships the converter is cut:
   changelog, converters guide, API reference.  While the converter is
   still in development, record the pending doc items in the PR instead
   of editing the docs worktrees.
9. **CI smoke test** — if applicable, add an import check in
   `.github/workflows/ci.yml`.
10. **`test_public_api.py`** — add the new class to `EXPECTED_EXPORTS`.
11. **`test_shim_loader.py`** — add to the expected shim set and the
    `(model_type, format)` mapping.

## Adding an agent-CLI lane (checklist)

An ACP lane fronts an agent CLI.  It is an upstream **model provider**, not a
place for the agent to run its own tools on the gateway host.

1. **Client tools go over MCP, not the prompt.** The transport serves the
   request's tools from `AcpMcpServer` as a per-session loopback endpoint
   named `llm_rosetta_tools`, passed in `session/new`. A `tools/call` becomes
   an IR tool call; the next request's result answers the pending MCP call.
   Never stuff tool definitions into the prompt.
2. **A session's MCP server list is fixed.** ACP cannot add or remove an MCP
   server inside a session. The ledger stores a digest of the session's tools;
   a request whose tools differ (added, removed, disabled, or a refreshed
   schema) closes the session and replays the history into a new one. Order
   alone is not a difference. Do not share one endpoint between sessions:
   each session's token selects its own `AcpToolBridge`.
3. **The agent's own tools stay refused.** Keep the ACP client advertising no
   filesystem or terminal capability, refuse every non-bridged permission
   request, and fail a turn that starts a built-in tool. An operator
   `acp_argv` override is the operator's own risk and must be documented.
4. **Declare `tools` in the shim's capabilities only when the core implements
   the bridge.** A lane that cannot bridge must not advertise tools. A request
   that declares tools without a declared conversation id is refused with
   `400`, because its result could not be matched to the suspended turn.
5. **Conformance test.** One fake-agent test must prove the whole rule in one
   lane: a bridged request tool returns a tool call and resumes on its result;
   a built-in file read and a built-in shell command are refused with no host
   side effect.

## Escalation

- Converter output mismatch → check IR types in `ir/model/`, then the
  converter's `*_ops.py` modules
- Shim/transform issue → check `shims/builtin/`, `contracts/transform.py` and `ir/primitives/`
- Gateway config issue → check `config/` resolution order
  (`loader` → `core` → `parse`/`typed`/`settings`, then `resolve`)
  (shim → type → name fallback)
- Ops audit issue → check `ops/base.py` execute flow, verify
  OpsContext wiring in `admin/__init__.py` (`setup_admin`)
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
   `pipelines/<name>.py` and serve it through
   `pipelines/typed.py::handle_typed_request`; a type supplies its
   pipeline, its resolver, its detector and any request/response
   hooks, and nothing else.  `handle_<name>(request, config)` stays the
   public entry point.
2. **Register descriptor** — in `model_types/__init__.py`, add a
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
   the same task) when the release that ships the model type is cut:
   changelog, model types guide.  While the model type is still in
   development, record the pending doc items in the PR instead of
   editing the docs worktrees.

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
   `ops/` (`data.py`, `security.py`, `keys.py`, or a new
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
6. **Export** — add the class to `ops/__init__.py` `__all__`.
7. **Unit tests** in `tests/gateway/ops/`.
8. **Update CLAUDE.md** ops module description if a new file was
   added.

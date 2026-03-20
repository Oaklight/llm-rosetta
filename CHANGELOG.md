# Changelog

## Unreleased

### Fixed

- **OpenAI Responses streaming tool calls**: Fixed missing `item_id → call_id` mapping, argument accumulation across dual StreamContexts, and event ordering (`output_text.done → content_part.done → output_item.done`) for full Codex CLI compatibility (#56)
- **OpenAI Chat streaming tool calls**: Fixed missing `index` field in streamed `tool_calls` delta chunks (#57)
- **`stream_options` leak into Responses API**: Removed `stream_options` emission from `ir_stream_config_to_p()` in the Responses converter — the Responses API does not accept this Chat Completions-only field. This caused upstream rejection when Chat-format clients (Kilo, OpenCode) were proxied to the Responses API (#58)
- **Google REST-format tool definitions ignored**: `request_from_provider()` now checks both SDK format (`config.tools`) and REST top-level (`tools`, `tool_config`, `generationConfig`) — previously only SDK format was handled, silently stripping tools from gateway-proxied requests (#59)

## v0.2.0 (2026-03-17)

### Added

- **LLM-Rosetta Gateway**: REST gateway application for cross-provider HTTP proxying
- CLI entry point (`llm-rosetta-gateway`) and package structure for the gateway
- Gateway config auto-discovery at `./config.jsonc`, `~/.config/llm-rosetta-gateway/config.jsonc`, `~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` flag to open config file in `$EDITOR` (falls back to nano/vi/vim)
- `--version` / `-V` flag showing current version
- ASCII art startup banner with `--no-banner` to suppress
- `add provider <name>` subcommand for adding provider entries to config
- `add model <name>` subcommand for adding model routing entries
- `init` subcommand to create a template `config.jsonc` at the XDG default location
- **Gateway providers module** (`providers.py`): centralized provider definitions with auth-header builders, URL templates, default base URLs, and API key env-var names
- **API key rotation**: round-robin `KeyRing` for comma-separated API keys per provider
- **Proxy support**: global `server.proxy` and per-provider `proxy` config for HTTP/SOCKS proxies; CLI `--proxy` flag overrides config
- **Model listing endpoints**: `GET /v1/models` and `GET /v1beta/models` — enables `client.models.list()` across all three SDKs (#54)
- **Standalone API test scripts** (`llm_api_simple_tests/`): 20 test scripts covering simple query, multi-round chat, image, function calling, and comprehensive scenarios
- Makefile `test-integration` and `test-gateway` targets

### Changed

- Bumped minimum Python to 3.10+; migrated to stdlib `typing` (removed `typing_extensions`)
- Applied `ruff` formatter across the entire codebase
- Updated Makefile with `lint`, `test`, and `build` targets
- Added `ty` (type checker) configuration
- Configured `ruff` lint rules (`E`, `F`, `UP`) in `pyproject.toml`

### Fixed

- Streaming crash with Anthropic provider when usage tokens are `null`
- Gateway provider `base_url` validation — fail early with clear error on config typos
- Added `socksio` to gateway dependencies for SOCKS proxy support
- Resolved all `ty` type checker diagnostics in `src/` (31 → 0) and `tests/` (1506 → 0)
- Resolved all `ruff` lint violations in `src/` and `tests/`
- Google `thought_signature` preservation through gateway round-trips (#51)
- OpenAI Responses converter now handles all 3 `input` formats: bare string, shorthand list, and structured list
- Special characters in model names for Google GenAI routes (#53)
- Streaming SSE events crash in gateway mode (#52)
- Serialize dict `tool_result` content to JSON for Anthropic API

---
title: Migrating to 0.14
---

# Migrating to 0.14

Release 0.14.0 removes the deprecated import paths and class aliases that 0.13.0 kept for one release (epic [#919](https://github.com/Oaklight/llm-rosetta/issues/919)). A removed name fails with a plain `ImportError`, `ModuleNotFoundError` or `AttributeError`; the [retired-names](#retired-names) diagnostic covers provider, shim and model-type names, not Python import paths. This page lists what is gone, what replaces it, and what still works. To rewrite a gateway config and database, use [`llm-rosetta-migrate`](#llm-rosetta-migrate); the [upgrade recipe](#upgrade-recipe) gives the order.

The rows up to [Removed helper modules, gateway stubs and `tool_ops` aliases](#removed-shims) were checked against `origin/master` (commit `756f084169`). The rows from [Removed facades](#removed-facades) on were checked against the head of PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656) (commit `0a8c0be374`): each removed name fails to import or raises, and each replacement imports.

## Removed converter packages {#removed-converter-packages}

| Removed import path | Use instead |
| --- | --- |
| `llm_rosetta.converters.openai_chat` | `llm_rosetta.converters.llm.chat_completions` |
| `llm_rosetta.converters.openai_responses` | `llm_rosetta.converters.llm.open_responses` |
| `llm_rosetta.converters.anthropic` | `llm_rosetta.converters.llm.messages` |
| `llm_rosetta.converters.google_genai` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_generate` | `llm_rosetta.converters.llm.google_generate` |
| `llm_rosetta.converters.google_interactions` | `llm_rosetta.converters.llm.google_interactions` |
| `llm_rosetta.converters.embedding` | `llm_rosetta.converters.embeddings` |

The `google_generate` and `google_interactions` rows are the old top-level locations; the same names live under `converters.llm` now. Submodule paths follow their package: `converters.openai_chat.converter` becomes `converters.llm.chat_completions.converter`.

## Removed class names {#removed-class-names}

| Removed name | Use instead |
| --- | --- |
| `OpenAIChatConverter` | `ChatCompletionsConverter` |
| `AnthropicConverter` | `MessagesConverter` |
| `GoogleGenAIConverter` | `GoogleGenerateConverter` |
| `GoogleConverter` | `GoogleGenerateConverter` |

The four names are gone from `llm_rosetta`, `llm_rosetta.converters`, `llm_rosetta.converters.llm`, and from the canonical packages (`converters.llm.chat_completions`, `converters.llm.messages`, `converters.llm.google_generate`). Replacements import from the same places. `ChatCompletionsConverter` and `MessagesConverter` are also exported from `llm_rosetta.converters.llm.<format>`.

## Removed ops and stream-context aliases {#removed-ops-aliases}

| Removed name | Use instead |
| --- | --- |
| `OpenAIChatConfigOps`, `OpenAIChatContentOps`, `OpenAIChatMessageOps`, `OpenAIChatToolOps` | `ChatCompletions*Ops` in `converters.llm.chat_completions` |
| `AnthropicConfigOps`, `AnthropicContentOps`, `AnthropicMessageOps`, `AnthropicToolOps` | `Messages*Ops` in `converters.llm.messages` |
| `OpenAIResponsesConfigOps` | `OpenResponsesConfigOps` |
| `OpenAIResponsesContentOps` | `OpenResponsesContentOps` |
| `OpenAIResponsesMessageOps` | `OpenResponsesMessageOps` |
| `OpenAIResponsesToolOps` | `OpenResponsesToolOps` |
| `OpenAIResponsesStreamContext` | `OpenResponsesStreamContext` |

The `OpenResponses*` names live in `llm_rosetta.converters.llm.open_responses`. `OpenAIResponsesConverter` is a real class (the OpenAI profile of the Open Responses converter) and is **not** removed.

## Removed helper modules, gateway stubs and `tool_ops` aliases {#removed-shims}

Removed in [#1647](https://github.com/Oaklight/llm-rosetta/pull/1647). The shims had said 0.16.0; they are gone in 0.14.0. Importing a removed path raises `ModuleNotFoundError`.

| Removed import path | Use instead |
| --- | --- |
| `llm_rosetta.converters.base.helpers.cache_breakpoints` | `llm_rosetta.ir.primitives.cache_breakpoints` |
| `llm_rosetta.converters.base.helpers.image_limit` | `llm_rosetta.ir.primitives.image_limit` |
| `llm_rosetta.converters.base.helpers.system_message_hoist` | `llm_rosetta.ir.primitives.system_message_hoist` |
| `llm_rosetta_gateway.auth` | `llm_rosetta_gateway.middleware.auth` |
| `llm_rosetta_gateway.headers` | `llm_rosetta_gateway.middleware.headers` |
| `llm_rosetta_gateway.embeddings` | `llm_rosetta_gateway.pipelines.embeddings` |
| `llm_rosetta_gateway.migrations` | `llm_rosetta_gateway.maintenance.migrations` |

| Removed `llm_rosetta.tool_ops` function | Use instead |
| --- | --- |
| `to_openai_chat` | `to_chat_completions` |
| `from_openai_chat` | `from_chat_completions` |
| `to_openai_responses` | `to_open_responses` |
| `from_openai_responses` | `from_open_responses` |
| `to_anthropic` | `to_messages` |
| `from_anthropic` | `from_messages` |

`tool_ops.to_provider` and the other provider-keyed functions no longer accept the old provider spellings (`openai_chat`, `anthropic`, `google-genai` and the rest of the [provider types](#provider-types) table). They raise `ValueError`, and the message names the replacement, for example `'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions')`.

## Removed facades (P4) {#removed-facades}

Removed in [#1648](https://github.com/Oaklight/llm-rosetta/pull/1648). Each module was a star re-export of the canonical module on the right. None of them existed in 0.13.0, so no released import path breaks; importing one raises `ModuleNotFoundError` (an `ImportError`).

| Removed import path | Use instead |
| --- | --- |
| `llm_rosetta.converters.base.helpers.tool_group` | `llm_rosetta.ir.primitives.tool_group` |
| `llm_rosetta.converters.base.helpers.truncate` | `llm_rosetta.ir.primitives.truncate` |
| `llm_rosetta.converters.base.tools.call_unwind` | `llm_rosetta.ir.primitives.call_unwind` |
| `llm_rosetta.converters.base.tools.allowed_tools` | `llm_rosetta.ir.tools.allowed_tools` |
| `llm_rosetta.converters.base.tools.batch` | `llm_rosetta.ir.tools.batch` |
| `llm_rosetta.converters.base.tools.call_id` | `llm_rosetta.ir.tools.call_id` |
| `llm_rosetta.converters.base.tools.error_marker` | `llm_rosetta.ir.tools.error_marker` |
| `llm_rosetta.converters.base.tools.intrinsic` | `llm_rosetta.ir.tools.intrinsic` |
| `llm_rosetta.converters.base.tools.orphan_fix` | `llm_rosetta.ir.tools.orphan_fix` |
| `llm_rosetta.converters.base.tools.strict_mode` | `llm_rosetta.ir.tools.strict_mode` |
| `llm_rosetta_gateway.middleware.sanitize` | `llm_rosetta_gateway.observability.sanitize` |

`from llm_rosetta.converters.base.tools import batch`, `intrinsic`, and `helpers.tool_group` now raise `ImportError`: the three were module attributes, not names the package `__init__` exports. The names that `converters.base.helpers` and `converters.base.tools` export themselves are unchanged. Names the star re-exports used to leak but the package `__init__` never exported (`ToolGroup`, `group_tools`, `tool_call_group`, `report_loss`, `report_drop`, `ToolChoice`, `ToolRef`, `truncate_with_digest`, `history_native_calls`, `UNANSWERED_TOOL_CALL`) are now only under `llm_rosetta.ir.*`.

## Removed name aliases (P5) {#removed-name-aliases}

Removed in PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656). The alias tables are deleted; a retired provider-type or shim name is never translated. It raises through the [retired-names](#retired-names) table, names its replacement, and ends with the migrate command.

| Removed name | Use instead |
| --- | --- |
| `llm_rosetta.provider_names` (module) | pass the canonical name; there is nothing to normalise |
| `canonical_provider_name`, `canonical_shim_name`, `normalize_provider_name` | the canonical name from the [retired-names](#retired-names) tables |
| `warn_legacy_provider_names`, `warn_legacy_shim_names` | none; a retired name raises |
| `LEGACY_PROVIDER_ALIASES`, `LEGACY_SHIM_ALIASES` | `llm_rosetta.retired_names.RETIRED_NAMES` |
| `legacy_names` (also in `llm_rosetta.api`) | `llm_rosetta.retired_names.RETIRED_NAMES` |
| the legacy members of `ProviderType` | the canonical formats |

The names are gone from `llm_rosetta`, `llm_rosetta.api` and `llm_rosetta.fidelity`. The boundaries that raise are `get_converter_for_provider`, `ConversionPipeline`, `convert`, `convert_response`, the embedding converters, `ProviderShim`, `provider.yaml` formats, and the gateway config.

The gateway no longer rewrites a retired name when it loads a config, so a config that still spells one does not start. The error names the file and the key and ends with the migrate command when the gateway is installed, for example:

```text
llm-rosetta-gateway: error: config.jsonc: providers.p.type: 'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions'); run 'llm-rosetta-migrate --config <path>' (add --write to apply), see the migration guide (https://llm-rosetta.readthedocs.io/en/latest/guide/migrating-to-0.14/)
```

`llm-rosetta-migrate` ships with the gateway distribution. In a core-only install (the library without `llm-rosetta-gateway`) a retired-name error ends with `see the migration guide (<url>)` instead, because the command is not there to run. The sentence is `llm_rosetta.retired_names.MIGRATE_HINT`.

The process exits 1 and prints no traceback. In code, the exception is `RetiredConfigName`, a `ValueError` whose message is `<config file>: <json path>: <retired-names message>`.

The refusal comes before any write. A config at an older `config_version` is not rewritten, and gets no `.pre-migration.bak`, when it also spells a retired name: the file keeps its bytes until you run the migrate command. A per-face shim value (`embeddings_shim`, `decisions_shim`, and the same key inside an `embeddings:` block) is checked like `shim:`: a retired value such as `openai` or `openai_chat` raises with its path, `providers.<name>.embeddings_shim`. A misspelled value that is not a retired name gets no rename hint and disables the endpoint as an unknown shim.

## Removed shim declarations (P6) {#removed-shim-declarations}

Removed in PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656). `ProviderShim(...)` raises a `TypeError` naming the replacement, and the loader refuses a `provider.yaml` that still uses the old keys.

| Removed `ProviderShim` keyword | Use instead |
| --- | --- |
| `base` | `format` |
| `from_transforms` | `pre_ir_transforms` |
| `to_transforms` | `post_ir_transforms` |
| `default_base_url` | `connection=ConnectionConfig(base_url=...)` |
| `default_api_key_env` | `connection=ConnectionConfig(api_key_env=...)` |
| `models_path`, `default_path`, `model_id_field` | `connection=ConnectionConfig(<same name>=...)` |
| `supports_custom_tools` | `tools=ToolsConfig(custom_tools=...)` |
| `max_tool_description_length` | `tools=ToolsConfig(max_description_length=...)` |
| `tool_search_mode` | `tools=ToolsConfig(search_mode=...)` |
| `multimodal_tool_result` | `tools=ToolsConfig(multimodal_result=...)` |

| Removed `provider.yaml` key | Use instead |
| --- | --- |
| `base:` | `format:` and `model_type:` |
| `default_base_url:` | `connection.base_url` |
| `default_api_key_env:` | `connection.api_key_env` |
| a connection key written at the top level (`base_url`, `models_path`, `default_path`, `model_id_field`, `auth_header`, `keyless`, `token_command`, and the other `ConnectionConfig` fields) | the same name under `connection:` |
| `supports_custom_tools`, `max_tool_description_length`, `tool_search_mode`, `multimodal_tool_result`, `tool_groups`, `intrinsic_tools` | `tools.custom_tools`, `tools.max_description_length`, `tools.search_mode`, `tools.multimodal_result`, `tools.tool_groups`, `tools.intrinsic_tools` |
| `transforms.py` exporting `from_transforms` / `to_transforms` | `pre_ir_transforms` / `post_ir_transforms` |

```yaml
# before
name: deepseek
format: chat_completions
default_base_url: https://api.deepseek.com
default_api_key_env: DEEPSEEK_API_KEY

# after
name: deepseek
model_type: llm
format: chat_completions
connection:
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
```

Other removals in the same step:

| Removed name | Use instead |
| --- | --- |
| `llm_rosetta.shims.provider_shim.resolve_base` (and `llm_rosetta.resolve_base`) | `resolve_format` |
| `llm_rosetta.shims.transforms` (module) | `llm_rosetta.contracts.transform` |
| `llm_rosetta.ir.tools.fix_orphaned_tool_calls_ir` | `normalize_tool_pairing` |
| `GoogleInteractionsToolOps.ir_function_call_to_p`, `ir_function_result_to_p` | `ir_tool_call_to_p`, `ir_tool_result_to_p` |

Three read-only properties of `ProviderShim` were kept for their readers until PR [#1661](https://github.com/Oaklight/llm-rosetta/pull/1661), which removes them. Reading one raises `AttributeError` naming its replacement:

| Removed property | Use instead |
| --- | --- |
| `ProviderShim.base` | `ProviderShim.format` |
| `ProviderShim.default_path` | `ProviderShim.connection.default_path` |
| `ProviderShim.max_tool_description_length` | `ProviderShim.tools.max_description_length` |

 The admin shim editor now validates a saved `provider.yaml` with the loader's own check, so it refuses `base:` with the same message as the loader.

## Removed model-type spellings (P7) {#removed-model-type-spellings}

Removed in PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656). `retired_names.ALIASES_ENABLED` and the model-type alias helpers (`LEGACY_MODEL_TYPES`, `canonical_model_type`, `legacy_model_type`) are deleted: every row of the retired-names table raises, and there is no switch. The model-type names are `llm`, `embeddings`, `rerank` and `decisions`.

A retired model type fails, naming the replacement and the migrate command, at: the typed converter registry and pipelines, `ProviderShim(model_type=...)`, a `provider.yaml` `model_type` and its `embedding:` block, the usage readers, the fidelity checker, config `type:` and `capabilities:`, the admin model editor, and `?type=` on `/models` (HTTP 400).

The gateway config loader no longer reads these keys; each fails at load and names the key that replaces it.

| Removed config key | Use instead |
| --- | --- |
| `embedding_providers` | `embeddings_providers` |
| `embedding_models` | `embeddings_models` |
| `default_embedding_format` | `default_embeddings_format` |
| a provider's `embedding_format`, `embedding_path`, `embedding_shim`, `embedding_enabled`, or its `embedding:` block | `embeddings_format`, `embeddings_path`, `embeddings_shim`, `embeddings_enabled`, or the `embeddings:` block |
| the `decision_*` and `default_decision_*` equivalents | the `decisions_*` and `default_decisions_*` names |
| a model `"type": "embedding"` / `"decision"`, or a `capabilities` entry with that spelling | `"embeddings"` / `"decisions"` |

The versioned `config_version` migrations still upgrade a document older than the current version, because they read their historical spellings through the retired-names table. Only a document already at the current version is rejected for an old spelling. The sections spelled with the current names (`embeddings_providers`, `rerank_providers`) stay readable.

## Retired tags in the usage ledger (P8) {#ledger-tags}

Removed in PR [#1656](https://github.com/Oaklight/llm-rosetta/pull/1656). `openai_chat`, `openai_responses`, `anthropic` and `google` are no longer rows of `STANDARD_SHAPES` or `ADDITIVE_CACHE_STANDARDS` (`llm_rosetta.contracts.usage`). The usage vocabulary names the current standards only: `chat_completions`, `open_responses`, `messages`, `google_generate`.

A database row written under a retired tag is not misread and does not stop the gateway. `shape_for` answers `None` and `cache_is_additive` answers `False` for it, so the startup total repair leaves the row's total as the old writer stored it. `backfill_total_tokens` logs one warning with the count of such rows, the retired tags and the command `llm-rosetta-migrate --db`. Run that command to rewrite the tags; see [below](#llm-rosetta-migrate).

## The extension contracts move to `llm_rosetta.contracts` {#contracts-home}

Changed in PR [#1659](https://github.com/Oaklight/llm-rosetta/pull/1659) (issue [#1657](https://github.com/Oaklight/llm-rosetta/issues/1657)). The modules a host or a shim author builds on now live in one package, `llm_rosetta.contracts`; see [The core surface](../api/index.md#the-core-surface). Of the old paths, only `llm_rosetta.routing`, `llm_rosetta.capabilities` and `llm_rosetta.shims.provider_shim` shipped in 0.13.0, and only those three keep a compat path. The other paths existed only in 0.14 development releases and have no forwarder: importing one raises `ModuleNotFoundError`.

| Old path | New path | What happens to the old path |
| --- | --- | --- |
| `llm_rosetta.transforms` (with `.body` and `.ir`) | `llm_rosetta.contracts.transform` | Removed |
| `_NamedTransform`, `_NamedIRTransform` | `llm_rosetta.contracts.transform.NamedTransform`, `NamedIRTransform` | Renamed; the new names are public |
| `llm_rosetta.shims.auth_exchange` | `llm_rosetta.contracts.auth` | Removed |
| `llm_rosetta.shims.quota_probe` | `llm_rosetta.contracts.quota` | Removed |
| `llm_rosetta.observability.usage` (and its submodules) | `llm_rosetta.contracts.usage` (the package only) | Removed. The usage names on `llm_rosetta.observability` itself still resolve, to the `contracts.usage` objects. |
| `llm_rosetta.routing` | `llm_rosetta.contracts.routing` | 0.13.0-compat forwarder of `ProviderType`, `ResolvedRoute`, `Router`. It emits a `DeprecationWarning` on import and is removed in 0.15.0. |
| `llm_rosetta.capabilities` | `ProviderShim`, `ReasoningCapability`: `llm_rosetta.contracts.shim`. `ConversionContext`: `llm_rosetta.converters.base`. The capability passes run inside the conversion pipeline and have no public home. | 0.13.0-compat forwarder of the twelve 0.13.0 names. It emits a `DeprecationWarning` on import and is removed in 0.15.0. |
| `llm_rosetta.types.ir.extensions` | `llm_rosetta.types.ir.extensions_experimental` | 0.13.0-compat forwarder of the ten 0.13.0 names (it was already deprecated in 0.13.0). It emits a `DeprecationWarning` on import and is removed in 0.15.0. |
| `llm_rosetta.capabilities.description_limit`, `.enforcement`, `.intrinsic_history`, `.tool_names` | None: the passes are internal (`llm_rosetta.ir.passes`) | Removed |
| `ProviderShim`, `ConnectionConfig`, `ToolsConfig`, `ReasoningCapability`, `VendorManifest`, `QuotaDeclaration`, `BatchDeclaration`, `SoftErrorPattern`, `TYPE_PARAM_KEYS` and the literal types, imported from `llm_rosetta.shims.provider_shim` | `llm_rosetta.contracts.shim` | Deprecated re-exports of the same objects, with no runtime warning; removed in 0.15.0. The registry functions (`register_shim`, `get_shim`, `list_shims`, …) stay in `llm_rosetta.shims.provider_shim`. |
| `SURFACE_VERSION` in `llm_rosetta.api` | `llm_rosetta.contracts.SURFACE_VERSION` | `llm_rosetta.api` re-exports the same object; nothing to change |

Upgrade shim packages together with core. The shim packages (`alcf`, `argo`, `asksage`, `free`, `opendesign`) are not on PyPI: the gateway installs them from their GitHub repositories through `shims.presets` (see the [shims guide](shims.md)). A copy built for an earlier 0.14 development release imports `llm_rosetta.transforms` or `llm_rosetta.shims.auth_exchange`, which are gone; the loader skips that package with a WARNING that names the missing module and asks for an upgrade of the shim package, and the gateway serves without its shims. Update the package from its card in the admin panel, or refresh the preset, after upgrading core.

A shim package's `transforms.py`, `auth.py` and `quota.py` must import from `llm_rosetta.contracts` only; see [What a shim may import](shims.md#what-a-shim-may-import).

`llm-rosetta-migrate --scan` (PR [#1658](https://github.com/Oaklight/llm-rosetta/pull/1658)) reports an import of a removed path in this table as an `import` item that names the replacement. It reports a path only when the path does not resolve in the installed core. `llm_rosetta.routing` and `llm_rosetta.capabilities` still resolve as forwarders, so the scan does not report them; their `DeprecationWarning` does. To find every old path, including the forwarders:

```bash
grep -rnE "llm_rosetta\.(transforms|shims\.(auth_exchange|quota_probe)|observability\.usage|routing|capabilities)\b|\b_Named(IR)?Transform\b" .
```

These rows were checked against the head of PR #1659 (commit `cdc6354af3`, version `0.14.0.dev35`): each removed path has no module spec, each forwarder imports with a `DeprecationWarning`, and each new path imports.

## Shim vendors that moved to contrib {#moved-to-contrib}

Fifteen vendors left core for the
[contrib shim repository](https://github.com/Oaklight/llm-rosetta-shim-contrib)
([#1660](https://github.com/Oaklight/llm-rosetta/issues/1660)); the rule that
decides what stays is in [Shim placement](shim-placement.md). Their shim names
did not change, so nothing here is a rename and nothing needs an edit:

| Vendor | Shim names | Why it left core |
|--------|------------|------------------|
| `opencode` | `opencode--chat_completions`, `opencode--messages`, `opencode--open_responses` (quota probe `opencode_go`) | subscription lane |
| `openai_chatgpt` | `openai_chatgpt--open_responses` (quota probe `openai_codex`) | OAuth (Sign in with ChatGPT) |
| `antigravity` | `antigravity--acp` | agent CLI lane |
| `kilo`, `openrouter`, `groq`, `nvidia`, `siliconflow`, `modelscope` | `kilo--chat_completions`; `openrouter--chat_completions`, `--messages`, `--open_responses`, `--rerank`; `groq--chat_completions`; `nvidia--chat_completions`; `siliconflow--chat_completions`, `--embedding`, `--rerank`; `modelscope--chat_completions` | aggregators and hosted endpoints |
| `zhipu`, `zai`, `xai`, `qwen`, `moonshot`, `volcengine` | `zhipu`, `zhipu--embedding`, `--rerank`; `zai--chat_completions`, `--messages`, `--embedding`, `--rerank`; `xai`; `qwen`; `moonshot`; `volcengine--chat_completions`, `--open_responses`, `--embedding` | model makers that start in contrib and are eligible for promotion |

**Gateway.** A config that names one of these shims — a provider `type:` /
`shim:` / key, a typed endpoint (`embeddings_shim`, an `embeddings: {shim}`
block), a legacy `<type>_providers` entry, or `quota.probe: openai_codex` —
makes the gateway add `contrib:<vendor>` to the
[install selection](shims.md#manifest-sources) by itself, with or without a
`shims` section, and log one INFO line per vendor. The `contrib` source is
added as well only when the config declares no `shims.sources`; with an
explicit list (a mirror, no egress) the vendor must come from a declared
source. A disabled provider counts
too. `quota.probe: openai_codex` on an `openai--*` provider resolves against
the probe `openai_chatgpt` now ships. When contrib cannot be reached and
nothing is cached yet, the gateway still starts: it logs a WARNING that names
the vendors and the shims that are missing, and disables the providers that use
them with a reason naming the actual cause: the shim is switched off by
`shims.disabled` (it comes back when removed from that list), the vendor was
not installed (with the reason, such as *not in the manifest* or *no valid
release*), or the install failed; they come back on the next load that
reaches the source. A vendor lost only through a face
(`embeddings_shim: siliconflow--embedding`) disables that endpoint with the
same wording. Under `shims.strict: true` that failure stops the load, as
for any source.

**Library.** `get_converter_for_provider("opencode--chat_completions")` and a
`ConversionPipeline` whose shim names a moved vendor raise a `ValueError` that
names `llm-rosetta-shim-contrib`. To use the shim without the gateway, check out
that repository and call `llm_rosetta.api.load_providers_from_dir(<checkout>/providers)`.
The table is `llm_rosetta.retired_names.MOVED_VENDORS`.

**`llm-rosetta-migrate`.** `--config` and `--scan` report each moved name as a
`note` — `--scan` also finds the probe ids `openai_codex` and `opencode_go`,
and the bare names `xai`, `qwen`, `zhipu` and `moonshot` where they are a whole
`type`/`shim` value, a provider key or an `install` item — ("now in llm-rosetta-shim-contrib; the gateway installs it
automatically"), which never counts in the exit code.

## Find the old names in your code {#find-old-names}

```bash
grep -rnE "converters\.(openai_chat|openai_responses|anthropic|google_genai|google_generate|google_interactions|embedding)\b|\b(OpenAIChat|Anthropic)(Converter|(Config|Content|Message|Tool)Ops)\b|\bGoogle(GenAI)?Converter\b|\bOpenAIResponses(Config|Content|Message|Tool)Ops\b|\bOpenAIResponsesStreamContext\b" .
```

## What still works {#still-works}

- **`llm_rosetta.gateway`** is still importable. It warns and forwards to `llm_rosetta_gateway`; its text says it is deprecated and removed in a later release, with no version. Import `llm_rosetta_gateway` directly now. The `llm_rosetta.observability` finder and the `llm-rosetta[gateway]` extra stay on the same terms.
- **`llm_rosetta.routing` and `llm_rosetta.capabilities`** still import, with exactly their 0.13.0 names. Each emits a `DeprecationWarning` on import and is removed in 0.15.0; see [the contracts table](#contracts-home).
- **Persisted internal keys** are not renamed: the `provider_metadata` tags `openai_chat`, `google` and `anthropic`, the converter tag, native tool definitions and the committed benchmark baselines keep their old spellings on purpose.
- **Retired provider, shim and model-type names** (next section) no longer resolve at all. They raise, and the error names the replacement.

## Retired names {#retired-names}

`llm_rosetta.retired_names.RETIRED_NAMES` lists 31 names that earlier releases renamed. Each row says what the name became. The table is an error table, not an alias table: every row raises `ValueError` and nothing is translated, for example:

```text
'openai_chat' was renamed to 'chat_completions' in 0.14 and the old spelling is no longer accepted (use 'chat_completions'); run 'llm-rosetta-migrate --config <path>' (add --write to apply), see the migration guide (https://llm-rosetta.readthedocs.io/en/latest/guide/migrating-to-0.14/)
```

For a shim that moved to a standalone package, the message also names the `shims.presets` key to declare. Every retired-name error ends with the migrate command (`retired_names.MIGRATE_HINT`), including the gateway's config guard.

### Provider types and formats {#provider-types}

The `type:` value of a provider entry, and the `format` argument of `convert()`, `ConversionPipeline` and `get_converter_for_provider`.

| Old name | Replacement |
| --- | --- |
| `google` | `google_generate` |
| `google-genai` | `google_generate` |
| `openai_chat` | `chat_completions` |
| `openai_responses` | `openai--open_responses` |
| `anthropic` | `messages` |
| `openai-chat` | `chat_completions` |
| `openai-responses` | `openai--open_responses` |
| `open-responses` | `open_responses` |
| `google-interactions` | `google_interactions` |

### Shim names {#shim-names}

The `shim:` value of a provider entry. The `shims.presets` column is the preset to declare when the shim now ships as a standalone package.

| Old name | Replacement | `shims.presets` |
| --- | --- | --- |
| `openai` | `openai--chat_completions` |  |
| `google` | `google--google_generate` |  |
| `google_generate` | `google--google_generate` |  |
| `google_interactions` | `google--google_interactions` |  |
| `anthropic` | `anthropic--messages` |  |
| `argo--openai_chat` | `argo--chat_completions` | `argo` |
| `argo--openai_responses` | `argo--open_responses` | `argo` |
| `minimax--openai_chat` | `minimax--chat_completions` |  |
| `openrouter--openai_chat` | `openrouter--chat_completions` |  |
| `volcengine--openai_chat` | `volcengine--chat_completions` |  |
| `volcengine--openai_responses` | `volcengine--open_responses` |  |
| `argo--anthropic` | `argo--messages` | `argo` |
| `minimax--anthropic` | `minimax--messages` |  |
| `openrouter--anthropic` | `openrouter--messages` |  |
| `alcf--metis` | `alcf_metis--chat_completions` | `alcf` |
| `alcf--minerva` | `alcf_minerva--chat_completions` | `alcf` |
| `alcf--sophia` | `alcf_sophia--chat_completions` | `alcf` |
| `asksage--anthropic` | `asksage--messages` | `asksage` |
| `asksage--openai_chat` | `asksage--chat_completions` | `asksage` |
| `asksage--openai_responses` | `asksage--open_responses` | `asksage` |

### Model types {#model-types}

The model-type spelling follows its endpoint.

| Old name | Replacement |
| --- | --- |
| `embedding` | `embeddings` |
| `decision` | `decisions` |

## `llm-rosetta-migrate` {#llm-rosetta-migrate}

`llm-rosetta-migrate` is a console script of the `llm-rosetta-gateway` distribution. It rewrites retired names in a gateway config, rewrites the persisted tags in `gateway.db`, and scans source for retired names. It reads `llm_rosetta.retired_names` only, so it keeps working after the alias tables are gone, and it never guesses: a name it cannot map is reported and left in place.

```bash
llm-rosetta-migrate --config config.jsonc                  # dry run: diff and report
llm-rosetta-migrate --config config.jsonc --write          # .bak copy, then rewrite
llm-rosetta-migrate --db /var/lib/llm-rosetta --write      # gateway.db, or the directory that holds it
llm-rosetta-migrate --config config.jsonc --check          # exit 3 when rewrites are pending
llm-rosetta-migrate --scan src/                            # advisory report, edits nothing
```

`--config`, `--db` and `--scan` can be repeated. With none of the three, the command looks for the gateway's config file.

**Config.** The edit is at text level: comments, key order and layout outside the rewritten names are untouched. It rewrites retired `type:` and `shim:` values, model `type:` and `capabilities:` entries, the `embedding_*` and `decision_*` keys and sections to their plural names, and adds the `shims.presets` entry when a shim moved to a standalone package. For example, `"type": "openai_chat"` becomes `"type": "chat_completions"`, `"shim": "argo--openai_chat"` becomes `"argo--chat_completions"` with `"shims": {"presets": ["argo"]}` added, and `embedding_providers` becomes `embeddings_providers`.

**Database.** It rewrites the retired tags in `request_log.source_provider`, `request_log.target_provider`, `error_dumps.source_provider` and `error_dumps.target_provider`: the retired format names, and the value `embedding`, which becomes `embeddings`. It also folds `metrics.counters.by_source_provider["embedding"]` into `embeddings`. A `by_target_provider` key `embedding` is reported as a note and not rewritten (the `--json` report lists notes under `notes`, apart from `items`), and `request_log.model_type` is unchanged. The write runs in one transaction after a backup, and the command refuses while another process holds the file (`--force` overrides). The rewrite is idempotent. On a copy of a production-sized database (268275 `request_log` rows, 775 `error_dumps` rows), the 334 rows tagged `embedding` became `embeddings` in both columns, with row counts, token sums, a checksum of the untouched columns and `integrity_check` unchanged, and a second run wrote nothing.

Persisted internal keys stay as they are by decision: the `provider_metadata` tags, the converter tag, native tool definitions and the committed baselines. A database row you do not migrate is not an error; the gateway starts and logs one warning (see [P8](#ledger-tags)).

**`--check` and the first start.** The gateway rewrites a config whose `config_version` is older than the current one (comments are not kept). So that a config the tool has finished is not rewritten again, the tool writes `config_version: 5` when the gateway's load-time steps would change nothing else (the flat `embeddings_*` keys it writes stay readable; the gateway folds them into blocks only when it migrates). When those steps would restructure the document, the tool reports a `pending` item and `--check` exits `3`; only the gateway can run them. A config with no `config_version` is not predicted.

**Modes.** The default is a dry run. `--write` applies the rewrites and makes a copy first: `<file>.bak`, or `<file>.bak-<timestamp>` when that name is taken. `--check` is a dry run that exits `3` when rewrites are pending, for CI. `--json` prints a machine-readable report.

**`--scan`.** The scan is advisory. Each finding is a textual mention of a retired name, not a proof of a use, and it is printed as `mentions ... (advisory)`, not as unmapped. A file that is also a `--config` target in the same run is skipped, because `--config` rewrites it. It reports retired names found in `.py`, `.json`, `.jsonc`, `.yaml` and `.yml` files under a path, and edits nothing. It keeps only strings that are retired wherever they appear: the current formats `google_generate` and `google_interactions`, and the bare vendor words `openai`, `google` and `anthropic` as shims, are not reported. A quoted string is a match, not a proof; `"anthropic"` may be your own variable.

**Exit codes.**

| Code | Meaning |
| --- | --- |
| `0` | Nothing to do, or everything mapped. A note (an item the tool leaves on purpose, such as a `by_target_provider` key) does not count, so a finished `--db` migration exits `0`. |
| `1` | Something needs a person: an unmapped item, or any `--scan` mention. |
| `2` | The command could not run: an unreadable file, a parse error, a database held by a running process, or a database locked by another writer (`--force` skips the open-file check, not a SQLite lock; the message is one line naming the lock). |
| `3` | `--check` found pending rewrites, or a load-time migration only the gateway can run. |

## Upgrade recipe {#upgrade-recipe}

1. Stop the gateway. `llm-rosetta-migrate --db` refuses a database that a running process holds.
2. Back up the config file and the data directory (`gateway.db`). `--write` also keeps a `.bak` copy, but a backup you made yourself does not depend on the command.
3. Dry run: `llm-rosetta-migrate --config <config> --db <data dir>`. Read the diff and the report. Exit `1` or `2` needs a person before you go on.
4. Write: `llm-rosetta-migrate --config <config> --db <data dir> --write`.
5. Start the gateway. A config that still holds a retired name stops at start with one `llm-rosetta-gateway: error: ...` line that names the file and the key.
6. Optional: run `llm-rosetta-migrate --scan <your code>` and read the findings. Library callers, shim packages and scripts are not rewritten by the command.

## The core surface {#core-surface}

0.14.0 is also the release of the gateway's strict core surface: the gateway imports core only through `llm_rosetta.api` and `llm_rosetta.contracts`, a shim imports only `llm_rosetta.contracts`, and `SURFACE_VERSION` (now `1`) rises on removals only. A host or shim author who imports other `llm_rosetta` modules is outside that contract. See [The core surface](../api/index.md#the-core-surface).

---
title: Changelog
---

# Changelog

All notable changes to LLM-Rosetta are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

### Added

- **The repairs the gateway runs after startup report what they did, and one can be re-run** (epic [#1027](https://github.com/Oaklight/llm-rosetta/issues/1027), PR [#1044](https://github.com/Oaklight/llm-rosetta/pull/1044)): the five data repairs that run once the socket is open — the provider name on rows written before that column existed, a key's `last_used`, the token totals that excluded cache tokens, the usage bucket on rows written before the bucket column, and an error dump's request-log id — kept a count in a local dict that was logged and dropped. They are now `Step`s in one registry with kept results, and two endpoints expose them: `GET /admin/api/maintenance` (per step: whether it ran, what it changed, how long it took, and why it failed) and `POST /admin/api/maintenance/<name>` to re-run one by name, which answers with the step's recorded failure rather than a 500 when the step fails. The two single-purpose routes that predate the surface — `POST /admin/api/keys/backfill-last-used` and `POST /admin/api/error-dumps/backfill-log-ids` — are unchanged for callers (same method, same `{"updated": n}` body) and are now handlers over the same steps instead of a second wiring. `GET /health/ready` gains a `maintenance` object (`state`, `not_started` / `running` / `done` / `partial`, the failed step names, and each step's `changed` and `seconds`) so a probe can tell "the repairs are done" from "the startup task finished"; a failed repair deliberately does not turn the probe into a 503, because each step is safe to retry and refusing traffic over one that failed is the outage the deferred policy exists to avoid. See [the admin API reference](api/admin.md#maintenance).

- **Citations convert across every dialect, on both legs, streaming included** (issue [#459](https://github.com/Oaklight/llm-rosetta/issues/459), PR [#1025](https://github.com/Oaklight/llm-rosetta/pull/1025)): a citation carried no *kind* — readers branched on which of the two nested dicts happened to be present, and each writer spoke its own dialect's spelling straight into the IR, so half the dialects could not read what the other half wrote (`open_responses` and `messages` never called their own reader at all). `CitationPart` now carries a `citation_type` discriminator, the fields every dialect shares (`url`, `title`, `cited_text`, `start_index`, `end_index`) flat on the part, and provider-private fields under `provider_metadata` — the same treatment `tool_group` (#999) and `intrinsic_kind` (#1010) got, with the pre-2026 nested spelling still *read* and never written; the reading lives in `converters/base/helpers/citations.py`, so no writer branches on shape. All five dialects read and write on both legs, and streaming has a carrier at last: a `citation_delta` event mirrors the non-streaming part, fed from `citations_delta` (Anthropic), `response.output_text.annotation.added` (Responses), a `text_annotation_delta` (Interactions) or `choices[].delta.annotations` (chat completions), with the base converter answering it by default with a warning so a dialect without a native form reports the loss instead of dropping it in silence. GenerateContent has no annotation at all — its citations are the candidate's `groundingMetadata`, rebuilt on the way out from the parts (sources become chunks, spans become the supports that index into them, and the `maps` wrapper round-trips), while its search-level metadata (the queries and the rendered entry point) moved from *every citation* to a single turn-level carrier. The conversion grid is measured against the live endpoints and pinned in `tests/converters/citation_matrix/`: the four dialects that spell `url_citation` — chat completions, Responses, Interactions, GenerateContent — convert freely among each other, and Anthropic's column is a **drop by measurement rather than omission**, because its validator accepts exactly its own five tags (`url_citation` and `text_citation` are rejected with a `400`), and requires each citation to point at a document or `search_result` block the message itself carries or at an `encrypted_index` the provider issued. **What converts is a URL-bearing citation, whatever its kind**: an Anthropic `search_result_location` or `web_search_result_location` leaves for any of the four, one way — nothing comes back. Anthropic's own shape is structural, so the reader splits its citations out of the text block they annotate and the writers hang them back on the preceding text block. A citation a target cannot render goes through `report_dropped_citation`, which prefers the conversion's warning sink and always writes the log line.

- **Ten built-in shims are on a removal schedule, and the gateway says so** (epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927), PR [#1036](https://github.com/Oaklight/llm-rosetta/pull/1036)): the ALCF, Argo and AskSage shims are moving to their own repositories in **0.16.0** — they are deployment-specific lanes whose upstreams change on their own schedule, so they are maintained as standalone shim packages a gateway declares through the top-level `shims` section ([#929](https://github.com/Oaklight/llm-rosetta/pull/929)) and consumes without a new core release. Nothing is deleted here; what is new is that the removal is *declared*, to you and to the code, from one manifest. `llm_rosetta.shims.removals` holds one entry per shim going away — the release that removes it, the `shims.remotes` entry that replaces it, and the older spellings that stop resolving with it — and `detect_shim_removals()` is **pure** (no I/O, no registry, no logging), so the startup warning, the admin panel and CI read the same answer from the same place, and it can later be a `LOAD_TIME` step in the maintenance module ([#1027](https://github.com/Oaklight/llm-rosetta/issues/1027)) without dragging state along. The gateway **logs one warning per affected provider**, naming the shim, the release and the source to declare instead; a shim that a declared source supplies is silent, because that source overrides the built-in at rebuild time — and the log going quiet is therefore the confirmation that a migration landed. A source that *failed* silences nothing, since it overrode nothing, and a **disabled** provider is still reported, because its config breaks the moment it is switched back on. **Settings → Shim Sources** lists the same advisories with a one-click "add the replacement source", which only appends to the remotes textarea — never saves — so the existing Save stays the single writer and the operator reviews exactly what would be written. **`make check-shim-removals`** holds the schedule to the code: while the version is below a release it passes, because the repository is *supposed* to still name those shims, and once the version reaches one it fails while the shim is registered *or* while anything still references it — an example, a Docker file, a UI mock-up, a helper script, the docs, or the source, where a leftover `LEGACY_SHIM_ALIASES` entry counts as much as the shim's own `provider.yaml`. The changelogs are exempt (their history is not rewritten), the manifest exempts itself (it names every scheduled shim by design), and a line carrying `shim-removal-ok` is exempt, which is how a 0.16.0 migration table is allowed to spell the old names. The migration itself is unaffected by all of this: the replacement shims keep the names, so declaring the source is the whole of it. See [Shims scheduled for removal](guide/shims.md#shims-scheduled-for-removal); the deletion is tracked as [#919](https://github.com/Oaklight/llm-rosetta/issues/919) / [#666](https://github.com/Oaklight/llm-rosetta/issues/666), with one follow-up issue per repository (`-argo#2`, `-asksage#2`, `-alcf#2`) covering the re-sync, the CI, and the first tag — which is why the presets the admin panel offers still carry no `ref`.

- **Tool `strict` and tool-choice allow-lists survive the IR** (epic [#462](https://github.com/Oaklight/llm-rosetta/issues/462)): two fields that Chat Completions and Responses share were being lost on the way through the IR, both silently. `strict` — the flag that asks a provider to validate a call's arguments against the tool's schema — was read by the Responses converter alone and kept only in `metadata["strict"]`, so a Chat request carrying `strict: true` reached a Responses upstream as `strict: false`, and a Responses request reaching Chat lost it entirely (as did Chat→Chat). It is now the first-class `ToolDefinition.strict`, which **three** of the five formats state (`chat_completions` and `messages` now read and write it; `open_responses` reads its own field and still accepts the old metadata spelling; the Google formats have no equivalent and drop the flag). Measured against the live upstreams: Chat and Responses both accept `strict: true`, and Anthropic accepts it too but answers `400` unless the schema sets `additionalProperties: false` — a client's schema is not ours to close, so we do not. The second field is the `tool_choice` allow-list, which four formats state in four different shapes and the IR held as an opaque Responses-shaped `provider_extensions` entry — so every other target dropped the restriction, and a Google target could not even keep a second name (`allowedFunctionNames` and Interactions' `allowed_tools` collapsed to the first, since `ToolChoice` had nowhere to put the rest). `ToolChoice.allowed_tools` now holds tool identities — `(name, tool_group)`, the same pair the IR uses everywhere else — and Responses, Google generateContent and Google Interactions all read and write it, listing every name. Where the source states a group, the naming pass resolves the restriction through it, so a tool whose bare name is shared still resolves and the group is spent rather than left to disagree with the name; a restriction naming no declared tool is still reported. Two formats can only widen it: Chat Completions and Anthropic Messages have no allow-list variant, so the choice degrades to its bare mode (`required` kept for a "must call one of these" choice) with a warning that the model may now call any declared tool — a restriction going away is not the same as an optional field being dropped. Chat Completions is the sharper case: the variant is in OpenAI's schema and the live API rejects it (`400 Missing required parameter: 'tool_choice.function.'`), so the shape is read — a client may hand one to the library and route it to a format that does state it — and never written. One more thing the live API taught us: an allow-list entry's `type` must match how the tool was declared, so it is derived from the declaration rather than carried in the IR, where a downgraded custom tool would leave it stale. Also fixes the IR tool example in the IR-types guide, which showed the nested wire shape instead of the flat IR one.

- **The usage report is readable from a shell** (issue [#1023](https://github.com/Oaklight/llm-rosetta/issues/1023), phase 3 of [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016)): `llm-rosetta-usage` prints what the admin panel shows — the same rolling windows, the same classes — for an operator who is not going to open a browser. It is a third reader of one report rather than a second implementation of it: the store is opened through `PersistenceManager`, the report is built by `UsageReport.build`, and `--json` emits that report rather than a rendering of it — the subject cut carries exactly the keys the admin API returns, so a script reading the command and one reading the panel see one shape, and a test compares the two. `--by subject,type,model` cuts the same read three ways, and every cut's rows add up to the store's total because the remainder is reported instead of dropped; `--top` bounds every table — including the subject table, the one that grows with every credential the gateway has ever seen — while each table's total keeps summing the whole of its cut. Two deliberate refusals: a store that does not exist is reported as missing rather than created (a mistyped path must not leave an empty database that reads as a store with no traffic), and the store is opened **read-only** — SQLite's own read-only mode, no backup, no integrity check, no table creation, no migrations — so a store whose schema is older than the installed build is refused with the two ways out named (start the gateway once, or pass `--migrate`, which opens it exactly as the gateway does: backup first, then the migrations), and a store the operator cannot write (a mounted volume, a snapshot copy) reads fine while its schema is current. Attribution reads the gateway's keys database read-only, two columns of one table, because `observability` may not import the gateway; without it every token is reported as unattributed and the command says so on stderr — distinguishing a missing database from an unreadable one — rather than letting an all-zero per-subject table read as "nobody spent this". Every table is ordered by the window it shows, the widest window breaking ties, so a row's position is explained by the numbers beside it. The internal-traffic subject and label moved from the gateway's keystore into `usage.report` (`INTERNAL_SUBJECT`, `INTERNAL_LABEL`) so the code that writes those rows and the code that groups them cannot drift apart, and `PersistenceManager.create(backup=False, migrate=False)` is the new pair of knobs — the second opens a store without writing to it at all, after the caller has checked `expected_schema_version()` against `schema_version()` itself — that lets a read-mostly caller read a store it does not own. See [Observability — The `llm-rosetta-usage` command](api/observability.md#the-llm-rosetta-usage-command).

- **The usage ledger counts the non-chat model types** (issue [#1019](https://github.com/Oaklight/llm-rosetta/issues/1019), phase 4 of [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016)): `/v1/embeddings`, `/v1/rerank` and `/v1/decision` are served by their own pipelines, bill the same upstreams, and wrote **no `request_log` row at all** — so "usage" meant chat's usage, and both the per-key panel and the dashboard's rolling total left those tokens out. Schema migration v6 adds `request_log.model_type` (`llm`, `embedding`, `rerank`, `decision`, `DEFAULT 'llm'`, so existing rows keep meaning chat with no rewrite), and the axis is named after the registry's vocabulary rather than "model family", which in LLM usage means a series of models. `usage/normalize.py` declares the ledger-column-to-IR-field map **per model type**, because a column does not carry the same name in every family: the decision IR reports `input_tokens`/`output_tokens` already, the embedding and rerank IR types have no output side, and rerank's cache counter is `cached_tokens`; an unknown type is refused rather than read as chat, and a family that reports no total still takes the additive rule. `UsageReport.by_model_type` cuts the same rows by type as well as by subject, and the dashboard's daily table (`/admin/api/token-usage`) returns each day's counters plus the same split and accepts a `model_type=` filter — the day's own fields keep their meaning, so the panel renders them unchanged. The three handlers record through `OpsProxyRequest`, the write path the chat routes already use, rather than growing a second one: two paths that fill a row differently is what [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014) was about. Note that the dashboard total and the per-key panel both start counting these requests on deploy — a token that no view counts is a token nobody can account for.

- **Usage metering is a subsystem of its own, outside the gateway** (epic [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016); phases [#1017](https://github.com/Oaklight/llm-rosetta/issues/1017) / [#1020](https://github.com/Oaklight/llm-rosetta/issues/1020), after the reconciliation fix in [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014)): `llm_rosetta.observability.usage` now owns what "how many tokens did we move, and for whom" means, and imports no gateway module — no HTTP, no config, no credentials. `semantics.py` declares once where each API standard keeps its cache counters and whether they are **additive** to the input (Messages) or a **subset** of it (OpenAI-, Google-shaped); the streaming extractor and the backfill repair both derive from that declaration, and a test fails if any converter disagrees with it, so the two encodings that [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014) caught disagreeing cannot drift apart again. `normalize.py` turns any family's usage into the ledger's columns with `token_total` (a provider total wins; otherwise the counters are added, with reasoning never added and cache added only where it is not already inside the prompt). `report.py` holds the classification that makes the views add up — **attributed**, **internal**, and an **unattributed** remainder carved by label — so `Σ panel rows == dashboard total == Σ ledger` holds by construction rather than by luck, and a library caller with no credentials gets the same numbers with everything in the remainder. New and public: `UsageLedger`, an in-memory ledger that records IR usage dicts and answers with the same `UsageReport` the SQLite path answers with for the same rows — no store, no schema — so a script, a batch job or a test can measure usage without running a gateway. The API Keys panel now shows the internal token's traffic and the remainder as their own rows, and marks a subject whose label is shared with another live key instead of reporting a figure the log cannot support; the dashboard's rolling total and the panel now sum the same `total_tokens` column. See [Observability — Usage](api/observability.md#usage).

- **Three drops that did not stay dropped** (second review round on [#1010](https://github.com/Oaklight/llm-rosetta/pull/1010)): the `open_responses` response writer appended the template's result unconditionally, so a server call it cannot render — which answers `{}` — shipped an item with no `type` ahead of the message item, where every other dialect was clean; the branch is now guarded (and passes the warnings sink), as its `_append_intrinsic_result_item` sibling already was. The orphan fixer collected answered tool ids from `assistant` and `tool` messages only, so once a history pair was degraded to an ordinary client call, a result riding in a `role:"user"` turn stopped counting as an answer and the fixer injected `[No output available yet]` *beside* the real result under the same id; answer ids now come from every message, whatever its role. And `_report_dropped_intrinsic` returned early when the caller passed no `warnings` sink and did not log, where the per-converter code it replaced logged unconditionally — the pipeline passes a sink, so the silence only hit library callers; the log line is now always written, with the sink preferred.

- **A server call is never degraded on the response or stream leg** (follow-up to the intrinsic refactor, from review of [#1010](https://github.com/Oaklight/llm-rosetta/pull/1010)): the invariant held for three of the five dialects. An `open_responses` **response** leg and a `chat_completions` **response and stream** leg still handed the client a `function_call` for a call the client never ran — and `open_responses` streamed what its own non-streaming leg degraded, so a client saw a function call for a server call exactly when it did not stream. The rule now lives in the base `BaseToolOps` template (`dropped_intrinsic_call` / `dropped_intrinsic_result`, empty plus a warning) instead of one converter at a time, so every dialect drops on those legs and only *history* degrades. A drop is falsy, so callers that appended a writer result unconditionally now skip it — that was also what could put an empty dict into a Chat Completions `tool_calls` list. Two more findings from the same review: two id-less intrinsic calls of different kinds shared one key and were rewritten as the last kind seen (an anonymous call is keyed by position now, and an anonymous result pairs with the call before it), and the degrade warning said "cannot be replayed by the target" for a case where the target does have the kind and the IR part is missing a field its form needs — it now distinguishes the two and reports an unpaired call the target requires a result for separately.

- **Intrinsic tools are derived tool calls, and cross-format server-tool history now replays** (issue [#839](https://github.com/Oaklight/llm-rosetta/issues/839), follow-ups [#996](https://github.com/Oaklight/llm-rosetta/issues/996) and [#994](https://github.com/Oaklight/llm-rosetta/issues/994)): the intrinsic kind is now the first-class field `intrinsic_kind` on `ToolCallPart`, `ToolResultPart`, `ToolDefinition`, `ToolCallStartEvent`, `ToolCallDeltaEvent` and `ToolResultEvent`, read through `intrinsic_kind_of()` with a fallback to the old `provider_metadata` / `metadata` spelling; `set_intrinsic_kind` is removed and `get_intrinsic_kind` / `get_definition_kind` remain as deprecated aliases. The kind is never inferred from a tool name any more. `BaseToolOps` owns the intrinsic dispatch and a converter overrides only hooks (`intrinsic_renderable`, `ir_intrinsic_call_to_p`, `ir_intrinsic_result_to_p`, `intrinsic_requires_result`), falling back to its ordinary client path. A new pipeline pass, `degrade_intrinsic_history`, replaces the five converter-local policies for a foreign kind in history: a call the target cannot replay is rewritten together with its result, by `tool_call_id`, into an ordinary function pair (marked `_downgraded_from`, named after the kind and qualified when the client declared a function of that name, reported once in the warnings); the response and stream legs never degrade a server call into a client call and drop what the target cannot render. `strip_intrinsic_tools` is removed, and the #996 call/result pairing decision moves to the shared `history_native_calls`. Replaying every cross-format path against the live upstreams surfaced and fixed: a server-tool id without the target's required prefix (`srvtoolu_`, `ws_`, `ci_`); an assistant turn's inline `tool_result` being ignored by the Open Responses writer; untyped search-result blocks rejected once degraded (now carried as JSON text); a `server_tool_use` with no paired result (Anthropic) being degraded as a pair; foreign code-execution payloads (`command` versus `{code, language}`, a plain output string versus a result object); and a result kept in a `role: "tool"` message instead of beside its call in the assistant turn. All twelve cross-format pairs among Anthropic, OpenAI Responses and Google Interactions now answer `200`. Behavior change: a server call of a kind Anthropic has no tool for (for example a Google `google_search` or an OpenAI `code_interpreter`) no longer reaches a Messages client as a fabricated `server_tool_use`; it is dropped from the response and stream with a warning. The Google Generate stream delta no longer repeats the start event's native part (Google sends the call whole, so the signature has one carrier), and the reasoning-signature origin is tied to the converter tag with a test pinning the foreign-origin drop. A unified warning system across the request, response and stream legs is tracked in [#1009](https://github.com/Oaklight/llm-rosetta/issues/1009). See the new "Intrinsic Tools Are Derived Tool Calls" section in the architecture guide.

- **Tool-call groups are first-class in the IR** (issue [#999](https://github.com/Oaklight/llm-rosetta/issues/999), parent [#953](https://github.com/Oaklight/llm-rosetta/issues/953)): a tool *call* now records the container it was made in through `ToolCallPart.tool_group` — and `ToolCallStartEvent.tool_group` for the streaming form — the same field a tool *definition* already used, instead of the private `provider_metadata["namespace"]` key. The Open Responses converter writes the field when it reads a `namespace` off the wire and reads it back out when serializing, on both the non-streaming and streaming paths; on the stream leg an upstream `function_call`'s namespace now reaches the `tool_call_start` event, which it previously did not. Reading is backward compatible — a call carrying only the old metadata key still resolves, through a new `tool_call_group` helper — and the legacy key is cleared whenever the first-class field is written, so the two spellings cannot drift apart.
- **Tool groups enabled for DeepSeek and Volcengine Responses shims** (issue [#998](https://github.com/Oaklight/llm-rosetta/issues/998), parent [#953](https://github.com/Oaklight/llm-rosetta/issues/953)): `deepseek--open_responses` and `volcengine--open_responses` now declare `tools.tool_groups: true`. Requests targeting either provider retain client-declared `namespace` tool containers instead of being flattened into prefixed function names, matching what both upstreams accept natively. The shim loader (`_load_single_provider`) now parses `tools.tool_groups` from `provider.yaml`. Addressing a container child through the nested `tool_choice` form is still not carried: the IR `ToolChoice` has no group slot, so the read leg falls back to `auto` and now reports the drop on the conversion warnings rather than doing it silently (parent [#953](https://github.com/Oaklight/llm-rosetta/issues/953)).
- **Tool groups for a target that carries them** (issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)): a shim can now declare `tools.tool_groups: true`, and the Responses writer emits the client's grouping as `{"type": "namespace", …}` containers again instead of folding the group into the tool names. The declaration is per provider and not derivable from the base format: OpenRouter's Chat Completions lane accepts the container while OpenAI's does not, and Anthropic and Google group only the tools their own MCP servers bring. Children go out under the client's own names. The group travels beside them on the wire, so the naming pass qualifies a child only against its own container's members, and a top-level `wait` no longer forces a sibling inside `tasks` to ship as `tasks_wait`. Verified live against DeepSeek and Volcengine: both accept a child named like a top-level tool, and both echo the namespace back on the call. The response leg reads that echo as half of the identity, `(name, namespace)`, so a shared name resolves to the container child rather than the top-level tool, and a group the library generated for a nameless container is dropped instead of handed to the client. A container the client gave no description to goes out with `""`, because OpenAI and OpenRouter reject an absent one with `400` while accepting the empty string. The contested-name machinery is removed in the same change: generated group ids left it unreachable, and eight collision shapes through the pipeline fire it in none of them.

- **Per-key token usage in the API Keys panel** (issue [#967](https://github.com/Oaklight/llm-rosetta/issues/967)): the admin API Keys tab now reports each key's request count and token totals over rolling 5h / 24h / 7d / 30d windows, switched with a segmented control built from the window list the endpoint returns. Attribution is by **credential**, not by label: `KeyStore.create_key` derives a stable bucket `uuid5(namespace, sha256(secret))` — the full hex, since the bucket is a grouping key and never displayed — once at creation, and recomputes it on rotation, so a rotated key starts a fresh bucket; the value rides on `KeyContext` and the proxy writes it to a new `request_log.api_key_bucket` column (schema migration v4). The report is one indexed `GROUP BY` with conditional per-window sums — no rollup table — and the request-log read names its columns instead of relying on physical order. Renaming a key therefore no longer breaks its usage; buckets with no live key are dropped, and unattributed rows (internal token, anonymous) store NULL rather than an empty string. Pre-existing keys get their bucket backfilled from the stored key hash; historical log rows are left unattributed rather than guessed from the label. A tooltip on the usage cells notes that rotation starts a new bucket. (The attribution this entry describes was extended by the usage-subsystem entry above: a row whose credential no longer exists is now reported as **unattributed** rather than dropped, so the panel still adds up to the dashboard, and the internal token has a bucket of its own.)
- **`scripts/asksage-token.py`**: AskSage's inference endpoints do not accept the account API key, only a short-lived access token from `POST {host}/user/get-token-with-api-key`, so an `asksage--*` provider could not be wired through `token_command` at all. The script performs that exchange and prints the token on stdout (diagnostics go to stderr), dropping into a provider's `token_command` the same way `alcf-token.py` does. `--status` reports the token's expiry from its JWT `exp` claim and `--probe` sends one minimal `/responses` request to confirm the token is actually accepted. Credentials are read from the environment only (`ASKSAGE_TOKEN_EMAIL`, `ASKSAGE_TOKEN_API_KEY`, falling back to the ANL sample client's `ASK_SAGE_API`) — never from argv, where the process table would expose them. Those names are deliberately not the shims' `api_key_env`: every `asksage--*` shim declares `connection.api_key_env: ASKSAGE_API_KEY`, and the gateway turns that variable into a provider's `api_key` before it reads `token_command`, then rejects the pair as mutually exclusive — so a `token_command` provider cannot be built at all while `ASKSAGE_API_KEY` sits in the gateway's environment. The two AskSage hosts are not interchangeable either: an Argonne API key is refused by `api.asksage.ai`'s exchange with `{"response": "Key doesnt exist", "status": 400}`, and since `GET {host}/server/openai/v1/models` answers without checking credentials (with a bogus, empty or absent auth header), a working model list is not evidence that a key works.
- **External shim sources** (epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927), issue [#666](https://github.com/Oaklight/llm-rosetta/issues/666)): a new top-level `shims` config section lets the gateway load provider shims from extra directories (`shims.paths`) and from remote archives fetched into a local cache (`shims.remotes`), so a shim set can live in its own repository and be consumed **without installing anything into the gateway image** — the official image ships no `pip` and no `git`. Remote fetches are git-less: a GitHub repository URL is rewritten to its codeload tarball endpoint, and any other host works via an explicit archive URL (optionally containing `{ref}`), so GitLab and self-hosted forges need no per-host code path. An omitted `ref` is resolved to a concrete commit SHA and recorded in the cache stamp (the config file is never rewritten); a pinned `ref`, a `sha256`, or `refresh: "never"` is fetched once, `refresh: "always"` always re-fetches, and the default `auto` re-fetches past `remote_ttl`. A failed fetch falls back to a stale cache with a warning. Extraction rejects absolute, traversal and symlink/hardlink members, and both the download and the extracted tree are size-capped. A failed source warns and is skipped (surfaced in the admin panel) unless `shims.strict: true` aborts startup. New public API — `llm_rosetta.shims.rebuild_shims()` / `resolve_remote()` and `llm_rosetta.gateway.apply_shim_sources()`; `create_app` applies the section automatically. The admin Settings tab gains a **Shim Sources** section to edit paths/remotes, see per-source load status, and reload or force a refetch. Note that a declared source may ship a `transforms.py` that is executed on load — the same trust level as installing a plugin package — so sources should be pinned and trusted.
- **OpenCode paid shims** (issue [#924](https://github.com/Oaklight/llm-rosetta/issues/924)): new `opencode` shim group — `opencode--chat_completions`, `opencode--messages`, `opencode--open_responses` — fronting OpenCode's Go subscription lane (`https://opencode.ai/zen/go/v1`); the Zen pay-as-you-go lane is the same shims with `base_url: https://opencode.ai/zen/v1`. Because the Go lane requires a stable `x-opencode-session` on every inference request, this adds a generic `connection.session_header` shim field: when set, the gateway injects that header with a value derived per conversation (reusing the affinity identity — the same conversation always gets the same id), and a client-supplied value is forwarded verbatim. The keyless free lane is tracked separately in issue [#884](https://github.com/Oaklight/llm-rosetta/issues/884).
- **Duplicate tool definition handling** (issue [#903](https://github.com/Oaklight/llm-rosetta/issues/903)): new IR-level pass `dedupe_tool_definitions` (in `capabilities.py`, run from `ConversionPipeline.convert_request`) collapses byte-identical duplicate tool definitions to the first occurrence, and reports same-name/different-definition collisions as warnings without dropping either. Tools harvested from a *named* namespace container are exempt — their namespace is part of their identity — but a container with no name of its own is not: it flattens its children to `namespace: ""`, which qualifies nothing, so its byte-identical children are duplicates like any other and now collapse too, instead of reaching a strict upstream as a repeated name (issue [#913](https://github.com/Oaklight/llm-rosetta/issues/913)). Anthropic, DeepSeek, xAI and Google reject a repeated tool name with `400`, while OpenAI and others accept it, so the same client body previously succeeded or failed depending on the target; identical duplicates now survive every target, and genuine collisions are attributable instead of surfacing as an opaque upstream error.
- **Free model sources — Kilo** (PR [#888](https://github.com/Oaklight/llm-rosetta/pull/888), issue [#883](https://github.com/Oaklight/llm-rosetta/issues/883), epic [#885](https://github.com/Oaklight/llm-rosetta/issues/885)): new `kilo--openai_chat` shim fronting Kilo's public gateway. New shim fields `display_name`, `free_source`, and `connection.keyless`; a keyless provider sends **no** auth header while no `api_key` is configured and falls back to Bearer when one is supplied, so the same shim serves both the free pool and a paid account. As a keyless entry the admin UI titles it **Kilo (Free)**, groups it under a new **Free Resource** section (hidden when no free source exists), adds a per-card **Refresh** that re-pulls the roster and offers new models for one-click apply, and shows a neutral third-party disclosure. The shim's `model_list_transform` exposes only the upstream's `isFree` slice.
- **Base formats in the provider-type picker** (PR [#915](https://github.com/Oaklight/llm-rosetta/pull/915), epic [#885](https://github.com/Oaklight/llm-rosetta/issues/885)): the admin Provider Type dropdown lists **Base Formats** (the five chat standards plus `open_responses`, from a single `BASE_FORMATS` constant next to `_PROVIDER_REGISTRY`) alongside **Providers**, naming the recommended vendor shim where it differs from the format. A bare format can now be created for a custom or compatible endpoint without hand-editing config.
- **External shim transforms** (PR [#892](https://github.com/Oaklight/llm-rosetta/pull/892), issues [#272](https://github.com/Oaklight/llm-rosetta/issues/272)/[#273](https://github.com/Oaklight/llm-rosetta/issues/273)): shims loaded from outside the package can now use relative imports in their `transforms.py`. The plugin loader registers each providers root as a synthetic package namespace keyed on the resolved path (so two roots with the same group/leaf names stay independent), rolls a failed load's modules back, and clears the namespace on reload.

- **Unified Ops audit layer** (PR [#856](https://github.com/Oaklight/llm-rosetta/pull/856), issue [#846](https://github.com/Oaklight/llm-rosetta/issues/846), sub-PRs [#851](https://github.com/Oaklight/llm-rosetta/pull/851)/[#852](https://github.com/Oaklight/llm-rosetta/pull/852)/[#853](https://github.com/Oaklight/llm-rosetta/pull/853)/[#854](https://github.com/Oaklight/llm-rosetta/pull/854)/[#855](https://github.com/Oaklight/llm-rosetta/pull/855)): new `gateway/ops/` package where every gateway mutation (data cleanup, password change, token rotation, API key CRUD, periodic prune) is a typed `OpsBase` subclass with automatic audit recording, including failed operations. Closes the audit gap where destructive admin actions left zero trace. Replaces the standalone `_record_telemetry()`/`_log_key_event()`/inline `ops_log.add()` call sites with `OpsProxyRequest`, `OpsClearData`/`OpsCleanupData`/`OpsTrimData`/`OpsVacuum`/`OpsRebuildMetrics`/`OpsClearOpsLog`/`OpsPrune`/`OpsPeriodicCleanup`, `OpsPasswordChange`/`OpsTokenRotate`/`OpsSessionLogoutAll`, and `OpsKeyCreate`/`OpsKeyUpdate`/`OpsKeyDelete`/`OpsKeyRotate`.
- **Persistence/DB module redesign** (PR [#869](https://github.com/Oaklight/llm-rosetta/pull/869), issue [#848](https://github.com/Oaklight/llm-rosetta/issues/848), sub-PRs [#859](https://github.com/Oaklight/llm-rosetta/pull/859)/[#860](https://github.com/Oaklight/llm-rosetta/pull/860)/[#861](https://github.com/Oaklight/llm-rosetta/pull/861)/[#863](https://github.com/Oaklight/llm-rosetta/pull/863)/[#864](https://github.com/Oaklight/llm-rosetta/pull/864)/[#866](https://github.com/Oaklight/llm-rosetta/pull/866)/[#867](https://github.com/Oaklight/llm-rosetta/pull/867)): async `config_lock()` (no more event-loop-blocking `fcntl.flock`); periodic WAL checkpoint (5 min, 64MB cap) with startup check; schema version table with a forward-migration registry and FK constraints; `RetentionPolicy`/`RetentionTracker` replacing four independent prune paths; monotonic `lifetime_total_requests`/`lifetime_total_errors` counters unaffected by prune; `KeyStore` converted from sync `sqlite3` to async `aiosqlite`; startup backup via `sqlite3.backup()` with integrity check.
- **Conversion pipeline performance** (PR [#847](https://github.com/Oaklight/llm-rosetta/pull/847), issue [#828](https://github.com/Oaklight/llm-rosetta/issues/828), sub-PRs [#842](https://github.com/Oaklight/llm-rosetta/pull/842)/[#843](https://github.com/Oaklight/llm-rosetta/pull/843)/[#844](https://github.com/Oaklight/llm-rosetta/pull/844)): 3-layer IR validation cache optimization (`pickle` cache keys, `GenerationalSet` eviction, epoch guard skipping per-entry hashing on unchanged lists) cuts warm-path `anthropic → anthropic` conversion from ~18ms to ~2.7ms. O(1) tool result name resolution in the Google converter (was O(n²)). New `scripts/bench_pipeline.py` / `make bench` benchmark suite with synthetic payloads and per-phase timing across all 10 converter pairs.
- **`ToolMessage.batch_id`** (PR [#841](https://github.com/Oaklight/llm-rosetta/pull/841), issue [#837](https://github.com/Oaklight/llm-rosetta/issues/837)): new `batch_id: NotRequired[str]` field for tracking which tool results belong to the same assistant tool-call turn. `assign_tool_batch_ids()` and `merge_tool_messages()` fix a structural bug where Anthropic and Google generateContent outbound converters (which require all tool results in a single message) silently relied on message adjacency to group parallel tool calls correctly.
- **Streaming convenience methods** (PR [#873](https://github.com/Oaklight/llm-rosetta/pull/873), issue [#872](https://github.com/Oaklight/llm-rosetta/issues/872)): `StreamProcessor.iter_events(chunks)` and `BaseConverter.iter_stream_from_provider(chunks, context)` iterate over upstream chunks and automatically flush deferred terminal events on exhaustion, so callers no longer need to remember `process_chunk(None)`.
- **Decision model paradigm** (PRs [#705](https://github.com/Oaklight/llm-rosetta/pull/705), [#709](https://github.com/Oaklight/llm-rosetta/pull/709), [#711](https://github.com/Oaklight/llm-rosetta/pull/711), [#718](https://github.com/Oaklight/llm-rosetta/pull/718)): new model category for probabilistic structured decisions. Three IR question types (`assertion`, `choice`, `score`) with converters backed by LLM structured output, reranker relevance, or embedding similarity. Gateway routes at `/v1/decision` and `/v1/systemone`.
- **Decision IR: shared entry abstraction** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889)): the three IR question types now unify on `criteria: list[DecisionEntry]` — `assertion` = 0 or 2 entries labeled `False`/`True`; `choice`/`score` = N entries keyed by `label`. Probability distributions are keyed by `str(label)` (Score drops `legend`); added `RefusalAnswer`, multimodal `state` (content parts), and optional `confidence`/`unknown_probability`, with derived values in `converters/decision/derived.py`.
- **OpenAI Decisions converter** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889)): `OpenAIDecisionsConverter` maps the IR to OpenAI's `POST /v1/decisions` wire (`input` + a `questions` **array** with `predicate`/`choice`/`score` and `name`; answers keyed by `name`, including `refusal`). Registered as the `openai_decisions` decision format; gateway route `/v1/decisions`; provider shim `openai_decisions`.
- **System One multimodal (`images[]`)** (issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889)): `SystemOneDecisionConverter` now reads/writes the System One family's top-level `images` extension (Cloudflare Clef, classifier.dev) — IR `state` content parts map to `state` (text) + `images` (inline base64 data URLs) and back, so multimodal decision evidence round-trips with the Jev-family providers.
- **Decision format rename** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889)): the System One decision format is now `systemone` (was `typesafe`) and the converter class is `SystemOneDecisionConverter` (was `TypeSafeDecisionConverter`) — named after the vendor-neutral System One standard shared by TypeSafe/Jev, Clef, Laya, and classifier.dev rather than a single vendor.
- **Decision source-format detection** (issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889)): the gateway decision handler now detects the source format from the route and body (`/v1/decisions` → `openai_decisions`, `/v1/systemone` → `systemone`, otherwise the body shape) instead of relying solely on `default_decision_format`.
- **Multi-provider routing** (PRs [#664](https://github.com/Oaklight/llm-rosetta/pull/664), [#663](https://github.com/Oaklight/llm-rosetta/pull/663), [#662](https://github.com/Oaklight/llm-rosetta/pull/662)): configure multiple upstream providers per model with weighted round-robin, API key affinity for prompt cache optimization, and per-request token usage tracking.
- **ALCF provider support** (PRs [#673](https://github.com/Oaklight/llm-rosetta/pull/673), [#675](https://github.com/Oaklight/llm-rosetta/pull/675), [#679](https://github.com/Oaklight/llm-rosetta/pull/679), [#689](https://github.com/Oaklight/llm-rosetta/pull/689), [#680](https://github.com/Oaklight/llm-rosetta/pull/680), [#681](https://github.com/Oaklight/llm-rosetta/pull/681)): provider shims for ALCF inference clusters with Globus OAuth2 token management, automatic token refresh on 401, and 30-day session-policy detection. `token_command` config field enables dynamic API key refresh for any provider with expiring tokens.
- **AskSage provider shims** (PR [#814](https://github.com/Oaklight/llm-rosetta/pull/814)): shims for AskSage OpenAI, Anthropic, and Gemini endpoints. New `connection.auth_header` field for per-shim auth header overrides.
- **Model type registry** (PRs [#730](https://github.com/Oaklight/llm-rosetta/pull/730), [#735](https://github.com/Oaklight/llm-rosetta/pull/735), [#736](https://github.com/Oaklight/llm-rosetta/pull/736), [#737](https://github.com/Oaklight/llm-rosetta/pull/737)): declarative registry for model types (LLM, embedding, rerank, decision). Adding a new type requires one descriptor — admin UI badges, provider modal, test menus, and routes are all data-driven from registry metadata.
- **Provider circuit breaker** (PR [#727](https://github.com/Oaklight/llm-rosetta/pull/727)): three-state circuit breaker that short-circuits requests to providers with sustained high error rates. Configurable per-provider, disabled by default.
- **Admin panel improvements**: hash-routed tab navigation with browser back/forward support; independent Settings tab; server ops log with event filters and retention policies; session auth via httponly cookies; log/error dump batch cleanup; token usage breakdown modal; multi-window rate limiting; multi-provider model editing with inline weight controls.
- **Routing intelligence** (PRs [#784](https://github.com/Oaklight/llm-rosetta/pull/784), [#785](https://github.com/Oaklight/llm-rosetta/pull/785), [#770](https://github.com/Oaklight/llm-rosetta/pull/770), [#676](https://github.com/Oaklight/llm-rosetta/pull/676)): routing loop detection via hop-count header, same-format provider affinity to avoid unnecessary conversion, cache-aware affinity routing, and rendezvous hashing for key affinity.
- **Fidelity verification** (PRs [#772](https://github.com/Oaklight/llm-rosetta/pull/772), [#796](https://github.com/Oaklight/llm-rosetta/pull/796)): persistent round-trip conversion baselines with regression detection. Always-on shadow diff for same-format routes.
- **Bidirectional tool namespace mapping** (PR [#753](https://github.com/Oaklight/llm-rosetta/pull/753)): collision-free tool name translation between provider naming conventions with round-trip preservation.
- **Observability** (PRs [#702](https://github.com/Oaklight/llm-rosetta/pull/702), [#755](https://github.com/Oaklight/llm-rosetta/pull/755), [#756](https://github.com/Oaklight/llm-rosetta/pull/756)): cache and reasoning token tracking, TTFB metrics, client disconnect detection, and a vendored tracing profiler replacing pyinstrument.
- **Tool-name collision surfacing** (issue [#907](https://github.com/Oaklight/llm-rosetta/issues/907)): the collision half of `dedupe_tool_definitions` now reaches the client and the error dump. When a request declares one tool name with differing definitions and the upstream rejects the request body with `400` or `422`, the gateway appends a `[llm-rosetta]` hint naming the conflicting tools to the `error.message` the client receives, so a bare "Tool names must be unique" is attributable to the client's own tool list instead of looking like a gateway fault. The structured map (`{name: distinct-definition count}`, capped at ten names) is recorded in the error dump under a new `tool_name_collisions` column (schema migration v3) and shown in the admin dump viewer. Of the parsed error document only `error.message` changes; the body is re-serialized, so whitespace and escaping may normalize while non-ASCII text stays readable. A non-JSON body, one whose envelope carries no string message, or one already carrying the marker is left untouched, as are `401`/`403`/`404`/`429` and `5xx`. The dump half is wired and unit-tested, and shows up in the admin viewer wherever the dump row itself is written; on a database created from 2026-10-08 onward it is not observable end to end, because the error-dump insert fails its `request_log` foreign key — the log row is written after the error path — so no dump, and therefore no collision map, is stored (issue [#964](https://github.com/Oaklight/llm-rosetta/issues/964)). Databases created before that foreign key existed are unaffected.

### Changed (**breaking**)

- **(breaking) The ALCF shims take canonical `alcf_<cluster>--<format>` names, and the picker's grouping is declared instead of parsed** (epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927), PR [#1035](https://github.com/Oaklight/llm-rosetta/pull/1035)): a shim name is `vendor--base` and the admin picker read it that way, but `alcf--metis` put a *deployment* where the wire format belongs — which is why the Provider Type dropdown showed `metis · chat_completions` (two call sites appended the base whenever a label was not one) and why the grammar had nowhere to put a second format per cluster. So the cluster moves into the provider segment (`alcf--metis` → `alcf_metis--chat_completions`, likewise `minerva` and `sophia`), and the three old spellings keep working through `LEGACY_SHIM_ALIASES`: `get_shim()` still resolves them and `correct_legacy_names` rewrites `providers.*.shim` / `.type` in a config on load, after the usual one-time `.pre-migration.bak`. They are removed with the rest of the legacy names in 0.16.0. Grouping stops being an inference at the same time: `provider.yaml` gains `group` (the top level) and `group_label` (its title), both falling back to the name split, so a shim that declares neither — including every third-party and external one — renders exactly as it always has. `display_name` goes back to its documented meaning, this shim's own label, and supplies the row label **only for a shim that also declares a grouping field**; for one that does not, the older reading is kept, where `display_name` titled the group — otherwise a vendor's name would be printed twice, once as the group title and again on its only row, and the shims that would move are precisely the third-party ones written against the schema this replaces. Kilo's `display_name: Kilo` becomes `group_label: Kilo` for the same reason. Every resolved value now carries the branch that produced it (`declared` / `derived` / the legacy `display_name`), so "an undeclared shim is unaffected by design" stays separable from "it happens to be unaffected today". The tree builder moves out of `tabs/providers.js` into a pure `core/shim-tree.js`, with `tests/js/shim-tree.test.mjs` (`make test-js`, run as a CI step) comparing it against an oracle of the pre-`group` builder. Also restores the per-vendor `--anthropic` aliases that [#917](https://github.com/Oaklight/llm-rosetta/pull/917) left behind: `argo--anthropic`, `asksage--anthropic`, `minimax--anthropic` and `openrouter--anthropic` have been unresolvable since it renamed the `anthropic` base to `messages`, and `test_every_shim_alias_target_is_registered` now fails if an alias ever points at a shim that is not registered.

- **(breaking) Google SDK shape moved `system_instruction` into `config`** (issue [#936](https://github.com/Oaklight/llm-rosetta/issues/936)): `GoogleGenerateConverter` with `output_format="sdk"` (the default, and what `convert()` / `convert_response()` return) emitted `{"model", "contents", "config", "system_instruction"}`. That top-level key made the dict unusable the way it was documented: `client.models.generate_content(**body)` raises `TypeError: unexpected keyword argument 'system_instruction'` on every `google-genai` version from 1.0.0 to 2.29.0, because `Models.generate_content()` takes only `model`, `contents` and `config` — and working around it by passing just those three silently dropped the system instruction from the wire. The shape is now exactly the `generate_content` keyword arguments, with the instruction inside `config`, where `GenerateContentConfig` declares it and where `llm_rosetta.types.google.GenerateContentRequest` already documented it. `request_from_provider` reads either placement. The REST shape (`output_format="rest"`, the gateway path) is unchanged byte for byte.

  **Migration:** for `convert()` / `convert_response()` / `request_to_provider(output_format="sdk")` output, read `body["system_instruction"]` as `body["config"]["system_instruction"]`.

  | Before | After |
  |--------|-------|
  | `{"model", "contents", "config", "system_instruction"}` | `{"model", "contents", "config": {"system_instruction": …}}` |

  A `generate_content(**body)` call that previously raised `TypeError` now works as written, and the system instruction reaches the model.

- **(breaking) `tool_ops` provider names standardized** (issue [#921](https://github.com/Oaklight/llm-rosetta/issues/921), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): the `tool_ops` convenience API now uses the **base converter types** as its canonical provider names — `anthropic`, `google_generate`, `google_interactions`, `chat_completions`, `open_responses`. Every previous spelling (`openai_chat`, `openai_responses`, `google`, `openai-chat`, `openai-responses`, `open-responses`, `google-genai`, `google-generate`, `chat-completions`, …) still resolves but emits a `DeprecationWarning` and is removed in **0.16.0** ([#919](https://github.com/Oaklight/llm-rosetta/issues/919)). Migration: `tool_ops.to_provider(tool, provider="openai-chat")` → `provider="chat_completions"`.

- **Gateway module reorganization** (PRs [#739](https://github.com/Oaklight/llm-rosetta/pull/739), [#738](https://github.com/Oaklight/llm-rosetta/pull/738), [#744](https://github.com/Oaklight/llm-rosetta/pull/744)): backend files moved into `gateway/middleware/` and `gateway/pipelines/` subpackages; admin JS into `js/core/`, `js/tabs/`, `js/components/`. **Old flat import paths (`gateway.auth`, `gateway.embeddings`, etc.) no longer work** — use `gateway.middleware.*` / `gateway.pipelines.*`. See `argo-proxy` [#179](https://github.com/Oaklight/argo-proxy/pull/179) for migration.

- **(breaking) Anthropic base renamed to `messages`; shim names canonicalised** (PR [#923](https://github.com/Oaklight/llm-rosetta/pull/923), issue [#917](https://github.com/Oaklight/llm-rosetta/issues/917)): the Anthropic Messages API base type is now the neutral `messages` — the format is a de-facto standard re-implemented by many providers, so it is no longer tied to the `anthropic` vendor. The converter module moved to `llm_rosetta.converters.messages` and the class to `MessagesConverter` (`AnthropicConverter` / `converters.anthropic` remain as deprecated aliases). Shim names are now canonical `<provider>--<base>`: `openai--chat_completions`, `openai--open_responses`, `anthropic--messages`, `google--google_generate`, `google--google_interactions`, and `<provider>--chat_completions` / `<provider>--open_responses` for the rest. The shim registry is canonical-only — a base *format* never resolves to a vendor shim (`get_shim("google_generate")` returns `None`; `google_generate` names the format while `google--google_generate` names the shim). Base converters are vendor-neutral: a bare `type: <base>` no longer carries an implied vendor `base_url` or API-key env var — those are shown only as an admin placeholder, so you must set `base_url` explicitly or use a `shim:`. To ease the transition the gateway rewrites legacy names in `providers.*.shim` / `providers.*.type` at startup, once and idempotently, saving a `.pre-migration.bak`; a name it cannot map is rejected with a `ValueError` pointing at the docs. Legacy names still warn and resolve at the library boundary until **0.16.0** ([#919](https://github.com/Oaklight/llm-rosetta/issues/919)), when they are removed.

  | Legacy | Canonical |
  |--------|-----------|
  | base `anthropic` | `messages` |
  | shim `anthropic` | `anthropic--messages` |
  | shim `openai` | `openai--chat_completions` |
  | shim `openai_responses` | `openai--open_responses` |
  | shim `google` / `google_generate` | `google--google_generate` |
  | shim `google_interactions` | `google--google_interactions` |
  | shim `<provider>--openai_chat` | `<provider>--chat_completions` |
  | shim `<provider>--openai_responses` | `<provider>--open_responses` |
  | type `openai_chat` | `chat_completions` |
  | type `google` / `google-genai` | `google_generate` |
  | base `google_generate` / `google_interactions` | unchanged |

### Changed

- **Tool groups are named where the target is known** (issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)): colliding tool names are qualified on the target leg rather than in the source converter that flattened the container, so the pass judges the names that actually go upstream. A container with no name of its own is given a short content-derived identifier (`g` + seven base32 characters, 32 bits, stable per request so the prompt cache holds) instead of flattening to an empty group, and the malformed container is reported once per request rather than once per process. Byte-identical children of a *generated* group fold into one tool (#913); a group the *client* named keeps them apart, as #955 settled and as DeepSeek and Volcengine require of a container's children. Nothing else moves — top-level and named-group collisions, the byte-identical collapse, and `tool_name_collisions` behave as before.

- **Intrinsics and baseline keyed on the provider, not the format name** (issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): the pipeline's two remaining literal-format-name decisions are gone. `ConversionPipeline._same_format` is now one named property comparing the two provider specs the caller passed; it gates both `baseline` and the intrinsic shortcut. It deliberately compares *specs* rather than converters — a bare `chat_completions` source and a `deepseek--chat_completions` target share a converter class but must stay distinct, because a same-format passthrough would skip the target shim's IR-level traits (its reasoning config, its transforms). Whether a client format can name a server tool natively moved from `self._source_provider == "chat_completions"` onto a converter attribute, `BaseConverter._SUPPORTS_NATIVE_SERVER_TOOLS` (`False` for Chat Completions), so the direction is unchanged but the classification now lives with the format that owns it — and is broader than the literal name it replaced: a chat-completions *shim* source (`deepseek--chat_completions`) now promotes a plain function named after a supported kind, where the `== "chat_completions"` comparison left it alone. Same format, same converter, just as unable to name a server tool; only reachable when the target shim declares that kind.

- **Tool grouping is an IR field now** (issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)): `ToolDefinition` gains `tool_group` and `tool_group_description`, and the group a tool was declared under is no longer parked in `metadata.namespace`. The word `namespace` is reserved from here on for the wire field the Responses API family uses for the same concept — the vendor-neutral Open Responses spec has no grouping concept at all — so it now appears only where a provider body is read or written. No behaviour change: containers are still flattened at the source and colliding children are still renamed for flat targets; deferring that decision to the target-aware stage, and giving tool *calls* a group of their own, follow on the same issue.

- **Tiered same-format preference** (issue [#911](https://github.com/Oaklight/llm-rosetta/issues/911), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): `server.prefer_same_format` no longer does a bare base-type string match. It picks an **anchor** first — the provider routing would select with the flag off, so the full route's counters advance exactly as they would there — then realigns to the client's format in tiers: same shim and endpoint as the anchor (failover ordering only, since a shim already fixes the format), then the same upstream **account** in the client's format, then any provider in that format, and finally the anchor itself as a cross-format fallback. This makes the preference do real work: a client whose anchor sits on a multi-format vendor stays on that vendor's account instead of moving to another account that merely speaks its format. Account identity is derived from the credential source, never declared — the same `token_command`, or the same key value (a literal key and an env-resolved key with the same value count as one key); only a hash of the credential is compared, keyless providers never group, and `base_url` is deliberately excluded so a vendor fronting several formats from different paths on one key is still one account. No new config key. The old `(model, source_provider)` sub-route cache is gone — the choice now depends on the client identity through the anchor — and a request always advances the full route's counter, so enabling the flag no longer shifts the anchor distribution — tier 3 can still serve the request from another account, which is the point.

- **Admin type picker: top level / base type split** (follows PR [#917](https://github.com/Oaklight/llm-rosetta/pull/917)): the provider modal's flat **Provider Type** `<select>` — which listed raw `vendor--base` shim names — is replaced by a **Top Level** dropdown and a **Base Type** dropdown on one row. The Top Level dropdown shows each vendor's logo and carries a `Top level | Tree` toggle: top-level view lists only first-level names, tree view expands each vendor's base types underneath with `tree`-style guides (logos on the root row only). Vendor-neutral base formats (`chat_completions`, `open_responses`, `messages`, `google_generate`, `google_interactions`) appear as their own childless roots; a vendor whose shim has no `--` separator (e.g. `moonshot`) shows its base type as the single child. Switching Base Type rewrites the provider's `type`; the view choice persists in `localStorage`. Provider cards now show the resolved base type as a badge beside the logo. In dark mode, logos are seated on a light chip instead of being inverted, because inverting shifted full-colour vendor marks (e.g. alcf).

- **Provider-name canonicalisation** (issue [#908](https://github.com/Oaklight/llm-rosetta/issues/908), PR [#914](https://github.com/Oaklight/llm-rosetta/pull/914)): `google_generate` is now the canonical name for the Google generateContent format; `google` and `google-genai` are deprecated aliases that auto-migrate at every boundary (gateway JSONC `type:`/`shim:`, request detection, library `source_provider`/`target_provider`, shim resolution) through a new shared mechanism (`llm_rosetta.provider_names`). The legacy spellings keep working and emit a `DeprecationWarning`; migration is normalisation at load/entry, not an on-disk rewrite. The bundled Google shim is registered under the canonical `google_generate` name. The deprecated spellings are planned for removal in **0.16.0** (see [#919](https://github.com/Oaklight/llm-rosetta/issues/919)).

- **`openai_chat` renamed to `chat_completions`** (issue [#910](https://github.com/Oaklight/llm-rosetta/issues/910), PR [#916](https://github.com/Oaklight/llm-rosetta/pull/916), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): the OpenAI Chat Completions base type is now the neutral `chat_completions`; the converter module moved to `converters/chat_completions/` and the class to `ChatCompletionsConverter`. The legacy `openai_chat` / `openai-chat` spellings auto-migrate at every boundary (deprecation warning), and `OpenAIChatConverter` / `converters.openai_chat` imports keep working. Planned removal of the legacy names in **0.16.0** (see [#919](https://github.com/Oaklight/llm-rosetta/issues/919)).

- **Responses collapsed to a single `open_responses` base** (issue [#909](https://github.com/Oaklight/llm-rosetta/issues/909), PR [#920](https://github.com/Oaklight/llm-rosetta/pull/920), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): `open_responses` is now the only Responses base; the OpenAI profile is a shim named `openai--open_responses` (`base: open_responses`) that opts in through shim settings (`response_id_prefix: resp_`, `preserve_fields`, `store_default: true`). The converter module moved to `converters/open_responses/` (old import paths kept via a deprecated compat package); `OpenAIResponsesConverter` is retained as a behavior-preserving preset. Legacy `openai_responses` auto-migrates with a deprecation warning; planned removal in **0.16.0** (see [#919](https://github.com/Oaklight/llm-rosetta/issues/919)). **Behavior change:** the two names are now the same base, so `prefer_same_format` may route between them and they no longer drop each other's provider-bound passthrough items (walks back part of #894).

- **`open_responses` differentiated from `openai_responses`** (issue [#493](https://github.com/Oaklight/llm-rosetta/issues/493)): the Open Responses spec is now the base converter and the OpenAI Responses API a derived profile. `OpenResponsesConverter` implements the vendor-neutral standard (no response-id prefix, stateless `store` default, spec-level echo fields); `OpenAIResponsesConverter` derives from it and restores OpenAI's `resp_` prefix, `store: true` default, and OpenAI-only lifecycle fields. The `OpenAIResponses*` ops classes are renamed to `OpenResponses*` with backward-compatible aliases. `open_responses` gains its own vendor-neutral shim (no default host, no prefix); the gateway no longer defaults it to OpenAI's URL and raises a clear error when no `base_url` is configured. The `allowed_tools` `tool_choice` variant round-trips losslessly instead of degrading to `auto`, and `compaction` items survive same-format round-trips (dropped when converting to a different dialect). Since the spec has no canonical host, the gateway now defaults `type: open_responses` to OpenRouter (`https://openrouter.ai/api/v1`, key env `OPENROUTER_API_KEY`); deployments relying on the old implicit OpenAI default, or targeting another host, must set `base_url` explicitly.
- **`converters/base/tools/` package** (PR [#871](https://github.com/Oaklight/llm-rosetta/pull/871), issue [#858](https://github.com/Oaklight/llm-rosetta/issues/858)): `converters/base/tools.py` restructured into a package consolidating all 8 tool-related helper modules (previously scattered under `helpers/`) — `call_id.py`, `content.py`, `orphan_fix.py`, `call_unwind.py`, `intrinsic.py`, `batch.py`, `schema.py`, `multimodal_patch.py` — plus `ops.py` for `BaseToolOps`. External import paths (e.g. `from llm_rosetta.converters.base.tools import sanitize_schema`) and old `helpers/` paths both preserved via re-export shims. No behavior change.
- **Google Generate intrinsic tool roundtrip** (PR [#874](https://github.com/Oaklight/llm-rosetta/pull/874)): `code_execution` call/result and `google_search`/`code_execution` tool definitions now round-trip through `tool_ops.py` (previously hardcoded in `message_ops.py`), bringing the converter to parity with Anthropic, OpenAI Responses, and Google Interactions on intrinsic tool A→IR→A fidelity. Supports both snake_case and camelCase provider keys.
- **Opaque reasoning data mapping unified across providers** (PR [#865](https://github.com/Oaklight/llm-rosetta/pull/865), issue [#857](https://github.com/Oaklight/llm-rosetta/issues/857)): `ReasoningPart.signature` is now reserved for verification signatures that coexist with visible text (Anthropic `thinking.signature`, Google `thoughtSignature`, Google Interactions `thought.signature`); `ReasoningPart.redacted_data` is reserved for opaque/encrypted blobs (Anthropic `redacted_thinking.data`, OpenAI Responses/Chat `encrypted_content`). OpenAI Responses and Chat previously stored `encrypted_content` under `signature`/`provider_metadata`; Google Generate previously stored `thoughtSignature` under `provider_metadata`. Streaming `ReasoningDeltaEvent.encrypted_content` is unaffected.
- **Intrinsic IR transforms** (PR [#875](https://github.com/Oaklight/llm-rosetta/pull/875), issue [#849](https://github.com/Oaklight/llm-rosetta/issues/849)): `BaseConverter` gains a `_INTRINSIC_IR_TRANSFORMS` class attribute for declaring format-level IR transforms that run even without a shim. Anthropic, Google Generate, and Google Interactions converters declare `hoist_late_system_messages()` (and `auto_cache_breakpoints()` for Anthropic) as intrinsic — these are format requirements, not provider quirks. OpenAI Chat/Responses keep empty defaults since they natively support inline system messages.
- **Transforms package** (PR [#877](https://github.com/Oaklight/llm-rosetta/pull/877), issue [#876](https://github.com/Oaklight/llm-rosetta/issues/876)): all transform primitives extracted from `shims/transforms.py` to a new top-level `llm_rosetta/transforms/` package (`ir.py` for IR-level, `body.py` for body-level). Eliminates the converter→shim cross-layer dependency. The old `shims/transforms.py` is a deprecated re-export shim — import from `llm_rosetta.transforms` instead.
- **Intrinsic tool type** (PR [#840](https://github.com/Oaklight/llm-rosetta/pull/840), issue [#181](https://github.com/Oaklight/llm-rosetta/issues/181)): replaced individual server tool type literals (`web_search`, `code_interpreter`, `file_search`) with a single `intrinsic` IR tool_type. The specific kind is stored in `provider_metadata["intrinsic_kind"]`. Intrinsic tools pass through in same-format conversions and are stripped in cross-format conversions. All five converters updated: Anthropic `server_tool_use` + `*_tool_result` blocks, OpenAI Responses `web_search_call`/`code_interpreter_call`/`file_search_call`/`shell_call`/`computer_call`, Google Generate `executableCode`/`codeExecutionResult`, Google Interactions all 6 server step types. Shared helpers in `converters/base/tools/intrinsic.py` (moved from `helpers/tool_intrinsic.py` by PR #871).
- **Shim transforms** (PRs [#684](https://github.com/Oaklight/llm-rosetta/pull/684), [#685](https://github.com/Oaklight/llm-rosetta/pull/685), [#716](https://github.com/Oaklight/llm-rosetta/pull/716), [#799](https://github.com/Oaklight/llm-rosetta/pull/799), [#812](https://github.com/Oaklight/llm-rosetta/pull/812)): new `default_tool_description` and `response_body_transforms` primitives; Argo and DeepSeek Responses API shims; `ProviderShim` fields grouped into `ConnectionConfig`/`ToolsConfig`.
- **Build system** (PR [#824](https://github.com/Oaklight/llm-rosetta/pull/824)): extracted `build-wheel` Makefile target from `deploy-dev`. Unified version parsing to `sed -n` across Makefile and CI for POSIX/macOS compatibility.

### Fixed

- **A failing startup repair no longer skips the four behind it** (epic [#1027](https://github.com/Oaklight/llm-rosetta/issues/1027), PR [#1044](https://github.com/Oaklight/llm-rosetta/pull/1044)): the five data repairs that run after the socket is open shared one `try`, so the first failure — a locked database, a column an older release never created — silently skipped everything after it. The two an operator opens the admin panel for, the token totals and the usage buckets, are the last two, so the failure that mattered most was the one that could hide the most. Each step now fails on its own: the failure is logged naming the step and recorded against it, the steps behind it still run, and the ones that succeeded still report what they changed. The change is verified where the defect lived — the test that asserted only that the readiness flag was cleared now also asserts that the later steps were called.

- **A citation could be lost without a trace, and Anthropic's citation vocabulary was modelled as a tag the live API rejects** (issue [#459](https://github.com/Oaklight/llm-rosetta/issues/459), PR [#1025](https://github.com/Oaklight/llm-rosetta/pull/1025)): the chat completions response writer promised in its docstring that "the caller reports the loss" and the caller had no `else`, so an Anthropic `char_location` aimed at chat completions left no sink entry and no log line while the same citation through the Messages writer reported properly; its history leg had a bare `pass` where the annotation belongs (measured: this endpoint *does* accept `annotations` on an assistant message in a request); and its stream reader had no annotation branch while the writer fell back to the base drop, so streamed citations were lost inbound and could not be sent outbound (measured: a search model streams them in `choices[].delta.annotations`, nested under `url_citation`). All three are wired, and the drop assertion is driven through the response writer rather than by calling the reporting helper — the helper is not what can silently regress, the call site is. Two conventions came out of the same review: a citation whose source dialect has no message-level span writes **no** span rather than `0/0` (which reads as "the first characters" rather than "unknown"), and the matrix's write cells pin the exact native spelling rather than truthiness, so a renamed kind or a leaked foreign tag fails. The Anthropic wire types are corrected with it: they modelled the whole citation vocabulary as a single `{"type": "text_citation"}`, a tag the API rejects with `400 Input tag 'text_citation' found using 'type' does not match any of the expected tags`, and now carry the five it accepts.

- **The API reference asked the reader to follow an empty link** (PR [#1040](https://github.com/Oaklight/llm-rosetta/pull/1040)): `IREmbeddingResponse`'s docstring writes a JSON path as `` `embeddings.{type}[][]` ``, and Markdown reads the trailing brackets as a *collapsed reference link* with an empty label — one no definition can satisfy — so the API reference build reported `unresolved autoref` and the built page carried an anchor with no target. Both mentions (the class and the module docstring) are code spans now, which is what they always were: the brackets are part of the path, not markup.

- **The `open_responses` intrinsic (server-tool) path was broken in both directions, streaming and non-streaming** (issue [#985](https://github.com/Oaklight/llm-rosetta/issues/985), PR [#987](https://github.com/Oaklight/llm-rosetta/pull/987)): an intrinsic call or result either vanished or was rewritten as a client call, so a `web_search` or `code_interpreter` turn could not survive the trip. On the streaming read, `web_search_call` / `code_interpreter_call` output items produced no IR events at all; `_handle_p_output_item_added_to_ir` now dispatches them to an intrinsic `ToolCallStartEvent` — the kind derived from the item type through the shared `_ITEM_TO_INTRINSIC_KIND` table, never from the tool name — and moves the `done` payload (`action` / `code` / `outputs`) as a `ToolCallDeltaEvent`, while a `*_call_output` item becomes a one-shot `ToolResultEvent` built by the same reader the non-streaming path uses. On the streaming write, an intrinsic `ToolCallStartEvent` used to become a client `function_call` and a `ToolResultEvent` was dropped; the writer now re-emits the provider's own server item type, keeps the payload across the delta, rebuilds the item at `output_item.done` / `response.completed`, and for a result replays the stored native item or rebuilds a `*_call_output` — a web-search result, which rides on the call item, and any unrenderable kind are dropped silently, and a server call is never downgraded to a client one. On the non-streaming write, `_build_server_item` now emits the fields the API requires — `id` on every server item (plus `status` on a web search, restored from the read side) and `code` / `container_id` / `outputs` for the interpreter — and no longer puts the client `call_id` / `arguments` pair on a server item. Implemented once in `OpenResponsesConverter` (the vendor-neutral base) so `OpenAIResponsesConverter` inherits it, and live-verified against the OpenAI Responses API on `gpt-5-nano`: every intrinsic kind pairs correctly in both modes.

- **An intrinsic result was paired with its call by kind alone, so the two writers drifted** (issue [#996](https://github.com/Oaklight/llm-rosetta/issues/996), fixed inside PR [#987](https://github.com/Oaklight/llm-rosetta/pull/987)): the result writer decided whether to drop an intrinsic result from the *kind* alone, while the call writer decided from `(kind, fields, leg)`, so a call the writer degrades to a client `function_call` could be left with no output at all, and a call the IR replays natively could be handed a `function_call_output` whose `call_id` no client call answers. Both writers now consult one shared predicate `_intrinsic_call_degrades_on_history`, and `ir_messages_to_p` pre-scans the message list into `degraded_intrinsic_calls` (sanitized `call_id` → bool) and threads it to the result writer, because a call and its result live in different messages: on the history leg a degraded call takes a matching `function_call_output` and a natively replayed call takes none, on the response leg the native `*_call_output` is kept, and `web_search` is always dropped. The duplicated `_INTRINSIC_KIND_TO_RESULT_ITEM` definition — a literal shadowed by a derived form — is merged into the derived form, so the call and result item maps cannot drift. Eight new tests drive each intrinsic kind through the message writer and assert the pairing contract, and both guards were mutation-checked (removing either turns three tests red).

- **The `google_generate` intrinsic (server-tool) path was broken in both directions, streaming and non-streaming** (issue [#986](https://github.com/Oaklight/llm-rosetta/issues/986), PR [#988](https://github.com/Oaklight/llm-rosetta/pull/988)): a code-execution turn raised a `ValidationError` on the non-streaming read; the non-streaming write emitted a client `functionCall` and dropped the result; the streaming read dropped the server parts; and the streaming write downgraded the call and lost the result. `executableCode` / `codeExecutionResult` now read into a validated IR response with the parts inline in order — the streaming reader emits an intrinsic `ToolCallStartEvent` (kind in provider metadata, native part carried for a same-dialect restore) plus an input delta so other formats receive the input, and a `codeExecutionResult` emits the one-shot `ToolResultEvent` paired by kind — and the writer re-emits Google's own part types, either immediately from the native part or rebuilt from the accumulated input at finish, with a `tool_result` handler writing `codeExecutionResult`. A server call is never written as a client `functionCall`, and an intrinsic kind Google cannot render is still dropped silently rather than downgraded; a truncated or non-object server-call flush is dropped instead of emitted, and a client result is kept out of the model turn. Live-verified with `gemini-2.5-flash` through the converter itself, both modes.

- **The `google_interactions` intrinsic (server-tool) path was broken in all four directions, and one failure cost a live `400` on replay** (issue [#992](https://github.com/Oaklight/llm-rosetta/issues/992), PR [#993](https://github.com/Oaklight/llm-rosetta/pull/993)): the non-streaming read dropped the server result and lost the call `signature`; the non-streaming write carried a `name` on the server call, which the API rejects with `400 Unknown parameter 'name'`; the streaming read emitted no intrinsic events; and the streaming write downgraded the call to a `function_call` and dropped the result. `_do_response_from_provider` now merges the `role: "tool"` server-result message into the assistant turn so the result survives in the response model; `p_server_call_to_ir` reads `step["signature"]` into `provider_metadata.google.thought_signature` and uses the canonical kind as the call identity, `ir_function_call_to_p` drops `name` and restores the signature, and `p_server_result_to_ir` keeps the payload in its wire shape (an array for `google_search_result`, a string for `code_execution_result` — a replay rejects the flattened string with `400 Expected an array, got string`); the stream reader and writer emit the intrinsic events and re-emit the provider step types, with the `tool_result` handler registered in this converter's own dispatch. Argument fragments are accumulated into one object, and an unrenderable intrinsic kind is dropped instead of fabricating a step type. The kind is always folded from the step type, never inferred from a step or tool name, and a server call is never downgraded to a client call. Live-verified with `gemini-3.7-flash` on `/v1beta/interactions`: a `google_search` and a `code_execution` turn round-trip through the IR and the library-written turn replays `200`.

- **An assistant turn carrying a server-tool result failed IR validation, and the result was dropped when it did not** (issue [#965](https://github.com/Oaklight/llm-rosetta/issues/965)): Anthropic returns a server-executed tool's result inline in the assistant turn, and the messages reader normalized every inline `tool_result` into a separate `role: "tool"` message — a rule written for request history, where a *client* result has to become a tool message. On a response that split produced a **list** where `ResponseChoice.message` holds exactly one message, so `[server_tool_use, web_search_tool_result, text]` raised `Value at 'choices[0].message' does not match any variant of ...`. The split now keys on **what the result is** rather than on the role: a client result still normalizes to a tool message, while a server result stays inline in the turn that produced it, which is what `AssistantContentPart` now admits. The response writer rebuilds that inline result instead of dropping it — measured before the fix, a round trip through IR emitted `[server_tool_use, text]` and the result block simply vanished. Two related defects surfaced in review and are fixed with it: the shared orphan repair counted an answer only when it sat in a `role: "tool"` message, so an inline server result left its call unanswered and a synthetic `[No output available yet]` placeholder was injected **in every target format** (all four writers call it) — an inline `tool_result` part now counts as an answer; and a client result replayed inline in an assistant turn lost its payload to that same placeholder, which the kind-keyed split restores. Verified against the live API: a captured web-search turn round-trips block for block, `caller` and the result payload included.

- **A streamed server-tool call was rewritten as a client tool call** (issue [#974](https://github.com/Oaklight/llm-rosetta/issues/974)): Anthropic returns a server-executed tool call as a `server_tool_use` block, but the streaming reader emitted the same `ToolCallStartEvent` for it and for a client `tool_use` without saying which it was, and the writer hardcoded `"type": "tool_use"` — so a same-format stream downgraded a server call to a client one. That is not cosmetic once both halves work: the client would replay a client call carrying a server-side `*_tool_result`, and Anthropic would read the turn as one where the *caller* executed the tool. The writer now mirrors `MessagesToolOps.ir_tool_call_to_p`, the same branch the non-streaming path has always had — `tool_type == "intrinsic"` re-emits `server_tool_use` with the canonical kind (resolved by `get_intrinsic_kind`, never inferred from the tool name), everything else re-emits `tool_use` — and the reader marks the event `tool_type="intrinsic"` with the kind under `provider_metadata["intrinsic_kind"]`, exactly as the non-streaming `ToolCallPart` carries it. Carrying the marker also closed two fields the block has and the stream dropped: `ToolCallStartEvent` gains `cache_hint`, and a non-`direct` `caller` now rides `provider_metadata` and is restored only on a client `tool_use`, matching the restriction `ir_tool_call_to_p` makes. Verified live against Anthropic web search: the captured `['server_tool_use', 'web_search_tool_result']` comes back as the same two blocks in order, with the call's name and `input` — including the `input_json_delta` that carried `{"query": ...}` — intact.

- **A config reload left `token_command` providers unseeded, so every request to them was rejected with 401 until a restart** (issue [#976](https://github.com/Oaklight/llm-rosetta/issues/976)): `_reload_gateway_config` — the path behind every admin save, the reload endpoint, and a shim-source change — rebuilds `GatewayConfig` and swaps it into app state, but token seeding and the refresh loops were started once, in `run_gateway`. A rebuilt `token_command` provider therefore came back holding the pending sentinel with no refresher: routing skipped it (its `ready` property is false), while the admin model-listing and connectivity-test handlers, which do not go through that gate, sent `Authorization: Bearer __token_pending__` upstream and reported the resulting 401 as "this provider may not support model listing". Observed on a production gateway whose three ALCF providers all use `token_command`: `/resource_server/metis/models` answered 200 for a freshly minted token but 401 through the gateway, and the deploy logged six "uses token_command" lines at reload and not one "Seeded API key" afterwards. `DeferredStartup.apply_config_reload` now re-attaches the manager to the replacement config, cancels the refresh tasks bound to the superseded `ProviderInfo` objects, and re-seeds the new providers off the request path; the admin handlers report "still starting up" while a provider is unseeded instead of forwarding the sentinel.
- **A streamed server-tool result never reached the IR, so it vanished from the client's stream** (issue [#972](https://github.com/Oaklight/llm-rosetta/issues/972)): Anthropic executes server-hosted tools and returns their results inside the assistant turn as a `*_tool_result` block — `web_search_tool_result`, `code_execution_tool_result`, and the `bash_` variants — delivered **whole in a single `content_block_start`** with no deltas to follow (captured live against `claude-haiku-4-5-20251001` with `web_search_20250305`). The streaming reader folded the neighbouring `server_tool_use` *call* block and emitted nothing at all for the result, so the client's stream had no carrier for it; the idempotent non-streaming path had mapped these blocks for years, but nothing produced one to exercise it until the intrinsic tool declarations made the capability reachable (#181 / `6c1b0f02cd`). The IR stream vocabulary had no event for a tool result either, so this was not a missing branch. There is now a one-shot `ToolResultEvent` — `tool_call_id`, `result`, `tool_type`, `is_error`, `cache_hint`, `block_index`, `provider_metadata` — mirroring `ToolResultPart` field for field; the Anthropic reader builds it with the same helper the non-streaming path uses, and the Anthropic writer replays the block at its original index, verbatim through the stored `_native` payload so fields the IR does not model (`caller`, the raw `content` shape) survive. Converters that cannot render it register no handler and drop it, which is the same best-effort degradation intrinsic results already have elsewhere. Verified end to end: the captured stream round-trips byte-for-byte, and the block's cache breakpoint is carried as `cache_hint` rather than lost. A rebuilt block (no `_native`) no longer emits `is_error` on a `*_tool_result`, because Anthropic rejects that field outright — `400 ... .is_error: Extra inputs are not permitted` — and the reader's dialect guard on `_native` is pinned so a foreign dialect's payload cannot be copied onto an Anthropic body.

- **Three lanes sent values or field names their upstreams reject — one of them crash-looping the endpoint** (issue [#961](https://github.com/Oaklight/llm-rosetta/issues/961)): `alcf--sophia` runs vLLM behind ALCF's Inference Service, which accepts `reasoning_effort` only at `low`/`medium`/`high` and answers **500**, not 400, for anything else — `none` included, despite vLLM documenting it as the way to disable thinking. The shim declared `effort_field` without an `effort_range`, so the IR ladder's `minimal`/`xhigh`/`max` reached the wire and a client asking for `max` broke the request outright. The same lane names its reasoning `reasoning` (vLLM renamed it from `reasoning_content`), which the Chat Completions reader does not look for, so its reasoning was dropped on both legs. It now declares `effort_range: [low, high]` — clamping `minimal` to `low` and `xhigh`/`max` to `high` — and renames the field on `message` and on `delta`; verified live end to end, where a request asking for `max` now sends `high` and returns 200 instead of 500, and the same response yields a thinking block plus text on the Anthropic side. `minimax--messages` is the opposite case: it has no effort parameter at all — its compatibility layer documents thinking as on/off/omitted, not a depth slider — which #757 had already measured, so it declares `effort_field: none` again instead of the `output_config.effort` PR #956 added; `none` rather than dropping the key, because an omitted field falls back to the loader default and the stray flat `reasoning_effort` goes back out. `minimax--chat_completions` does validate the value (`reasoning_effort="banana"` answers `400 invalid params`) and documents `low`…`max`, so it gains `effort_range: [low, max]`. `openrouter--chat_completions` had the same rename hole as ALCF — `message` only — so its streamed reasoning was dropped too, and it now covers `delta`. Each lane is pinned by tests, including pipeline-level checks that fail when a shim's `pre_ir_transforms` is emptied rather than passing silently.

- **Error dumps are written again on databases that enforce the `error_dumps` foreign key** (issue [#964](https://github.com/Oaklight/llm-rosetta/issues/964)): a dump names the request's log entry in `error_dumps.request_log_id`, but that row is written when the request finishes — after the error handling that captured the dump. With the FK present (any database created from 2026-10-08), the insert failed the constraint and `dump_error`'s catch-all swallowed it, so every dump taken before the response was returned was lost; only mid-stream dumps survived, and only because the log row is written before the stream is drained. The proxy now hands the handlers a store that queues those writes and replays them once the log row exists. Databases created before the FK existed never enforced it and kept working, which is why the loss was not visible everywhere. A dump write that still fails is now logged at `WARNING` with its traceback instead of `INFO`, so it cannot pass unnoticed again.

- **Reading an OpenAI Responses server item back to IR lost its payload** (follow-up to [#963](https://github.com/Oaklight/llm-rosetta/issues/963)): `p_tool_call_to_ir` built `tool_input` from `arguments`, but a native server item carries its payload under type-specific keys — a `web_search_call` nests it under `action`, a `code_interpreter_call` holds `code`/`outputs`/`container_id` as siblings. Both read back as `tool_input: {}`, so the query and the code were lost, and `tool_name` was the raw wire type (`web_search_call`) rather than the canonical kind that the definition side and `intrinsic_kind` already use. Three more things were wrong around it, all found on review: only the `search` action shape was lifted, so an `open_page` or `find` call still read back empty and replayed as a bare `web_search_call` with no action; the writer hardcoded `queries: [query]`, discarding the rest of a multi-query search; and it read `tool_input["query"]`, which a `file_search_call` never sets. The payload is now lifted whole — the action's own fields, with its discriminator carried aside and rebuilt on the way out — `tool_name` is the canonical kind, and every shape round-trips. Verified against a live `gpt-5-nano` response rather than a hand-written fixture.

- **A newer Anthropic code-execution version produced an intrinsic kind no shim declares** (follow-up to [#963](https://github.com/Oaklight/llm-rosetta/issues/963)): `code_execution_20250825` and the later versions run bash under the hood and report it under a different runtime name — the call is a `server_tool_use` named `bash_code_execution` and the result block type is `bash_code_execution_tool_result`. The read path took the wire spelling as the kind, so those landed on `bash_code_execution`, which no shim declares, and cross-format history degraded rather than mapping to the intrinsic kind — on Responses to a plain function call carrying the raw kind as its name, and on Messages to a `server_tool_use` still under that raw name. (Only the *definition* side is nameless: an intrinsic kind the target cannot express emits `{}`.) The aliases now fold onto the canonical `code_execution` on the call and the result side, and on the streaming reader as well as the block reader — folding only the non-streaming path had the two disagreeing about the same response. Verified against a live `claude-haiku-4-5` response, including that the API accepts the folded name on replay. Emitting `code_execution_20250522` stays correct: the API still accepts it, and a newer version a client declares round-trips verbatim through the existing `_native` payload.

- **A shim-spelled `target_provider` was invisible to three capability gates** (issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): `ConversionPipeline("openai_chat", "deepseek--chat_completions")` — naming the shim as the target *provider* instead of passing `target_shim=` — emitted no reasoning field at all for `{"reasoning_effort": "none"}`, because `enforce_reasoning`, `enforce_custom_tools` and `apply_ir_transforms` each read the raw `target_shim` argument (which is `None` in that shape) rather than the shim the pipeline had already resolved from the target provider. The same construction with `target_shim=<shim object>` emitted `{"thinking": {"type": "disabled"}}`, so the two spellings of the same target disagreed. All three now take the resolved shim — the one the response-id prefix and the shim-declared Responses traits already used. Sweeping every chat shim as the target provider shows the effect is wider than that one pair: seven shims now gain the target's reasoning config where nothing was emitted before (`deepseek--chat_completions`, `deepseek--open_responses`, `minimax--chat_completions`, `moonshot`, `volcengine--chat_completions`, `volcengine--open_responses`, `zhipu`), and six stop having their custom tools downgraded to a nameless function (`alcf--minerva`, `alcf--sophia`, `argo--chat_completions`, `argo--open_responses`, `openai--chat_completions`, `openai--open_responses`) — the same mechanism, through the custom-tools gate. The half a user notices is the request body rather than a capability field: every chat-completions shim carries `hoist_late_system_messages()`, so a late `{"role": "system"}` message stayed inline for an implied target where the explicit form hoists it into `<system>…</system>` on a user turn — reshaping what the upstream sees, or leaving it to reject the request — and a `custom` tool aimed at `alcf--sophia` (`custom_tools: true`) was downgraded to a function with a synthesised `input` schema instead of being kept. The target shim's own body transforms now run for an implied target too — `deepseek--chat_completions`, for instance, strips `n`, `logit_bias`, `seed` and the penalty fields, which previously reached the upstream intact when the shim was named as the target provider. It is bounded to callers that name a *shim* as the target provider: a bare-format target implies nothing (`resolve_shim("chat_completions")` is `None`), and the gateway always passes a shim name, so its requests are unchanged. Naming the shim as the target provider and passing `target_shim=` explicitly now produce identical bodies across all 32 chat shims.

- **The gateway dropped every intrinsic tool definition** (issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)): `gateway/proxy.py` builds the pipeline with `target_shim=route.shim_name` — a shim *name* — while `resolve_intrinsic_tools` read `ToolsConfig.intrinsic_tools` straight off that argument, so on a string it found nothing and dropped every `type="intrinsic"` tool definition regardless of what the provider's shim declared. The library path passes a shim object and was never affected, so the same request kept its intrinsic tools through `convert()` and lost them through the gateway. The gate now resolves a shim name the way its sibling gates do (`enforce_custom_tools`, `enforce_reasoning`), and the pipeline passes the target shim it already resolves for the other traits.

- **A Chat Completions chunk carrying both `content` and `reasoning_content` emitted the text delta first** (issue [#950](https://github.com/Oaklight/llm-rosetta/issues/950), PR [#951](https://github.com/Oaklight/llm-rosetta/pull/951)): some upstreams flush the tail of the reasoning together with the first answer token in a single `delta`, and `ChatCompletionsConverter._handle_p_choice_to_ir()` read `content` first, so the chunk produced `TextDeltaEvent` before `ReasoningDeltaEvent`. Serialized to Anthropic, the text block opened first and the leftover reasoning then had to open a second `thinking` block *after* it, which clients render as a broken-up answer (`text "基本" / thinking "。" / text "对，但我想把这一点坐实。"`). Live on xAI `grok-4-fast-reasoning`, which intermittently emits `{"content": "No", "reasoning_content": " is 7 times 13."}`: the block order was `thinking[0] / text[1] / thinking[2] / text[3]`, and is now `thinking[0] → text[1]` with the reasoning tail folded into the open thinking block. The reader now reads `reasoning_content` before `content`, which is what the non-streaming paths already produced — a message carrying both fields converted to `[thinking, text]` at the previous commit too, so streaming was the outlier — and it matches sdcb/chats, LiteLLM, LobeChat and new-api, all of which split a dual-field chunk with the reasoning first. The `is not None` guard (PR [#393](https://github.com/Oaklight/llm-rosetta/pull/393), issue [#357](https://github.com/Oaklight/llm-rosetta/issues/357)) and the empty-reasoning drop in the Anthropic writer are untouched, and `tool_calls` still come last. The reorder sits in the reader, so every downstream writer sees it, not only Anthropic: `chat_completions` swaps the two deltas, `open_responses` swaps `output_text.delta` and `reasoning_summary_text.delta`, `google_generate` moves the `thought` part ahead of the text part, and `google_interactions` is unchanged. The fix is per-chunk — a stream whose deltas keep alternating between the two fields still yields one block per reasoning run, because reordering *across* chunks is out of scope (no gateway does it, and it would invalidate thinking signatures); the surviving behaviour is called out in the code comment and in the issue's known-limitation section. Official DeepSeek (`deepseek-reasoner`, `deepseek-chat` with thinking) and the opencode-go lanes (`deepseek-v4.1-flash`, `deepseek-v4-pro`) were probed across ~14 streams including tool loops and never emit dual-field chunks, so the change is a no-op for them.

- **DeepSeek thinking mode rejected a cross-format history that replays without reasoning** (issue [#948](https://github.com/Oaklight/llm-rosetta/issues/948)): DeepSeek's thinking mode requires the model's reasoning to be replayed once a request involves tools — per the vendor docs, "for requests carrying the `tools` parameter, the `reasoning_content` must be fully passed back to the API in all subsequent requests — even for turns where the model did not perform a tool call". A request whose message list does not end on a `user` turn and omits the field on any assistant message returns `400 The reasoning_content in the thinking mode must be passed back to the API`. An agent harness that replays history across a format boundary (Claude Code → gateway `messages` → `chat_completions`) drops the model's reasoning for some turns, so the converted history tripped this. The conversion was faithful — the field was missing upstream because it was missing in the source request — so the `deepseek--chat_completions` shim now applies a new `backfill_reasoning_content()` body transform: once the history is tool-bearing and does not end on a `user` turn, it fills `reasoning_content: ""` on every assistant message that lacks it, tool call or not. Live matrix on `deepseek-flash`: a history ending on a tool result (with or without the `tools` parameter) and one ending on an assistant answer with `tools` present both fail as-is and pass once every assistant message carries the field; filling only the tool-call message, or only the trailing turn, still fails; ending on a `user` message needs nothing; and `reasoning_content: ""` is accepted everywhere, including with `reasoning_effort` set, thinking explicitly disabled, and on `deepseek-v4-pro`. Replaying the captured failing request body end to end returns 400 before the change and 200 after.

- **`tool_result.is_error` silently dropped by both OpenAI targets** (issue [#879](https://github.com/Oaklight/llm-rosetta/issues/879)): `is_error: true` on a tool result survives `messages` (native `is_error`), `google_interactions` (native `is_error`) and `google_generate` (which encodes the bit as `response.error` instead of `response.output`), but `chat_completions` and `open_responses` have no field to carry it — they emitted the payload unchanged and returned an empty `ctx.warnings`, so the loss was invisible end to end. The flag is not decorative: with a payload that does not itself name the failure (a tool that exits non-zero and prints nothing), a model reads the result as success and stops, where the flag alone makes it retry — reproduced live on `gpt-5-nano`, where the same neutral `"(no output)"` yields `content='DONE'` without the flag and a fresh `deploy` tool call with it. Both targets now append a `[Tool error]` text block to the result (`{"type": "text"}` for Chat Completions, `{"type": "input_text"}` for Responses, both accepted by `custom_tool_call_output` and `function_call_output`) and record a warning naming the call. A bare string payload is normalized to a one-block list followed by the marker, so the emitted shape does not depend on what the IR carried; only `true` triggers it, `false` and absent are untouched, and an empty payload yields the marker alone. The marker is deliberately plain text rather than a machine-parseable envelope — nothing recovers it on the way back, because once the text names the failure the flag no longer changes model behaviour. Both converted bodies verified live against the real API (HTTP 200 on `/chat/completions` and `/responses`), and the block-list shape was exercised against eight OpenAI-compatible endpoints (OpenAI, DeepSeek, xAI, MiniMax, Zhipu, Volcengine, OpenRouter, SiliconFlow) — all accepted it, none required a bare string. Both OpenAI **sources** also read a non-standard `is_error` off a foreign body when one is present, so the converse direction is not silent either; the one case with nowhere to put it — an intrinsic result whose call becomes a native server item on Responses — now reports a warning instead of letting the flag vanish.
- **`open_responses` shims claimed the OpenAI profile without checking the upstream** (issue [#899](https://github.com/Oaklight/llm-rosetta/issues/899)): #909's collapse onto a single `open_responses` base copied the same opt-in OpenAI-profile block — `store_default: true` plus `preserve_fields` for `access_programs`, `billing`, `content_filters`, `moderation`, `tool_usage` — onto `argo`, `asksage`, `deepseek`, `volcengine` and `opencode`, so five providers claimed server-side storage the vendor-neutral spec does not have. Probing each upstream settles it per provider: OpenRouter answers `store: true` with `400 store: true is not supported on this proxy. Each response request is independent.`, DeepSeek ignores the field and always reports `store: false` alongside a bare UUID id, and OpenCode never echoes `store` and returns a `resp_`-prefixed id for its GPT models against a UUID for its DeepSeek ones. A shim-level trait set cannot describe a multi-vendor proxy like OpenCode at all. ARGO and AskSage turned out to be the mirror image: both return `resp_`-prefixed ids and populate echo fields, so both gain the `response_id_prefix` they were missing, but both ignore `store` (ARGO always reports `store: false`; AskSage reports `false` on some models and omits the field on others), so both lose `store_default`. Volcengine goes the other way, which is why "delete `store_default` everywhere" would have been the wrong conclusion: ARK returns `resp_`-prefixed ids and is the one non-OpenAI provider that genuinely stores server-side — omitting `store` on the wire yields `store: true`, and `store: false` is honored when sent — so it keeps `store_default: true`, gains the prefix, and loses the five lifecycle fields it never emits. `deepseek--open_responses` and `opencode--open_responses` now declare no profile traits, `openrouter--open_responses` records why it declares none, and `openai--open_responses` keeps the full profile (verified live). New developer tool `scripts/dev/probe/probe_responses_shims.py` compares each shim's declared traits against its live upstream and exits non-zero on a contradiction, which is how all six were settled.
- **Internal `_llm_rosetta_provider_metadata` carrier leaked out of the library** (issue [#941](https://github.com/Oaklight/llm-rosetta/issues/941)): converters attach an internal `_llm_rosetta_provider_metadata` key to provider-format objects so an in-process IR→provider→IR hop can recover provider-specific values (a Google `thoughtSignature`, opaque-value origins). Only the gateway stripped it before dispatch, so a caller that used the library directly — `get_converter_for_provider(...).request_to_provider(ir)` or `ConversionPipeline.convert_request()` — got a body the upstream refused: Anthropic returned `400 ... Extra inputs are not permitted`, OpenAI Responses `400 Unknown parameter: 'input[1]._llm_rosetta_provider_metadata'`, and Google `400 ... Unknown name "_llm_rosetta_provider_metadata"`. Chat Completions silently ignored the key, which is why the leak went unnoticed. `request_to_provider`, `response_to_provider`, and `messages_to_provider` now strip the carrier, so their output is directly sendable; the ops-level `ir_*_to_p` / `p_*_to_ir` functions keep it as the in-process round-trip mechanism. One deliberate behaviour change follows: a Google refusal still round-trips through the ops level, but reading a **clean wire body** back now yields a text part, since Google never emits the carrier on the wire. The gateway's strip is kept as a defensive guard and delegates to the shared `converters.base.helpers.strip_internal_metadata` helper. The carrier key is namespaced (`_llm_rosetta_provider_metadata`) so it cannot collide with a caller's own `tool_input` or tool-schema key.
- **Cross-provider opaque reasoning/signature handling** (issue [#881](https://github.com/Oaklight/llm-rosetta/issues/881), issues [#880](https://github.com/Oaklight/llm-rosetta/issues/880)/[#930](https://github.com/Oaklight/llm-rosetta/issues/930)): an opaque reasoning value — Anthropic `thinking.signature` / `redacted_thinking.data`, OpenAI `encrypted_content`, Google `thoughtSignature` / `function_call.signature` — is only valid for the provider that issued it, but the IR stored it under provider-agnostic names with no record of origin, so a converter emitted it onto whatever target it was handed. Where the target validates the field the request was rejected: `openai_responses → anthropic` emitted a foreign `encrypted_content` as `redacted_thinking.data` (`400 Invalid data in redacted_thinking block`), and `anthropic → google` on Gemini 3 replayed `functionCall` history with no usable signature (`400 ... missing a thought_signature`). Converters now record the origin under a reserved `provider_metadata["_opaque_origin"]` key; Anthropic and OpenAI **drop** a foreign opaque value, while a Google target emits the real signature when it is Google-origin and otherwise injects the documented dummy `skip_thought_signature_validator` on the **first** `functionCall` of each model turn. Google reasoning readers also mirror the signature into `provider_metadata.google.thought_signature`, so a Google value still survives an **ops-level** hop through a foreign converter (Google → Anthropic → Google) — the top-level `*_to_provider` API strips that carrier, see [#941](https://github.com/Oaklight/llm-rosetta/issues/941). Covers the non-streaming request/response paths **and** the streaming writers. Verified live: the injected dummy returns HTTP 200 on both Gemini 3.7 and Gemini 2.5, an arbitrary signature is still rejected on Gemini 3, and a foreign `encrypted_content` routed to Anthropic no longer produces `redacted_thinking`; same-provider round trips still preserve the real value. One deliberate behaviour change on this edge: an Anthropic hop now also drops a reasoning block that carries **only** an opaque value and no visible text, where it previously emitted it — that block was structurally lossless but wire-invalid (Anthropic rejects a foreign opaque field), so the round trip is intentionally lossy there. Origin is tracked at the **base-format** level (`messages`, `open_responses`, `google_generate`, ...), not per shim.
- **`google_generate` streaming outbound dropped `thoughtSignature`; `open_responses` had no tool-call metadata carrier** (issue [#930](https://github.com/Oaklight/llm-rosetta/issues/930), folded into [#881](https://github.com/Oaklight/llm-rosetta/issues/881)): the GG streaming writer registered a tool call but discarded its `provider_metadata`, so a streamed `functionCall` lost the signature — and a Google Interactions stream translated to a GG target hit the Gemini-3 missing-signature `400`. The signature now rides through `StreamContext` and is written on the flushed `functionCall`, with the dummy injected when absent. `OpenResponsesToolOps.ir_tool_call_to_p` / `p_tool_call_to_ir` now ferry `provider_metadata` (like the Messages and Google Generate ops), so a Google-signed tool call survives an `open_responses` hop.
- **Google `thoughtSignature` lost on the `google_generate` streaming read and write paths** (issue [#939](https://github.com/Oaklight/llm-rosetta/issues/939)): a Google thinking turn returns its signature on a dedicated trailing part whose `text` is an empty placeholder, and the streaming reader dropped it there — emitting a stray empty `text_delta` in its place — while the streaming writer never wrote `thoughtSignature` at all, so a `google_generate → google_generate` streaming round trip lost the signature the non-streaming path kept. The reader now emits a signature-only reasoning delta for such a part and skips the placeholder text; the writer writes `thoughtSignature` on the thought part, origin-gated like the non-streaming path (the Google namespace carrier first, then an own-origin flat signature; a foreign signature is dropped while the visible reasoning text still goes out). This also settles the `ReasoningDeltaEvent` contract: `reasoning` and `signature` are independent fields and one event may carry both, so the Anthropic and Interactions writers now emit **both** deltas for a combined event instead of dropping the visible text. Verified live: a streamed Gemini 3.7 thinking turn is captured, replayed with the signature intact, and accepted with HTTP 200.


- **Unvalidated `google_output_format`** (issue [#901](https://github.com/Oaklight/llm-rosetta/issues/901), follow-up to PR [#900](https://github.com/Oaklight/llm-rosetta/pull/900)): `ConversionPipeline` selected the REST body only on an exact `"rest"` match and silently fell through to the SDK shape — the one the REST endpoint rejects with HTTP 400 `Unknown name "config"` — for any other value, so `"REST"`, `"restful"` or a typo produced a body that only failed upstream, with no warning. The parameter is now typed `Literal["rest", "sdk"]` and any other value raises `ValueError` at construction for Google Generate targets — a value the target never reads is still ignored, so non-Google routes are unaffected; `GoogleGenerateConverter.request_to_provider()` validates its own `output_format` argument the same way, so callers that bypass the pipeline are covered too.
- **`google_generate` target emitted the SDK body** (PR [#900](https://github.com/Oaklight/llm-rosetta/pull/900), issue [#882](https://github.com/Oaklight/llm-rosetta/issues/882)): `ConversionPipeline` only selected the Generative Language REST output format when the target provider string was exactly `"google"`. The canonical converter name `google_generate` — accepted by `get_converter_for_provider` — fell through to the SDK body shape (`config: {...}`), which the REST endpoint rejects with HTTP 400 `Unknown name "config"`. Both names now select the REST output format (shim names resolving to `google_generate`, e.g. `asksage--google_generate`, are normalised first); `google_output_format="sdk"` remains an explicit opt-out.
- **Gateway dropped shim connection defaults** (PR [#886](https://github.com/Oaklight/llm-rosetta/pull/886)): `GatewayConfig` passed the base converter type to `build_provider_info`, so the shim name was lost and every shim's `connection.*` defaults (`base_url`, `api_key_env`, `auth_header`, `models_path`) were silently ignored. A provider configured via `type: <shim>--<base>` without an explicit `base_url` hit the base type's host, and shim auth-header overrides (e.g. AskSage's `x-access-tokens`) were dropped.
- **Creating a keyless (free-source) provider from the admin UI** (PR [#915](https://github.com/Oaklight/llm-rosetta/pull/915)): the provider route required an `api_key`, so a free-source entry was rejected with a `400` and never saved. A keyless shim is now allowed without one; `keyless` is computed after every api_key source, so an unrelated base-type env var (e.g. a stray `OPENAI_API_KEY`) can no longer turn the provider into a keyed one, and an empty `api_key`/`base_url` no longer shadows a shim default.
- **Anthropic round-trip fidelity** (PR [#850](https://github.com/Oaklight/llm-rosetta/pull/850), issue [#845](https://github.com/Oaklight/llm-rosetta/issues/845)): mid-conversation `system` messages were silently dropped (when a top-level `system` exists) or hoisted to the wrong position (when it doesn't) — fixed via a shared `hoist_system_in_converter()` step that now runs unconditionally in Anthropic, Google Generate, and Google Interactions converters, independent of shim configuration. `redacted_thinking` blocks were dropped entirely because the dispatch table didn't recognize the content type — added `ReasoningPart.redacted_data` and bidirectional conversion. `caller: {type: direct}` and `is_error: false` were injected into tool results on every round-trip even when absent from the source — now only emitted when explicitly present.
- **OpenAI Responses reasoning items with multiple turns** (PR [#832](https://github.com/Oaklight/llm-rosetta/pull/832)): a turn with several reasoning items (think → tool call → think again) collapsed them into one on the outbound leg — the id latched onto the first item while `encrypted_content` from later items overwrote it, producing an id/blob mismatch that Argo's native `/v1/responses` rejects with `400 invalid_encrypted_content`. Each reasoning item now tracks its own id and encrypted content via a `ReasoningItemState` list.
- **OpenAI Responses custom tool outputs reverting to `function_call_output`** (PR [#833](https://github.com/Oaklight/llm-rosetta/pull/833)): replayed request history lost the `custom_tool_call_output` type because the tool-type lookup relied on a context map only populated on the response leg, never during request parsing. `ToolResultPart` now carries `tool_type` directly so same-format `openai_responses → openai_responses` history round-trips without growing fidelity warnings.
- **Anthropic parallel tool-call stream indices** (PR [#835](https://github.com/Oaklight/llm-rosetta/pull/835)): converting a Chat Completions SSE stream with multiple interleaved tool calls into Anthropic's format reused content-block index `0` for every tool and routed argument fragments to the wrong block. Each tool call now gets its own Anthropic block index, argument deltas route to the correct block, and all open blocks are closed at stream end.
- **`gateway` sync/async compatibility with downstream consumers** (PR [#836](https://github.com/Oaklight/llm-rosetta/pull/836)): `create_app`, `setup_admin`, `dump_error`, and `_flush_now` became coroutines when persistence moved to `aiosqlite`, silently breaking `argo-proxy>=3.3.3` (sync caller, no upper version bound) — `create_app` crashed at startup, the others created coroutines that were never awaited. New `@sync_compat` decorator runs these via `asyncio.run()` in a sync context or schedules a `Task` in an async context, preserving the `async def` signature for type checkers. Backward-compat shim modules added for `gateway.headers`/`gateway.auth`/`gateway.embeddings`.
- **Streaming missing terminal events** (PR [#870](https://github.com/Oaklight/llm-rosetta/pull/870), issue [#838](https://github.com/Oaklight/llm-rosetta/issues/838)): when OpenAI-compatible upstreams omit the `usage` object (`stream_options.include_usage` unset), `StreamEndEvent` was never emitted, causing downstream converters to never flush `message_delta` (with `stop_reason`) and `message_stop`. The `@anthropic-ai/sdk` stream helper rejected with "stream ended without producing a Message." Fixed by deferring `StreamEndEvent` when `finish_reason` arrives without `usage`, then flushing on the next chunk or via a `None` EOF sentinel.
- **Provider shim audit** (PR [#758](https://github.com/Oaklight/llm-rosetta/pull/758)): systematic audit against latest API docs with live verification. Key fixes: Anthropic adaptive thinking for Claude 5 models, OpenAI `max` effort support, Volcengine `ARK_API_KEY` env var, MiniMax/Moonshot/Zhipu/OpenRouter reasoning config corrections.
- **Provider affinity routing** (PR [#823](https://github.com/Oaklight/llm-rosetta/pull/823)): `AffinityRoundRobinStrategy` used Python's process-randomized `hash()`, breaking session affinity across restarts and multi-worker deployments. Replaced with `hashlib.sha256`.
- **OpenAI Responses streaming** (PRs [#765](https://github.com/Oaklight/llm-rosetta/pull/765), [#766](https://github.com/Oaklight/llm-rosetta/pull/766), [#788](https://github.com/Oaklight/llm-rosetta/pull/788), [#792](https://github.com/Oaklight/llm-rosetta/pull/792), [#800](https://github.com/Oaklight/llm-rosetta/pull/800), [#810](https://github.com/Oaklight/llm-rosetta/pull/810)): string tool outputs no longer JSON-coerced; reasoning preserved with same-turn text/tools; DeepSeek `reasoning_content` extraction; response-level fields and tool definitions round-trip through IR; `.done`-only function_call arguments captured correctly.
- **Anthropic `budget_tokens` with adaptive thinking** (PR [#805](https://github.com/Oaklight/llm-rosetta/pull/805)): strip `budget_tokens` when `thinking.type` is `adaptive` (rejected by Anthropic API).
- **Tool schema sanitization** (PRs [#747](https://github.com/Oaklight/llm-rosetta/pull/747), [#825](https://github.com/Oaklight/llm-rosetta/pull/825)): preserve multi-branch `anyOf`/`oneOf` unions; strip `required` when `properties` is missing.
- **Performance** (PRs [#778](https://github.com/Oaklight/llm-rosetta/pull/778), [#782](https://github.com/Oaklight/llm-rosetta/pull/782), [#783](https://github.com/Oaklight/llm-rosetta/pull/783), [#774](https://github.com/Oaklight/llm-rosetta/pull/774), [#777](https://github.com/Oaklight/llm-rosetta/pull/777)): async SQLite persistence, O(1) prune queries and metadata eviction, fd exhaustion prevention, CLOSE_WAIT socket leak fix.
- **Admin UI** (PRs [#816](https://github.com/Oaklight/llm-rosetta/pull/816), [#820](https://github.com/Oaklight/llm-rosetta/pull/820), [#821](https://github.com/Oaklight/llm-rosetta/pull/821)): Models tab blank after routing loop PR, favicon SVG color, provider routing column layout, dark theme text visibility, toast z-index, settings popup/modal conflicts.
- **Shim loading** (PRs [#667](https://github.com/Oaklight/llm-rosetta/pull/667), [#674](https://github.com/Oaklight/llm-rosetta/pull/674)): `max_tool_description_length` and `hoist_system_messages` not loaded from provider YAML; CI guard test to prevent future field omissions.

### Infrastructure

- **The shim registry is no longer left empty between test files** (PR [#1040](https://github.com/Oaklight/llm-rosetta/pull/1040)): `tests/test_shim_loader.py` empties the registry before and after every test — its tests register synthetic shims and assert on the whole registry — and nothing put the built-ins back, so the process was left with none. Every other file resolves shims through `get_shim`, and a file collected after it saw `None` for every name. It stayed invisible because it depends on collection order: the files that read the registry all come first alphabetically, and the ones after it reload it in their own fixtures. `pytest tests/test_shim_loader.py tests/test_provider_name_alias.py` failed 21 of 147; a module-scoped teardown now rebuilds once per file, and the file leaves the process as it found it.

- **Test Release workflow** (PR [#686](https://github.com/Oaklight/llm-rosetta/pull/686)): publish dev versions to Test PyPI with automated Docker dev image builds.
- **Vendored zerodep updates** (PRs [#658](https://github.com/Oaklight/llm-rosetta/pull/658), [#710](https://github.com/Oaklight/llm-rosetta/pull/710), [#798](https://github.com/Oaklight/llm-rosetta/pull/798), [#780](https://github.com/Oaklight/llm-rosetta/pull/780)).
## v0.13.0 — 2026-09-08

### Added

- **Google Interactions API converter** (Issue [#581](https://github.com/Oaklight/llm-rosetta/issues/581)): new `converters/google_interactions/` module with full bidirectional conversion for Google's Interactions API (`/v1beta/interactions`). Supports typed steps (user_input, model_output, thought, function_call, function_result), `thinking_level` ↔ IR effort mapping, Interactions SSE streaming lifecycle (`interaction.created/completed`, `step.start/delta/stop`), and MCP server tool definitions. Consecutive assistant-role steps are merged into single `AssistantMessage`. No IR type changes required.
- **Rename `google_genai` → `google_generate`** (PR [#641](https://github.com/Oaklight/llm-rosetta/pull/641)): the generateContent converter module, classes, and shim base are renamed to avoid confusion with the `google-genai` SDK which covers both APIs. `GoogleGenAIConverter` and `from llm_rosetta.converters.google_genai` remain as deprecated aliases.
- **Google Interactions gateway route** (PR [#647](https://github.com/Oaklight/llm-rosetta/pull/647)): `POST /v1beta/interactions` endpoint registered in the gateway. `GoogleInteractionsConverter` exported from top-level package.
- **"Adding a new converter" checklist** in CLAUDE.md (PR [#647](https://github.com/Oaklight/llm-rosetta/pull/647)): 11-item gate checklist covering module, tests, exports, auto-detect, shim, gateway route, docs, CI smoke test, and test registry updates.
- **Relocate oversized tool descriptions** (PR [#625](https://github.com/Oaklight/llm-rosetta/pull/625)): when a custom tool description exceeds a configurable threshold, the full text is moved into a late system message to prevent upstream 400 errors. Threshold configurable at model, provider, and shim levels.
- **Configurable Nuitka flags and optimized binary builds** (PRs [#639](https://github.com/Oaklight/llm-rosetta/pull/639), [#640](https://github.com/Oaklight/llm-rosetta/pull/640)): added `NUITKA_EXTRA_FLAGS` variable for binary size experiments; baked winning flag combination (LTO, stripped docstrings, nofollow exclusions) into defaults; dropped pyinstrument from binary builds.
- **Cross-format round-trip tests** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649)): 20 tests verifying Interactions ↔ OpenAI Chat / Anthropic / google_generate request and response fidelity.
- **Expand `tool_ops` convenience API** (PR [#653](https://github.com/Oaklight/llm-rosetta/pull/653)): add `google_interactions` provider support (was the only converter missing) and expose the full `BaseToolOps` lifecycle — `choice_to_provider`/`choice_from_provider`, `call_to_provider`/`call_from_provider`, `result_to_provider`/`result_from_provider`, `config_to_provider`/`config_from_provider`. 70 tests covering all 5 providers.
- **Dashboard multi-select batch download** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655)): checkbox selection and bulk action bar for profiling, content capture, and error dump tables. Error dumps support batch download and batch delete with selection persisted across pagination.
- **Request log ↔ Error dump cross-linking** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655)): 4xx/5xx request log entries show a jump-to-dump button; error dump rows show a jump-to-log button. Jumps clear filters, navigate to the correct page, and highlight the target row. "Match Logs" button and auto-startup backfill link unlinked dumps via timestamp proximity (±0.1s) with model alias resolution.
- **API key last-used tracking** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655)): new `last_used` column in API keys table, auto-migrated on startup. Updated in the auth hook with 5-minute throttle. Backfill from request log on startup and via manual "Refresh Last Used" button.
- **Capability badge SVG icons** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655)): model table capability badges now use `cap-badge` class with inline SVG icons (text, vision, tools, reasoning). Model table column widths rebalanced (Model 19%, Type 15% centered, Capabilities 23%).

### Fixed

- **Google Interactions provider URL registry** (PR [#648](https://github.com/Oaklight/llm-rosetta/pull/648)): `google_generate` and `google_interactions` URL templates were missing from the gateway provider registry, causing 404s on upstream requests.
- **Google Interactions tool call status** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649)): Google generateContent returns `STOP` for tool calls — the converter now detects `function_call` steps and overrides status to `requires_action`.
- **Google Interactions streaming** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649)): added `google_generate` and `google_interactions` to `SSE_FORMATTERS` registry; added IR→provider stream handlers; synthesize `step.start`/`step.stop` events when upstream omits `ContentBlockStartEvent`; defer `interaction.completed` to `stream_end` so usage data is included.
- **Google Interactions thinking passthrough** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649)): set `include_thoughts=True` in IR when `thinking_level` is enabled, so the upstream Google API returns thought content.


- **Profiler deferred for streaming responses** (PR [#633](https://github.com/Oaklight/llm-rosetta/pull/633)): pyinstrument profiler now runs for the full stream lifecycle instead of stopping when the handler returns the `StreamingResponse`.
- **Auto-rebuild metrics counters on startup** (PR [#643](https://github.com/Oaklight/llm-rosetta/pull/643)): detect counter drift after ungraceful shutdown and auto-rebuild from the request log.
- **OpenAI Responses `status` field on input items** (PR [#650](https://github.com/Oaklight/llm-rosetta/pull/650)): add `"status": "completed"` to all input item types. Fixes Volcengine (doubao) 400 `MissingParameter` errors.
- **Security: deleted/rotated API keys could still authenticate** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652)): the auth hook had a `config_fallback` dict (built from `server.api_keys` at startup) that was never invalidated by admin panel key deletion/rotation. Revoked keys could bypass keystore validation via this stale fallback. Removed the `config_fallback` path entirely — SQLite keystore is now the sole API key authority.
- **Security: credential_visible default and persistence** (PR [#654](https://github.com/Oaklight/llm-rosetta/pull/654)): default flipped from `true` to `false`, forced off when no admin password is set. Backend 403 guard on the PUT endpoint and GET reveal endpoint. Admin UI toggle disabled when no password is configured. Sensitive fields (admin_password, api_keys) masked in config API responses.
- **Server Time label** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655)): renamed "System Time" to "Server Time" for clarity. Timezone display changed from abbreviation (CDT) to GMT±N format throughout.

### Changed

- **API key management is now SQLite-only** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652)): `server.api_keys` and `server.api_key` config-file fields are no longer parsed or used for authentication. All API key management now goes exclusively through the admin panel + SQLite keystore. Existing keys in config files are silently ignored. The `keystore.import_from_config()` migration path has been removed.

### Internal

- **Vendor zerodep `jsonx` module** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652)): replaced the hand-rolled `_strip_jsonc_comments` regex in `gateway/config.py` with the vendored `jsonx` parser. Adds support for `#` comments, trailing commas, and better error line-number remapping.

## v0.12.0 — 2026-09-03

### Added

- **Namespace tool flattening** (PR [#626](https://github.com/Oaklight/llm-rosetta/pull/626)): flatten `type: "namespace"` tool containers (used by Codex) into individual IR tools with namespace metadata and cross-namespace dedup. Qualified naming keeps tool names within the 64-char limit.
- **`additional_tools` input item support** (PR [#623](https://github.com/Oaklight/llm-rosetta/pull/623)): extract tool definitions from Codex's Responses API `additional_tools` input items. Namespace tools are automatically expanded via the #626 pipeline.
- **Custom tool tri-state** (PR [#624](https://github.com/Oaklight/llm-rosetta/pull/624)): `supports_custom_tools` is now `None | bool` — `None` defers to shim default, explicit `False` forces downgrade even when the shim says `True`. Gateway config `supports_custom_tools: false` now works correctly.
- **Configurable `data_dir`** (PR [#619](https://github.com/Oaklight/llm-rosetta/pull/619)): gateway persistence storage location is now configurable via `--data-dir` CLI flag or `server.data_dir` in config. Keys DB and request log DB default to this directory.
- **Admin logo picker** (PR [#630](https://github.com/Oaklight/llm-rosetta/pull/630)): compact icon button with searchable dropdown. Curated popular provider icons shown first, full list fetched on-demand from jsdelivr (`@lobehub/icons-static-svg`). Supports custom URL fallback, keyboard navigation.
- **Admin layout editor** (PR [#629](https://github.com/Oaklight/llm-rosetta/pull/629)): drag-and-drop developer tool at `design/ui/layout-editor.html` for prototyping provider modal field layouts with preview mode and HTML export.
- **Provider `models_path` and `logo` config fields** (PR [#617](https://github.com/Oaklight/llm-rosetta/pull/617)): per-provider model listing endpoint path override and logo URL in gateway config and admin UI.
- **Reasoning capability enforcement** in pipeline: block `reasoning` tool choice when the upstream model does not declare reasoning capability.
- **Admin chart redesign** (PR [#600](https://github.com/Oaklight/llm-rosetta/pull/600)): replace line charts with step (throughput) + scatter (latency) for more accurate visualization.
- **Admin accessibility improvements** (PRs [#603](https://github.com/Oaklight/llm-rosetta/pull/603), [#606](https://github.com/Oaklight/llm-rosetta/pull/606)): keyboard navigation for model toggles, focus traps, ARIA attributes.

### Fixed

- **Anthropic streaming tool call binding** (PR [#628](https://github.com/Oaklight/llm-rosetta/pull/628)): use `chunk["index"]` to resolve tool call ID for `input_json_delta` instead of last-registered key. Fixes interleaved parallel tool calls producing incorrect tool IDs.
- **Admin hint popup clipping** (PR [#631](https://github.com/Oaklight/llm-rosetta/pull/631)): switch from CSS absolute positioning to JS-driven fixed positioning with smart above/below detection. Popups no longer clipped by modal `overflow-y: auto`.
- **`tool_call_id` sanitization** at provider output boundary: sanitize IDs that contain characters not accepted by downstream providers.
- **Defensive `model_list_transform`** when `internal_id` is absent in model config.
- **Admin error dump buttons** not responding to clicks.
- **Admin provider list view** redesigned as compact single-line rows.
- **Logging import** moved to module level to avoid repeated imports.

### Changed

- **Admin provider modal layout** rearranged: Logo + Provider Name row, API Key full width, Base URL + Models Listing Path row (69/31), Proxy URL + Timeout row (69/31). Hint text removed from Models Path and Timeout fields.
- **Shim `ReasoningCapability`** redesigned with `thinking_modes`, `effort_range`, and `visibility_modes` fields (PR [#614](https://github.com/Oaklight/llm-rosetta/pull/614)).
- **Argo shim** uses `argo:` prefix in model display names and shows upstream provider in fetch dialog.
- **Admin responsive layout** unified provider card markup with responsive list stages (PR [#612](https://github.com/Oaklight/llm-rosetta/pull/612)).

## v0.11.2 — 2026-08-30

### Added

- **Native `tool_search` passthrough** (PR [#593](https://github.com/Oaklight/llm-rosetta/pull/593)): vendor `sparse_search`, add `tool_search_mode` to shim schema, and support native `tool_search` passthrough in both streaming and non-streaming Responses API paths.
- **Provider connectivity test** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): new `POST /admin/api/config/providers/<name>/test-connectivity` endpoint probes a provider's base URL and each configured endpoint (models, embedding, rerank) for reachability. Shows raw and normalized URLs to help diagnose double-prefix issues. "Test" button added to provider cards in the admin UI.
- **`.well-known/change-password` redirect** (PR [#579](https://github.com/Oaklight/llm-rosetta/pull/579)): `GET /.well-known/change-password` returns a 302 redirect to `/admin#change-password`, auto-opening the settings panel. Enables browser and password manager integration per the [web standard](https://web.dev/articles/change-password-url).

### Fixed

- **Duplicate chat finish events** (PR [#592](https://github.com/Oaklight/llm-rosetta/pull/592)): deduplicate `finish_reason` when upstream repeats it on the same choice index (e.g. OpenAI with `stream_options.include_usage`). Adds per-choice finish tracking to `StreamContext`. Fixes [#589](https://github.com/Oaklight/llm-rosetta/issues/589).
- **Embedding route `upstream_model` remapping** (PR [#586](https://github.com/Oaklight/llm-rosetta/pull/586)): the embedding-specific route now applies `upstream_model` name remapping from `model_upstream_names`, matching the chat fallback route. Previously model aliases (e.g. `argo:text-embedding-3-small` → `v3small`) were ignored, causing upstream 404s.
- **Double version prefix in embedding/rerank URLs** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): `ProviderInfo` now auto-detects when `base_url` ends with a version segment (e.g. `/v1`) that would duplicate the `url_template` path start, and strips it. Fixes `base_url: "https://api.openai.com/v1"` + `embedding_path: "/v1/embeddings"` producing `/v1/v1/embeddings`.
- **Logo icon centering** (commit [f4c89e3](https://github.com/Oaklight/llm-rosetta/commit/f4c89e3)): center stone silhouette in icon SVGs, switch to transparent background with `prefers-color-scheme` media query for automatic dark/light theme adaptation.

### Changed

- **Unified endpoint URL construction** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): embedding and rerank handlers now use `ProviderInfo.upstream_url()` (template-based) instead of ad hoc f-string concatenation, matching the chat path architecture.
- **Vendor updates** (PR [#594](https://github.com/Oaklight/llm-rosetta/pull/594)): httpclient 0.4.5→0.4.6, sse 0.3.2→0.3.3.

## v0.11.1 — 2026-08-29

### Fixed

- **OpenAI Responses reasoning encrypted state** (PR [#576](https://github.com/Oaklight/llm-rosetta/pull/576)): forced same-format Responses→Responses streaming conversion now preserves `encrypted_content` and the source reasoning item ID from `response.output_item.done`. Previously the IR round-trip dropped both, breaking clients that replay completed reasoning state (e.g. `store: false` / ZDR flows). Fixes [#575](https://github.com/Oaklight/llm-rosetta/issues/575).
- **Google GenAI reasoning config round-trip** (PRs [#582](https://github.com/Oaklight/llm-rosetta/pull/582), [#583](https://github.com/Oaklight/llm-rosetta/pull/583), [#584](https://github.com/Oaklight/llm-rosetta/pull/584)): parse `thinkingConfig` from REST `generationConfig` on inbound, map reasoning effort to `thinkingLevel`, forward `summary`/`include_thoughts` across all converters.
- **Responses API reasoning summary forwarding** (PR [#582](https://github.com/Oaklight/llm-rosetta/pull/582)): forward `reasoning.summary` in outbound Responses API requests.

### Changed

- **Gateway `create_app` composable** (PR [#578](https://github.com/Oaklight/llm-rosetta/pull/578)): refactored `create_app` via `GatewayExtensions` for extensibility.
- **IR `ReasoningDeltaEvent`** now declares `encrypted_content` and `provider_metadata` fields, consistent with `ToolCallStartEvent`.

## v0.11.0 — 2026-08-28

### Changed

- **Admin UI color system redesign** (PRs [#564](https://github.com/Oaklight/llm-rosetta/pull/564), [#566](https://github.com/Oaklight/llm-rosetta/pull/566), [#571](https://github.com/Oaklight/llm-rosetta/pull/571)): replaced the single light/dark theme with a two-scheme system — **Minimal** (Vercel-inspired pure B&W) and **Emerald** (Neon-inspired green accent), each with light and dark modes (4 total combinations). Architecture moved from JS-driven `THEMES` object to CSS-driven compound selectors (`[data-scheme][data-mode]`). Settings UI now has separate Scheme and Mode selectors.
- **Admin UI typography and consistency** (PR [#570](https://github.com/Oaklight/llm-rosetta/pull/570)): consolidated font sizes from 10 discrete steps to 7. Extracted `.btn-disabled` class to replace inline disabled styles. Normalized elevation model — removed box-shadow from settings sections, unified to border-hover pattern. Standardized form input font-family (mono for code inputs, sans for selects).
- **Admin UI responsive layout** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)): tab bar overflow scroll on mobile, table-scroll wrappers, content max-width 1400px, provider card grid 280px min for 4-column layout, settings popup padding reduced on mobile.
- **Admin UI CSS variable architecture** (PR [#571](https://github.com/Oaklight/llm-rosetta/pull/571)): migrated scheme-specific structural differences (table header typography, badge radius, chart bar colors) from CSS selector overrides to custom properties. Adding a new scheme now requires only one variable block in `base.css`.

### Fixed

- **Admin UI CSS bugs** (PR [#564](https://github.com/Oaklight/llm-rosetta/pull/564)): fixed undefined `var(--hover)` in fetch-models list, consolidated duplicate `.btn-danger` definitions, added `--purple` to dark theme, replaced all hardcoded hex/rgba colors with CSS variables and `color-mix()`.
- **Error dump coverage** (PRs [#572](https://github.com/Oaklight/llm-rosetta/pull/572), [#573](https://github.com/Oaklight/llm-rosetta/pull/573)): added `dump_error()` to 4 previously uninstrumented failure paths — request-phase conversion errors (400), non-streaming connection errors (502), response-phase conversion errors (502), and mid-stream error chunks. Extracted `DumpContext` dataclass for cleaner parameter passing.
- **Emoji empty-state icons replaced** with SVG line icons (bar-chart, camera, folder) matching the minimal design language.
- **Rosetta Stone SVG favicon** — replaced the emoji favicon (🔀) with the project's Rosetta Stone silhouette, served as both inline `<link rel="icon">` and server-side `/favicon.ico`.
- **Admin UI config path abbreviated** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — shows filename only with full path in tooltip. System clock now shows timezone abbreviation.
- **Admin UI toast centered** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — toast notifications now centered horizontally instead of pinned to bottom-right.
- **OpenAI Responses reasoning input lifecycle** (PR [#569](https://github.com/Oaklight/llm-rosetta/pull/569)): fixed output-only fields (`status: "completed"`, synthetic `rs_` IDs) leaking into Responses request input items. Unproven cross-format reasoning (from Chat/Anthropic/Google) is now omitted rather than assigned a fake identity. Proven Responses-origin reasoning preserves the real ID and summary but strips output-only status. Fixes [#568](https://github.com/Oaklight/llm-rosetta/issues/568).
- **Admin UI "Allowed Shims" hidden** (PR [#574](https://github.com/Oaklight/llm-rosetta/pull/574)): removed the confusing "Allowed Shims" column and modal input from the API keys page — "shim" is an internal concept. Backend storage unchanged (defaults to `["*"]`).

### Added

- **Admin UI accessibility** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)): ARIA roles (`tablist`/`tab`/`tabpanel`, `radiogroup`/`radio`, `dialog`), arrow key navigation for tabs and segmented controls, focus trap for modals and settings popup.
- **Rosetta Stone header logo** — SVG silhouette icon in the admin panel header.
- **GitHub repo icon** — `design/logo/out/rosetta-icon-github.svg` for GitHub repository settings.
- **Admin UI i18n** — added Scheme/Mode selector labels in English and Chinese.
- **Design demos** — theme comparison demos on `design/ui` branch (Minimal, Emerald, Vercel, Neon, Metallic styles).

## v0.10.0 — 2026-08-26

### Added

- **Standalone Nuitka binaries** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): pre-compiled single-file executables for 6 platforms — linux-x86_64 (glibc + musl), linux-arm64 (glibc + musl), macOS arm64, and Windows x86_64. No Python runtime required. Includes pyinstrument profiling support.
- **Binary-based Docker images** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): three image variants — `alpine` (musl binary, ~21 MB, default), `glibc` (busybox:glibc, ~25 MB), and `python` (pip-based, ~80 MB). Alpine variant tagged as `:latest` and `:<version>`.
- **Makefile build targets** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): `build-binary`, `build-binary-musl`, `build-docker-alpine`, `build-docker-glibc`, `build-docker-python` for local and CI builds.

### Changed

- **Docker privilege model** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): replaced su-exec/PUID/PGID with Docker-native `USER appuser` + `--user` flag. Use `docker run --user $(id -u):$(id -g)` for custom UID mapping.
- **Docker default image** — `:latest` now points to the Alpine binary image (~21 MB) instead of the Python-based image (~80 MB). The Python image is still available as `:<version>-python`.

## v0.9.0 — 2026-08-21

### Added

- **Rerank IR types and converter family** (PR [#506](https://github.com/Oaklight/llm-rosetta/pull/506)): 5 TypedDict IR types (`IRRerankRequest`, `IRRerankResponse`, `RerankDocument`, `RerankResultItem`, `RerankUsageInfo`) and 3 converters (`JinaRerankConverter`, `CohereRerankConverter`, `VoyageRerankConverter`) covering all major rerank API format families.
- **Embedding IR types and converter family** (PR [#510](https://github.com/Oaklight/llm-rosetta/pull/510)): 6 IR types (`IREmbeddingRequest`, `IREmbeddingResponse`, `EmbeddingItem`, `EmbeddingUsageInfo`, `EmbeddingTaskType`, `EmbeddingEncodingFormat`) and 4 converters (`OpenAIEmbeddingConverter`, `JinaEmbeddingConverter`, `VoyageEmbeddingConverter`, `CohereEmbeddingConverter`).
- **Gateway rerank proxy** (PRs [#511](https://github.com/Oaklight/llm-rosetta/pull/511), [#512](https://github.com/Oaklight/llm-rosetta/pull/512)): `/v1/rerank` and `/v2/rerank` endpoints with IR-based cross-format conversion. Config-driven routing via `rerank_providers`, `rerank_models`, `default_rerank_format`. `/v2/rerank` auto-detects Cohere source format.
- **Gateway embedding proxy with IR conversion** (PR [#517](https://github.com/Oaklight/llm-rosetta/pull/517)): convert `/v1/embeddings` from passthrough to IR-based conversion mode (OpenAI↔Cohere↔Jina↔Voyage). Backward compatible — configs without `embedding_providers` still use passthrough.
- **`UpstreamTimeoutError`** (PR [#513](https://github.com/Oaklight/llm-rosetta/pull/513)): distinguish upstream timeout (504) from connection error (502) across all transport methods. Rerank and embedding handlers return proper 504 on timeouts.
- **`x-rosetta-conversion: passthrough` header** — returned when response conversion fails, allowing clients to detect fallback to raw upstream format.
- **Rerank API format documentation** — bilingual (en/zh) documentation of Jina, Cohere, Siliconflow, and Voyage rerank API formats with provider lineage and IR mapping tables.
- **Embedding API format documentation** — bilingual (en/zh) documentation of OpenAI, Cohere, Jina, and Voyage embedding API formats.
- **ConversionPipeline passthrough mode** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): `force_conversion` parameter (default `True`). When `False` and source == target, the pipeline skips IR round-trip — fixes information loss when proxying same-format traffic (e.g. Claude Code → gateway → Anthropic upstream).
- **Converter instance caching** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): `get_converter_for_provider()` caches instances in a module-level dict. Eliminates per-request converter allocation.
- **Fidelity checker** (`fidelity.py`, PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): compare original and round-tripped bodies to detect IR conversion loss. Two modes: `"critical"` (per-format field check, ~0.01ms) and `"full"` (recursive leaf-level diff). Wired into pipeline passthrough path via `fidelity_mode` parameter for background monitoring.
- **`StreamProcessorProtocol`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): shared `Protocol` for `StreamProcessor` and `PassthroughStreamProcessor` with terminal chunk detection.
- **Rerank source format auto-detection** (PR [#522](https://github.com/Oaklight/llm-rosetta/pull/522)) — detect Voyage format from `top_k` in request body; `/v2/rerank` implies Cohere.
- **Embedding source format auto-detection** (PR [#521](https://github.com/Oaklight/llm-rosetta/pull/521)) — detect embedding source format from request body fields (`input_type` → Cohere/Jina/Voyage; `encoding_format` candidates disambiguate).
- **Admin `disabled_tabs` parameter** (PR [#505](https://github.com/Oaklight/llm-rosetta/pull/505)): `setup_admin(disabled_tabs=["metrics"])` hides admin UI tabs at initialization time.
- **Shim `multimodal_tool_result` capability** (PRs [#523](https://github.com/Oaklight/llm-rosetta/pull/523), [#524](https://github.com/Oaklight/llm-rosetta/pull/524)): `ProviderShim` can now declare `multimodal_tool_result: true/false` in YAML to override the converter's class-level default. The flag is wired through `ConversionContext.options` in both `convert()` and `ConversionPipeline`. Chat converter threads it to `_convert_tool_result_with_packing` so multimodal content is preserved natively when the provider supports it.
- **Streaming refusal events** (PR [#528](https://github.com/Oaklight/llm-rosetta/pull/528)): `response.refusal.delta` / `response.refusal.done` SSE events in the OpenAI Responses converter. New `RefusalDeltaEvent` IR stream type with full p→ir→p round-trip support. Completes issue [#431](https://github.com/Oaklight/llm-rosetta/issues/431).

### Fixed

- **Google GenAI multimodal tool result handling** (PR [#525](https://github.com/Oaklight/llm-rosetta/pull/525)): structured content blocks (`list[ContentPart]`) in tool results are now preserved natively instead of being flattened via `json.dumps`/`str()`. Dict content uses `json.dumps` (not `str()` which produced invalid Python repr). A `_is_content_block_list` guard distinguishes typed content blocks from plain data lists.
- **Chat converter multimodal content loss** (PR [#524](https://github.com/Oaklight/llm-rosetta/pull/524)): `_do_request_to_provider` was not passing `supports_multimodal_tool_result` to `ir_messages_to_p`, so shim overrides had no effect on the real request path. Additionally, `_convert_tool_result_with_packing` always stripped images from tool messages even when the flag was True — images were packed but never injected back, causing silent content loss.
- **Test ordering flakiness** (PR [#523](https://github.com/Oaklight/llm-rosetta/pull/523)): `test_shims.py` fixture now saves/restores the global shim registry instead of clearing it, preventing cross-module test failures.

### Changed

- **Unified `convert()` and `ConversionPipeline`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): both now support dual-shim (source + target) transforms and response conversion. `ConversionPipeline` delegates to `convert()` internally, eliminating code divergence.
- **Unified outbound transport** (PR [#516](https://github.com/Oaklight/llm-rosetta/pull/516)): merge `send_request` and `send_passthrough` into a single `send(provider_info, url, body)` method. Move URL construction and stream-flag injection from transport to proxy layer. Net -68 lines.
- **Documentation restructured** — split into three top-level tabs: IR Type System, Library, Gateway. API Reference merged into each tab. Removed standalone API Reference tab.

## v0.8.2 — 2026-08-09

### Added

- **Per-provider/model timeout overrides** (PR [#502](https://github.com/Oaklight/llm-rosetta/pull/502)): configure upstream timeout at provider and model granularity instead of global-only. Resolution: `model.timeout > provider.timeout > server.upstream_timeout`. Admin UI timeout inputs in both provider and model modals.
- **Prompt cache preservation** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): wire `hoist_late_system_messages` IR transform to all 15 provider shims. Mid-conversation system/developer messages are rewritten as user-role `[System: ...]` envelopes to keep the prompt cache prefix stable.
- **Per-provider hoist toggle** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): `hoist_system_messages` boolean in gateway config, per-provider override via admin UI checkbox with (i) hint popup.
- **SQLite API key storage** (PR [#496](https://github.com/Oaklight/llm-rosetta/pull/496)): migrate API key storage from plaintext config to SQLite with hash-based validation.

### Changed

- **Complexity reduction** — refactor base converter helpers, Google GenAI converter, OpenAI Responses converter, gateway config, and SOCKS5 test handler to reduce cyclomatic complexity.
- **complexipy v6 pre-commit hook** — enable complexipy v6 pre-commit hook; bump to v6.2.0 with threshold raised to 30.
- **Config override resolution** (PR [#501](https://github.com/Oaklight/llm-rosetta/pull/501)): unify per-provider toggle resolution to Pattern C — `config.resolve()` uses shim defaults as fallback, downstream types simplified from `bool | None` to `bool`.

### Fixed

- **Hint popup hover** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): admin UI hint popups are now hoverable for text selection (CSS `::before` bridge over the icon-popup gap).
- **Abort-path test doubles** (PR [#497](https://github.com/Oaklight/llm-rosetta/pull/497)): replace `_FakeContext`/`_FakeProcessor` with real `StreamContext` to eliminate interface drift risk.

### Removed

- **Backward-compat aliases** (PR [#492](https://github.com/Oaklight/llm-rosetta/pull/492)): remove `to_provider()` compat aliases from openai_responses and google_genai converters.
- **Dead compat shims** — remove dead backward-compat shims and compat aliases from base converter and converters.
- **Stale docs** — remove stale base converter READMEs.
- **Stray file** — remove stray `validate.py` from project root.

## v0.8.1 — 2026-08-07

### Fixed

- **Custom tool grammar shape** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489)): Fix grammar-constrained custom tools (e.g. Codex `apply_patch`) failing on Chat Completions upstreams. Responses API uses a flat `format` shape (`{type, syntax, definition}`); Chat Completions nests under `format.grammar`. Added bidirectional, idempotent shape converters at the Chat boundary.
- **Streaming null union members** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489)): Fix `AttributeError` crash on streaming custom tool call deltas. Providers serialize all union members on each delta with inactive ones set to `null`; `dict.get("type", "function")` returns `None` for a present-but-null key. Type is now recovered from the populated payload, falling back to the context-registered type.

### Changed

- **Public API for tool call order** (PR [#490](https://github.com/Oaklight/llm-rosetta/pull/490)): Replace direct `_tool_call_order` private access from all 4 converters with public methods on `StreamContext`: `tool_call_ids`, `tool_call_count`, `resolve_tool_call_id_by_index()`, `get_tool_call_index()`.
- **O(1) tool call index lookup** — `get_tool_call_index()` uses a reverse dict instead of linear `list.index()` scan.

## v0.8.0 — 2026-08-06

### Spec Compliance

Systematic pass across all four converters to ensure output matches official API specs. Driven by [llm-comply](https://github.com/Oaklight/llm-comply) compliance testing.

- **Anthropic usage fields** — emit all spec-required usage fields (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`). Always emit `stop_sequence` and `stop_details` on responses.
- **Anthropic caller, citations, container** — add `caller`, `citations`, and `container` fields to Anthropic response output.
- **Anthropic streaming `message_start.input_tokens`** ([#424](https://github.com/Oaklight/llm-rosetta/issues/424), PR [#425](https://github.com/Oaklight/llm-rosetta/pull/425)): Fix `message_start` always reporting `input_tokens=0` by swapping `UsageEvent`/`StreamStartEvent` emission order.
- **OpenAI Chat `logprobs`** — always emit `logprobs` on response choices (required nullable field).
- **OpenAI Chat `finish_reason`** — include `finish_reason: null` on all streaming chunks; map IR refusal reason to `stop` explicitly.
- **OpenAI Chat `annotations`** — nest annotations under `url_citation` and always emit the field.
- **OpenAI Responses `response.in_progress`** — emit `response.in_progress` streaming event per spec.
- **Google `responseId` / `modelVersion`** — include `responseId` and `modelVersion` in streaming chunks.
- **Google `ModalityTokenCount`** — filter non-standard modality values; normalize schema type case between Google and IR formats.
- **Google streaming usage** — flush pending usage on `stream_end` for cross-format streaming; extract `_build_stream_usage_metadata` helper.

### Cross-Format Refusal Handling

- **Complete refusal support** ([#429](https://github.com/Oaklight/llm-rosetta/issues/429)): All 4 converters handle refusal bidirectionally:
    - OpenAI Chat ([#430](https://github.com/Oaklight/llm-rosetta/issues/430)): `refusal` field always present (nullable), streaming `delta.refusal` accumulated.
    - Anthropic ([#432](https://github.com/Oaklight/llm-rosetta/issues/432)): Structured `stop_reason: "refusal"` + `stop_details` (category/explanation), including streaming.
    - Open Responses ([#431](https://github.com/Oaklight/llm-rosetta/issues/431)): `RefusalContent` type parsed and produced.
    - Google ([#433](https://github.com/Oaklight/llm-rosetta/issues/433)): `promptFeedback.blockReason` handling, missing `finishReason` values (`BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`), refusal round-trip via `_provider_metadata` marker.
- **IR RefusalPart** ([#427](https://github.com/Oaklight/llm-rosetta/pull/427)): `RefusalPart` added to `AssistantContentPart` union — refusal responses no longer fail IR validation with 502.

### Response Identity & Metadata

- **Shim-driven response ID prefix** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410), PR [#420](https://github.com/Oaklight/llm-rosetta/pull/420)): `ProviderShim` declares `response_id_prefix` in `provider.yaml`. Converter strips source prefix on ingest, adds target prefix on output. Enables clean cross-format response ID mapping (e.g. `chatcmpl-xxx` ↔ `resp_xxx`). OpenAI (`chatcmpl-`), Anthropic (`msg_`), and OpenAI Responses (`resp_`) prefixes declared.
- **`completed_at` timestamp** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410)): Responses converter sets `completed_at` to Unix timestamp on completed responses (previously always `null`).
- **Function call item ID preservation** — preserve `function_call` item ID across all conversion paths.
- **Strip `_provider_metadata` at HTTP boundary** ([#422](https://github.com/Oaklight/llm-rosetta/issues/422), PR [#423](https://github.com/Oaklight/llm-rosetta/pull/423)): Internal `_provider_metadata` fields no longer leak into outbound HTTP requests or downstream responses.

### Responses API Streaming

- **Unified `output_index`** ([#418](https://github.com/Oaklight/llm-rosetta/issues/418), PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419)): Single monotonic counter on `OpenAIResponsesStreamContext` replaces fragmented per-handler calculation. All output item types allocate indices through `next_output_index()`.
- **Reasoning output items** ([#407](https://github.com/Oaklight/llm-rosetta/issues/407), [#408](https://github.com/Oaklight/llm-rosetta/issues/408), PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419)): Non-streaming reasoning items always include `id` and `status` fields. Streaming reasoning has full lifecycle events. Deterministic reasoning ID generation via SHA-256 hash.
- **Reasoning item order** ([#437](https://github.com/Oaklight/llm-rosetta/issues/437), PR [#438](https://github.com/Oaklight/llm-rosetta/pull/438)): Fix `response_to_provider` output ordering — reasoning items now correctly precede message items.
- **Message phase preservation** ([#440](https://github.com/Oaklight/llm-rosetta/issues/440), PR [#441](https://github.com/Oaklight/llm-rosetta/pull/441)): Responses API `phase` field (`commentary`/`final_answer`) preserved through all conversion paths and streaming. Pipeline bridges phase across streaming contexts.
- **SSE `[DONE]` terminator** ([#409](https://github.com/Oaklight/llm-rosetta/issues/409)): Gateway emits `[DONE]` as the final SSE event after `response.completed`.

### IR & Architecture

- **Provider passthrough events** — new IR types for non-stream and streaming provider passthrough items, allowing converters to forward provider-specific data without loss.
- **Empty reasoning content** — OpenAI Chat converter preserves empty reasoning content instead of dropping it.
- **Multimodal tool result payload duplication** ([#480](https://github.com/Oaklight/llm-rosetta/issues/480), PR [#482](https://github.com/Oaklight/llm-rosetta/pull/482)): When targeting OpenAI Chat, images in tool results were emitted twice — once as a real image block in the synthetic user message, and once as inert base64 text via `json.dumps()` in the `role: "tool"` body — exactly doubling the payload and tripping upstream request-size limits. Packed blocks are now stripped from the tool message; blocks that fail to pack are retained.

### Gateway

- **CORS preflight auth bypass** (PR [#404](https://github.com/Oaklight/llm-rosetta/pull/404), [#405](https://github.com/Oaklight/llm-rosetta/pull/405)): Skip auth on CORS preflight requests; hardened to require both `Origin` + `Access-Control-Request-Method` headers.
- **CORS headers on auth errors** — add CORS headers to auth error responses.
- **Bearer token fallback** — accept `Bearer` token as fallback for all auth strategies.
- **Embedding request IDs** — align embedding request IDs with upstream format.
- **Terminal event on upstream stream abort** ([#481](https://github.com/Oaklight/llm-rosetta/issues/481), PR [#483](https://github.com/Oaklight/llm-rosetta/pull/483)): When an upstream connection dropped mid-stream, the gateway closed the SSE socket without any terminal event — clients waiting on one reported a bare `stream closed before response.completed` with no trace of the cause. The gateway now emits a format-appropriate terminal event carrying the upstream reason (`response.failed` + `[DONE]` for Responses, an error chunk + `[DONE]` for Chat, `event: error` for Anthropic, an error object for Google). Skipped when the stream already ended or the client disconnected. Adds `StreamProcessor.source_context`, plus `StreamContext.next_sequence_number` and `StreamContext.outbound_response_id`.
- **Structured JSON logging** (PR [#468](https://github.com/Oaklight/llm-rosetta/pull/468)): Configurable `debug.log_format` (`json`/`text`/`auto`). JSON mode emits one JSON object per line with UTC ISO 8601 timestamps, structured extras promoted to top-level keys via allowlist. `auto` resolves to `json` for non-TTY, `text` for interactive. Hot-reload via admin API.
- **In-band upstream error surfacing** (PR [#454](https://github.com/Oaklight/llm-rosetta/pull/454)): When an upstream reports a request error inside a 200 SSE stream (e.g. Argo sends `event: error` for over-limit tools), the error chunk was previously swallowed and the client received a successful but empty response. Now detects bare error envelopes, emits a format-appropriate error event, and stops the stream cleanly.
- **Admin custom tools toggle** (PR [#467](https://github.com/Oaklight/llm-rosetta/pull/467)): Expose `supports_custom_tools` checkbox in admin provider settings with (i) hint tooltip.
- **Configurable timeouts** (PR [#463](https://github.com/Oaklight/llm-rosetta/pull/463)): `server.upstream_timeout` and `server.read_timeout` config options (both default 300s).
- **Root redirect** (PR [#461](https://github.com/Oaklight/llm-rosetta/pull/461)): `server.root_redirect` config option for redirecting `GET /` to admin panel.
- **Anonymous access opt-in** — `server.open_on_no_keys` allows anonymous access when no API keys are configured.
- **Admin modal polish** — CSS fixes, flatten hint tooltip, i18n alignment.


### Shims & Transforms

- **Auto-inject Anthropic cache breakpoints** ([#464](https://github.com/Oaklight/llm-rosetta/issues/464), [#465](https://github.com/Oaklight/llm-rosetta/issues/465), PR [#469](https://github.com/Oaklight/llm-rosetta/pull/469)): Cross-format requests (OpenAI/Gemini → Anthropic) that lack cache semantics now get up to 4 `cache_hint` breakpoints injected automatically via the `auto_cache_breakpoints` IR transform. Breakpoints placed on: last tool definition, system instruction tail, and last two user messages. Mounted on `argo--anthropic` and `openrouter--anthropic` shims. Two modes: `none_only` (default, skips if any hint exists) and `fill_gaps` (fills per-segment independently).
- **Custom tool downgrade for Chat upstreams** ([#460](https://github.com/Oaklight/llm-rosetta/issues/460), PR [#486](https://github.com/Oaklight/llm-rosetta/pull/486)): `supports_custom_tools` flag on `ProviderShim` (default `False`). When targeting a Chat upstream that doesn't support `{type: "custom"}` tool definitions, `enforce_custom_tools()` downgrades them to `{type: "function"}` with synthetic parameters on request, and `restore_custom_tool_calls()` restores the original type on response.

### Converters

- **Native custom tool support in OpenAI Chat** — `openai_chat` converter natively handles `custom` tool type, including `set_tool_call_type()` public API on `ConversionContext`.
- **`custom_tool_call_output` item type** — OpenAI Responses converter supports `custom_tool_call_output` input items.
- **BaseConverter template method refactor** — `BaseConverter` converted to template method pattern with `__init_subclass__` enforcement of `_PASSTHROUGH_RESTORE_KEY`.

### CI & Documentation

- **On-demand compliance testing** — add [llm-comply](https://github.com/Oaklight/llm-comply) GitHub Actions workflow for schema/spec-based compliance testing against the gateway.
- **Compliance Testing section in README** — link to llm-comply, hosted service, and CI workflow.

## v0.7.3 — 2026-07-25

### Added

- **Per-model enable/disable toggle** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Models can now be individually enabled or disabled via an ON/OFF pill toggle in the admin panel. Disabled models are excluded from routing (`_parse_models` skips `enabled: false`). Backend routes: `toggle_model`, `bulk_update_models` (batch enable/disable/delete).
- **Embedding test menu** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Embedding models now show dedicated test options — Embedding, Batch (array of texts), Matryoshka (user-specified dimensions), and Multimodal (image). Matryoshka uses a custom modal instead of a native `prompt()` dialog.
- **URL template admin UI** for provider and model cards — configure custom upstream URL templates directly from the admin panel.

### Changed

- **Model modal three-tab layout** ([#389](https://github.com/Oaklight/llm-rosetta/pull/389)): Redesigned model edit/add modal with tabbed interface — General (name+provider, segmented LLM/Embedding control, pill-style capability chips), Routing (URL template with expand-link for stream), Transforms (flatten system + reasoning config). Replaces long-scroll single-panel form.
- **Model table UI restyle** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Checkbox column for multi-select with bulk action bar (Enable/Disable/Delete). Clone and Delete moved into ⋯ dropdown menu. Test button unified across LLM and embedding models.
- **Atomic config writes** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387)): `write_config` now uses `tempfile.mkstemp` + `fsync` + `os.replace` for crash-safe, cross-platform atomic writes. Removes all platform-specific locking code (`fcntl`/`msvcrt`). Readers never see a partially-written file.
- **Cross-process config serialization** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387)): New `config_lock(path)` context manager using `.lock` sidecar files with `fcntl.flock` (Unix) / `msvcrt.locking` (Windows). Protects against multiple gateway instances sharing the same config file. All 14 admin route handlers wrapped to serialize read-modify-write cycles.

### Fixed

- **Windows compatibility** ([#381](https://github.com/Oaklight/llm-rosetta/pull/381)): Gateway no longer imports Unix-only `fcntl` module at top level.
- **Preserve upstream User-Agent header** ([#385](https://github.com/Oaklight/llm-rosetta/pull/385)): The gateway now passes through the client’s `User-Agent` header to upstream providers instead of dropping it.
- Filter null values from usage token details before IR validation.
- Align `test_auth` with `open_on_no_keys` behavior ([#388](https://github.com/Oaklight/llm-rosetta/pull/388)).

### Security

- **Block requests when no API keys configured** ([#383](https://github.com/Oaklight/llm-rosetta/pull/383)): When `api_keys` is empty and `open_on_no_keys` is not set, API requests are now rejected with 403 instead of silently passing through.

## v0.7.2 — 2026-07-20

### Added

- **`custom_head` injection for admin panel** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378)): `setup_admin()` accepts an optional `custom_head` HTML fragment injected before `</head>`. Downstream projects can inject `<style>`/`<script>` tags to customize admin UI without modifying the reference `admin.html`. Cached per value — no per-request overhead.
- **`branding` dict for admin panel identity** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378)): `setup_admin(..., branding={title, subtitle, version, links, attribution})` customizes the header, login screen, and settings footer. Serialized as `window.__branding` via `custom_head`; consumer script in `admin.html` patches the DOM. Element IDs: `brandTitle`, `brandLoginTitle`, `brandFooterName`, `brandFooterLinks`. Without branding, the default llm-rosetta identity is unchanged.

### Changed

- Bump vendored `httpclient` 0.4.4 → 0.4.5 — fixes fd leak where `close()` did not close `_async_writer`, preventing `__del__` from cleaning up leaked async streaming responses.
- **Extract `ConfigIO` protocol for admin panel config I/O** ([#376](https://github.com/Oaklight/llm-rosetta/pull/376)): Admin routes now use a `ConfigIO` protocol instead of importing `load_config`/`load_config_raw`/`write_config` directly. Default `JsoncConfigIO` implementation preserves existing behavior; downstream projects (e.g. argo-proxy) can supply alternative implementations via `setup_admin(..., config_io=...)`. Internal helpers `_get_config_path` and `_get_config_io` now raise descriptive `RuntimeError` on missing values instead of returning `None`, removing 16 redundant guard blocks across route handlers.
- Replace Unicode emoji (🔍) with inline SVG in the content capture table for consistent cross-platform rendering.

### Fixed

- Escape `</` in branding JSON serialization to prevent `<script>` tag breakout when branding values contain `</script>`.

## v0.7.1 — 2026-07-16

### Fixed

- **Tool schema sanitization for Anthropic and Google** ([#372](https://github.com/Oaklight/llm-rosetta/issues/372)): Anthropic rejects the OpenAPI `nullable` extension in tool parameter schemas (e.g. from Pydantic-generated JSON Schema). New `convert_nullable_to_type_array()` helper recursively converts `"nullable": true` to standard JSON Schema `"type": [T, "null"]`. Anthropic converter now strips `title` fields and converts `nullable` to type arrays; Google GenAI converter strips `title` (keeps `nullable` — Google supports it). Also handles the edge case where `nullable: true` appears alongside `anyOf`/`oneOf` without a `type` field.
- **`flatten_system` checkbox layout and i18n** in the gateway admin panel.

### Changed

- Bump vendored `validate` 0.6.0 → 0.6.1 (dataclass instance support).
- Restrict Dependabot to LLM SDK dependencies only.

### Added

- SDK type coverage scanner and manual CI workflow for tracking provider SDK type alignment.

## v0.7.0 — 2026-07-10

### Added

- **Anthropic `cache_control` preservation** ([#362](https://github.com/Oaklight/llm-rosetta/pull/362)): New `cache_hint` field on IR parts (`TextPart`, `ImagePart`, `FilePart`, `ReasoningPart`, `ToolCallPart`, `ToolResultPart`, `ToolDefinition`) enables round-tripping Anthropic's block-level `cache_control` through the IR pipeline. The Anthropic converter reads `cache_control` → `cache_hint` on ingest and writes it back on output; non-Anthropic converters silently ignore `cache_hint`, ensuring cross-format safety.
- **`flatten_system_content()` transform** ([#370](https://github.com/Oaklight/llm-rosetta/issues/370)): New body-level transform factory that flattens system message content arrays to plain strings. OpenAI Chat converter now outputs structured content arrays for system messages (preserving block boundaries for `cache_hint`); `flatten_system_content()` downgrades to plain strings for upstream compatibility. Per-model `flatten_system` gateway config with auto-detection for Gemini models. Admin panel toggle included.

### Fixed

- **OpenAI SDK 2.45+ compatibility**: Added `cache_write_tokens` field to `InputTokensDetails` (Responses API) and `PromptTokensDetails` (Chat Completions API) TypedDict replicas to match upstream SDK changes.

### Changed

- **Transform fields renamed** — `from_transforms` → `pre_ir_transforms`, `to_transforms` → `post_ir_transforms` on `ProviderShim`. Old names accepted as backward-compatible aliases in both constructor kwargs and `transforms.py` exports.
- **`system_instruction` unified to `list[TextPart]`** ([#364](https://github.com/Oaklight/llm-rosetta/issues/364)): The canonical IR form of `system_instruction` is now `list[TextPart]` instead of `str`. A single string `"You are helpful."` is represented as `[TextPart(type="text", text="You are helpful.")]`. This ensures consistent structure across all converters and enables block-level metadata (e.g. `cache_hint` for Anthropic prompt caching) to flow through the IR pipeline. All 4 converters updated. **Breaking**: code that reads `ir_request["system_instruction"]` as `str` must handle `list[TextPart]`.

## v0.7.0a1 — 2026-06-27

### Added

- **Hybrid profiling system** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339)): Always-on `perf_counter` phase timing in `ConversionPipeline.profile` (source_to_ir_ms, ir_transforms_ms, ir_to_target_ms, etc.) plus on-demand per-request pyinstrument deep profiling via admin API. `DeepProfiler` context manager in `llm_rosetta.profiling`; new `[profiling]` optional dependency group. Admin endpoints: `POST /admin/api/profiling/enable`, `GET /admin/api/profiling/results`, `GET /admin/api/profiling/results/<index>`, `POST /admin/api/profiling/disable`, `DELETE /admin/api/profiling/results`
- **Profiling admin UI** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339)): New "Profiling" section in admin dashboard with enable/disable controls, result listing, flamegraph download (single and bulk), and restart hint
- **Error dump capability** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341)): Fire-and-forget error dump system that captures full request context on upstream/conversion failures. Image offload before hashing for content-based dedup, zlib compression, 10K entry cap with cascade pruning. Four trigger points covering upstream errors, stream header errors, stream chunk errors, and conversion errors. New functions `dump_error()`, `offload_images()`, `compute_body_hash()`, `compress_body()`, `decompress_body()` exported from `llm_rosetta.observability`
- **Metrics rebuild** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340)): `POST /admin/api/metrics/rebuild` endpoint and "Rebuild Counters" button in admin dashboard. Reconstructs all metrics counters from request log history using batched iteration and atomic swap to avoid exposing half-rebuilt state
- **Observability package** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341)): Extracted `MetricsCollector`, `RequestLog`, `RequestLogEntry`, `PersistenceManager`, and `ProfilerState` from `gateway/admin/` into a new top-level `llm_rosetta.observability` package. These modules are framework-agnostic and can be used by any LLM proxy consumer (e.g. argo-proxy) without depending on the gateway's config system or HTTP server. The `gateway/admin/` modules now re-export from `observability/` for full backward compatibility

### Fixed

- **Metrics breakdown by provider name** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340)): Dashboard breakdown section now groups by provider display name instead of API type, which was merging all Anthropic-format providers into one row
- **Config file write safety**: `write_config()` now uses file locking for cross-process safety
- **Vendored httpserver updated to 0.2.1**: Returns proper HTTP error responses instead of silent disconnects on malformed requests
- **Vendored SSE updated to 0.3.2**: Uses constructor arguments for parser initialization instead of post-init mutation

### Changed

- **Dev tool versions pinned**: `ruff==0.15.20` and `ty==0.0.54` pinned in `[project.optional-dependencies]` to prevent CI drift from upstream tool releases

## v0.7.0a0 — 2026-06-25

### Added

- **ConversionPipeline class** ([#322](https://github.com/Oaklight/llm-rosetta/pull/332)): High-level orchestration class encapsulating the full Phase 1→2→4 conversion lifecycle. `convert_request()`, `convert_response()`, `create_stream_processor()` with `on_ir_ready` callbacks for metadata store integration. One-shot guard prevents accidental reuse
- **Routing layer** ([#323](https://github.com/Oaklight/llm-rosetta/pull/331)): `ResolvedRoute` frozen dataclass and `Router` protocol in the core library. `GatewayConfig.resolve()` consolidates model lookup, provider type, shim binding, capabilities, and reasoning overrides into a single typed result
- **Capabilities module** ([#335](https://github.com/Oaklight/llm-rosetta/pull/336)): `capabilities.py` with `enforce_reasoning()` (pre-IR) and `enforce_vision()` (post-IR) — platform-level capability enforcement separated from provider-specific shim transforms
- **IRTransform system** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334)): `TransformContext` dataclass, `IRTransform` callable type, `apply_ir_transforms()` executor, and `_NamedIRTransform` wrapper. IR-level transforms are now declarative on `ProviderShim.ir_transforms`, separate from body-level `Transform`
- **IR transform factories**: `strip_non_vision_images()`, `truncate_images(max, pattern)`, `unwind_parallel_tool_calls(pattern)` — factory functions producing `IRTransform` callables
- **Message-level transform primitives** ([#328](https://github.com/Oaklight/llm-rosetta/pull/333)): `replace_message_field()`, `default_message_field()`, `strip_fields_for_model()` for nested field operations on `messages[]`
- **Transport layer** ([#321](https://github.com/Oaklight/llm-rosetta/pull/329)): `UpstreamTransport` protocol, `HttpTransport` implementation, `UpstreamResponse`/`UpstreamStream` types, `HttpClientPool`, `send_passthrough()` for non-conversion endpoints
- **`resolve_shim()` public function**: Promoted from private `_resolve_shim()` to public API on `provider_shim.py`

### Breaking Changes

- **`ProviderShim` fields removed**: `max_images`, `max_images_pattern`, `unwind_parallel_tool_calls`, `unwind_parallel_tool_calls_pattern` deleted — these capabilities are now declared via `ir_transforms` tuple using factory functions (`truncate_images()`, `unwind_parallel_tool_calls()`)
- **`apply_shim_to_ir()` behavior changed**: No longer hardcodes image/unwind operations; reads `shim.ir_transforms` declaratively. Renamed to `apply_ir_transforms()` (old name is a deprecated alias)
- **Gateway handler signatures changed**: `handle_non_streaming` and `handle_streaming` take `route: ResolvedRoute` instead of 6 separate parameters (`source_provider`, `target_provider`, `model`, `target_shim_name`, `reasoning_config_override`, `model_capabilities`)

### Refactored

- **Pipeline renamed** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334)): `apply_shim_to_ir()` → `apply_ir_transforms()`, `setup_shim_context()` → `configure_context()`. Old names emit `DeprecationWarning`
- **Gateway proxy.py**: Handlers use `ConversionPipeline` internally. `_resolve_target_transforms`, `process_stream_chunk` deleted
- **Embeddings handler**: Uses `transport.send_passthrough()` instead of reaching into `HttpTransport._pool`. Migrated from `resolve_model()` to unified `resolve()` API and replaced inline telemetry with shared `_record_telemetry()`
- **Auth functions renamed**: `_openai_auth` → `openai_auth` etc. (dropped underscore, public API)
- **Removed `GatewayConfig.resolve_model()`**: Legacy 5-tuple API superseded by `resolve()` which returns `ResolvedRoute` + `ProviderInfo`. Duplicate `DEFAULT_CAPABILITIES` class variable removed

### Fixed

- **Restored legacy `converters/base/` import paths** ([#317](https://github.com/Oaklight/llm-rosetta/pull/317)): Backward-compatible shim modules at old paths (`converters.base.tools`, `.schema`, `.tool_content`, `.cache`)
- **`sanitize_schema` strips `exclusiveMinimum`/`exclusiveMaximum`** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337)): Google GenAI API rejects JSON Schema draft 6+ numeric constraints in tool definitions
- **Stop emitting `reasoning.type` for OpenAI Responses API** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337)): OpenAI and Volcengine Responses APIs reject `reasoning.type` — reasoning is controlled via `reasoning.effort` only. Historical bug from v0.6.8

## v0.6.12 — 2026-06-23

### Fixed

- **Restored legacy `converters/base/` import paths** ([#310](https://github.com/Oaklight/llm-rosetta/issues/310)): The v0.6.11 helpers/ reorganization unintentionally broke import paths that external callers relied on. `sanitize_schema`, `extract_part_ids`, `log_orphan_warnings`, `fix_orphaned_tool_calls_ir`, and `strip_orphaned_tool_config` are re-exported from `converters.base.tools` again, and compatibility shim modules at `converters.base.schema`, `converters.base.tool_content`, and `converters.base.cache` redirect to their new `helpers/` locations. The canonical import path remains `llm_rosetta.converters.base.helpers`; existing code importing from the old paths (e.g. `from llm_rosetta.converters.base.tools import sanitize_schema`) keeps working without changes. The cache singletons are shared across both paths

## v0.6.11 — 2026-06-21

### Added

- **Admin panel provider UX enhancements** ([#292](https://github.com/Oaklight/llm-rosetta/pull/292)): Three improvements to the provider tab:
    - **Multi-key entry list**: API key field auto-detects comma-separated keys (rotation) and switches to multiple `<input type="password">` entries. `+ Add key` button always visible for manual promotion. Eye toggle and copy button in unified footer
    - **Provider search bar**: appears when provider count exceeds 6, filters by name, type, and base URL
    - **Grid/list view toggle**: two icon buttons switch between card grid and compact single-column list view. Preference persisted in localStorage
- **Request ID propagation** ([#296](https://github.com/Oaklight/llm-rosetta/pull/296), [#122](https://github.com/Oaklight/llm-rosetta/issues/122)): Every proxy request generates or honours an `X-Request-ID` header. Propagated to upstream providers, included in all response headers (including error responses), and logged with `[request_id]` prefix for end-to-end traceability
- **Enhanced health check endpoints** ([#297](https://github.com/Oaklight/llm-rosetta/pull/297), [#127](https://github.com/Oaklight/llm-rosetta/issues/127)):
    - `/health` — returns uptime, request counts, errors in the last hour, and per-provider health snapshot (success rate, avg latency, last error). Always HTTP 200; `status` field shows `"ok"` or `"degraded"`
    - `/health/live` — always 200 (Kubernetes liveness probe)
    - `/health/ready` — 200 when all providers healthy, 503 when any provider is critically unhealthy (Kubernetes readiness probe)
- **CORS restriction on admin API** ([#294](https://github.com/Oaklight/llm-rosetta/pull/294), [#233](https://github.com/Oaklight/llm-rosetta/issues/233)): `/admin/api/*` endpoints no longer send `Access-Control-Allow-Origin: *`. New config option `server.admin_cors_origins` (list, default `[]`) allows explicit origin allow-listing. `/v1/*` proxy endpoints unchanged
- **Image count enforcement via shim** ([#301](https://github.com/Oaklight/llm-rosetta/pull/301), [#299](https://github.com/Oaklight/llm-rosetta/issues/299)): `ProviderShim` gains `max_images` and `max_images_pattern` fields. When set, images exceeding the limit are replaced with `[image omitted due to limit]` text placeholders (oldest first, most recent kept). Argo OpenAI shim declares `max_images: 50` with pattern `^(gpt|o\d)` — only GPT/o models are truncated; Gemini and Claude through the same provider pass through unaffected
- **Vision capability enforcement** ([#314](https://github.com/Oaklight/llm-rosetta/pull/314), [#313](https://github.com/Oaklight/llm-rosetta/issues/313)): Models without `vision` capability now have all images automatically stripped and replaced with `[image not available]` instead of being forwarded to upstream where they cause opaque errors (e.g. DeepSeek's "unknown variant `image_url`"). Gateway logs a warning with image count and model name
- **Unix domain socket support** ([#315](https://github.com/Oaklight/llm-rosetta/pull/315)): Gateway can listen on a Unix socket instead of TCP via `--socket/-S` CLI flag or `server.socket` config field. Enables secure deployments on shared multi-user hosts where `127.0.0.1` still exposes the service to all local users. Socket file is restricted to owner-only (`0600`) and cleaned up on shutdown
- **Parallel tool call unwind** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300)): `ProviderShim` gains `unwind_parallel_tool_calls` and `unwind_parallel_tool_calls_pattern` fields. When enabled, parallel tool calls (multiple `tool_call` parts in one assistant message) are unwound into sequential call-result pairs before forwarding. Argo OpenAI shim enables this with pattern `^gemini` — Gemini models through Argo get sequential pairs; GPT/o models pass through unchanged

### Changed

- **`converters/base/` reorganized into helpers/ subpackage** ([#311](https://github.com/Oaklight/llm-rosetta/pull/311), [#312](https://github.com/Oaklight/llm-rosetta/pull/312), [#310](https://github.com/Oaklight/llm-rosetta/issues/310)): Utility functions extracted from the flat `converters/base/` directory into `converters/base/helpers/`. Abstract base classes (the Ops pattern contract) stay at the top level; implementation utilities (`cache`, `schema`, `tool_orphan_fix`, `tool_content`, `tool_call_unwind`, `image_limit`, `reasoning`) move to `helpers/`. `tools.py` reduced from 428→185 lines (pure ABC). `reasoning_helpers.py` moved from `converters/` root. `orphan_fix.py` renamed to `tool_orphan_fix.py` for consistent `tool_*` prefix. `helpers/__init__.py` re-exports public functions
- **Retired Argo `_normalize_thinking` transform** ([#304](https://github.com/Oaklight/llm-rosetta/pull/304), [#192](https://github.com/Oaklight/llm-rosetta/issues/192)): Removed dead code from Argo Anthropic shim — the `_normalize_thinking` function, `_BUDGET_RATIO`, and `_ADAPTIVE_THINKING_MODELS` were replaced by declarative `reasoning.model_overrides` in `provider.yaml` but the code and 19 tests remained
- **Speculative extension types marked experimental** ([#302](https://github.com/Oaklight/llm-rosetta/pull/302), [#71](https://github.com/Oaklight/llm-rosetta/issues/71)): `SystemEvent`, `BatchMarker`, `SessionControl`, `ToolChainNode` moved from `types.ir.extensions` to `types.ir.extensions_experimental`. Old import path still works but emits `DeprecationWarning`. Types removed from default `types.ir` namespace; available via `from llm_rosetta.types.ir import experimental`
- **Admin panel i18n**: Chinese translation updated from "服务商" to "服务方" (more neutral for mixed commercial and self-hosted providers)
- **Request log timestamps** ([#298](https://github.com/Oaklight/llm-rosetta/pull/298)): Now show date and time (e.g. "06/19, 20:25:29") instead of time only

### Fixed

- **Admin panel auth flash** ([#291](https://github.com/Oaklight/llm-rosetta/pull/291)): Eliminated flash of unauthenticated content when `admin_password` is configured. Main UI is hidden via CSS (`body.auth-pending`) until the async auth check completes
- **Admin password unresolved env var** ([#293](https://github.com/Oaklight/llm-rosetta/pull/293)): Gateway now refuses to start if `admin_password` contains an unresolved `${...}` placeholder, preventing a predictable literal string from being used as the password
- **`is_image_part` type guard for OpenAI format** ([#306](https://github.com/Oaklight/llm-rosetta/pull/306)): `is_image_part()` now matches both `type: "image"` (IR canonical) and `type: "image_url"` (OpenAI format retained in IR), fixing image truncation being silently skipped for OpenAI-format requests
- **Tool result images counted for truncation** ([#308](https://github.com/Oaklight/llm-rosetta/pull/308), [#299](https://github.com/Oaklight/llm-rosetta/issues/299)): `truncate_images()` now scans images inside `tool_result.result` lists, not just direct message content. Fixes requests that had ≤50 images at IR level but exceeded 50 after the OpenAI Chat converter unpacked tool result images into synthetic user messages. Also optimized `deepcopy` to only copy affected messages instead of the entire conversation
- **Argo Gemini parallel tool call failures** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300)): All Gemini models through Argo gateway failed with "function response parts ≠ function call parts" when Claude Code made parallel tool calls. Root cause: Argo's internal OpenAI→Gemini conversion doesn't merge separate tool result messages into a single `functionResponse` Content block. Fixed by unwinding parallel tool calls to sequential pairs at IR level before forwarding

## v0.6.10 — 2026-06-18

### Added

- **Process-level conversion cache** ([#276](https://github.com/Oaklight/llm-rosetta/issues/276), [#279](https://github.com/Oaklight/llm-rosetta/pull/279), [#281](https://github.com/Oaklight/llm-rosetta/pull/281), [#283](https://github.com/Oaklight/llm-rosetta/pull/283)): Per-entry LRU cache with access-refreshed TTL (default 30 min) for tool conversion, schema sanitization, and IR validation. Eliminates repeated work for unchanged tool definitions and messages across conversation turns
    - **Hub-and-spoke architecture**: conversion caches (spokes) are converter-specific; IR validation cache (hub) is converter-agnostic and shared across all converters
    - **Per-entry caching**: individual tools and messages cached by content hash — partial tool changes only re-convert the changed entries, and cross-agent tool overlap shares cache entries
    - **Incremental message validation**: only newly appended messages are validated; previously-seen messages are skipped via the IR validation hub
    - **Mutation detection**: `check_integrity()` on test teardown catches accidental in-place mutation of cached objects; optional `verify=True` mode for runtime self-healing
    - **Benchmark**: 4.4× warm-path speedup (3250 µs → 527 µs local); 33% TTFB reduction in production (11.4 ms → 7.6 ms)
- **`validate_tools()`** ([#283](https://github.com/Oaklight/llm-rosetta/pull/283)): New standalone IR validation function for tool definition lists, symmetric with `validate_messages()`
- **OpenRouter Anthropic shim** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): OpenRouter's Anthropic-compatible Messages endpoint is now a first-class provider type. The single `openrouter` shim is split into `openrouter--openai_chat` (Chat Completions) and `openrouter--anthropic` (Messages API), letting OpenRouter route Claude models through the native Anthropic format
- **Admin panel per-model reasoning override** ([#288](https://github.com/Oaklight/llm-rosetta/pull/288)): The model edit modal now displays the effective reasoning config (`thinking_type`, `budget_tokens_ratio`, `disabled_strategy`) with a source badge (provider / model_override / config) and inline editing. Overrides are persisted to `config.jsonc` and resolved at runtime with priority: config override > shim model_override > shim provider default
- **`budget_tokens_default_ratio` reasoning capability** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): `ReasoningCapability` gains a `budget_tokens_default_ratio` field. When a provider requires `thinking.type=enabled` but the caller omits `budget_tokens`, a default is derived as `min(max(1024, max_tokens × ratio), max_tokens - 1)` instead of falling back to the unsupported `adaptive` type

### Changed

- **`_convert_tools_from_p` no longer abstract** ([#281](https://github.com/Oaklight/llm-rosetta/pull/281)): Default implementation in `BaseConverter` handles all providers (including Google's list/None return). Per-converter overrides removed — 90 lines of duplicated code eliminated
- **Complete Claude thinking model_overrides** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Added per-model thinking overrides for the Anthropic and Argo shims based on tested support matrices — Haiku 4.5 (`enabled`+budget), Opus 4.7/4.8 (`adaptive`-only), Sonnet 4 on Argo (`enabled`+budget)
- **Model "Clone" replaces "Copy"** ([#290](https://github.com/Oaklight/llm-rosetta/pull/290)): The model row's clone action now opens a prefilled model modal (provider, capabilities, upstream model, and effective reasoning config) with a blank name, matching the provider row's "Clone" behavior — instead of copying a YAML snippet to the clipboard. The model name in the table remains click-to-copy

### Fixed

- **Haiku 4.5 `adaptive` thinking 400 errors** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Haiku 4.5 supports extended thinking but only accepts `thinking.type=enabled` + `budget_tokens`, not `adaptive`. The previous fallback to `adaptive` when no budget was provided caused 400 errors on Anthropic Official, Argo, and OpenRouter. The new `budget_tokens_default_ratio` derives a budget instead
- **Haiku 4.5 `effort` parameter 400 errors** ([#289](https://github.com/Oaklight/llm-rosetta/pull/289)): The `effort` parameter (`output_config.effort`) is only supported on Opus 4.5/4.6/4.7/4.8 and Sonnet 4.6 — not Haiku. Anthropic Official rejected `reasoning_effort` on Haiku 4.5 with a 400. The Haiku model_override now sets `effort_field: none` to drop the unsupported field while keeping the working `thinking.type=enabled` + budget path
- **OpenRouter Anthropic reasoning effort field** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): The `openrouter--anthropic` shim uses `output_config.effort` (Anthropic format) instead of the OpenAI Chat `reasoning_effort` field
- **`.env` secret leakage in Docker builds**: Docker build context no longer includes `.env` files, preventing API keys from being baked into image layers

## v0.6.9 — 2026-06-13

### Added

- **API key rotate**: New `POST /admin/api/keys/<id>/rotate` endpoint generates a fresh key value while preserving the same id and label. The admin panel shows a "Rotate" button with inline confirmation and a one-time copy modal for the new key. Request logs are unaffected — they associate by label, not key value
- **Model type selector in Fetch from Provider modal**: Users can now choose between LLM and Embedding when batch-adding models. LLM shows capability checkboxes (text, vision, tools, reasoning); Embedding auto-sets `['embedding']`
- **Model type selector in Add/Edit Model modal**: Replaces the old embedding checkbox + mutual-exclusion logic with the same Model Type radio pattern

### Changed

- **API key length upgraded**: Default generated keys increased from 36 characters (`rsk-` + 32 hex) to 52 characters (`rsk-` + 48 hex), matching OpenAI's key length (192-bit entropy)

### Fixed

- **SSE streaming proxy compatibility** ([#274](https://github.com/Oaklight/llm-rosetta/issues/274), [#275](https://github.com/Oaklight/llm-rosetta/pull/275)): Vendored `httpserver` v0.1.1 — SSE (`text/event-stream`) streaming responses now use `Transfer-Encoding: chunked` instead of raw byte flushing with `Connection: close`. Fixes Go-based reverse proxies (notably NPS `httputil.ReverseProxy`) misinterpreting SSE data as chunked encoding, producing `invalid byte in chunk length` errors and intermittent connection failures under concurrent load. Upstream fix: [Oaklight/zerodep#101](https://github.com/Oaklight/zerodep/pull/101)
- **Admin panel active tab not loading after login**: `initApp()` now triggers data loading for the currently active tab after successful authentication, fixing the issue where the Request Log tab appeared empty until manually switched away and back
- **Uppercase model type radio labels**: Added `text-transform: none` to `.fetch-type-radios label` to prevent `.form-group label` CSS from uppercasing "Embedding" to "EMBEDDING"

## v0.6.8 — 2026-06-11

### Added

- **Shim-driven reasoning configuration** ([#244](https://github.com/Oaklight/llm-rosetta/issues/244), [#245](https://github.com/Oaklight/llm-rosetta/pull/245)): Reasoning effort mapping is now declarative. Provider shims declare a `ReasoningCapability` in `provider.yaml` — specifying `disabled` strategy (`omit` or `thinking_disabled`), `effort_field`, `effort_map`, and `max_effort` cap — instead of hardcoded converter branches. New shared `reasoning_helpers.py` provides `normalize_reasoning_input()` and `apply_reasoning_config()` used by all four converters
- **Expanded reasoning effort ladder** ([#245](https://github.com/Oaklight/llm-rosetta/pull/245)): IR `ReasoningEffortLevel` expanded to six levels: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Input normalization accepts `none` (maps to `mode: disabled`) and provider-native values (`xhigh`, `max`) as first-class efforts. Provider shims declare `effort_map` to convert IR levels to provider-specific strings and `max_effort` to cap the highest level emitted
- **`block_index` on IR stream delta events** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): `TextDeltaEvent`, `ReasoningDeltaEvent`, and `ToolCallDeltaEvent` now carry an optional `block_index` field, preserving the provider's content block index through IR round-trips
- **`cache_creation_tokens` in `UsageInfo`** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): New field on the `UsageInfo` TypedDict for Anthropic cache creation token counts
- **Model-level `thinking_type` in shim reasoning config** ([#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` gains a `thinking_type` field to force the outbound `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID (e.g. Argo `claudeopus47 → thinking_type: adaptive`). The `_normalize_thinking` transform is retired — thinking type normalization is now declarative via shim YAML
- **Anthropic `provider_metadata` on tool calls, tool results, and reasoning blocks** ([#257](https://github.com/Oaklight/llm-rosetta/pull/257)): The Anthropic converter now serializes `provider_metadata` as `_provider_metadata` on `tool_use`, `tool_result`, and `thinking` blocks during IR→provider conversion, and reads it back during provider→IR. Fixes Google `thought_signature` being lost in cross-provider round-trips (Anthropic client → Google upstream), which caused Gemini 2.5+ to reject requests with 400 "missing thought_signature"
- **Response reasoning losslessness across converters** ([#263](https://github.com/Oaklight/llm-rosetta/pull/263)): Reasoning content is now preserved through response-side IR→provider conversion in all converters that previously dropped it:
    - **Google GenAI**: `p_reasoning_to_ir` now captures `thoughtSignature` into `provider_metadata` instead of discarding it; `message_ops` delegates to `content_ops.p_reasoning_to_ir()` instead of constructing a bare `ReasoningPart` inline
    - **Anthropic**: `ir_text_to_p` / `p_text_to_ir` now round-trip `_provider_metadata` on text blocks, matching the treatment already applied to reasoning and tool blocks
    - **OpenAI Chat**: `_build_choice_to_provider` now collects `ReasoningPart` content and emits it as `reasoning_content` on the response message, instead of silently dropping reasoning parts
- **Provider-specific reasoning field normalization** ([#264](https://github.com/Oaklight/llm-rosetta/pull/264)): Shim transforms and config for MiniMax, OpenRouter, and Volcengine reasoning fields:
    - **MiniMax**: `thinking_type: adaptive` (rejects `enabled`); `_inject_reasoning_split` to_transform auto-sets `reasoning_split: true` when thinking is requested; `_parse_think_tags` from_transform extracts `<think>` tags from content as fallback
    - **OpenRouter**: `_rename_reasoning_field` from_transform renames `message.reasoning` → `message.reasoning_content` (OpenRouter uses non-standard field name)
    - **Volcengine**: `thinking_type: enabled` (rejects `adaptive`; overrides base converter's `auto → adaptive` default)

### Changed

- **`_build_ir_usage` return type tightened to `UsageInfo`** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): All four converter overrides now return `UsageInfo` instead of `dict[str, Any]`, and `_build_provider_usage` accepts `Mapping[str, Any]` instead of `dict[str, Any]`. Removes all usage-related `ty: ignore` comments
- **Anthropic stream usage handlers deduplicated** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): `_handle_message_start_from_p` and `_handle_message_delta_from_p` now call `_build_ir_usage()` instead of duplicating cache field extraction inline (−21 lines)

### Fixed

- **Anthropic stream block index desync after thinking block** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): During Anthropic→IR→Anthropic streaming round-trip, text deltas after a thinking block used index 0 instead of the correct block index (e.g. 1). The Anthropic `from_p` path now copies `chunk["index"]` onto IR delta events, and the `to_p` path prefers the explicit `block_index` over the context fallback. Fixes Claude CLI "Content block is not a text block" errors
- **Cross-provider stream block boundary synthesis** ([#250](https://github.com/Oaklight/llm-rosetta/issues/250), [#251](https://github.com/Oaklight/llm-rosetta/pull/251)): When converting IR streams from providers without content block events (OpenAI Chat, OpenAI Responses, Google GenAI) to Anthropic format, the serializer now emits synthetic `content_block_stop` / `content_block_start` at content-type transitions (e.g. reasoning → text). Previously text deltas could land inside a synthetic thinking block. Added `current_block_type` tracking to `StreamContext`
- **Stream usage detail propagation** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): Cache and detail token fields (`cache_read_tokens`, `cache_creation_tokens`, `prompt_tokens_details`, `completion_tokens_details`, `cachedContentTokenCount`) are now preserved through all four converters' streaming paths. Previously these fields were dropped during stream round-trips
- **OpenAI Chat `thinking.type=auto` passthrough** ([#258](https://github.com/Oaklight/llm-rosetta/pull/258)): IR `mode: "auto"` is not a valid upstream value for OpenAI Chat's `thinking.type`. The OpenAI Chat converter now maps `auto` → `adaptive` before emitting the `thinking` object, and applies the same shim `thinking_type` override + `enabled` → `adaptive` safety fallback that the Anthropic path uses
- **`thinking_type=enabled` fallback when `budget_tokens` missing**: When a shim declares `thinking_type: enabled` but the request has no `budget_tokens` (required by Anthropic for `type: "enabled"`), the converter now automatically falls back to `type: "adaptive"` instead of emitting an invalid payload. Applied to both Anthropic and OpenAI Chat converter paths
- **Unsigned Anthropic reasoning blocks in Argo history** ([#268](https://github.com/Oaklight/llm-rosetta/issues/268), [#269](https://github.com/Oaklight/llm-rosetta/pull/269)): `ReasoningCapability` now supports `unsigned_reasoning_blocks: as_is | preserve`. The `argo--anthropic` shim uses `preserve` so prior assistant `thinking` blocks without a usable signature are not forwarded to Argo, avoiding 400 errors while preserving the reasoning content in `provider_metadata.anthropic.unsigned_reasoning_blocks`

## v0.6.7 — 2026-06-04

### Fixed

- **Embedding endpoint upstream_model alias**: The `/v1/embeddings` passthrough handler now substitutes the `upstream_model` name into the request body before forwarding, matching the behavior of the chat completions proxy handler. Previously model aliases (e.g. `bge-m3` → `BAAI/bge-m3`) were ignored, causing upstream model-not-found errors.
- **Admin test timer leak**: The elapsed-time counter is now tracked globally and cleared when a new test starts, preventing multiple timers from writing alternating values to the same display element.
- **Admin test timeout auto-cancel**: When the browser-side 120s timeout fires, the server-side task is now explicitly cancelled via the API instead of being left running.
- **Server-side test task timeout**: Added `asyncio.wait_for()` with a 120s timeout to `_run_test_task`, so hung upstream calls are terminated server-side instead of lingering until the 300s cleanup window.

## v0.6.6 — 2026-06-03

### Added

- **Admin status bar total requests**: Lifetime request counter shown as the first footer segment with locale-aware thousand separators; per-segment hover tooltips (en/zh) explain each metric
- **Vendor httpclient URL-encoded form data**: `httpclient` v0.4.2 — when `data` is a dict without files, encode as `application/x-www-form-urlencoded` instead of requiring explicit serialization

### Changed

- **Schema sanitization module split**: JSON Schema sanitization extracted from `converters/base/tools.py` into its own `converters/base/schema.py` module for clearer separation of concerns
- **Cyclomatic complexity reduction**: Reduced cognitive complexity across tool ops (cross-converter `extract_part_ids`/`log_orphan_warnings` reuse), gateway auth (`check_admin_auth`), proxy streaming (`process_stream_chunk`), config parsing, logging, and admin routes
- **complexipy threshold**: Raised `max-complexity-allowed` from 15 to 25; added `complexipy-pre-commit` hook definition (commented out) for future enablement

### Fixed

- **Admin footer i18n**: Status bar footer now re-renders on language switch instead of requiring a page refresh
- **Docker non-semver build**: `make build-docker V=dev-test` no longer fails — non-semver `V` values fall back to installing from local wheel instead of `pip install ==<version>`

## v0.6.5 — 2026-06-02

### Added

- **API key label filter** — new dropdown on the Request Log tab to filter entries by API key name
- **Client IP logging** — extracts client IP from `X-Forwarded-For` / `X-Real-IP` / TCP peer address and displays it in a new "Client IP" column on the Request Log tab
- **System clock** — live-updating clock in the admin header for correlating log timestamps with current time
- **Dual-threshold log retention** — success and error request log entries are pruned independently; errors get their own cap (`error_max`) so rare failures are not evicted by a flood of successful traffic
- **DB sizing footer** — admin panel footer shows on-disk database size, entry counts per class, and retention caps

### Fixed

- **Provider filter** — filter now correctly matches entries by provider display name, with three-tier fallback (`target_provider_name` → `target_provider` → API type for legacy NULL rows) to handle backfill gaps and disabled providers
- **`/health` info leak** — endpoint no longer exposes the full provider and model list to unauthenticated callers; now returns only `{"status": "ok"}`
- **i18n completeness** — added missing Chinese translations for footer stats, system time label, filter options, and Client IP column header

### Changed

- **Shim directory layout** — provider shims now support grouped subdirectories (e.g. `argo/anthropic/`, `argo/openai_chat/`)
- **Schema migration** — `_migrate_add_columns()` is now generic, adding any missing nullable columns in a single pass
- **CI** — switched to pre-commit for lint/type checks, pinned ty version

## v0.6.4 — 2026-05-20

### Added

- **Tinyleaf-style settings popup**: Replace the modal-overlay settings dialog with a lightweight centered popup — click outside or press Escape to dismiss, theme and language via `<select>` dropdowns with instant apply, About section with version and project links (GitHub, PyPI, Docker Hub, Docs)
- **Lightweight host IP detection endpoint**: `GET /admin/api/diagnostics/host-ip` reads `/proc/net/route` only (microsecond-level, no network calls); proxy URL placeholders auto-update with the correct Docker host IP on page load
- **Admin login persistence**: Login state stored in `localStorage` with 30-minute inactivity auto-logout, logout button in header, password manager compatibility (proper `<form>`, `autocomplete` attributes)
- **Inline delete confirmation**: Two-step confirm for models, API keys, and request logs replaces native `confirm()` dialogs
- **Test modal improvements**: Cancel button with elapsed timer, chart empty state message, Clone button for providers/models
- **Mobile responsiveness**: Responsive header with wrapping, horizontally scrollable tabs and tables

### Fixed

- **Argo Anthropic response normalization**: Detect and convert OpenAI Chat Completions format responses from Argo's `/v1/messages` endpoint to Anthropic Messages format
- **Model-level `thinking_type` in shim reasoning config** ([#254](https://github.com/Oaklight/llm-rosetta/issues/254), [#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` supports `thinking_type` to force `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID. Argo `claudeopus47 → thinking_type: adaptive` via `model_overrides`. `_normalize_thinking` transform retired — thinking type normalization is now declarative
- **Inline confirm i18n and onclick restore**: Add missing `confirm.sure`/`confirm.yes` translation keys; restore original `onclick` handler after confirmation reverts
- **Reverse proxy caching**: Add `Cache-Control: no-cache, no-store, must-revalidate` on all admin API responses; switch test polling to POST
- **Login overlay loop**: Prevent login overlay from dismissing password manager autofill popups
- **C901 complexity**: Extract `_format_connection_error` helper from `fetch_upstream_models`

### Security

- **Admin login rate limiting**: 5 failed attempts trigger a 5-minute IP lockout

### Changed

- **Settings UI simplified**: Themes reduced to Light/Dark; theme and language selectors moved from header dropdowns into the settings popup

## v0.6.3 — 2026-05-17

### Added

- **Full `custom_tool_call` support for OpenAI Responses API**: Handle the `type: "custom"` tool type end-to-end — request ingestion (coerce to IR `type: "function"` with `_passthrough` for round-trip), response parsing (`custom_tool_call` items with plain-text `input`), and streaming (`response.custom_tool_call_input.delta/done` events). Cross-provider degradation synthesizes a single-string-param JSON Schema so custom tools remain usable on Anthropic/Google
- **`tool_type` field on IR `ToolCallStartEvent`**: Streaming events now carry `tool_type` ("function", "custom", etc.) so converters can emit the correct provider-specific event types
- **Argo shims with `model_id_field` and `upstream_model` alias**: New `argo_openai`, `argo_anthropic`, `argo_google` provider shims that rewrite the model field name for Argo-proxied endpoints. Includes thinking normalization transform for `argo_anthropic`
- **Async server-side test tasks**: Admin panel test requests now run in background tasks, preventing browser connection pool exhaustion on slow models
- **Admin login rate limiting**: Brute-force protection on the admin login endpoint

### Fixed

- **Stored XSS in admin UI**: Escape single quotes in the `esc()` helper to prevent injection via provider/model names
- **`custom_tool_call` streaming type loss in gateway**: `OpenAIResponsesStreamContext.from_base()` now copies `_tool_call_types`, fixing custom tools falling back to `function_call` event types during IR→provider streaming
- **Admin UI regressions**: Fix infinite recursion in fetch models checkbox handler, allow API key editing regardless of `credential_visible` setting, remove prefix real-time preview input lag, fix fetch models prefix losing selections, abort test requests on modal close
- **Reasoning test `max_tokens` too small**: Enforce `budget_tokens >= 1024` for reasoning capability tests
- **httpclient AsyncClient serialization lock**: Update vendored httpclient to v0.4.1, use per-task AsyncClient for test self-calls to avoid deadlock
- **ty type-check errors**: Resolve compatibility issues with ty 0.0.32+

### Changed

- **Admin routes split into subpackage**: Refactored monolithic `routes.py` into `routes/` with dedicated modules for auth, config, keys, observability, and testing
- **CI switched to pre-commit**: Linting now uses `pre-commit run --all-files` (ruff + ty); complexipy suspended pending upstream fix

## v0.6.2 — 2026-05-15

### Added

- **Admin password protection**: `server.admin_password` in config enables a login overlay for the admin panel, using HMAC-based session tokens
- **Credential visibility control**: `server.credential_visible: false` hides API key viewing/copying across the admin UI
- **Provider cascade delete**: Deleting a provider now shows affected models and cascade-deletes them

### Fixed

- **Base URL overwrite**: Switching provider type no longer overwrites user-entered base URLs
- **Request log collapse**: Expanded error detail rows persist across auto-refresh

### Changed

- **Zero-dependency on Python ≥3.11**: Replaced PyYAML with vendored zerodep yaml module

## v0.6.1 — 2026-05-15

### Added

- **`/v1/embeddings` passthrough endpoint**: Proxy embedding requests directly to upstream providers without IR conversion — the OpenAI embeddings format is universal across compatible providers. Includes metrics and request log instrumentation
- **`/v1/models` enriched response**: Model listing now includes `api_standard` (e.g. `"openai_chat"`, `"anthropic"`) and per-model `capabilities` fields
- **"Fetch from Provider" in admin panel**: Query upstream `/v1/models` (or equivalent) endpoint from the Models tab, browse available models with checkboxes, and bulk-add with optional prefix. Already-existing models shown as disabled
- **Model management enhancements**: Provider filter dropdown and model name search in the Models tab
- **Embedding capability and test type**: `embedding` capability in the model editor (mutually exclusive with `vision`/`tools`). Embedding models get a single Test button that POSTs to `/v1/embeddings` and displays dimension count
- **Reasoning capability and test type**: `reasoning` capability with dedicated test that sends `reasoning_effort: "low"`. Mutually exclusive with `embedding`
- **Admin panel tab persistence**: Active tab stored in `localStorage`, survives page refresh

### Fixed

- **Missing event loop in SOCKS5 proxy tests**: Use `asyncio.new_event_loop()` as fallback when prior tests have closed the default event loop
- **Type assertion for httpclient response in fetch_upstream_models**: Resolve ty type-check error for `AsyncClient.get()` return type

## v0.6.0 — 2026-05-15

### Added

- **Provider shim layer with declarative YAML directory**: Shims are now defined as `provider.yaml` + optional `transforms.py` files under `shims/providers/<name>/`, automatically discovered and registered at import time
- **Transform mechanism for provider-specific field adaptation**: Three composable primitives — `strip_fields()`, `rename_field()`, `set_defaults()` — handle field-level differences between a provider's API dialect and its base standard
- **7 new built-in provider shims**: xAI (Grok), Qwen (DashScope), Moonshot (Kimi), MiniMax, Zhipu (GLM), OpenRouter, Volcengine — each with provider-specific transforms where needed
- **Gateway proxy applies shim transforms**: The gateway request/response pipeline now applies `to_transforms` on outbound requests and `from_transforms` on inbound responses and stream chunks
- **Provider logos in admin panel**: Provider shims can declare a `logo` URL (SVG), displayed in the admin panel provider cards
- **SOCKS5 proxy support restored**: Updated vendored `httpclient` from zerodep v0.3.1 to v0.4.0, which includes full SOCKS5 proxy support (RFC 1928/1929, with username/password authentication). Both `--proxy socks5://...` CLI flag and `"proxy": "socks5://..."` config entries now work for all upstream requests

### Changed

- **Shim system refactored to declarative YAML**: Replaced programmatic `builtins.py` with a directory-based system (`shims/providers/*/provider.yaml` + `transforms.py`). Adding a new provider now requires only YAML + optional Python, no changes to core code
- **Vendored `validate` updated to zerodep v0.5.0**: Adds `FieldValidator` and `model_validator` for field-level transform+validate pipelines

### Removed

- **`ModelShim` class removed**: Model-level metadata removed in favor of simpler provider-only shims. The `ProviderShim` dataclass no longer has a `models` field

### Refactored

- **Zero-dependency gateway** ([#178](https://github.com/Oaklight/llm-rosetta/pull/178)): Replaced Starlette + uvicorn + httpx with vendored zerodep `httpserver` and `httpclient` modules. The `[gateway]` extra now has zero external runtime dependencies

### Fixed

- **Deep-merge properties in schema flattening** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Fix `$ref`/`$defs` resolution to deep-merge properties and strip orphaned `required` entries
- **Unconditional usage fallback and StreamContext merge** ([#176](https://github.com/Oaklight/llm-rosetta/pull/176)): Guard against missing usage data and ensure StreamContext state is properly merged

### Known Issues

- **Google tool schema `required` validation** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Some Anthropic tool schemas have `required` entries referencing properties not defined in the schema, causing Google API to reject with `INVALID_ARGUMENT`

## v0.5.3 — 2026-04-25

### Added

- **OpenAI Chat converter: thinking config support** ([#170](https://github.com/Oaklight/llm-rosetta/pull/170)): The OpenAI Chat converter now handles `reasoning_config` in IR requests, mapping to OpenAI's `reasoning_effort` parameter. Enables thinking/extended thinking configuration when routing through the Chat Completions API
- **OpenAI Chat converter: `reasoning_content` field handling**: Non-streaming and streaming responses from reasoning models (e.g., o1, o3) now correctly extract the `reasoning_content` field and convert it to IR `ReasoningPart`, preserving chain-of-thought content during cross-provider conversion
- **Upstream error body in admin request log**: When an upstream provider returns an error, the response body is now included in the admin request log entry, making it easier to diagnose upstream failures without checking server logs
- **Copy entry buttons for providers and models in admin page**: Provider and model entries in the admin panel now have copy/duplicate buttons for quickly creating new entries based on existing configurations

### Fixed

- **`FilePart` excluded from `UserContentPart`** ([#160](https://github.com/Oaklight/llm-rosetta/issues/160), [#162](https://github.com/Oaklight/llm-rosetta/pull/162)): `UserContentPart` union type did not include `FilePart`, causing `validate_ir_request()` to reject any user message containing file content (e.g., PDF attachments sent by Claude Code as Anthropic `document` blocks). The bidirectional conversion logic was already implemented for Anthropic (`document`), Google (`inlineData`), and OpenAI Responses (`input_file`) — only the type definition was missing
- **`google_genai/content_ops.py` unconditional `httpx` import** ([#163](https://github.com/Oaklight/llm-rosetta/issues/163)): Replaced `httpx` with `urllib.request` in the Google GenAI content converter for image URL downloads. `httpx` was only declared as a `[gateway]` optional dependency but was imported unconditionally, causing `ModuleNotFoundError` when installed without `[gateway]` extra
- **Emoji icons replaced with SVG in API key management**: API key action buttons in the admin panel used emoji characters that rendered inconsistently across platforms. Replaced with inline SVG icons and added a key visibility toggle button
- **API key column layout shift**: Fixed CSS layout issue where the API key column width changed when toggling key visibility, causing adjacent buttons to shift position
- **Wheel path glob collision with extras brackets**: Quoted the wheel file path in CI install commands to prevent shell glob expansion when the filename contains `[extras]` bracket syntax

### Refactored

- **SQLite persistence backend**: Replaced the JSONL-based request log and JSON-based metrics persistence with a unified SQLite backend. Provides better write durability, atomic operations, and eliminates log rotation complexity. Vendored `persistdict` from zerodep (v0.4.1) as the key-value storage layer

### CI/Build

- **Install smoke tests**: Added CI smoke tests that verify `pip install` succeeds for both `llm-rosetta` (core) and `llm-rosetta[gateway]` variants, catching missing or circular dependencies early

## v0.5.2 — 2026-04-19

### Fixed

- **Streaming round-trip event inflation** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Fixed multiple scenarios where `Provider A → IR → Provider B` streaming conversion produced more output events than input events:
    - OpenAI Chat, Anthropic, and Google GenAI converters emitted redundant `content_block_end` events when no content block was open, inflating the output stream
    - Google GenAI compound chunks (text + finish in the same SSE frame) triggered duplicate text and finish events. Deferred text/finish payloads via `StreamContext.pending_text` / `pending_finish` so they merge into a single event
    - Tool call events generated spurious `content_block_start` / `content_block_end` wrappers in non-Anthropic targets. Suppressed via `_started` lifecycle guard

### Refactored

- **Unified `stream_response_to_provider` dispatch** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Extracted identical dispatch logic (10-entry `_TO_P_DISPATCH` table + dispatch skeleton) from all 4 provider converters into `BaseConverter`. Each converter now only implements a provider-specific `_post_process_to_provider` hook (OpenAI Chat injects envelope fields; OpenAI Responses injects `sequence_number`). Net reduction: ~27 lines
- **`StreamContext` buffer convenience methods**: Added `buffer_usage()` / `pop_pending_usage()` / `buffer_finish()` / `pop_pending_finish()` to replace manual set-and-clear patterns across all converters

### Changed

- **Pinned dev tooling versions**: `ty>=0.0.31` and `ruff>=0.15.0` now declared in `pyproject.toml` dev dependencies. CI no longer installs them separately — uses versions from `pip install -e ".[all]"`
- **Converter tests added to CI**: `tests/converters/` (1086+ tests) now runs in GitHub Actions alongside `tests/test_types/`
- **Roundtrip inflation regression test**: New pytest-parametrized test suite (`tests/converters/test_roundtrip_inflation.py`, 15 cases) verifies `len(output_events) <= len(input_events)` for all 4 providers across text, reasoning, tool call, and compound scenarios

## v0.5.1 — 2026-04-15

### Added

- **`tool_ops` convenience API** ([#148](https://github.com/Oaklight/llm-rosetta/issues/148)): New top-level `llm_rosetta.tool_ops` module for standalone tool definition conversion without instantiating full converter pipelines. Provides `to_provider()` / `from_provider()` unified dispatch and per-provider shortcuts (`to_openai_chat()`, `to_anthropic()`, etc.). All imports are lazy
- **Multi-key API management**: Admin panel now supports multiple API keys per gateway with per-key labels, create/reveal/delete operations, and usage tracking in request logs
- **Gateway API key authentication**: Configurable API key (`server.api_key`) protects AI request endpoints (`/v1/*`). Supports format-native credential extraction — OpenAI `Authorization: Bearer`, Anthropic `x-api-key`, Google `x-goog-api-key` / `?key=` query param. When no key is configured, all requests pass through (backward compatible)
- **Provider enable/disable**: Each provider now supports an `enabled` field (default `true`). Disabled providers and their models are silently excluded from routing
- **Docker support**: Official `Dockerfile`, `docker-compose.yml`, and Makefile targets (`build-docker`, `push-docker`, `run-docker`) for containerized deployment. Alpine-based image with non-root user, config volume mount, and PUID/PGID support
- **Admin panel enhancements**:
    - Provider toggle switches (enable/disable without deleting)
    - Model search and column sorting
    - Provider rename with automatic model reference updates
    - Network diagnostics button (connectivity check + proxy test)
    - Model testing with collapsible raw request/response details and image preview for vision tests
    - Embedded test image (base64 data URI) to avoid external network downloads
    - `reasoning_effort: 'low'` for reasoning model tests to limit token budget

### Changed

- **Admin panel authentication removed from gateway**: Admin panel endpoints (`/admin/*`) no longer require the gateway API key. Admin access control is delegated to the reverse proxy (e.g. Caddy, Nginx). The gateway API key now only authenticates AI request endpoints (`/v1/*`)
- **C901 cyclomatic complexity enforced at threshold 15**: Progressive reduction from 25 → 20 → 15 across all converters and gateway modules. Extracted cross-provider consistency helpers (`_build_ir_usage`, `_build_provider_usage`, `_convert_tools_from_p`, `_apply_tool_config`) with identical names across all 4 converters
- **`BaseConverter` abstract methods**: Four new abstract methods formalize the cross-provider helper pattern. Preserve-mode hooks documented as convention for providers supporting lossless round-trip
- **Vendored `validate.py` updated to zerodep v0.4.2**: Internal refactor of monolithic `_validate()` into focused helpers; no functional changes

### Fixed

- **User-Agent header for image URL downloads**: Google GenAI content converter now sends `User-Agent: llm-rosetta/1.0 (image fetch)` when downloading image URLs for inline base64 conversion, preventing 403 Forbidden from servers like Wikimedia
- **Image URL download with proxy support**: Image downloads in the Google GenAI converter now respect `HTTPS_PROXY` / `HTTP_PROXY` environment variables
- **Empty content fallback for reasoning models**: Admin panel test results now correctly handle `content: ""` (from reasoning models where all `max_tokens` are consumed by reasoning tokens) instead of showing raw JSON
- **Config file not found error**: Gateway now shows a friendly error message when the config file doesn't exist, instead of a Python traceback
- **ty type checker compatibility**: Added `ty: ignore` annotations for TypedDict vs `dict[str, Any]` mismatches and `FinishReason` Literal type narrowing
- **Google converter crash when thinking consumes all tokens** ([#152](https://github.com/Oaklight/llm-rosetta/issues/152)): Gemini 2.5 Pro with small `max_tokens` could have all tokens consumed by thinking, producing a response with no content parts. The converter now falls back to an empty assistant message instead of failing IR validation

## v0.5.0 — 2026-04-12

### Added

- **Gateway Admin Panel**: Built-in web admin panel at `/admin/` for managing gateway configuration, monitoring traffic, and inspecting request logs without editing config files or restarting the server
    - **Configuration tab**: Visual management of providers (add, edit, rename, delete) and model routing with capabilities (text/vision/tools)
    - **Dashboard tab**: Real-time metrics with summary cards (total requests, error rate, active streams, uptime), rolling 60-second throughput and latency charts, per-provider breakdown
    - **Request Log tab**: Filterable request log with model, provider, and status filters, paginated view with color-coded status codes
    - **8 themes**: Light, Indigo Dark, Dracula, Nord, Solarized, Osaka Jade, One Dark, Rosé Pine — persisted in localStorage
    - **i18n**: English and Chinese language support with localStorage persistence
- **File-based persistence**: Metrics counters (JSON) and request log (JSONL) are automatically saved to disk alongside the config file. Data survives server restarts. Log rotation with gzip compression (2 MB limit, 3 backups)
- **Provider rename**: Renaming a provider automatically updates all model routing references
- **API key security**: Masked keys on provider cards, reveal-on-demand with visibility toggle and copy button in edit modal. Masked values are never written back to config

### Changed

- **Provider names decoupled from API standard types**: Provider names are now user-defined strings (e.g. `"my-openai"`, `"OpenRouter_anthropic"`) instead of being constrained to the 4 standard type identifiers. A separate `type` field specifies the API standard (`openai_chat`, `openai_responses`, `anthropic`, `google`)
- Extracted `write_config()` to `config.py` for shared use by CLI and admin panel

## v0.4.2 — 2026-04-11

### Changed

- **`ReasoningConfig.enabled` replaced with `mode` field**: The boolean `enabled` field has been replaced by `mode: Literal["auto", "enabled", "disabled"]`. This aligns the IR more closely with provider semantics (Anthropic's three-way `thinking.type`, OpenAI Responses' `reasoning.type`). Omitting `mode` retains the previous "provider default" behavior. The `effort` field now lives directly in `ReasoningConfig` rather than being nested

### Fixed

- **Responses API `developer` role mapping**: The OpenAI Responses API uses `role: "developer"` (equivalent to Chat's `"system"`). Previously this role was passed through to IR unchanged, causing validation failures. Now correctly mapped to IR `"system"` during Provider→IR conversion
- **Google GenAI `additionalProperties` rejection**: Google's function_declarations API rejects the `additionalProperties` JSON Schema keyword. Added `extra_strip_keys` parameter to `sanitize_schema()` so providers can strip provider-specific unsupported keywords. Google tool_ops now strips `additionalProperties` recursively from nested schemas
- **Google GenAI `prompt_tokens_details` format mismatch**: Google returns modality token details as `list[ModalityTokenCount]` (e.g. `[{"modality": "TEXT", "token_count": 42}]`) but IR expects `dict[str, int]` (e.g. `{"text_tokens": 42}`). Added bidirectional conversion helpers `_modality_list_to_dict()` and `_dict_to_modality_list()`. Handles both SDK (`token_count`) and REST API (`tokenCount`) field names
- **Cross-format tool call ID prefix mapping**: The Responses API enforces `fc_` prefix on tool call IDs, but Chat uses `call_` and Anthropic uses `toolu_`. Added automatic prefix mapping during Responses conversion to prevent validation failures in cross-format scenarios
- **Adaptive thinking fallback**: When converting IR reasoning config to Anthropic format, `mode: "enabled"` without `budget_tokens` now correctly falls back to `{"type": "adaptive"}` with a warning, instead of producing an invalid `{"type": "enabled"}` without the required `budget_tokens`

## v0.4.1 — 2026-04-10

### Added

- **`force_conversion` parameter for `convert()`**: New `force_conversion: bool = False` keyword-only parameter. When `True`, the full source→IR→target pipeline runs even when source and target providers match, ensuring parameter normalization (e.g. `max_tokens` → `max_completion_tokens` for OpenAI Chat). Default `False` preserves existing passthrough behavior

### Fixed

- **Vendored `validate.py` updated from zerodep v0.4.1**: Applied pyupgrade fixes — `Callable` imported from `collections.abc` instead of `typing` (UP035), `@functools.cache` replaces `@functools.lru_cache(maxsize=None)` (UP033)
- Removed unused `sys` import in benchmark script
- Applied `ruff format` to benchmark scripts

### Changed

- Removed incorrect "Related Projects" section from README — LLM-Rosetta is an independent project, not part of the ToolRegistry ecosystem

## v0.4.0 — 2026-04-09

### Added

- **Metadata preservation for lossless A→IR→A round-trip** (Issue [#60](https://github.com/Oaklight/llm-rosetta/issues/60), PR [#119](https://github.com/Oaklight/llm-rosetta/pull/119)): New `MetadataMode` (`"strip"` / `"preserve"`) option in `ConversionContext` that captures provider-specific fields during `from_provider` and re-injects them during `to_provider`, enabling lossless round-trip conversion. Helper methods on `ConversionContext`: `store_request_echo()`, `store_response_extras()`, `store_output_items_meta()`, `get_echo_fields()`, `get_output_items_meta()`. Per-provider coverage:
    - **OpenAI Responses**: captures/restores 28+ echo fields (temperature, tools, reasoning, truncation, etc.), per-output-item metadata (id, status, annotations, logprobs), `RESPONSES_REQUIRED_DEFAULTS` dict for spec-required fields with sensible defaults, `sequence_number` on all SSE events
    - **Anthropic**: preserves `stop_sequence`, `container`, citations, and OpenRouter extension usage fields
    - **OpenAI Chat**: now re-emits `refusal` and `annotations` fields in `response_to_provider` (previously dropped)
    - **Google GenAI**: preserves `promptTokensDetails` and `cachedContentTokenCount` in usage metadata
    - **Gateway**: automatically enables preserve mode for both streaming and non-streaming paths; bridges metadata between `from_ctx` and `to_ctx` during streaming

### Fixed

- **Open Responses spec compliance for streaming and non-streaming**: Added required fields to all SSE events (`item_id`, `logprobs`, `annotations`, `status`, `sequence_number`, `output_index`, `content_index`), usage detail breakdowns (`output_tokens_details`, `input_tokens_details`), message item IDs and status for non-streaming output items, `function_call` status field in tool_ops, `service_tier` default to `"default"` (string, not null per spec), `completed_at` in required defaults, `created_at` fallback to current time when not provided, normalized echoed tools with `strict: null`, and metadata bridging from `from_ctx` to `to_ctx` in gateway streaming. All 6 Open Responses compliance tests now pass (schema + semantic)

## v0.3.1 — 2026-04-07

### Fixed

- **`service_tier: None` and `system_fingerprint: None` causing validation errors** (PR [#118](https://github.com/Oaklight/llm-rosetta/pull/118)): OpenAI upstream returns these fields as `null`, but the existence check (`if "key" in dict`) passed and assigned `None` to IR's `NotRequired[str]` field. Changed to value-not-None check in both OpenAI Chat and OpenAI Responses converters. Discovered via [Oaklight/argo-proxy#99](https://github.com/Oaklight/argo-proxy/issues/99)
- **Base `StreamContext` missing provider-specific attributes in Responses streaming** (PR [#118](https://github.com/Oaklight/llm-rosetta/pull/118)): When a gateway passes a base `StreamContext` to `OpenAIResponsesConverter.stream_response_to_provider()`, the method accesses `accumulated_text`, `output_item_emitted`, etc. that only exist on `OpenAIResponsesStreamContext`. Added auto-upgrade via `from_base()` classmethod with metadata caching to preserve state across calls

## v0.3.0 — 2026-04-07

### Added

- **Multimodal tool result support across all 4 converters** (Issue [#92](https://github.com/Oaklight/llm-rosetta/issues/92), PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Tools can now return multimodal content (text + images + files) as `ToolResultPart.result`. Three providers (Anthropic, OpenAI Responses, Google GenAI) support this natively; content blocks are converted through each provider's `content_ops` layer. See provider support matrix below
- **Lossless multimodal tool result roundtrip for OpenAI Chat** (Issue [#92](https://github.com/Oaklight/llm-rosetta/issues/92), PR [#108](https://github.com/Oaklight/llm-rosetta/pull/108)): OpenAI Chat Completions only accepts `content: string` for tool messages. Implements a dual encoding strategy — tool message keeps `json.dumps(result)` as data fallback, plus a synthetic user message carries visual content (`image_url` parts) wrapped in `<tool-content call-id="...">` XML tags. Unpacking recovers multimodal structure from the synthetic message (preferred) or falls back to JSON parsing if the synthetic message was trimmed by agent frameworks
- **`extract_all_text()` helper function** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Extracts text from both `TextPart` and `ReasoningPart` content — useful for thinking models (e.g. gemini-2.5-flash) that may place answers in reasoning parts rather than text parts
- **`generate_chart` example tool** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): New multimodal tool in `examples/tools.py` returning `[TextPart, ImagePart]` with inline base64 PNG, plus `multimodal_tools_spec` combining all 3 example tools
- **Multimodal integration tests across all 4 provider SDKs** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Two new test scenarios per provider — (A) tool returning multimodal content (text + image), (B) image input combined with tool calls. All 30 tests pass against official APIs: OpenAI Chat 9/9, OpenAI Responses 6/6, Anthropic 8/8, Google GenAI 7/7
- **Runtime IR validation via vendored zero-dependency validator** (Issue [#91](https://github.com/Oaklight/llm-rosetta/issues/91)): `validate_ir_request()`, `validate_ir_response()`, and `validate_ir_messages()` utilities validate IR structures against their TypedDict definitions at runtime. All 4 converters now validate output in `request_from_provider()` and `response_from_provider()`. Replaces manual `BaseMessageOps.validate_messages`. Includes Python <3.11 compatibility for `typing_extensions.TypedDict`
- **Constants validation tests**: 39 new tests across 4 `test_constants.py` files verifying that all reason mapping values are valid IR finish reasons, mapping coverage is complete, event type constants are well-formed, and ID generation produces correct formats
- **Finish reason mapping test coverage**: 38 tests validating reason mapping correctness as a safety net for the constants refactoring
- **`ConversionContext` base class for conversion pipelines** (Issue [#106](https://github.com/Oaklight/llm-rosetta/issues/106), PR [#111](https://github.com/Oaklight/llm-rosetta/pull/111)): New `ConversionContext` dataclass with `warnings: list[str]`, `options: dict[str, Any]`, and `metadata: dict[str, Any]` — a structured context container for non-streaming conversions. New `BaseConverter.create_conversion_context(**options)` factory method mirrors the existing `create_stream_context()`. All 6 non-streaming `BaseConverter` methods now accept an optional `context: ConversionContext` keyword parameter; converter implementations sync warnings to `context.warnings`. Gateway proxy creates a shared context per request and passes it through the full source→IR→target→response pipeline

### Fixed

- **Contextual error messages for tool conversion failures** (Issue [#85](https://github.com/Oaklight/llm-rosetta/issues/85), PR [#110](https://github.com/Oaklight/llm-rosetta/pull/110)): When `p_tool_definition_to_ir()` fails on a malformed or unsupported tool definition, the `ValueError` now includes `type=` and `name=` context so users can identify which tool caused the issue. Applied to all 4 converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI) with unit tests
- **OpenAI Responses `tool_choice` format** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Was using Chat Completions format (`{"type": "function", "function": {"name": "..."}}`); now uses Responses format (`{"type": "function", "name": "..."}`)
- **OpenAI Responses tool call ID round-trip** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Responses API uses `fc_` prefix IDs while IR uses `call_` prefix. The Responses `id` is now preserved in `provider_metadata` separately from `call_id`, enabling lossless round-trip conversion
- **OpenAI Responses reasoning item round-trip** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Reasoning models (e.g. gpt-5-nano) emit reasoning items with `id` (rs_ prefix), structured `summary` arrays, and `encrypted_content`. These are now preserved through `provider_metadata` for lossless round-trip — fixes 400 errors when reasoning items were sent back without their original `id`
- **IR validation accepts `None` for optional response fields** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): `logprobs` and `system_fingerprint` in `IRResponse` now accept `None` values (previously only accepted missing keys)
- **OpenAI Responses `content_filter` finish reason mapped to wrong status** (Issue [#90](https://github.com/Oaklight/llm-rosetta/issues/90)): `content_filter` was incorrectly mapped to `"completed"` status in `response_to_provider` and `stream_response_to_provider`. Now correctly maps to `"incomplete"` status with `incomplete_details.reason = "content_filter"`
- **Anthropic streaming missing `refusal` reason mapping**: The streaming `reason_map` was missing the `refusal` entry present in the non-streaming path, causing Anthropic refusal stop reasons to be silently dropped during streaming. Fixed as a side effect of the constants extraction (Issue [#64](https://github.com/Oaklight/llm-rosetta/issues/64)) — both paths now share the same `ANTHROPIC_REASON_FROM_PROVIDER` dict

### Changed

- **`ReasoningConfig.effort` expanded to 5-level enum** (Issue [#100](https://github.com/Oaklight/llm-rosetta/issues/100)): Effort levels now include `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"`. Provider-specific mappings: Anthropic maps to `thinking.type="adaptive"` with `thinking.effort`; OpenAI Chat/Responses clamp `"minimal"`→`"low"` and `"max"`→`"high"` (with warnings); Google GenAI maps to `thinking_config.thinking_level`
- **`ReasoningConfig.type` replaced with `ReasoningConfig.enabled`** (Issue [#70](https://github.com/Oaklight/llm-rosetta/issues/70)): The `type: Literal["enabled", "disabled"]` field is replaced with `enabled: bool` to avoid shadowing the Python built-in `type` and provide a more natural API
- **Merged duplicate IR concepts** (Issue [#69](https://github.com/Oaklight/llm-rosetta/issues/69)): Removed `candidate_count` from `GenerationConfig` — use `n` instead (Google GenAI converter maps `n` ↔ `candidate_count` internally). Unified `system_instruction` type from `str | list[dict]` to `str`
- **Normalized `ImagePart`, `FilePart`, `AudioPart` to canonical forms** (Issue [#68](https://github.com/Oaklight/llm-rosetta/issues/68)): Each part now has exactly two canonical forms — URL reference + structured inline data (e.g. `image_data`) — plus a unified `provider_ref: dict[str, Any]` for provider-specific references. Removed redundant top-level `data`/`media_type` fields and replaced `file_id`/`audio_id` with `provider_ref`
- **IR type fields changed from `Iterable` to `list`; function parameters to `Sequence`** (Issue [#67](https://github.com/Oaklight/llm-rosetta/issues/67)): TypedDict fields now use `list` for indexable, serialization-friendly semantics; function parameters use `Sequence` (covariant, read-only). Also fixes a latent generator-consumption bug in `strip_orphaned_tool_config`
- **`StreamContext` now inherits from `ConversionContext`** (Issue [#106](https://github.com/Oaklight/llm-rosetta/issues/106), PR [#111](https://github.com/Oaklight/llm-rosetta/pull/111)): `StreamContext` is a subclass of `ConversionContext` (IS-A relationship), unifying the context model for streaming and non-streaming paths. File renamed: `base/stream_context.py` → `base/context.py`
- **`StreamContext` converted to dataclass with provider subclass** (Issue [#65](https://github.com/Oaklight/llm-rosetta/issues/65)): `StreamContext` is now a `@dataclass` with typed fields (eliminates defensive `getattr`/`hasattr` patterns). OpenAI Responses-specific state extracted into `OpenAIResponsesStreamContext` subclass. New `BaseConverter.create_stream_context()` factory method

### Refactored

- **Warnings single-source convergence** (Issue [#113](https://github.com/Oaklight/llm-rosetta/issues/113), PR [#115](https://github.com/Oaklight/llm-rosetta/pull/115)): All 4 converter `request_to_provider` methods now use `ConversionContext` as the single accumulation point for warnings. Eliminates the dual-write pattern where warnings were written to both a local list and `context.warnings`. The returned warnings list IS the same object as `context.warnings` — no duplication possible
- **`ProviderMetadataStore` replaces global metadata cache** (Issue [#112](https://github.com/Oaklight/llm-rosetta/issues/112), PR [#117](https://github.com/Oaklight/llm-rosetta/pull/117)): The module-level `_provider_metadata_cache` dict in `proxy.py` is replaced with `ProviderMetadataStore` — a class with TTL-based expiration (30 min), max-size eviction (10k entries), and explicit lifecycle management. The store is created per-app in `create_app()` and passed via `app.state`, eliminating implicit global mutation. `close_clients()` renamed to `close_resources()` to also clear the store on shutdown
- **Shrink public API export surface** (Issue [#114](https://github.com/Oaklight/llm-rosetta/issues/114), PR [#116](https://github.com/Oaklight/llm-rosetta/pull/116)): Reduced `__all__` exports across converter packages to only the primary converter class, removing internal implementation details (`*MessageOps`, `*ContentOps`, `*ConfigOps`, `*ToolOps`, `*Constants`) from the public API. Internal modules remain importable for advanced use but are no longer promoted as public surface
- **Extracted stream event handlers from monolithic methods** (Issue [#63](https://github.com/Oaklight/llm-rosetta/issues/63)): Replaced 8 monolithic `if`/`elif` stream methods (~1,781 lines) across all 4 converters with individual handler methods dispatched via class-level handler tables. Public API unchanged
- **Extracted shared utility functions in OpenAI Responses converter** (Issue [#66](https://github.com/Oaklight/llm-rosetta/issues/66)): `resolve_call_id()` and `build_message_preamble_events()` extracted from `converter.py` into `utils.py` with dedicated unit tests
- **Extracted per-provider constants for reason mappings and magic values** (Issue [#64](https://github.com/Oaklight/llm-rosetta/issues/64)): Inline reason mapping dicts, SSE event type string literals, status-to-reason conditional logic, and ID generation patterns across all 4 converters are now centralized in per-provider `_constants.py` modules. Includes `AnthropicEventType` and `ResponsesEventType` classes, `REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` dicts, and `generate_tool_call_id()` / `generate_message_id()` helpers

## v0.2.6 — 2026-03-29

### Fixed

- **Chat Completions tool message ordering after Responses API conversion** *([@caidao22](https://github.com/caidao22))*: Codex CLI interleaves `function_call_output` with other items (e.g. user warnings) in Responses API format — valid there since items match by `call_id`. But after IR → Chat Completions conversion, the interleaved messages break the OpenAI Chat API constraint that `role: "tool"` messages must immediately follow their `assistant` `tool_calls`, causing upstream 400 errors. Added `_reorder_tool_messages()` post-processing in `OpenAIChatMessageOps.ir_messages_to_p()` that groups tool responses back to their corresponding assistant messages
- **Orphaned `tool_choice`/`tool_config` stripped when no tools defined** *([@caidao22](https://github.com/caidao22))*: Codex context compaction can drop all tool definitions while keeping `tool_choice` (e.g. `"auto"`), causing upstream APIs to reject with *"tool_choice is set but no tools are provided"*. Added `strip_orphaned_tool_config()` in all four converters — part of the same Codex compaction fix family as `fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing) and `_reorder_tool_messages` (tool message ordering). Also extended `fix_orphaned_tool_calls_ir` to Google GenAI converter for completeness (Issue [#87](https://github.com/Oaklight/llm-rosetta/issues/87))
- **Stream event ordering**: `UsageEvent` is now emitted before `FinishEvent` in all four provider converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI). Previously `FinishEvent` was processed first, causing `response.completed` to carry `output_tokens=0` — downstream consumers (e.g. Codex token tracking) saw stale usage data. For cross-chunk scenarios (OpenAI Chat sends `finish_reason` and `usage` in separate chunks), `FinishEvent` now defers `response.completed` to `StreamEndEvent` which merges any pending usage
- **Parallel tool calls merged into one in Anthropic/Google → Chat streaming**: Anthropic and Google GenAI `stream_response_from_provider` emitted `ToolCallStartEvent` and `ToolCallDeltaEvent` without `tool_call_index`. When routing to Chat Completions, all parallel tool calls defaulted to index 0, causing the client SDK to merge them into a single call. Anthropic now derives `tool_call_index` from `context._tool_call_order` position; Google computes it from registration order in context (#88, #89)
- **Missing `id` field on Responses `function_call` output**: Non-streaming `response_to_provider` was missing the `id` field on `function_call` output items. Streaming used a synthetic `fc_` prefix that could leak into IR via `p_tool_call_to_ir` fallback path. Unified both paths to use `call_id` directly as `id` (no prefix)
- **Responses streaming `item_id` and empty `tool_call_id` resolution** *([@caidao22](https://github.com/caidao22))*: Added `item_id` tracking to `StreamContext` (`tool_call_item_id_map`, bidirectional mapping). Responses `stream_response_to_provider` now emits `item.id` on `output_item.added` and `item_id` (not `call_id`) on `function_call_arguments.delta/done` events. Defense-in-depth: resolves empty `tool_call_id` by `tool_call_index` via context (Issue [#86](https://github.com/Oaklight/llm-rosetta/issues/86))
- **Non-function tool names mangled with type prefix** *([@caidao22](https://github.com/caidao22))*: Non-function IR tool definitions (e.g. `type="custom"`, `name="apply_patch"`) were converted with a type prefix (`custom_apply_patch`), breaking tool_call matching since the client expects the original name. Both OpenAI Chat and Responses converters now use `ir_tool["name"]` directly (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))

## v0.2.5 — 2026-03-23

### Fixed

- **Anthropic `input_schema` missing `type` for parameterless tools**: MCP tools with no parameters produce `input_schema: {}`, but Anthropic requires `"type"` to be present. Now defaults to `{"type": "object"}` when the schema dict lacks a `type` field — fixes `tools.0.custom.input_schema.type: Field required` errors when routing Google GenAI or OpenAI Responses tool calls to Anthropic upstream
- **Google GenAI camelCase field handling across the full converter stack**: Gemini CLI and the Google REST API use camelCase (`inlineData`, `fileData`, `mimeType`, `fileUri`, `functionCall`, `functionResponse`, `finishReason`, `usageMetadata`, `responseMimeType`, `responseSchema`, `thinkingConfig`, `maxOutputTokens`, `stopSequences`, etc.), but the converter only accepted snake_case. All P→IR methods in content_ops, config_ops, tool_ops, message_ops, and converter now accept both conventions; all IR→P methods now output camelCase for REST API compatibility
- **Image/audio/file data lost during Google→IR conversion**: `p_part_to_ir` checked for `inline_data` (snake_case) but Gemini CLI sends `inlineData` (camelCase) — binary content was silently dropped with a `不支持的Part类型` warning. Fixed by normalizing camelCase keys at the dispatch entry point
- **Cross-format image conversion failure (Google → OpenAI/Anthropic)**: Google's `p_image_to_ir` produces `ImagePart` with top-level `data` + `media_type` fields, but OpenAI Chat, Anthropic, and OpenAI Responses `ir_image_to_p` only checked `image_url` and nested `image_data` — threw `ValueError`. All three target converters now handle top-level fields as a fallback path (Issue [#68](https://github.com/Oaklight/llm-rosetta/issues/68))
- **Google GenAI tool_call_id reconciliation**: Google `functionCall` has no ID field, so UUIDs are generated during P→IR. But Gemini CLI assigns its own IDs to `functionResponse` (format: `name_timestamp_index`), creating a mismatch. New `_reconcile_tool_call_ids` method matches tool results to tool calls by function name, fixing orphaned tool_call errors
- **tool_call_id exceeds OpenAI 40-character limit**: Generated IDs used `call_{name}_{8hex}` format — MCP tool names like `mcp_toolregistry-hub-server_datetime-now` produced 54-char IDs. Shortened to `call_{24hex}` (fixed 29 chars)
- **Google→IR role mapping for tool results**: `functionResponse` parts produced `role: "user"` IR messages, so `fix_orphaned_tool_calls_ir` (which checks `role: "tool"`) couldn't detect them. Now separates `functionResponse` into `role: "tool"` messages with explicit `"tool": "user"` in `_IR_TO_GOOGLE_ROLE`
- **Mixed content message ordering**: When a Google message contains both `functionResponse` and `inlineData`, the content parts were emitted before tool results, breaking OpenAI's required `assistant(tool_calls) → tool(response)` ordering. Tool results now precede content parts in the split
- **Google built-in tools (googleSearch, codeExecution)**: `p_tool_definition_to_ir` now returns `None` for tool entries without a `name` field; converter skips them instead of producing empty `function.name` errors
- **Gateway: Starlette `on_shutdown` deprecation**: Replaced deprecated `on_shutdown` parameter with `lifespan` async context manager — fixes compatibility with Starlette 0.38+ which removed `on_shutdown`/`on_startup`

### Added

- **StreamContext**: `get_tool_call_args()` and `get_pending_tool_calls()` methods for querying accumulated tool call state during streaming

### Changed

- **`BaseToolOps.p_tool_definition_to_ir` return type**: Now `ToolDefinition | list[ToolDefinition] | None` to support unconvertible tool entries

### Added (Documentation)

- **Provider & CLI Compatibility Matrix**: New guide page documenting real-world issues found during live integration testing with Gemini CLI, Claude Code, and OpenCode through format-converting proxies

## v0.2.4 — 2026-03-22

### Added

- **`fix_orphaned_tool_calls()` utilities**: Public functions in `converters/openai_chat/tool_ops.py`, `converters/openai_responses/tool_ops.py`, and `converters/anthropic/tool_ops.py` that detect mismatched tool calls/results and fix them bidirectionally — injecting synthetic placeholder results for orphaned calls **and** removing orphaned results without matching calls. OpenAI (Chat & Responses) and Anthropic strictly require this pairing (return 400 otherwise); only Google Gemini is lenient. Automatically applied at the IR level during `request_to_provider()` for all strict-pairing converters; emits `WARNING`-level log when orphaned tool calls or results are detected (#82, #84)

### Fixed

- **Anthropic→IR role normalization for `tool_result` messages**: Anthropic places `tool_result` blocks in `role: "user"` messages, but IR uses `role: "tool"` (like OpenAI). The Anthropic converter now normalizes pure `tool_result` user messages to `role: "tool"`, and splits mixed `tool_result` + text messages into separate `role: "tool"` and `role: "user"` IR messages. This fixes `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls in cross-format conversions (e.g. Anthropic → OpenAI Chat) (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))
- **OpenAI Responses→IR role normalization for `function_call_output` items**: `function_call_output` and `mcp_call_output` items were grouped into `role: "user"` IR messages, but IR uses `role: "tool"` for tool results. The Responses converter now groups these items into `role: "tool"` messages, fixing `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls when converting Responses → other formats (e.g. Responses → OpenAI Chat) (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))

### Added (Documentation)

- **Provider Dialect Differences guide**: New section in the Converters guide (EN + ZH) documenting tool schema sanitization, orphaned tool call handling, and Google camelCase/snake_case differences

## v0.2.3 — 2026-03-22

### Fixed

- **Tool schema sanitization applied to all converters**: `_sanitize_schema()` was previously only called in the OpenAI Chat converter. Google GenAI, OpenAI Responses, and Anthropic converters now also sanitize tool parameter schemas before sending to upstream, preventing rejections from strict endpoints like Vertex AI (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **Non-standard `ref` and `$schema` keywords stripped**: OpenCode's built-in tools use a bare `ref` field (without `$` prefix) and `$schema` at the top level, both rejected by Vertex AI. Added to the unsupported keywords blocklist (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **`$ref`/`$defs` resolved by inlining**: JSON Schema `$ref` references are now resolved by inlining the referenced definition from `$defs`/`definitions`, and both keys are removed from the output. Supports nested and chained references (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **Streaming tool call arguments not accumulated**: OpenAI Chat, Anthropic, and Google GenAI converters registered tool calls in `StreamContext` but never called `append_tool_call_args()` to accumulate argument deltas during streaming. This caused tool call arguments to arrive empty at upstream (e.g., MCP tools returning `'query' is a required property`). Only the OpenAI Responses converter was correct (Issue [#81](https://github.com/Oaklight/llm-rosetta/issues/81))
- **OpenAI Chat streaming tool call ID resolution**: Delta-only chunks (carrying `index` but no `id`) produced an empty-string `tool_call_id`. Now resolves the effective ID from `StreamContext._tool_call_order` using the chunk index (Issue [#81](https://github.com/Oaklight/llm-rosetta/issues/81))

### Changed

- **`sanitize_schema` extracted to `converters/base/tools.py`**: The schema sanitization utility (previously `_sanitize_schema` private to `openai_chat/tool_ops.py`) is now a public shared function in `converters/base/tools.py`, exported via `converters.base`. All 4 converter `tool_ops.py` files import from the shared location instead of cross-importing from `openai_chat` (Issue [#66](https://github.com/Oaklight/llm-rosetta/issues/66))

## v0.2.2 — 2026-03-22

### Fixed

- **Missing `content_block_stop` in Anthropic SSE output**: When converting OpenAI Chat streaming responses to Anthropic SSE format, `content_block_stop` events were not emitted before `message_delta`, causing Claude Code to silently discard response content. The Anthropic converter now emits `content_block_stop` for any open content block when processing a `FinishEvent` (Issue [#77](https://github.com/Oaklight/llm-rosetta/issues/77))
- **Upstream preflight chunk misinterpreted as stream end**: Argo API sends a preflight chunk with `choices: []` and empty `id`/`model` before actual content. The OpenAI Chat converter now only treats empty-choices chunks as stream-end after the stream has actually started (`context.is_started` guard) (Issue [#77](https://github.com/Oaklight/llm-rosetta/issues/77))

## v0.2.1 — 2026-03-20

### Added

- **Gateway request/response body logging**: configurable debug logging with colorized output, body sanitization and truncation — enable via config (`"debug": {"verbose": true, "log_bodies": true}`), env vars (`LLM_ROSETTA_VERBOSE`, `LLM_ROSETTA_LOG_BODIES`), or `--verbose` CLI flag
- **Google `output_format="rest"` for `request_to_provider()`**: pass `output_format="rest"` to get a REST API–ready request body with `tools`/`tool_config` at top level and generation params wrapped in `generationConfig` — eliminates the need for manual SDK→REST fixups

### Changed

- **Gateway modularization**: split `app.py` (1057 lines) into `proxy.py` (proxy engine, SSE handling, upstream requests), `cli.py` (CLI entry point, argparse, subcommands), and a slimmed `app.py` (route handlers, app factory, ~210 lines)
- **Moved Google REST body fixup to core**: `_fixup_google_body()` logic moved from `gateway/proxy.py` into `GoogleGenAIConverter._to_rest_body()`, removing duplicated SDK→REST transforms from the gateway and all 6 REST examples

### Fixed

- OpenAI Responses streaming: added missing `id`/`object`/`model` fields to `response.completed`, `output_index`/`content_index` to text delta events, and proper lifecycle events (`output_item.added`, `content_part.added`, `content_part.done`, `output_item.done`) (Issue [#56](https://github.com/Oaklight/llm-rosetta/issues/56))
- OpenAI Chat streaming: `tool_calls` entries now always include the required `index` field, defaulting to `0` when not explicitly provided by the upstream IR event (Issue [#57](https://github.com/Oaklight/llm-rosetta/issues/57))
- OpenAI Chat streaming: usage-only chunk now includes `"choices": []` to satisfy clients that validate every `chat.completion.chunk` must contain a `choices` array (Issue [#55](https://github.com/Oaklight/llm-rosetta/issues/55))
- `stream_options` (Chat Completions-only field) no longer leaks into OpenAI Responses API requests — the Responses converter's `ir_stream_config_to_p()` was incorrectly emitting `stream_options`, causing upstream rejection when Chat-format clients (Kilo, OpenCode) were proxied to the Responses API (Issue [#58](https://github.com/Oaklight/llm-rosetta/issues/58))
- Google GenAI converter now handles tools and tool_config in REST-format requests (top-level fields) in addition to SDK format (`config.tools`) — previously only SDK format was recognized, silently stripping tool definitions from gateway-proxied requests (Issue [#59](https://github.com/Oaklight/llm-rosetta/issues/59))
- Google camelCase `functionDeclarations` not parsed: `p_tool_definition_to_ir()` now handles both `functionDeclarations` (camelCase/REST) and `function_declarations` (snake_case/SDK), and extracts all declarations instead of only the first. Also added camelCase support for `functionCallingConfig`/`allowedFunctionNames` and `toolConfig` in request parsing — fixes Gemini CLI tool calling through the gateway (Issue [#61](https://github.com/Oaklight/llm-rosetta/issues/61))
- Google streaming tool calls split into two chunks: `stream_response_to_provider()` now defers `tool_call_start` and emits the complete `function_call` (name + args) in a single chunk on `tool_call_delta`, matching the Google API's native format (Issue [#62](https://github.com/Oaklight/llm-rosetta/issues/62))

## v0.2.0 — 2026-03-18

### Added

- **Standalone API test scripts** (`llm_api_simple_tests/`): 20 test scripts (5 per provider) using official SDKs directly, covering simple query, multi-round chat, image, function calling, and comprehensive scenarios — added as a git submodule from [Oaklight/llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests)
- **LLM-Rosetta Gateway**: REST gateway application for cross-provider HTTP proxying
- CLI entry point (`llm-rosetta-gateway`) and package structure for the gateway
- Gateway config auto-discovery at `./config.jsonc`, `~/.config/llm-rosetta-gateway/config.jsonc`, `~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` flag to open config file in `$EDITOR` (falls back to nano/vi/vim)
- `--version` / `-V` flag showing current version
- ASCII art startup banner with `--no-banner` to suppress
- `add provider <name>` subcommand for adding provider entries to config (with `--api-key`, `--base-url` flags or interactive prompts; known providers auto-fill defaults)
- `add model <name>` subcommand for adding model routing entries (with `--provider` flag or interactive prompt)
- **Gateway providers module** (`providers.py`): centralized provider definitions with auth-header builders, URL templates, default base URLs, and API key env-var names
- **API key rotation**: round-robin `KeyRing` for comma-separated API keys per provider
- **Proxy support**: global `server.proxy` and per-provider `proxy` config for HTTP/SOCKS proxies; CLI `--proxy` flag overrides config
- Makefile `test-integration` target using `proxychains` (if available) for integration tests
- `init` subcommand to create a template `config.jsonc` at the XDG default location (`~/.config/llm-rosetta-gateway/`)
- **Model listing endpoints**: `GET /v1/models` (compatible with both OpenAI and Anthropic SDKs) and `GET /v1beta/models` (Google GenAI SDK format) — enables `client.models.list()` across all three SDKs (Issue [#54](https://github.com/Oaklight/llm-rosetta/issues/54))

### Changed

- Bumped minimum Python to 3.10+; migrated to stdlib `typing` (removed `typing_extensions`)
- Applied `ruff` formatter across the entire codebase
- Updated Makefile with `lint`, `test`, and `build` targets
- Added `ty` (type checker) configuration
- Configured `ruff` lint rules (`E`, `F`, `UP`) in `pyproject.toml`; ignore `UP007` (Union syntax) and `E501` (line length)
- Modernized typing imports across `src/`, `tests/`, `examples/`, and `scripts/` — replaced `typing.Dict`, `List`, `Tuple`, `Optional`, `Type` with stdlib builtins

### Fixed

- Streaming crash with Anthropic provider when usage tokens are `null` — `TypeError: NoneType + int` in all converters (replaced `.get("*_tokens", 0)` with `.get("*_tokens") or 0`)
- Gateway provider `base_url` validation — fail early with clear error on config typos like `https:example.com` (missing `//`)
- Added `socksio` to gateway dependencies for SOCKS proxy support (`httpx[socks]`)
- Added missing `__init__.py` for `types` package
- Updated `git clone` URL from `llmir` to `llm-rosetta` in documentation
- Resolved all `ty` type checker diagnostics in `src/` (31 → 0):
    - Fixed `is_part_type()` TypeGuard narrowing — replaced with specific type guard functions (`is_text_part`, etc.)
    - Added missing TypedDict fields: `provider_metadata` on `TextPart`/`ReasoningPart`, `file_id` on `ImagePart`/`FilePart`
    - Fixed `IRRequest.messages` type from `Required[Message]` to `Required[Iterable[Message]]`
    - Used `cast()` to bridge `dict[str, Any]` intermediates to TypedDict return types
    - Fixed dict literal type inference conflicts in converter response builders
- Resolved all `ty` type checker diagnostics in `tests/` (1506 → 0):
    - Added `cast()` wrappers on dict literals passed to functions expecting TypedDict parameters (`GenerationConfig`, `IRRequest`, `IRResponse`, `ToolDefinition`, `ToolChoice`, etc.)
    - Narrowed `Message | ExtensionItem` union results with `cast(list[Any], ...)` or `cast(Message, ...)`
    - Converted `Iterable` content fields to `list` for subscript and `len()` access
    - Added `assert ... is not None` guards before subscripting optional return types
    - Fixed `FinishReason` from bare string to TypedDict form `{"reason": "stop"}`
    - Fixed `IRResponse.object` literal from `"chat.completion"` to `"response"`
- Resolved all `ruff` lint violations in `src/` and `tests/` (UP035 deprecated imports, F401 unused imports)
- Google `thought_signature` preservation through gateway round-trips — newer Google models require `thoughtSignature` echoed back in function call parts; the gateway now caches `provider_metadata` (including `thought_signature`) keyed by `tool_call_id` and re-injects it on subsequent requests for both streaming and non-streaming modes (Issue [#51](https://github.com/Oaklight/llm-rosetta/issues/51))
- OpenAI Responses converter now handles all 3 `input` formats: bare string (`"input": "hello"`), shorthand list (`[{"role": "user", "content": "hi"}]`), and structured list — previously only the structured format was supported, causing the OpenAI Python SDK's shorthand items to be silently dropped and producing empty IR messages when cross-converting to Anthropic or Google providers

---

## 2026-03-15 — Rebrand to LLM-Rosetta

### Changed

- **Project renamed from LLMIR to LLM-Rosetta** across all code, docs, and configuration
- Python import package name set to `llm_rosetta` (underscore); `pyproject.toml` updated accordingly
- Documentation fully rewritten with Zensical for both English (`docs_en`) and Chinese (`docs_zh`)
- README (EN/ZH) updated with new branding, badges, and `pyproject.toml` metadata

---

## 2026-03-06 — Streaming & StreamContext

### Added

- **`StreamContext`** for stateful stream chunk processing across all 4 providers
- `stream_response_from_provider()` and `stream_response_to_provider()` methods on all converters
- `accumulate_stream_to_assistant_message()` helper function
- Stream abstract methods (`stream_response_to_provider`, `stream_response_from_provider`) added to `BaseConverter`
- 4 new IR stream event types: `StreamStart`, `StreamEnd`, `ContentBlockStart`, `ContentBlockEnd`
- `ReasoningDeltaEvent` and `tool_call_index` field on IR stream types
- Cross-provider streaming examples for all provider pairs (SDK and REST variants)
- Local file cache and retry logic for image downloads in examples

### Changed

- Stream method signatures updated with optional `context` parameter
- Deprecated `from_provider` methods removed; `auto_detect` updated to new API
- Obsolete single-provider example scripts removed (replaced by cross-provider examples)
- `_normalize()` extracted to `BaseConverter` as a shared utility

### Fixed

- camelCase fallback for Google GenAI REST stream/response fields
- Anthropic stream converter: `thinking_delta`, `signature_delta`, `tool_call_id` handling
- OpenAI Chat stream converter: `reasoning_content`, empty string, `tool_call_index` handling
- Missing `__init__.py` for test package discovery
- `from_provider` calls in `google_genai_rest_e2e` integration test

---

## 2026-02-14 — Cross-Provider Examples & Stream Converters

### Added

- **Stream converters** for all 4 providers: OpenAI Chat, Anthropic, Google GenAI, OpenAI Responses
- Stream converter unit tests for all providers
- **6 cross-provider conversation examples** (SDK-based): OpenAI Chat ↔ Anthropic, OpenAI Chat ↔ Google GenAI, OpenAI Chat ↔ OpenAI Responses, Anthropic ↔ Google GenAI, Anthropic ↔ OpenAI Responses, Google GenAI ↔ OpenAI Responses
- Common resources module for cross-provider conversation examples
- Image URL to inline base64 conversion helpers for Google GenAI compatibility
- OpenAI Responses E2E integration tests (REST + SDK)
- Unit tests for OpenAI Responses Ops classes and converter
- Examples README in English and Chinese

### Changed

- **OpenAI Responses converter** restructured to Bottom-Up Ops Pattern
- Post-refactor cleanup: removed deprecated utils and empty directories

### Fixed

- Image URLs converted to inline base64 for Google GenAI provider compatibility

---

## 2026-02-13 — Bottom-Up Ops Architecture

### Added

- **Google GenAI converter** rebuilt with Bottom-Up Ops Pattern
- TypedDict replicas of **OpenAI Responses API** types
- TypedDict replicas of **Google GenAI SDK** types
- Google GenAI REST and SDK E2E integration tests
- Unit tests for `google_genai` converter Ops classes
- Anthropic SDK and REST E2E integration tests
- OpenAI Chat E2E tests split into SDK and REST versions
- **GitHub Actions** CI/CD workflows and Dependabot configuration

### Changed

- **Anthropic converter** redesigned with bottom-up Ops architecture
- Imports updated to use new `google_genai` converter module
- Old `google/` converter and legacy tests removed

---

## 2026-02-12 — Converter Redesign

### Added

- TypedDict replicas of **Anthropic SDK** types
- TypedDict replicas of **OpenAI Chat** types with backward compatibility and tests
- Legacy body converter design preserved as historical reference

### Changed

- **OpenAI Chat converter** redesigned with bottom-up Ops architecture
- Ruff lint errors fixed across entire codebase

---

## 2026-01-06 — Layered Architecture & Documentation

### Added

- English and Chinese documentation structures initialized (`docs_en`, `docs_zh`)
- Comprehensive error handling documentation
- OpenAI Chat Converter integration tests
- Comprehensive mock implementations for `BaseConverter` test class
- File handling functionality in base converter
- Provider-to-IR mapping documentation

### Changed

- Converter base refined with layered abstract template
- All 4 converters restructured with layered architecture (Anthropic, OpenAI Chat, OpenAI Responses, Google GenAI)
- Type annotations updated for IR content/part conversion methods
- IR type system reorganized and enhanced
- English translations added to code comments and docstrings

### Fixed

- Reasoning content field assertion corrected
- File content handling in OpenAI Chat Completions converter

---

## 2026-01-05 — Auto-Detection & Package Maturity

### Added

- **`detect_provider()`** for automatic provider format auto-detection
- **`convert()`** convenience function for one-step format conversion
- `developer` role support in message validation
- Comprehensive validation tests for `BaseConverter`, Anthropic, Google GenAI, and OpenAI converters
- Tool call and tool definition conversion tests
- pytest configuration and `pytest-cov` dependency
- Competitive analysis document

### Changed

- **Package renamed** from `llm-provider-converter` to `llm-rosetta`
- IR format usage standardized across all providers
- Message creation standardized using `Message` class in examples
- Test suite migrated from unittest to pytest
- Common logic extracted into shared utility modules

### Fixed

- Standalone tool calls without current message context in OpenAI Responses converter
- Google GenAI Pydantic model handling reordered for tuple compatibility
- OpenAI content handling logic simplified for single text parts

---

## 2026-01-04 — Examples & Packaging

### Added

- `pyproject.toml` for package configuration
- Multi-turn chat example with tool integration
- Anthropic handover in multi-turn chat example
- Google GenAI function calling in multi-turn chat example

### Changed

- Utility functions moved from converters to IR types module
- OpenAI Chat converter code formatting improved
- Deprecated multi-provider query and weather tool modules removed

---

## 2025-12-24 — Initial Implementation

### Added

- **IR type system**: intermediate representation types for messages, content parts, tools, configs, request/response
- **`BaseConverter`** abstract class for LLM provider conversion
- **`AnthropicConverter`**: bidirectional Anthropic Messages API conversion
- **`OpenAIChatConverter`**: bidirectional OpenAI Chat Completions API conversion
- **`OpenAIResponsesConverter`**: bidirectional OpenAI Responses API conversion
- **`GoogleGenAIConverter`**: bidirectional Google GenAI SDK format conversion
- Comprehensive test suites for all 4 converters
- Package initialization and exports
- Weather tool example with mock data

---

## 2025-12-09 — Research & Design

### Added

- Initial project structure
- LLM provider message typing schemas documentation and comparison
- Provider messages IR design documentation
- MCP support comparison across providers (OpenAI, Anthropic, Google)
- Google GenAI Interactions API type analysis
- Multi-provider query example function
- OpenAI Responses API support in query examples

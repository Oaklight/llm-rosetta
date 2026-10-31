---
title: Admin Panel
---

# Admin Panel

The gateway ships a built-in web console for configuration, traffic and diagnostics — no config file to edit by hand, no restart to apply a change.

Open **`http://localhost:8765/admin/`**.

## The console

The console is a left rail of sections — **Dashboard**, **Providers**, **Models**, **API Keys** and **Observability** — with the mode, language, settings and logout buttons in one row at the bottom, and a status bar along the bottom edge that reports the request counters. Below 900 px the rail folds into a top bar.

`Alt+1` … `Alt+5` jump to a section, `[` and `]` step through them, Back and Forward move between the sections you visited, and `?` opens the shortcut list.

## Providers

The **Providers** section manages every upstream the gateway can call.

![Providers section](images/admin-providers.png)

The header filters by model type — **All**, **LLM**, **Embedding**, **Rerank**, **Decision** — and searches by name, shim or base URL; the count beside it is how many providers the filters leave. The grid/list toggle switches between cards and a compact table, and the choice is stored in `localStorage`. **+ Add Provider** opens the form; **+ Free resource** lists the free lanes the gateway knows about, and picking one is the same as adding that provider.

Each card shows the logo and name, the capability badges for the model types the provider serves, an enable switch and the number of models on it. A disabled provider greys its logo and dims its name; the card stays readable. The actions are **Edit**, **Test** and a **⋯** menu holding **Clone**, **Refresh** (when the shim declares a model-listing path), **Quota** (when the shim ships a probe) and **Delete**.

The card omits the type row and the base URL on purpose: one provider can serve several model types at once, which the badges say better than a single cell, and the URL is in the name's tooltip and in the form.

### Adding and editing

The provider form is a right-side drawer with a sticky title and action bar. It holds the name, the provider type or shim, the base URL, the API key, an optional proxy and the logo.

- **Rename** rewrites every model reference automatically.
- **Clone** pre-fills the drawer with the source provider's settings and the name `<name> Clone` (`<name> 克隆` in Chinese).
- **Delete** opens a type-to-confirm dialog that warns about the cascade: the provider's models go with it.

An API key field auto-detects comma-separated values and switches to a multi-entry list with one password row per key. **+ Add key** appends a row at any time; each row has an eye toggle and a copy button, and the footer offers view-all and copy-all for the whole comma-separated value. Masked keys on cards are never written back to the config file.

### Signing in to a shim-declared credential

A provider whose shim ships `auth.py` has no key to type in — the exchange mints the credential through a sign-in that needs a browser. Editing such a provider replaces the API-key box with a **Credential** section that reads `GET /admin/api/config/providers/<name>/credential`:

| State | What the panel shows |
|-------|----------------------|
| connected | the account the exchange's summary names, the access token's expiry, **Manage usage** (when the shim declares a page) and **Sign out** |
| needs setup | no credential yet — **Install a record** |
| unreadable | a record exists but cannot be read — install a replacement or sign out |

**Install a record** takes the JSON the CLI prints (`llm-rosetta-gateway credential <provider>`, see the [CLI reference](cli.md#credential-provider)) and hands it to the exchange over `PUT …/credential`; a record that does not verify is refused with the exchange's own message. **Sign out** calls `DELETE`, which revokes the grant upstream when the vendor allows it and takes the token out of the gateway's key ring immediately. The panel never renders a token: only the subject and expiry the exchange chooses to expose.

## Models

The **Models** section is the model routing table: every configured model, the capabilities it declares, and the provider that serves it.

![Models section](images/admin-models.png)

- **Filters** — provider, a status facet (**All statuses**, **Usable**, **Partial**, **Unusable**, each counting the rows the other filters leave), and model type.
- **Badges** are icon-only; the capability name is the tooltip. A model served by more than one provider says **Multiple providers**, and the cell opens a list with each provider's own switch, weight and **Test**.
- **Actions** — **Test** and **⋯** appear when a row is hovered; the enable switch never hides, because it is the row's state.
- **A row opens a side panel** with the capabilities, every provider with its weight, and **Edit** in the same panel — there is no separate model modal.
- **Ordering** — the header stays put while the table scrolls, and a model no provider can serve sorts below the usable rows.

Two actions are not in the table: **+ Add Model** in the header, and **Fetch from Provider**, which queries the upstream `/v1/models` endpoint, lists what it returns with checkboxes, and bulk-adds the selection with an optional name prefix. Its failure messages are diagnostic rather than generic:

- **Connection refused** — the upstream service is not running, or a firewall blocks the connection
- **Timeout** — the upstream host is not reachable from the gateway
- **DNS failure** — the Base URL hostname does not resolve

Renaming a model onto a name that already exists merges the providers instead of failing: the confirm dialog adds the new provider to `providers[]` rather than replacing the entry that already held the name.

### Model capabilities

Each model can declare capabilities that determine the available test types and are returned in the `/v1/models` response:

| Capability | Description | Badge Color |
|------------|-------------|-------------|
| `text` | Text generation | Blue |
| `vision` | Image comprehension | Green |
| `tools` | Function/tool calling | Orange |
| `embedding` | Text embeddings (mutually exclusive with vision/tools) | Teal |
| `reasoning` | Extended thinking / chain-of-thought (mutually exclusive with embedding) | Purple |

### Model testing

**Test** on a model row runs a quick test; the types on offer follow the declared capabilities:

| Test Type | Description | Requires |
|-----------|-------------|----------|
| Text | Simple text query | `text` |
| Vision | Image comprehension (embedded test image) | `vision` |
| Stream | Streaming text response | `text` |
| Tools | Function calling with a weather tool | `tools` |
| Embedding | POST to `/v1/embeddings`, displays dimension count | `embedding` |
| Reasoning | Text query with `reasoning_effort: "low"` | `reasoning` |

Embedding models show a single Test button, since only one test type applies.

A model served by several providers is tested against the one picked in its provider list, pinned with the `X-Rosetta-Provider-Pin` header — which the gateway honours only for admin-internal tokens. While a test runs, **Cancel** aborts it and an elapsed timer ticks beside the status. Test payloads use `max_tokens=256`. Results show the extracted reply, with collapsible sections for the raw request, the raw response body and, for a vision test, the image preview.

### Delete confirmations

Destructive operations use two confirmation patterns:

- **Provider delete** — a type-to-confirm dialog that names the cascade effects.
- **Model delete**, **API key delete** and **Clear Log** — an inline two-step confirmation: the button becomes **Sure? Yes** and reverts after 3 seconds.

## API Keys

The **API Keys** section issues and revokes the keys that call `/v1/*`. Each key has a label, an allowed-shim list (`*` for any), a creation time and a last-used time. A new key is shown once, in full; the list keeps only its id and label. Deleting a key takes effect on the next request — no restart.

## Dashboard

The **Dashboard** is the landing section and reports live traffic.

![Dashboard](images/admin-dashboard.png)

**Summary cards** — Total Requests, Error Rate, Active Streams, Uptime and Tokens (24h), the last with a link to its breakdown.

**Rolling charts** — Throughput (req/s) and Latency (ms) over the last 60 seconds. With no traffic they say "No data yet" rather than drawing a flat line.

**Requests histogram** — the request log folded into buckets, with a **5h** / **24h** toggle. Success stacks on the baseline and errors cap the bar, so the height stays the request count and red only ever means errors; hovering a bucket breaks it down.

**Token usage** — clicking a token count opens a modal with prompt / completion / total, cache creation and cache read (Anthropic-style providers), reasoning tokens, and the rolling 24-hour figures.

**Per-Provider Breakdown** — requests grouped by target provider in a two-column list. **Rebuild Counters** recomputes the totals from the request log when a counter has drifted.

**Network Diagnostics** — three checks from the gateway process:

| Item | Description |
|------|-------------|
| **IP Location** | The gateway's outgoing public IP and geolocation, from an external lookup |
| **Host IP** | The Docker host's gateway IP (e.g. `172.29.0.1`), read from the container's default route via `/proc/net/route`. Only shown inside a container. |
| **Google** | Basic connectivity check to `google.com` |

## Observability

**Observability** is one panel with five views of the same traffic.

![Observability — requests](images/admin-observability.png)

| View | Shows |
|------|-------|
| **Requests** | One row per proxied call — the request log |
| **Ops Log** | Server-level events: lifecycle, config, keys, data mutations, security, health |
| **Error Dumps** | The bodies behind a failed request |
| **Profiling** | Toggle for the phase profiler |
| **Capture** | Toggle for request/response capture |

### Filters

Every view shares one filter vocabulary:

- **Time** — All time, Last 1h, Last 24h, Last 7d, Last 30d, or a custom range
- **Status** — All, OK (2xx/3xx), 4xx, 5xx, or a custom code
- **Provider**, **Model**, **API Key** — the same lists the config defines

### Request rows

| Column | Description |
|--------|-------------|
| Time | Request timestamp |
| Model | Model name from the request |
| Source → Target | Source format and target provider |
| Mode | `stream`, or a blank cell for a non-streaming call |
| API Key | Label of the key that called |
| Client IP | Peer address |
| Status | HTTP status, colour-coded |
| Tokens | Prompt and completion counts, plus cache reads where the provider reports them |
| Duration | End to end, and output tokens per second where token usage is known |

A stream's duration is the span of the streamed body, not the time to headers, so the rate beside it reflects what the client actually waited for.

**Pause refresh** freezes the list while you read it. **Clear Log** empties the view behind an inline confirmation.

### Error dumps

An error dump opens full screen at `#dump/<id>` — behind the same login gate as the rest of the panel — with copy and download per section, and a bulk selection (**select all**, **invert**) scoped to the filtered result rather than the page. The header box reflects the rows the filter leaves.

### Server ops log

The ops log records mutations and security events with structured metadata.

#### Event categories

| Category | Event Types | Description |
|----------|-------------|-------------|
| Lifecycle | `startup`, `shutdown` | Server start and graceful stop |
| Configuration | `config_reload`, `admin_setup` | Config hot-reload, admin panel init |
| API Keys | `key_create`, `key_update`, `key_delete`, `key_rotate` | Key management operations |
| Data Mutations | `data_cleared`, `data_cleanup`, `data_trimmed`, `data_vacuumed`, `data_rebuilt` | Database maintenance operations |
| Security | `password_changed`, `token_rotated`, `session_logout_all` | Authentication and session events |
| Health | `health_status_change` | Provider health state changes |

#### Severity levels

- **info** — non-destructive operations (key creation, config reload, vacuum)
- **warning** — destructive data mutations (clear, trim, cleanup) and security events (password change, token rotation)

An operation that fails is recorded with elevated severity automatically.

#### Source filtering

| Source | Operations |
|--------|-----------|
| `admin` | Admin panel actions (clear data, session logout) |
| `auth` | Password change, token rotation |
| `keys` | API key create/update/delete/rotate |
| `persistence` | Background prune, periodic cleanup |
| `gateway` | Startup, shutdown |
| `config` | Configuration reload |

#### ops_log vs request_log

The **request log** records every proxy request (high volume, traffic-focused). The **ops log** records mutations and security events (low volume, audit-focused). They are queried independently, and auto-refresh follows the active view.

### Retention

Retention lives in **Settings → Logging & data** and applies per store — request log, error dumps and the ops log each have their own cap and age:

- **Count-based**: the maximum number of entries to keep (the oldest are pruned)
- **Age-based**: entries older than a threshold are cleaned up

Both thresholds apply independently — whichever triggers first wins.

## Settings

**Settings** is a full-width console with a left sub-nav, not a drawer of cards. Each line is one setting and the control is sized to its content instead of stretched across the page. Server-scoped groups save on change; the groups that touch the database keep an explicit **Save**.

![Settings — security](images/admin-settings-security.png)

| Pane | Holds |
|------|-------|
| **Appearance** | Colour scheme, mode, language, auto-refresh interval and the theme override |
| **Behavior** | Console output level, error detail, credential reveal, the global proxy URL |
| **Rate limiting** | Switch, algorithm (sliding window, token bucket, fixed window, GCRA) and scope (global, per IP, per key, per model) |
| **Logging & data** | Request log cap and age, error dump cap, export and cleanup, ops log retention |
| **Security** | Admin token and its rotation, **Logout All**, the password form and the session lifetimes |
| **Shims** | Declared sources and the loaded-shim inventory |
| **About** | Config file path, versions and links |

The **global proxy URL** lives here rather than on the Providers page: it is server-scoped, and it applies to every provider that does not override it.

### Theme override

**Appearance → Theme override** appends CSS to the admin page after the built-in schemes and after any downstream fragment. It saves to `server.admin_custom_css` and applies on save and on every later page load, so an operator can restyle the panel from the console instead of shipping a `custom_head` from code. A rule that sets a theme token needs the scheme's own selector.

### Session lifetimes

**Security → Sessions** holds two answers, and only one of them applies to a given session:

- **Sign out after inactivity** — the window a session slides on every authenticated request (`server.admin_session.idle_minutes`, default 30, `0` = never). The panel's own idle timer reads the same number.
- **Stay signed in for** — the persistent-cookie lifetime a remembered sign-in gets (`server.admin_session.remember_days`, default 30, `0` = the login page does not offer it). A remembered session ignores the idle window and ends at its own lifetime.

Both apply on the next request, and both are reported by `/admin/api/auth-check` before sign-in, so the login card can say which one is in force.

### Shims { #shim-sources }

The **Shims** pane declares external sources and lists what loaded.

![Settings — shims](images/admin-shims.png)

**Declared sources** — the top-level `shims` section (see [External Shim Sources](configuration.md#external-shim-sources)): **Directories** (one per line), **Preload presets** (catalog keys such as `argo, alcf, asksage`), **Remotes** (a JSON array), and the remaining knobs — cache directory, re-fetch TTL and fail-startup-on-a-bad-source. **Save** persists the section and re-applies it; **Reload** re-applies the current config; **Refetch** forces every remote to be downloaded again, bypassing its cache and `refresh` mode. A source that fails is reported here without stopping the gateway.

**Preset catalog.** Above the remotes list, a catalog offers every standalone shim package the gateway knows about — ALCF, Argo and AskSage. Choosing one and pressing **Add source** appends its expanded `shims.remotes` entry to the JSON list below — it does **not** save, so the entry goes through the same review and the same **Save** as one typed by hand. The **Preload presets** field is the shorter spelling: catalog keys that expand server-side, with an explicit remote overriding its preset when the URLs match. An entry whose `url` is already in the list is not added a second time, and the button says so. See [Standalone shim packages](../guide/shims.md#standalone-shim-packages).

**Loaded shims.** Below the sources, the inventory lists every registered shim, grouped by the source it came from — the group head carries the source's state, resolved ref, count and any failure, so "what loaded" and "where from" are one answer. A shim in a declared local directory can be opened, edited and deleted here; an edit that does not load is rolled back to the previous file, and deletion is refused while a configured provider still names the shim. A shim can also be **disabled** and re-enabled through `shims.disabled` — a declaration rather than a deletion, so the same definition comes back. Built-in and remote shims are read-only, with the reason on the row; so is a definition sitting at the root of a declared source, because the loader does not read it and removing "its directory" would take the whole source.

**Removal advisories.** Below the inventory, the advisories name each provider in the config whose shim is going away, the release that removes it and the source that replaces it — with the same **Add source** action. The list is derived per request from the config and the load report, so it is current without a restart.

!!! warning
    A declared shim directory or archive runs third-party Python
    (`transforms.py`) at load time. Only add sources you trust, and prefer
    pinning a tag or commit SHA.

## Themes

The panel ships two colour schemes, **Minimal** and **Emerald**, each in a light and a dark mode. The rail's mode button cycles light / dark / system, and **Settings → Appearance** carries the same choice as selects. The choice is stored per browser profile in `localStorage` (`llm-rosetta-scheme` and `llm-rosetta-mode`) and applied before the first paint, so a stored scheme does not flash the default. The login card carries the same mode and language buttons, because both live in `localStorage` and a login page in the wrong language is otherwise a dead end.

A scheme declares token values only; the components decide where those values land. If you are adding a scheme, see [Admin Themes](../contributing/admin-themes.md): the contract is enforced by `make test-js`, and `make theme-matrix` renders every scheme × mode × width against a live panel and saves screenshots.

## Authentication

The admin panel does **not** require the gateway API key.

Set `server.admin_password` in your config to enable built-in password protection for the admin panel:

```jsonc
{ "server": { "admin_password": "${ADMIN_PASSWORD}" } }
```

See [Configuration — Admin Panel Security](configuration.md#admin-panel-security) for details.

With password protection enabled, the console shows a login card: the product mark, one password field with a reveal button, a Caps Lock hint and a submit button that goes busy. A **Stay signed in** checkbox appears only when the gateway offers it, and it follows the last choice. The server's own message is shown verbatim, so a lockout names its remaining seconds instead of "Invalid password".

![Login card](images/admin-login.png)

!!! info "No content flash"
    While the browser verifies the stored session, the admin UI stays hidden (`body.auth-pending`) until authentication is confirmed. This prevents a brief flash of the interface before the login card appears.

- Authentication uses **HttpOnly + SameSite=Lax** session cookies with `Path=/admin` — no token is exposed to JavaScript.
- Without **Stay signed in**, the cookie dies with the browser and the session ends after the idle window (default 30 minutes without activity; mouse movement, keyboard input, scrolling and clicks all count).
- With it, the cookie carries a `Max-Age` of the remembered lifetime (default 30 days) and the session ignores the idle window — "stay signed in" and "idle timeout" are two promises, so only one applies to a given session.
- A **Logout** button sits in the rail footer. A password change mints the replacement session with the lifetime the user asked for, so remembering survives it.
- Sessions live in memory: a gateway restart ends every one, remembered or not.
- For programmatic API access, the `X-Admin-Token` header is still accepted as a fallback.

Alternatively, protect the admin panel using a reverse proxy:

- **Caddy**: Use `basicauth` directive
- **Nginx**: Use `auth_basic` directive
- **Traefik**: Use BasicAuth middleware

Gateway API keys (managed in the **API Keys** section) protect AI request endpoints (`/v1/*`). See [Configuration — Gateway API Keys](configuration.md#gateway-api-keys) for details.

## Internationalization

The console supports English and Chinese (中文). The language button sits in the rail footer, and on the login card. Every interface string — tooltips, placeholders, aria-labels and toasts included — goes through the translation layer, and the selection persists in `localStorage`.

!!! note "Chinese label update"
    The Chinese translation for "provider" was updated from **服务商** to **服务方** to better reflect the local usage convention.

## Data Persistence

Metrics and request log data are persisted in a single SQLite database alongside the config file:

```
~/.config/llm-rosetta-gateway/
    config.jsonc
    data/
        gateway.db    # SQLite database — request log + metrics (WAL mode)
```

### How it works

- **Request log** entries are written directly to `gateway.db` on each request
- **Metrics counters** are saved to `gateway.db` after each update
- On **startup**, persisted data is loaded back — metrics and logs survive restarts
- **WAL mode** ensures crash-safe writes without blocking reads

### Retention

Up to **5,000** request log entries are retained. When the limit is reached, the oldest entries are automatically pruned.

### Legacy migration

On first startup, if `request_log.jsonl` or `metrics.json` exist in the data directory from a previous installation, they are automatically imported into `gateway.db` and renamed to `*.migrated`.

## Accessing Host Services from Docker

When the gateway runs inside a Docker container and needs to reach a service on the host machine (e.g. another API proxy listening on `localhost:44501`), you cannot use `127.0.0.1` as the Base URL — that refers to the container itself.

Instead, use the **Host IP** shown in the Dashboard's Network Diagnostics section. This is the Docker bridge gateway address (typically `172.x.0.1`) and is automatically detected from the container's default route. The provider proxy URL placeholder in the admin panel dynamically shows this detected IP so you can copy it directly.

### Firewall configuration (ufw)

If the host runs `ufw`, Docker container traffic is blocked by default even though the container and host share a bridge network. You need to explicitly allow the Docker subnet.

The recommended approach is to allow the entire Docker private address range (`172.16.0.0/12`) in one rule. This covers all Docker networks — default bridge, compose-created networks, and any future networks — so you never have to add per-network rules:

```bash
sudo ufw allow from 172.16.0.0/12 comment 'Docker all subnets'
```

This is safe because `172.16.0.0/12` is an [RFC 1918](https://datatracker.ietf.org/doc/html/rfc1918) private address range that is not routable on the public internet. Only containers running on the same host can use these addresses.

!!! warning "Why per-network rules are fragile"
    Docker assigns subnets dynamically. If you only allow a specific `/16` (e.g. `172.29.0.0/16`), a container started with `docker run` instead of `docker compose` may land on a different subnet (e.g. `172.17.0.0/16`) and lose access to host services. The `/12` rule avoids this entirely.

If the firewall is blocking traffic, "Fetch from Provider" and model tests will show timeout errors. Check the Network Diagnostics Host IP and add the corresponding ufw rule to resolve it.

## Caching / Reverse Proxy

The admin panel sets `Cache-Control: no-cache, no-store, must-revalidate` on all `/admin/api/*` responses. This prevents aggressive reverse proxy caches (e.g. Caddy with the Souin plugin) from serving stale dashboard metrics or request log data.

Model test polling also uses POST requests instead of GET, which avoids being cached by intermediaries.

## Mobile / Responsive

The console is responsive — below 900 px the rail becomes a top bar, the settings sub-nav folds into a bar above the pane, and data tables scroll horizontally so no column is clipped.

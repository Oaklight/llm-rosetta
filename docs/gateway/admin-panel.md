---
title: Admin Panel
---

# Admin Panel

The gateway includes a built-in web-based admin panel for managing configuration, monitoring traffic, and inspecting request logs — all without editing config files or restarting the server.

Access it at: **`http://localhost:8765/admin/`**

## Configuration

The **Configuration** tab lets you manage providers, models, and server settings through a visual interface.

![Configuration tab](images/admin-configuration.png)

### Server Settings

Set a global proxy URL that applies to all providers unless overridden per-provider.

### Providers

Each provider card shows its name, API standard type, base URL, masked API key, and optional proxy. You can:

- **Add** a new provider with the "+ Add Provider" button
- **Edit** an existing provider (name, type, base URL, API key, proxy)
- **Rename** a provider — all model references are updated automatically
- **Delete** a provider

When editing, API keys are shown as a password field with a visibility toggle and copy button. Masked keys displayed on cards are never written back to the config file.

### Model Routing

Below the providers section, a model routing table lists all configured models with their target provider and capabilities (text, vision, tools). You can:

- **Add** a new model with provider assignment
- **Edit** model capabilities inline
- **Delete** a model routing entry

## Dashboard

The **Dashboard** tab provides real-time metrics about gateway traffic.

![Dashboard tab](images/admin-dashboard.png)

### Summary Cards

- **Total Requests** — cumulative request count since startup (or last persistence load)
- **Error Rate** — percentage of non-2xx responses
- **Active Streams** — currently active streaming connections
- **Uptime** — time since the gateway started

### Time-Series Charts

Two rolling 60-second charts show:

- **Throughput (req/s)** — request rate over time
- **Latency (ms)** — average response time per second

### Per-Provider Breakdown

A table showing request counts grouped by target provider, useful for identifying traffic distribution.

## Request Log

The **Request Log** tab shows individual requests flowing through the gateway.

![Request Log tab](images/admin-request-log.png)

Each entry includes:

| Column | Description |
|--------|-------------|
| Time | Request timestamp |
| Model | Model name from the request |
| Source -> Target | Source API format and target provider |
| Mode | Streaming or non-streaming |
| Status | HTTP status code (color-coded) |
| Duration | End-to-end latency |

### Filtering

Use the dropdown filters at the top to narrow by:

- **Model** — filter by specific model
- **Provider** — filter by target provider
- **Status** — show only successful (2xx/3xx) or error (4xx/5xx) responses

Click **Clear Log** to remove all entries from the current view.

## Themes

The admin panel ships with 8 themes, selectable from the dropdown in the top-right corner:

| Theme | Style |
|-------|-------|
| Light | Default, clean white background |
| Indigo Dark | Dark with indigo accents |
| Dracula | Popular dark theme |
| Nord | Arctic-inspired pastel palette |
| Solarized | Ethan Schoonover's color scheme |
| Osaka Jade | Dark with jade green accents |
| One Dark | Atom editor's dark theme |
| Rose Pine | Muted rose and pine tones |

![One Dark theme example](images/admin-dark-theme.png)

Theme selection is stored in `localStorage` and persists across browser sessions.

## Internationalization

The admin panel supports English and Chinese (中文). Switch languages using the language dropdown in the top-right corner. The selection persists in `localStorage`.

## Data Persistence

Metrics and request log data are automatically persisted to disk alongside the config file:

```
~/.config/llm-rosetta-gateway/
    config.jsonc
    data/
        metrics.json              # cumulative counters
        request_log.jsonl         # recent requests (JSONL)
        request_log.1.jsonl.gz    # rotated backup
        request_log.2.jsonl.gz    # older backup
```

### How it works

- **Request log** entries are flushed to `request_log.jsonl` every 10 seconds
- **Metrics counters** are saved to `metrics.json` every 30 seconds (atomic write)
- On **shutdown**, a final flush ensures no data is lost
- On **startup**, persisted data is loaded back — metrics and logs survive restarts

### Log rotation

When `request_log.jsonl` exceeds 2 MB:

1. Existing backups shift up (`.1.jsonl.gz` -> `.2.jsonl.gz`, etc.)
2. Current log is compressed to `.1.jsonl.gz`
3. Log file is truncated

Maximum 3 compressed backups are kept. All operations use Python's stdlib (`gzip`, `json`, `os`) for cross-platform compatibility.

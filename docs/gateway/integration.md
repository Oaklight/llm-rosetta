---
title: Gateway Integration Guide
---

# Gateway Integration Guide

The llm-rosetta gateway can be embedded into downstream projects as a proxy pipeline + admin panel. Instead of running the standalone `llm-rosetta-gateway` CLI, your project imports the gateway's internals and wires them into its own HTTP application, config system, and startup lifecycle.

This guide covers the integration surface: the `ConfigIO` protocol, `setup_admin()` parameters, telemetry wiring, and branding customization. The canonical integration example is [argo-proxy](https://github.com/Oaklight/argo-proxy), which wraps the gateway with its own YAML config, model registry, and authentication layer.

## ConfigIO Protocol

### Why

The standalone gateway reads and writes JSONC config files. Downstream projects typically have their own config format — YAML, TOML, environment variables, or even a database. The `ConfigIO` protocol abstracts file I/O so the admin panel can read and write config without knowing the underlying format.

### Protocol Definition

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ConfigIO(Protocol):
    def load(self, path: str) -> dict[str, Any]:
        """Read config with env-var substitution (for runtime use)."""
        ...

    def load_raw(self, path: str) -> dict[str, Any]:
        """Read config without env-var substitution (for edit round-trips)."""
        ...

    def save(self, path: str, data: dict[str, Any]) -> None:
        """Write config back to disk."""
        ...
```

- `load()` returns the config with `${ENV_VAR}` placeholders resolved — used at runtime.
- `load_raw()` returns the config with placeholders intact — used when the admin panel reads config for editing and must write it back without losing placeholders.
- `save()` persists the admin panel's edits back to disk.

### Injecting Runtime State

If your project generates providers or models at runtime (e.g. discovered from an upstream API), inject them in `load_raw()`:

```python
class MyConfigIO:
    def __init__(self, config, model_registry):
        self._config = config
        self._registry = model_registry

    def load(self, path: str) -> dict[str, Any]:
        raw = load_my_config(path)
        return self._inject_runtime_state(raw)

    def load_raw(self, path: str) -> dict[str, Any]:
        raw = load_my_config(path)
        return self._inject_runtime_state(raw)

    def _inject_runtime_state(self, data: dict[str, Any]) -> dict[str, Any]:
        # Inject providers and models that are managed by your project,
        # not stored in the config file
        data["providers"] = build_providers_from_registry(self._registry)
        data["models"] = build_models_from_registry(self._registry)
        data.setdefault("server", {}).update({
            "host": self._config.host,
            "port": self._config.port,
        })
        return data

    def save(self, path: str, data: dict[str, Any]) -> None:
        # Strip runtime-injected fields that shouldn't be persisted
        data.pop("providers", None)
        data.pop("models", None)
        write_my_config(path, data)
```

!!! warning "Strip runtime fields in `save()`"
    If your `load_raw()` injects providers/models that are managed upstream (not user-editable), `save()` **must** strip them before writing. Otherwise, the admin panel's "Save" button would persist stale copies of runtime-generated data into the config file.

## setup_admin() Parameters

The `setup_admin()` function initializes the admin panel state on your app instance. It is typically called during app startup after your app's own config and transport are ready.

```python
from llm_rosetta.gateway.admin import setup_admin

def setup_admin(
    app,
    config: GatewayConfig,
    config_path: str | None,
    config_io: ConfigIO | None = None,
    custom_head: str | None = None,
    branding: dict[str, Any] | None = None,
    disabled_tabs: list[str] | None = None,
) -> None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `app` | `App` | Your HTTP application instance |
| `config` | `GatewayConfig` | Parsed gateway configuration |
| `config_path` | `str | None` | Path to config file on disk. `None` disables persistence and config editing. |
| `config_io` | `ConfigIO | None` | Custom config adapter. Defaults to `JsoncConfigIO` when `None`. |
| `custom_head` | `str | None` | HTML fragment injected before `</head>` in the admin page |
| `branding` | `dict | None` | Dict to customize the admin panel identity |
| `disabled_tabs` | `list[str] | None` | Tab IDs to hide from the admin UI (e.g. `["metrics"]`). `None` shows all tabs. |

### Full Example

```python
from llm_rosetta.gateway.admin import setup_admin
from llm_rosetta.gateway.admin.routes import register_admin_routes

# 1. Register admin routes before setup
register_admin_routes(app)

# 2. Initialize admin state
config_io = MyConfigIO(my_config, model_registry)
setup_admin(
    app,
    gateway_config,
    str(config_path) if config_path else None,
    config_io=config_io,
    custom_head='<script>console.log("custom admin");</script>',
    branding={
        "title": "My Project",
        "subtitle": "gateway admin",
        "version": __version__,
        "links": [
            {"label": "GitHub", "url": "https://github.com/me/myproject", "icon": "github"},
            {"label": "PyPI", "url": "https://pypi.org/project/myproject/", "icon": "pypi"},
            {"label": "Docs", "url": "https://myproject.readthedocs.io", "icon": "docs"},
        ],
        "attribution": "Powered by llm-rosetta gateway",
    },
)
```

## Telemetry Integration

The gateway's `handle_streaming()` and `handle_non_streaming()` do **not** record telemetry — they only handle request conversion and upstream forwarding. The caller is responsible for recording metrics and request log entries.

### Recording Metrics and Request Log

After each proxy call, record the result:

```python
import time
from llm_rosetta.observability import MetricsCollector, RequestLog, RequestLogEntry

# These are set up by setup_admin() and attached to app
metrics: MetricsCollector = app.metrics
request_log: RequestLog = app.request_log

t0 = time.monotonic()
response, profile = await handle_non_streaming(
    route, provider_info, body,
    transport=transport,
    metadata_store=store,
    extra_headers=extra_headers,
    capture_state=app.capture_state,  # for content capture
    persistence=app.persistence,       # for error dumps
)
duration_ms = (time.monotonic() - t0) * 1000

# Record metrics
metrics.record_request(
    model=model,
    source=source_provider,
    target=route.target_provider,
    status_code=response.status_code,
    duration_ms=duration_ms,
    is_stream=False,
    provider_name=route.provider_name,
    error_detail=error_detail,
)

# Record request log entry
entry = RequestLogEntry.create(
    model=model,
    source_provider=source_provider,
    target_provider=route.target_provider,
    target_provider_name=route.provider_name,
    is_stream=False,
    status_code=response.status_code,
    duration_ms=duration_ms,
    error_detail=error_detail,
)
request_log.add(entry)
```

### Passing capture_state and persistence

The `capture_state` and `persistence` parameters enable content capture (for the admin panel's request inspector) and error dumps (for debugging upstream failures), respectively:

```python
response, profile = await handle_streaming(
    route, provider_info, body,
    transport=transport,
    metadata_store=store,
    extra_headers=extra_headers,
    capture_state=app.capture_state,   # in-memory content capture
    entry_id=pre_entry_id,             # for stream profile write-back
    request_log=app.request_log,       # for stream profile write-back
)
```

Both are created and attached to `app` automatically by `setup_admin()`.

### Admin Test Button

The admin panel has a "Test" button that sends a request through the gateway to verify provider connectivity. For this to work, the app must expose its bind address:

```python
app._bind_host = host
app._bind_port = port
```

Set these after calling `setup_admin()` and before the server starts accepting connections.

## Branding Customization

### Branding Dict

The `branding` dict maps to UI elements across the admin panel:

| Key | Where It Appears | Default |
|-----|-----------------|---------|
| `title` | Header bar, browser tab | `"llm-rosetta"` |
| `subtitle` | Header bar (smaller text) | `"gateway admin"` |
| `version` | Settings footer | _(none)_ |
| `links` | Settings footer (icon buttons) | _(none)_ |
| `attribution` | Settings footer ("Powered by" line) | _(none)_ |

### Footer Links

Each entry in the `links` list is a dict with `label`, `url`, and optionally `icon`:

```python
"links": [
    {"label": "GitHub",  "url": "https://github.com/me/project", "icon": "github"},
    {"label": "PyPI",    "url": "https://pypi.org/project/pkg/",  "icon": "pypi"},
    {"label": "Docker",  "url": "https://hub.docker.com/r/me/img","icon": "docker"},
    {"label": "Docs",    "url": "https://docs.example.com",       "icon": "docs"},
    {"label": "Custom",  "url": "https://example.com"},  # no icon
]
```

Available icon names: `github`, `pypi`, `docker`, `docs`. Omit `icon` for a text-only link.

### Using custom_head for Deeper UI Changes

The `custom_head` parameter injects raw HTML before `</head>` in the admin page. This is useful for CSS overrides, JavaScript patches, or MutationObserver-based UI modifications.

Example — hiding provider edit/delete buttons for managed providers:

```python
_ADMIN_CUSTOM_HEAD = """\
<script>
document.addEventListener('DOMContentLoaded', function(){
  var _managed = {'my-upstream-openai':1, 'my-upstream-anthropic':1};
  new MutationObserver(function(){
    var cards = document.querySelectorAll('.provider-card');
    if (!cards.length) return;
    cards.forEach(function(card){
      var nm = card.querySelector('.name');
      if (!nm) return;
      var pName = nm.textContent.trim();
      if (!_managed[pName]) return;
      // Hide toggle and action buttons for managed providers
      var toggle = card.querySelector('.toggle');
      if (toggle) toggle.style.display = 'none';
      var actions = card.querySelector('.actions');
      if (actions) actions.style.display = 'none';
    });
    // Hide "Add Provider" button entirely
    var addBtn = document.querySelector('button[onclick*="openProviderModal"]');
    if (addBtn) addBtn.style.display = 'none';
  }).observe(document.body, {childList: true, subtree: true});
});
</script>"""
```

!!! note
    `configData` in `admin.html` is `let`-scoped — you cannot access it from `window` in injected scripts. If you need to read config state, use the admin API endpoints (`/admin/api/config`) instead.

## Provider Configuration

### Read-Only Providers

When providers are managed by your project (e.g. auto-discovered from an upstream service), mark them as read-only:

```python
def build_providers(config):
    return {
        "my-openai": {
            "type": "openai_chat",
            "api_key": config.api_key,
            "base_url": config.openai_base_url,
            "readonly": True,  # prevents admin panel edits
        },
    }
```

The `readonly: True` flag tells the admin panel to disable edit controls for that provider.

### Model Capabilities

Each model entry can declare its capabilities, which appear in the admin panel's model table:

```python
def build_models(registry):
    models = {}
    for alias, model_id in registry.available_models.items():
        models[alias] = {
            "provider": resolve_provider(model_id),
            "capabilities": ["text", "vision", "tools", "reasoning"],
        }
        # Optional: map gateway alias to a different upstream model name
        if model_id != alias:
            models[alias]["upstream_model"] = model_id
    return models
```

Available capability values: `text`, `vision`, `tools`, `reasoning` (or any custom string — they are displayed as-is in the admin UI).

### Stripping Provider/Model Changes on Save

For managed deployments where providers and models are upstream-controlled, your `ConfigIO.save()` should strip any admin-panel edits to those sections:

```python
def save(self, path: str, data: dict[str, Any]) -> None:
    # Remove runtime-managed sections
    data.pop("providers", None)
    data.pop("models", None)

    # Map gateway-format keys back to your project's config format
    if server := data.pop("server", None):
        if "host" in server:
            data["host"] = server["host"]
        if "port" in server:
            data["port"] = server["port"]

    write_my_config(path, data)
```

## Common Pitfalls

### Circular Imports

If your `ConfigIO` imports from bridge modules that themselves import from config, you'll hit circular import errors at startup. Use lazy imports inside method bodies:

```python
class MyConfigIO:
    def _inject_runtime_state(self, data):
        # Lazy import to avoid circular dependency
        from ..bridge import build_providers, build_models
        data["providers"] = build_providers(self._config)
        data["models"] = build_models(self._registry)
        return data
```

### Path vs str for config_path

The `config_path` parameter to `setup_admin()` must be `str`, not `pathlib.Path`. The admin panel serializes it to JSON for the frontend. Pass `str(config_path)` if you have a `Path` object:

```python
setup_admin(
    app,
    gateway_config,
    str(config_path) if config_path else None,  # not Path object
    config_io=config_io,
)
```

### anthropic_stream_mode and Admin Test Button

The admin panel's test button sends a **non-streaming** request. If your project forces streaming for Anthropic (e.g. `anthropic_stream_mode = "force"`), the test button may behave unexpectedly. Make sure non-streaming requests can still go through for test purposes, or handle the `"retry"` mode that falls back to streaming when the upstream returns a "streaming required" error.

### configData Scope in admin.html

The admin page's `configData` variable is `let`-scoped inside an IIFE. Scripts injected via `custom_head` run in the global scope and cannot access `configData` directly. If you need config state in your injected scripts, fetch it from the admin API:

```javascript
// In custom_head script — don't try to read configData directly
fetch('/admin/api/config')
  .then(r => r.json())
  .then(cfg => {
    // Now you have the config
  });
```

### Register Routes Before setup_admin

Call `register_admin_routes(app)` **before** `setup_admin()`. The route registration sets up the HTTP endpoints, while `setup_admin()` initializes the state those endpoints depend on. Reversing the order won't crash, but the routes will reference uninitialized state during the brief window between registration and setup.

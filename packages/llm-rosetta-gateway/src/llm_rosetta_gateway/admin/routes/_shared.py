"""Shared helpers used by multiple admin route modules."""

from __future__ import annotations

import functools
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, overload
from collections.abc import AsyncGenerator

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ...admin_api import admin_context, catalogue

if TYPE_CHECKING:
    from ...admin_api import ConfigIO, GatewayConfig

logger = logging.getLogger("llm-rosetta-gateway")

_ENV_VAR_RE = re.compile(r"^\$\{.+\}$")
_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

#: The longest string a shim may put in a field label, hint or placeholder.
#: These are rendered verbatim in the admin origin, so the bound is what keeps
#: a malformed shim from filling the panel.
_FIELD_TEXT_LIMIT = 200
#: How many declared fields the panel will render.  A credential set with more
#: than a handful of rows is a record, and the paste box exists for that.
_FIELD_LIMIT = 8
_FIELD_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def exchange_field_text(value: Any) -> str | None:
    """Return *value* when it is a non-empty string, clipped to the bound."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:_FIELD_TEXT_LIMIT] if text else None


def declared_fields(shim: Any) -> list[dict[str, Any]]:
    """The form inputs a shim declares, in the panel's own shape.

    Two sources, one list: ``fields:`` in the shim's ``provider.yaml`` (already
    validated and merged with its vendor manifest at load) and, for
    compatibility, what an ``auth.py`` exchange declares in code.  The data
    wins per key, so a YAML entry can reword a field an exchange also
    describes; the order follows the shim first.
    """
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in getattr(shim, "fields", ()) or ():
        cleaned = _normalize_field(entry)
        if cleaned is None or cleaned["key"] in seen:
            continue
        seen.add(cleaned["key"])
        fields.append(cleaned)
    for entry in exchange_fields(getattr(shim, "auth", None)):
        if entry["key"] in seen:
            continue
        seen.add(entry["key"])
        fields.append(entry)
    return fields[:_FIELD_LIMIT]


def _normalize_field(entry: Any) -> dict[str, Any] | None:
    """Shape one declared input for the panel, or ``None`` when it cannot be.

    Both sources go through this: the shim's YAML (validated at load) and an
    exchange's ``fields()`` (arbitrary Python).  Every value is coerced to a
    clipped string, the key must be a plain identifier, and anything else is
    dropped rather than guessed at.  Nothing here is a secret: these are the
    setup inputs stored on the provider entry, so the credential itself stays
    in ``api_key`` or the exchange's own record.
    """
    if not isinstance(entry, dict):
        return None
    key = entry.get("key")
    label = exchange_field_text(entry.get("label"))
    if not isinstance(key, str) or not _FIELD_KEY_RE.match(key) or label is None:
        return None
    field: dict[str, Any] = {"key": key, "label": label}
    for name in ("placeholder", "help"):
        text = exchange_field_text(entry.get(name))
        if text is not None:
            field[name] = text
    required = entry.get("required")
    # ``True`` from Python, ``"true"`` from a YAML-ish declaration: both mean
    # the same thing, and anything else means no.
    if required is True or (isinstance(required, str) and required.strip() == "true"):
        field["required"] = True
    return field


def exchange_fields(exchange: Any) -> list[dict[str, Any]]:
    """The inputs an exchange declares in code, in the panel's own shape.

    The exchange is trusted code, but the panel renders what it returns inside
    the admin origin, so every entry is normalized and a broken one is dropped.
    """
    reader = getattr(exchange, "fields", None)
    if reader is None:
        return []
    try:
        declared = reader()
    except Exception:  # noqa: BLE001 - a broken declaration is not a 500
        return []
    if not isinstance(declared, (list, tuple)):
        return []
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in declared:
        field = _normalize_field(entry)
        if field is None or field["key"] in seen:
            continue
        seen.add(field["key"])
        fields.append(field)
        if len(fields) >= _FIELD_LIMIT:
            break
    return fields


# The old panel wrote this word into the key slot to say "a command mints the
# credential".  It is a display marker, never a key, and storing it beside a
# real command makes the provider unloadable.
_TOKEN_COMMAND_MARKER = "token_command"


@overload
def _qp(request: Any, key: str) -> str | None: ...


@overload
def _qp(request: Any, key: str, default: str) -> str: ...


def _guard(tab_id: str, handler: Any) -> Any:
    """Wrap a route handler to return 404 when its tab is disabled.

    Read-only endpoints (``get_config``, ``get_metrics``,
    ``get_request_key_labels``) are intentionally left unguarded —
    disabled tabs are a UI convenience, not security isolation, and
    other components (e.g. argo-proxy's ``ArgoConfigIO``) may depend
    on them regardless of tab visibility.
    """

    @functools.wraps(handler)
    async def _guarded(request: Any, **kw: Any) -> Any:
        if tab_id in admin_context(request).disabled_tabs():
            return JSONResponse(
                {"error": "Tab disabled", "disabled": True}, status_code=404
            )
        return await handler(request, **kw)

    return _guarded


def _qp(request: Any, key: str, default: str | None = None) -> str | None:
    """Extract a single query param value (httpserver convenience)."""
    vals = request.query_params.get(key)
    if vals:
        return vals[0]
    return default


def parse_json_body(request: Any) -> tuple[dict[str, Any], Response | None]:
    """Parse the JSON body from a request.

    Returns:
        A ``(body, None)`` tuple on success, or ``({}, error_response)``
        on failure (invalid JSON).
    """
    try:
        body = request.json()
    except Exception:
        return {}, JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    return body, None


def retired_type_error(body: Any) -> Response | None:
    """A 400 answer when the request body names a retired model type.

    ``type: embedding`` and ``type: decision`` are no longer accepted; the
    answer names the replacement and the ``llm-rosetta gateway migrate`` command.
    Returns ``None`` when the body names no retired type.
    """
    message = (
        catalogue().retired_type_message(body.get("type"))
        if isinstance(body, dict)
        else None
    )
    if message is None:
        return None
    return JSONResponse(dict(error=message), status_code=400)


class ConfigMutationContext:
    """Holds state for a :func:`config_mutate` session.

    Attributes:
        data: The raw config dict loaded from disk.  Mutate in place.
        error: Set by the context manager if load/save/reload fails.
            Callers should check this after the ``with`` block.
        new_config: The reloaded :class:`GatewayConfig` after a
            successful commit.  ``None`` until commit + reload completes.
    """

    __slots__ = ("data", "error", "new_config", "warnings", "_committed")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.error: Response | None = None
        self.new_config: GatewayConfig | None = None
        self.warnings: list[dict[str, Any]] = []
        self._committed: bool = False

    def commit(self) -> None:
        """Mark the mutation for save + reload on context exit."""
        self._committed = True


@asynccontextmanager
async def config_mutate(
    request: Any,
) -> AsyncGenerator[ConfigMutationContext, None]:
    """Async context manager for the lock -> load -> mutate -> save -> reload cycle.

    Usage::

        async with config_mutate(request) as ctx:
            if ctx.error:
                return ctx.error
            # ... validate and mutate ctx.data ...
            if bad:
                return JSONResponse(...)  # exits without saving
            ctx.commit()
        if ctx.error:
            return ctx.error
        # ctx.new_config is now available

    The save and reload only happen when :meth:`ConfigMutationContext.commit`
    has been called **and** no exception was raised.  Early returns from the
    handler (e.g. validation errors) skip the save automatically.
    """
    ctx = ConfigMutationContext()
    config_path = _get_config_path(request)

    async with catalogue().config_lock(config_path):
        try:
            ctx.data = _get_config_io(request).load_raw(config_path)
        except Exception as exc:
            ctx.error = JSONResponse(
                {"error": f"Failed to read config: {exc}"}, status_code=500
            )
            yield ctx
            return

        yield ctx

        if not ctx._committed:
            return

        try:
            _get_config_io(request).save(config_path, ctx.data)
        except Exception as exc:
            ctx.error = JSONResponse(
                {"error": f"Failed to write config: {exc}"}, status_code=500
            )
            return

    # Reload happens outside the lock
    try:
        ctx.new_config = await _reload_gateway_config(request, config_path)
        ctx.warnings = admin_context(request).provider_validation_warnings()
    except Exception as exc:
        ctx.error = JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )


def _expand_env(value: Any) -> Any:
    """Return *value* with ``${ENV}`` placeholders resolved, without mutating it.

    The loader substitutes placeholders in the document text before parsing;
    the admin works on the raw document, so a preflight build has to do the
    same or a numeric field written as ``${UPSTREAM_TIMEOUT}`` would fail here
    while the real load succeeds.
    """
    if isinstance(value, str):
        return _ENV_PLACEHOLDER_RE.sub(
            lambda m: os.environ.get(m.group(1), m.group(0)), value
        )
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _config_preflight_error(
    data: dict[str, Any], data_dir: str | None = None
) -> str | None:
    """Build every enabled provider in *data* and return the first refusal.

    ``config_mutate`` saves first and reloads second, so a document the loader
    rejects used to be on disk before the 500 came back — and the next process
    start failed on it.  A provider save or an enable can be checked first.
    Only the providers are built, not the whole ``GatewayConfig``: the admin
    edits the *raw* document (``${ENV}`` placeholders intact), and a full
    build would resolve server settings the raw text does not.

    *data_dir* is the resolved data directory.  The profile-duplicate check
    needs it: without it two entries whose ``acp.home`` is defaulted could
    appear distinct when they are not.
    """
    providers = data.get("providers") or {}
    enabled = {
        str(name): _expand_env(cfg)
        for name, cfg in providers.items()
        if isinstance(cfg, dict) and cfg.get("enabled", True) is not False
    }
    if not enabled:
        return None
    return catalogue().provider_preflight_error(enabled, data_dir)


def _mask_api_key(value: str) -> str:
    """Mask a literal API key, leaving env-var placeholders intact."""
    if _ENV_VAR_RE.match(value):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _mask_proxy_url(url: str | None) -> str | None:
    """Mask userinfo credentials in a proxy URL, leaving the rest intact."""
    if not url:
        return url
    if _ENV_VAR_RE.match(url):
        return url
    # Match scheme://user:pass@host...
    m = re.match(r"^(https?://)([^:]+):([^@]+)@(.+)$", url)
    if m:
        return f"{m.group(1)}{m.group(2)}:***@{m.group(4)}"
    return url


def _sanitize_server_section(server: dict[str, Any]) -> dict[str, Any]:
    """Strip or mask sensitive fields from a server config dict."""
    safe = dict(server)
    safe.pop("admin_password", None)
    if "api_key" in safe:
        safe["api_key"] = _mask_api_key(safe["api_key"])
    if "api_keys" in safe:
        safe["api_keys"] = [
            {**entry, "key": _mask_api_key(entry.get("key", ""))}
            for entry in safe["api_keys"]
        ]
    if "proxy" in safe:
        safe["proxy"] = _mask_proxy_url(safe["proxy"])
    return safe


def _get_config_path(request: Any) -> str:
    """Return the config file path stored on the app object."""
    path = admin_context(request).config_path()
    if path is None:
        raise RuntimeError("No config file path available")
    return path


def _get_config_io(request: Any) -> ConfigIO:
    """Return the :class:`ConfigIO` adapter stored on the app object."""
    io = admin_context(request).config_io()
    if io is None:
        raise RuntimeError("No ConfigIO adapter configured on this application")
    return io


def _install_shim_report(request: Any, report: Any, signature: str) -> None:
    """Record a freshly applied shim report on the app.

    The update verdicts cached on ``shim_update_check`` describe the previous
    report — which pinned other commits — so they are dropped with it.
    """
    admin_context(request).install_shim_report(report, signature)


async def _reload_gateway_config(
    request: Any, config_path: str, *, force_shims: bool = False
) -> GatewayConfig:
    """Re-read config from disk and swap it in, then refresh the admin theme.

    The reload itself is the gateway's (``AdminContext.reload_config``); the
    theme is admin's own, so it is applied here.
    """
    from .. import apply_theme

    new_config = await admin_context(request).reload_config(
        config_path, force_shims=force_shims
    )
    # A theme saved from the admin panel must reach the next page load.
    apply_theme(admin_context(request), getattr(new_config, "admin_custom_css", ""))
    return new_config


def _apply_auth_fields(
    entry: dict[str, Any], body: dict[str, Any], existing: dict[str, Any] | None
) -> None:
    """Store the setup inputs an exchange declared, as the entry's own fields.

    The panel sends what the operator typed into the fields the shim declared;
    the values are validated here rather than trusted, because the entry is
    read by the exchange and written to the config file.  Validation *refuses*
    rather than repairs: a value the entry cannot hold is an error the operator
    can see and fix, whereas a silently dropped or truncated one is a field
    that saves, reloads empty, and fails at the vendor with no trace of why.

    An absent key means "keep what is stored", the same rule the command
    fields follow: the editor always sends them, and a client that does not
    must not wipe an account by saving something else.  An empty mapping means
    "the operator cleared them", which is how switching to a shim with no
    declared inputs drops the previous shim's values.

    Raises:
        ValueError: The payload is not a mapping, names a key this entry
            cannot hold, or carries a value that is not a string or is longer
            than :data:`_FIELD_TEXT_LIMIT`.
    """
    if "auth_fields" not in body:
        if existing and isinstance(existing.get("auth_fields"), dict):
            entry["auth_fields"] = dict(existing["auth_fields"])
        return
    value = body["auth_fields"]
    if value is None:
        entry.pop("auth_fields", None)
        return
    if not isinstance(value, dict):
        raise ValueError("auth_fields must be a mapping of field key to value")
    if len(value) > _FIELD_LIMIT:
        raise ValueError(
            f"auth_fields carries {len(value)} entries; at most {_FIELD_LIMIT} "
            "are accepted"
        )
    cleaned: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not _FIELD_KEY_RE.match(key):
            raise ValueError(f"auth_fields names the key {key!r}, which is not usable")
        if not isinstance(raw, str):
            raise ValueError(f"auth_fields[{key!r}] must be a string")
        text = raw.strip()
        if not text:
            continue
        if len(text) > _FIELD_TEXT_LIMIT:
            raise ValueError(
                f"auth_fields[{key!r}] is {len(text)} characters; at most "
                f"{_FIELD_TEXT_LIMIT} are accepted"
            )
        cleaned[key] = text
    if cleaned:
        entry["auth_fields"] = cleaned
    else:
        entry.pop("auth_fields", None)


def _apply_optional_int(entry: dict[str, Any], body: dict[str, Any], key: str) -> None:
    """Set or clear an optional integer field on a provider entry."""
    if key in body:
        val = body[key]
        if val not in (None, ""):
            entry[key] = int(val)
        else:
            entry.pop(key, None)


#: Provider keys the ACP lane owns.  They are written by hand in
#: ``config.jsonc`` — the provider editor has no ACP pane — so a save that
#: rebuilt the entry from the editor's fields alone would move the lane back to
#: HTTP, or to the default profile.  ``transport`` names the lane, ``acp``
#: holds the profile and the agent command, and the two flat spellings stay
#: for an entry written before ``acp.argv``.
#
#: A body that names ``acp`` replaces the whole block; the keys inside it are
#: not merged.  Send the block as it should be stored, or omit it to keep what
#: the file holds.  ``{}`` means "nothing to change" so an API client cannot
#: clear the install pin by sending an empty object; only ``null`` removes it.
_ACP_PROVIDER_KEYS = ("transport", "acp", "acp_argv", "acp_env")


def _apply_acp_fields(
    entry: dict[str, Any], body: dict[str, Any], existing: dict[str, Any] | None
) -> None:
    """Carry a hand-written ACP lane through a provider save.

    A key the editor sends is stored, or cleared when it sends an explicit
    empty value; a key it does not send keeps what the file already holds.
    Without this an ordinary edit from the panel dropped ``acp.home``,
    ``acp.argv`` and ``transport: acp`` — the next request then took the HTTP
    path.
    """
    for key in _ACP_PROVIDER_KEYS:
        if key in body:
            value = body[key]
            if key == "acp":
                if value is None:
                    entry.pop(key, None)
                elif isinstance(value, dict) and not value:
                    if existing and key in existing:
                        entry[key] = existing[key]
                else:
                    entry[key] = value
            elif value in (None, "", {}, []):
                entry.pop(key, None)
            else:
                entry[key] = value
        elif existing and key in existing:
            entry[key] = existing[key]


def _optional_provider_keys() -> list[str]:
    """Return provider config keys that should be set-or-cleared.

    Derives format/path keys from the model type registry so that new
    types are picked up automatically without hardcoding.
    """
    keys: list[str] = [
        "models_path",
        "logo",
        "quota",
        # The credential source, for providers that mint it from a command
        # (ALCF, AskSage).  Without these a save rewrote the entry without
        # them and the provider quietly fell back to a static key.
        "token_command",
        "token_refresh_interval",
        # Names the variable the command reads the operator's key from; part of
        # the same credential-source block, and just as invisible to the editor.
        "token_command_env",
    ]
    for desc in catalogue().model_types():
        if desc.is_llm:
            continue
        if desc.config_format_key:
            keys.append(desc.config_format_key)
        if desc.config_path_key:
            keys.append(desc.config_path_key)
        if desc.config_shim_key:
            keys.append(desc.config_shim_key)
    return keys


def _apply_optional_provider_fields(
    entry: dict[str, Any], body: dict[str, Any]
) -> None:
    """Set or clear optional provider-level fields on *entry*.

    Truthy values in *body* are stored; keys explicitly sent as empty
    are removed so the admin UI can clear them.

    The per-endpoint ``<type>_enabled`` flags are the exception: ``False`` is
    a value, not an absence, so it is stored as sent.  Any other falsy value
    (``""``, ``null``) removes the key, which is how the editor says "enabled"
    without writing a key that only restates the default.
    """
    for opt_key in _optional_provider_keys():
        val = body.get(opt_key)
        if val:
            entry[opt_key] = val
        elif opt_key in body:
            entry.pop(opt_key, None)
    for flag in _typed_enable_keys():
        val = body.get(flag)
        if isinstance(val, bool):
            entry[flag] = val
        elif flag in body:
            entry.pop(flag, None)
    _apply_face_blocks(entry, body)


def _apply_face_blocks(entry: dict[str, Any], body: dict[str, Any]) -> None:
    """Store each typed endpoint's ``<type>:`` block as the editor sent it.

    A provider entry is an account; each typed endpoint is a face of it, and
    its block holds the settings that belong to the face alone — the shim, the
    path, the switch, and the ``base_url`` when the face lives on another
    vendor than the entry's chat lane (#1595).  An empty block is removed
    rather than written, so a capability the provider dropped leaves no empty
    mapping behind.

    Only a mapping is accepted: the parser reports a block of any other type
    and ignores it, and storing one here would make the editor's own mistake
    survive a save it could have refused.
    """
    for desc in catalogue().model_types():
        if desc.is_llm or desc.name not in body:
            continue
        block = body[desc.name]
        if not isinstance(block, dict):
            continue
        cleaned = {
            key: value
            for key, value in block.items()
            if value not in (None, "") or key == "enabled"
        }
        if cleaned:
            entry[desc.name] = cleaned
        else:
            entry.pop(desc.name, None)


def _typed_enable_keys() -> list[str]:
    """The ``<type>_enabled`` keys the registry's model types declare."""
    return [
        f"{desc.name}_enabled" for desc in catalogue().model_types() if not desc.is_llm
    ]


def _shim_ships_credential_exchange(shim: Any) -> bool:
    """Whether *shim* mints the upstream credential itself in ``auth.py``.

    Such a provider has no key to type into the editor, exactly like a
    ``token_command`` one, so the save path must not demand one.
    """
    return shim is not None and getattr(shim, "auth", None) is not None


def _shim_ships_credential_command(shim: Any) -> bool:
    """Whether *shim* declares the command that mints the credential.

    The argv form of the same fact: the provider has no key of its own, and
    the save path must not demand one either.
    """
    connection = getattr(shim, "connection", None)
    return bool(connection is not None and connection.token_command)


def _credential_command(body: dict[str, Any], existing: dict[str, Any] | None) -> Any:
    """Return the command that owns the credential after this save, if any.

    The body wins; an absent key means "keep what is stored", which is how the
    provider editor behaves (it has no field for the command).
    """
    command = body.get("token_command")
    if command is None and existing:
        command = existing.get("token_command")
    return command


def _credential_fields(
    body: dict[str, Any], api_key: str, existing: dict[str, Any] | None
) -> dict[str, Any]:
    """The credential slots of a provider entry.

    A command owns the credential when there is one, so an empty key is not a
    key: storing ``"api_key": ""`` beside ``token_command`` gives the loader two
    credential sources where it accepts one, and the provider comes back
    unusable.  The command itself is not something the editor can express, so
    an absent key in *body* means "keep what is stored"; a caller that wants to
    clear it sends ``clear_api_key: true``, which also works when the key was
    not in the body at all.
    """
    fields: dict[str, Any] = {}
    # The old panel's display marker is not a credential.  Dropping it here
    # repairs an entry saved by that panel the next time it is edited: the
    # command still owns the credential, and the loader no longer sees two
    # sources where it accepts one.
    if api_key == _TOKEN_COMMAND_MARKER:
        api_key = ""
    # Switching away from a static key removes it.  The editor sends this
    # flag, not an empty key: an empty key has always meant "keep what is
    # stored", and overloading it would clear a key on every ordinary save.
    # The flag itself is never copied into the entry.
    if body.get("clear_api_key") is True:
        api_key = ""
    # An explicit ``token_command`` — including the empty list the editor sends
    # for a source that stores no command — says this provider has no static
    # key.  Storing ``""`` anyway would defeat the shim's own ``api_key_env``:
    # the loader only fills that default when the entry leaves ``api_key``
    # unset.
    if api_key or _credential_command(body, existing) is None:
        fields["api_key"] = api_key
    if existing is not None:
        for carry in ("token_command", "token_refresh_interval", "token_command_env"):
            if carry not in body and carry in existing:
                fields[carry] = existing[carry]
    return fields


def _build_provider_entry(
    body: dict[str, Any],
    api_key: str,
    base_url: str,
    existing_providers: dict[str, Any],
    resolve_name: str,
) -> dict[str, Any]:
    """Build a provider entry dict from request body, resolving masked keys."""
    if "***" in api_key and resolve_name in existing_providers:
        api_key = existing_providers[resolve_name].get("api_key", api_key)

    entry: dict[str, Any] = {"base_url": base_url}
    entry.update(
        _credential_fields(body, api_key, existing_providers.get(resolve_name))
    )

    provider_type = body.get("type")
    if provider_type:
        entry["type"] = provider_type

    # The service the entry stands on, as the editor's preset picker shows it.
    # Nothing at runtime reads it: each face carries its own shim.  It is kept
    # so the editor reopens on the same preset, and so a face set by hand to
    # another service is still told apart from one that follows.  A shim name,
    # so anything but a non-blank string is dropped rather than stored.
    preset = body.get("preset")
    if isinstance(preset, str) and preset.strip():
        entry["preset"] = preset.strip()

    if "proxy" in body:
        proxy = body["proxy"]
        if proxy:
            entry["proxy"] = proxy

    for tpl_key in ("url_template", "stream_url_template"):
        tpl_val = body.get(tpl_key, "")
        if tpl_val:
            entry[tpl_key] = tpl_val

    # `enabled` is the provider's own switch — the LLM endpoint's on/off, since
    # chat is the provider rather than one of its endpoints.  An explicit
    # `true` drops the key (it is the default); an absent one keeps what is
    # stored, so a card toggle is not undone by an unrelated edit.  The
    # auto-disable marker is never carried over: the entry is rebuilt, and the
    # next reload revalidates it.
    if isinstance(body.get("enabled"), bool):
        if not body["enabled"]:
            entry["enabled"] = False
    elif resolve_name in existing_providers:
        existing_enabled = existing_providers[resolve_name].get("enabled")
        if existing_enabled is not None:
            entry["enabled"] = existing_enabled

    for flag in (
        "supports_custom_tools",
        "hoist_system_messages",
        "preflight_token_count",
    ):
        if flag in body:
            entry[flag] = bool(body[flag])
    if "timeout" in body and body["timeout"] not in (None, ""):
        entry["timeout"] = float(body["timeout"])
    _apply_optional_int(entry, body, "max_tool_description_length")

    # Optional provider-level fields: set when truthy, clear when
    # explicitly sent as empty (so the admin UI can remove them).
    _apply_optional_provider_fields(entry, body)
    # The exchange's declared setup inputs live beside those fields but are a
    # dict, so they get their own validated writer.
    _apply_auth_fields(entry, body, existing_providers.get(resolve_name))
    # The ACP lane's settings are hand-written beside the fields the editor
    # knows; a save keeps them unless it names one itself.
    _apply_acp_fields(entry, body, existing_providers.get(resolve_name))

    return entry


def _resolve_models_path(provider_cfg: dict, config: Any, name: str) -> str | None:
    """Return explicit models_path from provider config or shim, if any."""
    path = provider_cfg.get("models_path")
    if not path:
        from llm_rosetta.api import get_shim

        shim_name = (
            config.provider_shim_names.get(name)
            if hasattr(config, "provider_shim_names")
            else None
        )
        shim = get_shim(shim_name) if shim_name else None
        path = shim.connection.models_path if shim else None
    if path and path.startswith("http://"):
        logger.warning(
            "models_path for %s uses plain HTTP — auth headers will be sent "
            "over an unencrypted connection",
            name,
        )
    return path


def _handle_provider_rename(
    data: dict[str, Any], rename_from: str, name: str
) -> Response | None:
    """Handle provider rename: remove old entry, update model refs."""
    providers = data.get("providers", {})
    if rename_from not in providers:
        return JSONResponse(
            {"error": f"Original provider '{rename_from}' not found"},
            status_code=404,
        )
    if name in providers:
        return JSONResponse(
            {"error": f"Provider '{name}' already exists"},
            status_code=409,
        )
    del providers[rename_from]
    models = data.get("models", {})
    for model_name, model_val in models.items():
        if isinstance(model_val, str) and model_val == rename_from:
            models[model_name] = name
        elif isinstance(model_val, dict) and model_val.get("provider") == rename_from:
            model_val["provider"] = name
    return None

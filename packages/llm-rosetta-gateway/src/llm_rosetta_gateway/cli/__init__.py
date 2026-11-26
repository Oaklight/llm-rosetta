"""CLI entry point and subcommands for llm-rosetta gateway."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any, NoReturn

import asyncio
from collections.abc import Sequence

from llm_rosetta.api import CommandSpec
from llm_rosetta.api import __version__ as _core_version
from llm_rosetta.api import (
    CORE_COMMAND_NAMES,
    register_command,
    resolve_command,
    run_cli,
)

from .. import __version__ as _gateway_version
from ..shims import ShimSourceError

if TYPE_CHECKING:
    from ..shims import ShimLoadReport

from .banner import print_banner, print_shim_sources
from ..config import (
    PATHS_TO_TRY,
    GatewayConfig,
    JsoncConfigIO,
    discover_config,
    load_config,
    load_config_raw,
    write_config,
)
from ..config.errors import RetiredConfigName
from ..observability.logging import get_logger, setup_logging
from ..config.providers import (
    get_default_api_key_env,
    get_default_base_url,
    known_provider_types,
)
from ..shims.shim_sources import apply_shim_sources, describe_shim_sources

logger = get_logger()

# ---------------------------------------------------------------------------
# Editor helper
# ---------------------------------------------------------------------------


def _open_in_editor(config_path: str | None = None) -> None:
    """Open a config file in the user's preferred editor."""
    paths = [config_path] if config_path else list(PATHS_TO_TRY)

    editors: list[str] = []
    env_editor = os.getenv("EDITOR")
    if env_editor:
        editors.append(env_editor)
    editors += ["notepad"] if os.name == "nt" else ["nano", "vi", "vim"]

    for path in paths:
        if path and os.path.exists(path):
            for editor in editors:
                try:
                    subprocess.run([editor, path], check=True)
                    return
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    print(
                        f"Error: failed to open {editor} for {path}: {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    print("Error: no config file found to edit. Searched:", file=sys.stderr)
    for p in paths:
        print(f"  {p}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE: dict[str, Any] = {
    "providers": {},
    "models": {},
    "server": {"host": "0.0.0.0", "port": 8765},
}


def _load_or_create_config(path: str) -> tuple[dict[str, Any], str]:
    """Load existing config (raw, no env substitution) or create a scaffold."""
    if os.path.isfile(path):
        return load_config_raw(path), path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data: dict[str, Any] = json.loads(json.dumps(_CONFIG_TEMPLATE))
    return data, path


def _write_jsonc(path: str, data: dict[str, Any]) -> None:
    write_config(path, data)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    """Create a template config.jsonc at the XDG default location."""
    config_path = args.config or PATHS_TO_TRY[1]  # XDG: ~/.config/…
    if os.path.isfile(config_path):
        print(f"Config already exists at {config_path}", file=sys.stderr)
        print("Use --edit / -e to modify it, or remove it first.", file=sys.stderr)
        sys.exit(1)

    template = {
        "providers": {
            "chat_completions": {
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
            },
            "messages": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "base_url": "https://api.anthropic.com",
            },
            "google_generate": {
                "api_key": "${GOOGLE_API_KEY}",
                "base_url": "https://generativelanguage.googleapis.com",
            },
        },
        "models": {
            "gpt-4o": "chat_completions",
            "claude-sonnet-4-20250514": "messages",
            "gemini-2.0-flash": "google_generate",
        },
        "server": {"host": "0.0.0.0", "port": 8765},
    }

    _write_jsonc(config_path, template)
    print(f"Created config at {config_path}")
    print("Edit it to add your API keys, then run: llm-rosetta-gateway")


def _cmd_add_provider(args: argparse.Namespace) -> None:
    config_path = discover_config(args.config) or PATHS_TO_TRY[0]
    data, path = _load_or_create_config(config_path)

    name: str = args.name
    # A vendor shim (reachable under a vendor spelling like ``anthropic`` or
    # ``openai``) supplies a connection default; a bare base format is
    # vendor-neutral, so its base_url must be given explicitly.
    from llm_rosetta.api import get_shim

    shim = get_shim(name)
    env = (shim.connection.api_key_env if shim else None) or get_default_api_key_env(
        name
    )
    default_key = f"${{{env or f'{name.upper()}_API_KEY'}}}"
    default_url = (shim.connection.base_url if shim else None) or get_default_base_url(
        name
    )

    # api_key: CLI flag > interactive > auto-default
    api_key: str = args.api_key or ""
    if not api_key:
        if sys.stdin.isatty():
            api_key = input(
                f"API key env placeholder for '{name}' [{default_key}]: "
            ).strip()
        if not api_key:
            api_key = default_key

    # base_url: CLI flag > interactive > auto-default
    base_url: str = args.base_url or ""
    if not base_url:
        if default_url:
            base_url = default_url  # known provider — use default silently
        elif sys.stdin.isatty():
            base_url = input("Base URL (required): ").strip()
    if not base_url:
        print(
            "Error: --base-url is required for non-standard providers.", file=sys.stderr
        )
        sys.exit(1)

    data.setdefault("providers", {})[name] = {"api_key": api_key, "base_url": base_url}
    _write_jsonc(path, data)
    print(f"Added provider '{name}' to {path}")


def _cmd_add_model(args: argparse.Namespace) -> None:
    config_path = discover_config(args.config) or PATHS_TO_TRY[0]
    data, path = _load_or_create_config(config_path)

    model_name: str = args.name
    providers = data.get("providers", {})
    provider: str = args.provider or ""
    if not provider:
        if providers:
            choices = list(providers.keys())
            print(f"Available providers: {', '.join(choices)}")
            provider = input(f"Provider for '{model_name}': ").strip()
        else:
            provider = input(f"Provider for '{model_name}': ").strip()
    if not provider:
        print("Error: provider is required.", file=sys.stderr)
        sys.exit(1)
    if provider not in providers:
        print(
            f"Warning: provider '{provider}' not yet in config. "
            f"Add it with: llm-rosetta-gateway add provider {provider}",
            file=sys.stderr,
        )

    data.setdefault("models", {})[model_name] = provider
    _write_jsonc(path, data)
    print(f"Added model '{model_name}' -> '{provider}' to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cmd_set_password(args: argparse.Namespace) -> None:
    """Set or change the admin password directly in the config file.

    Intended for forgot-password recovery — does not require the current
    password.  The gateway must be restarted for the change to take effect
    (unless it happens to hot-reload the config).
    """
    import getpass

    config_path = discover_config(args.config)
    if config_path is None or not os.path.isfile(config_path):
        print(f"Config file not found: {config_path or '(auto-detect failed)'}")
        sys.exit(1)

    new_pw = getpass.getpass("New admin password: ")
    if not new_pw:
        print("Password cannot be empty.")
        sys.exit(1)
    if len(new_pw) < 4:
        print("Password must be at least 4 characters.")
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if new_pw != confirm:
        print("Passwords do not match.")
        sys.exit(1)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        print(f"Failed to read config: {exc}")
        sys.exit(1)

    data.setdefault("server", {})["admin_password"] = new_pw

    try:
        write_config(config_path, data)
    except Exception as exc:
        print(f"Failed to write config: {exc}")
        sys.exit(1)

    print(f"Admin password updated in {config_path}")
    print("Restart the gateway for changes to take effect.")


_KNOWN_PROVIDERS = known_provider_types()


def _resolve_data_dir(config_path: str, args: argparse.Namespace) -> str:
    """Resolve the data directory from CLI flag, config file, or default."""
    if getattr(args, "data_dir", None):
        return args.data_dir
    raw = load_config(config_path)
    configured = raw.get("server", {}).get("data_dir")
    if configured:
        if not os.path.isabs(configured):
            return os.path.join(os.path.dirname(config_path), configured)
        return configured
    return os.path.join(os.path.dirname(config_path), "data")


def _cmd_db_cleanup(args: argparse.Namespace) -> None:
    """Run age-based database cleanup."""
    from llm_rosetta_gateway.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    async def _do_cleanup() -> dict:
        pm = await PersistenceManager.create(data_dir)
        try:
            r = await pm.cleanup_by_age(args.max_age_days)
            if r["request_log_deleted"] > 0:
                await pm.set_rebuild_flag()
            return r
        finally:
            await pm.close()

    result = asyncio.run(_do_cleanup())

    total = (
        result["request_log_deleted"]
        + result["error_dumps_deleted"]
        + result["dump_bodies_deleted"]
    )
    if total == 0:
        print(f"Nothing to clean up — no records older than {args.max_age_days} days.")
        return

    freed_mb = result["freed_bytes"] / (1024 * 1024)
    print(f"Cleaned up records older than {args.max_age_days} days:")
    print(f"  Request logs deleted:  {result['request_log_deleted']}")
    print(f"  Error dumps deleted:   {result['error_dumps_deleted']}")
    print(f"  Dump bodies deleted:   {result['dump_bodies_deleted']}")
    print(f"  Space freed:           {freed_mb:.1f} MB")
    print(
        f"  DB size:               {result['size_before'] / (1024 * 1024):.1f} MB → {result['size_after'] / (1024 * 1024):.1f} MB"
    )


def _cmd_db_cleanup_logs(args: argparse.Namespace) -> None:
    """Run age-based request log cleanup."""
    from llm_rosetta_gateway.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    async def _do_cleanup_logs() -> dict:
        pm = await PersistenceManager.create(data_dir)
        try:
            r = await pm.cleanup_logs_by_age(args.max_age_days)
            if r["deleted"] > 0:
                await pm.set_rebuild_flag()
            return r
        finally:
            await pm.close()

    result = asyncio.run(_do_cleanup_logs())

    if result["deleted"] == 0:
        print(f"Nothing to clean up — no logs older than {args.max_age_days} days.")
        return

    freed_mb = result["freed_bytes"] / (1024 * 1024)
    print(f"Cleaned up logs older than {args.max_age_days} days:")
    print(f"  Log entries deleted:  {result['deleted']}")
    print(f"  Space freed:          {freed_mb:.1f} MB")
    print(
        f"  DB size:              {result['size_before'] / (1024 * 1024):.1f} MB"
        f" → {result['size_after'] / (1024 * 1024):.1f} MB"
    )


def _cmd_db_cleanup_errors(args: argparse.Namespace) -> None:
    """Run age-based error dump cleanup."""
    from llm_rosetta_gateway.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    async def _do_cleanup_errors() -> dict:
        pm = await PersistenceManager.create(data_dir)
        try:
            return await pm.cleanup_error_dumps_by_age(args.max_age_days)
        finally:
            await pm.close()

    result = asyncio.run(_do_cleanup_errors())

    total = result["error_dumps_deleted"] + result["dump_bodies_deleted"]
    if total == 0:
        print(
            f"Nothing to clean up — no error dumps older than {args.max_age_days} days."
        )
        return

    freed_mb = result["freed_bytes"] / (1024 * 1024)
    print(f"Cleaned up error dumps older than {args.max_age_days} days:")
    print(f"  Error dumps deleted:  {result['error_dumps_deleted']}")
    print(f"  Dump bodies deleted:  {result['dump_bodies_deleted']}")
    print(f"  Space freed:          {freed_mb:.1f} MB")
    print(
        f"  DB size:              {result['size_before'] / (1024 * 1024):.1f} MB"
        f" → {result['size_after'] / (1024 * 1024):.1f} MB"
    )


def _cmd_db_export_errors(args: argparse.Namespace) -> None:
    """Export error dumps to a tar.gz file."""
    from llm_rosetta_gateway.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    start = f"{args.start}T00:00:00Z" if args.start else None
    end = f"{args.end}T23:59:59Z" if args.end else None

    output = args.output or "error-dumps.tar.gz"
    partial = f"{output}.part"
    output_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(output_dir):
        print(f"Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    async def _do_export() -> int:
        pm = await PersistenceManager.create(data_dir)
        written = 0
        try:
            # Written as it is read, so the archive of a large store never
            # sits in memory — the same reason the admin route streams it.
            # The archive lands in *output* only once its last chunk is
            # written, so a failed export leaves no half archive behind.
            with open(partial, "wb") as f:
                async for chunk in pm.iter_export_error_dumps(start=start, end=end):
                    f.write(chunk)
                    written += len(chunk)
        finally:
            await pm.close()
        return written

    try:
        written = asyncio.run(_do_export())
    except BaseException:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise
    os.replace(partial, output)
    print(f"Exported {written} bytes to {output}")


def _write_record_file(path: str, text: str) -> None:
    """Write the record so only its owner can read it.

    The record *is* the credential — an access token, a refresh token and an
    ID token — so it is written ``0600`` through a temporary file in the same
    directory, and never left half-written.

    The temporary name must be unpredictable: a fixed one (``.<name>.tmp``)
    is derivable from the path the operator typed, and ``--out /tmp/…`` is
    exactly the kind of path used on a shared host.  Someone who can write in
    that directory could then plant a ``0644`` file at the name — which
    ``O_CREAT`` does not re-mode — or a symlink, which ``O_CREAT`` follows, and
    the full credential would land where they chose.  ``mkstemp`` gives a
    random name created ``O_CREAT|O_EXCL`` at ``0600``, so there is nothing to
    inherit and nothing to follow; ``os.replace`` then publishes it atomically.
    """
    import tempfile

    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{base}.", suffix=".tmp")
    try:
        # mkstemp already asks for 0600; ask again rather than trust a umask.
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
            # The rename can otherwise reach disk before the data, leaving a
            # correctly named and moded *empty* record after a crash.
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _cmd_credential(args: argparse.Namespace) -> None:
    """Run a shim's interactive sign-in and emit its credential record.

    The record has to be produced where a browser exists, and the gateway host
    is not always that machine.  This command runs the exchange's ``login()``
    here, prints what it produced, and leaves handing it to the gateway to the
    operator — the admin API's ``PUT …/credential``.  Nothing is copied to the
    gateway by this command.
    """
    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    try:
        config, report = _load_startup_config(config_path, args)
    except OSError as exc:
        # Discovery found a file (or --config named one) that cannot be read.
        # That is a typing mistake, not a crash report.
        print(f"Cannot read config file {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except RetiredConfigName as exc:
        _exit_with_config_error(exc)
    _report_shim_sources(report)

    name: str = args.provider
    pinfo = getattr(config, "providers", {}).get(name)
    if pinfo is None:
        known = ", ".join(sorted(getattr(config, "providers", {}))) or "(none)"
        print(
            f"Unknown provider '{name}'. Configured providers: {known}",
            file=sys.stderr,
        )
        sys.exit(1)

    exchange = getattr(pinfo, "auth_exchange", None)
    if exchange is None:
        print(
            f"Provider '{name}' mints no credential of its own: its shim ships "
            "no auth.py, and a static key or token_command owns the credential.",
            file=sys.stderr,
        )
        sys.exit(1)

    can_login = getattr(exchange, "can_login", None)
    if can_login is not None and not can_login():
        print(
            f"Provider '{name}' has no interactive sign-in: its exchange does "
            "not implement login().",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        record = exchange.login(pinfo.auth_context)
    except Exception as exc:  # noqa: BLE001 - report the exchange's own words
        # The contract says the record must not appear in the exception, but a
        # vendor's error body is not under our control and this message is
        # printed.  Scrub what credential shapes we know on the way out.
        from ..observability.sanitize import scrub_credential_patterns

        print(f"Sign-in failed: {scrub_credential_patterns(str(exc))}", file=sys.stderr)
        sys.exit(1)

    text = json.dumps(record, indent=2, sort_keys=True)
    if getattr(args, "out", None):
        _write_record_file(args.out, text)
        print(f"Credential record written to {args.out} (mode 0600).")
    else:
        print(text)

    print(
        "Hand this record to the gateway now: admin panel → Providers → "
        f"'{name}' → Credential, or PUT /admin/api/config/providers/{name}"
        "/credential. It holds live tokens — do not paste it anywhere else, "
        "and do not leave it lying around: the ID token is verified on arrival.",
        file=sys.stderr,
    )
    state_dir = getattr(pinfo, "auth_state_dir", None)
    if state_dir:
        print(
            f"The sign-in also stored a copy in this machine's record directory "
            f"({state_dir}) — delete it there if the gateway is elsewhere.",
            file=sys.stderr,
        )


def _apply_shim_sources_or_exit(
    raw_config: dict[str, Any], config_path: str
) -> ShimLoadReport | None:
    """Register configured external shim sources, before the config is built.

    Must run before ``GatewayConfig``, whose provider-type resolution reads the
    shim registry.  With no ``shims`` section there is nothing to apply, and
    skipping entirely leaves shims registered in-process via ``register_shim``
    untouched — unless a provider names a shim whose vendor moved out of core
    (#1660): the gateway installs that vendor without a section.  A malformed
    section raises ``ValueError``; a ``strict`` source failure raises
    ``ShimSourceError`` — either way the process exits.
    """
    from ..config.settings import configure_extensions_for
    from ..shims.moved import moved_references

    # Where extensions live is chosen once, before the first shim load.  An
    # extensions directory that already holds extensions needs a rebuild even
    # with no ``shims`` section, so they are loaded.
    server = raw_config.get("server")
    extensions = configure_extensions_for(
        server.get("data_dir") if isinstance(server, dict) else None, config_path
    )
    has_extensions = extensions is not None and os.path.isdir(extensions)
    if (
        not raw_config.get("shims")
        and not moved_references(raw_config)
        and not has_extensions
    ):
        return None
    try:
        return apply_shim_sources(raw_config, config_path)
    except (ShimSourceError, ValueError) as exc:
        logging.basicConfig(level=logging.ERROR)
        logger.error("Shim source error: %s", exc)
        sys.exit(1)


def _report_shim_sources(report: ShimLoadReport | None) -> None:
    """Log the shim-source outcome (called once a log handler is attached).

    Each remote source gets one line naming the leg it loaded from, so an
    operator can tell a primary fetch from a mirror that took over.  A source
    that failed because nothing could be reached gets the remedies next to it.
    """
    if report is None:
        return
    for status in report.failures():
        logger.warning(
            "Shim source failed (%s %s): %s",
            status.kind,
            status.location,
            status.error,
        )
        if status.unreachable:
            logger.warning(
                "  %s is unreachable; declare a second manifest URL for it "
                "(shims.sources[].fallback), install the extension into the "
                "extensions directory by hand, or restore access to the host",
                status.location,
            )
    for status in report.sources:
        if status.kind == "remote" and status.ok:
            logger.info(
                "Shim source %s: loaded via %s%s",
                status.location,
                status.leg or "primary",
                " (cache)" if status.from_cache else "",
            )
    if report.sources:
        loaded = sum(s.shim_count for s in report.sources if s.ok)
        mirror = sum(
            1
            for s in report.sources
            if s.kind == "remote" and s.ok and s.leg == "mirror"
        )
        logger.info(
            "Shim sources: %d external source(s), %d shim(s) loaded%s",
            len(report.sources),
            loaded,
            f", {mirror} via mirror" if mirror else "",
        )


def _exit_with_config_error(exc: Exception) -> NoReturn:
    """Print a config error as one line, without a traceback, and exit 1."""
    print(f"llm-rosetta-gateway: error: {exc}", file=sys.stderr)
    sys.exit(1)


def _load_startup_config(
    config_path: str, args: argparse.Namespace
) -> tuple[GatewayConfig, ShimLoadReport | None]:
    """Read the document this process is about to serve, with the CLI overrides.

    Read through the load-time steps rather than the bare parser: the document
    on disk may have been written by an older release, and serving it unmigrated
    drops whatever a migration would have merged in — for a ``config_version: 0``
    file that is the whole legacy rerank/embedding pool (#1046).  The config
    carries the registry those steps ran, which is what the admin surface lists
    the load-time set from.
    """
    reader = JsoncConfigIO()
    raw_config = reader.load(config_path)

    # CLI --proxy overrides config-level server.proxy
    if args.proxy:
        raw_config.setdefault("server", {})["proxy"] = args.proxy
    if args.data_dir:
        raw_config.setdefault("server", {})["data_dir"] = args.data_dir
    if args.tls_cert:
        raw_config.setdefault("server", {})["tls_cert"] = args.tls_cert
    if args.tls_key:
        raw_config.setdefault("server", {})["tls_key"] = args.tls_key

    shim_source_report = _apply_shim_sources_or_exit(raw_config, config_path)
    # Recover only providers whose standalone shim failed to load.  This runs
    # before identity resolution so healthy providers can still serve repairs.
    from ..ops import OpsContext, OpsProviderValidation

    recovery = OpsProviderValidation(
        OpsContext(config_io=reader, config_path=config_path),
        raw_config,
        report=shim_source_report,
        persist=True,
    )
    recovery_result = asyncio.run(recovery.execute())
    if recovery_result.get("disabled"):
        logger.warning("Provider recovery disabled: %s", recovery_result["disabled"])
    try:
        config = GatewayConfig(
            raw_config, maintenance=reader.last_load_registry, config_path=config_path
        )
    except RetiredConfigName as exc:
        raise exc.located(path=config_path) from None
    # The ops log is not open yet, so the run cannot record itself.  Leave the
    # result on the config; run_gateway writes the row once the log exists
    # (issue #1386: the startup disable was the one missing from the audit).
    config.provider_validation_result = recovery_result
    return config, shim_source_report


_SERVE_PROG = "llm-rosetta-gateway"

#: What each command is, for the dispatcher's help text.  The entry points in
#: pyproject.toml name the same targets; ``main()`` registers these explicitly so
#: a host without package metadata (the Nuitka binary) still finds them.
_OWNER = "llm-rosetta-gateway"
_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "serve",
        "Start the gateway server (the default)",
        "llm_rosetta_gateway.cli:serve_main",
    ),
    CommandSpec(
        "init",
        "Create a template config.jsonc",
        "llm_rosetta_gateway.cli:init_main",
    ),
    CommandSpec(
        "add",
        "Add a provider or model to the config",
        "llm_rosetta_gateway.cli:add_main",
    ),
    CommandSpec(
        "credential",
        "Run a shim's interactive sign-in and print its credential record",
        "llm_rosetta_gateway.cli:credential_main",
    ),
    CommandSpec(
        "password",
        "Set or change the admin password (forgot-password recovery)",
        "llm_rosetta_gateway.cli:password_main",
    ),
    CommandSpec(
        "db",
        "Database maintenance: cleanup and error-dump export",
        "llm_rosetta_gateway.cli:db_main",
    ),
    CommandSpec(
        "usage",
        "Report token usage and cost from the gateway database",
        "llm_rosetta_gateway.observability.usage.cli:main",
    ),
    CommandSpec(
        "migrate",
        "Migrate a config and database off the names 0.14.0 retires",
        "llm_rosetta_gateway.maintenance.migrate_0140:main",
    ),
)

#: ``set-password`` is what 0.13.0 called ``password``.
_ALIASES = {"set-password": "password"}


def register_commands() -> None:
    """Register the gateway's commands with the ``llm-rosetta`` dispatcher.

    Package metadata declares the same commands, but a host that has none (the
    Nuitka binary) relies on this call.  Registering twice is a no-op.
    """
    for spec in _COMMANDS:
        register_command(spec, owner=_OWNER)


def _serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_SERVE_PROG,
        description="llm-rosetta Gateway — cross-provider LLM proxy",
        epilog="commands: "
        + ", ".join(spec.name for spec in _COMMANDS if spec.name != "serve")
        + f"  (see '{_SERVE_PROG} <command> --help')",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to JSONC config file (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=(f"%(prog)s {_gateway_version} (llm-rosetta {_core_version})"),
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner",
    )
    parser.add_argument(
        "--edit",
        "-e",
        action="store_true",
        help="Open the config file in $EDITOR for editing",
    )
    parser.add_argument("--host", default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--socket",
        "-S",
        default=None,
        help="Listen on a Unix domain socket instead of TCP (e.g. /run/user/1000/rosetta.sock)",
    )
    parser.add_argument(
        "--tls-cert",
        default=None,
        help="Path to TLS certificate PEM file (requires --tls-key)",
    )
    parser.add_argument(
        "--tls-key",
        default=None,
        help="Path to TLS private key PEM file (requires --tls-cert)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="HTTP/SOCKS proxy URL for upstream requests (overrides config)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for gateway.db and other persistent data (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging; overrides config and --log-level",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    return parser


def _command_parser(name: str, description: str) -> argparse.ArgumentParser:
    """Return the parser of a gateway command that reads a config file."""
    parser = argparse.ArgumentParser(
        prog=f"llm-rosetta {name}", description=description
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config file (auto-discovered if omitted)",
    )
    return parser


def serve_main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Start the gateway server (the default)."""
    from ..app import create_app

    args = _serve_parser().parse_args(None if argv is None else list(argv))

    # --- edit mode ---
    if args.edit:
        _open_in_editor(args.config)
        return 0

    # --- normal server startup ---
    if not args.no_banner:
        print_banner()

    config_path = discover_config(args.config)
    if config_path is None:
        # Minimal fallback logging so the error is visible before setup_logging
        logging.basicConfig(level=logging.ERROR)
        logger.error(
            "No config file found. Searched:\n  %s\n"
            "Provide one with --config or create a config at one of the above paths.\n"
            "Tip: use 'llm-rosetta-gateway init' to create a template config.",
            "\n  ".join(PATHS_TO_TRY),
        )
        sys.exit(1)

    if not os.path.isfile(config_path):
        logging.basicConfig(level=logging.ERROR)
        logger.error(
            "Config file not found: %s\n"
            "Tip: use 'llm-rosetta-gateway init --config %s' to create one.",
            config_path,
            config_path,
        )
        sys.exit(1)

    try:
        config, shim_source_report = _load_startup_config(config_path, args)
    except RetiredConfigName as exc:
        _exit_with_config_error(exc)

    # Resolve verbosity: CLI --verbose wins, then config/env, then --log-level
    verbose = args.verbose or config.verbose

    setup_logging(
        verbose=verbose,
        log_bodies=config.log_bodies,
        log_format=config.log_format,
    )

    # Report the shim-source outcome now that a handler is attached — the
    # sources are applied before setup_logging (they must be registered before
    # GatewayConfig), so their own log records would otherwise be dropped.
    _report_shim_sources(shim_source_report)
    if not args.no_banner:
        print_shim_sources(describe_shim_sources(shim_source_report))

    host = args.host or config.host
    port = args.port or config.port
    socket_path = args.socket or config.socket

    # Build TLS context when cert/key are configured
    ssl_context = None
    if config.tls_cert and config.tls_key:
        import ssl

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(config.tls_cert, config.tls_key)

    logger.info("Config loaded from %s", config_path)
    if socket_path:
        scheme = "https+unix" if ssl_context else "unix"
        logger.info("Starting llm-rosetta gateway on %s:%s", scheme, socket_path)
    else:
        scheme = "https" if ssl_context else "http"
        logger.info("Starting llm-rosetta gateway on %s://%s:%d", scheme, host, port)
    logger.info("Configured providers: %s", list(config.providers.keys()))
    logger.info("Configured models: %s", list(config.models.keys()))
    if verbose:
        logger.info("Verbose logging enabled (DEBUG level)")
    if config.log_bodies:
        logger.info("Request/response body logging enabled")

    async def _run() -> None:
        app = await create_app(
            config, config_path=config_path, shim_source_report=shim_source_report
        )
        await run_gateway(app, host, port, socket=socket_path, ssl_context=ssl_context)

    from ..app import run_gateway

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    return 0


def init_main(argv: Sequence[str] | None = None) -> int:
    """Create a template config.jsonc"""
    parser = _command_parser("init", "Create a template config.jsonc")
    args = parser.parse_args(None if argv is None else list(argv))
    _cmd_init(args)
    return 0


def add_main(argv: Sequence[str] | None = None) -> int:
    """Add a provider or model to the config"""
    parser = _command_parser("add", "Add a provider or model to the config")
    add_sub = parser.add_subparsers(dest="add_type")

    _provider_list = ", ".join(_KNOWN_PROVIDERS)
    prov_parser = add_sub.add_parser("provider", help="Add a provider entry")
    prov_parser.add_argument(
        "name",
        help=f"Provider type. Built-in types: {_provider_list}",
    )
    prov_parser.add_argument(
        "--api-key", default=None, help="API key or ${ENV_VAR} placeholder"
    )
    prov_parser.add_argument("--base-url", default=None, help="Provider base URL")

    model_parser = add_sub.add_parser("model", help="Add a model routing entry")
    model_parser.add_argument("name", help="Model name (e.g. gpt-4o)")
    model_parser.add_argument("--provider", default=None, help="Target provider name")

    args = parser.parse_args(None if argv is None else list(argv))
    if args.add_type == "provider":
        _cmd_add_provider(args)
    elif args.add_type == "model":
        _cmd_add_model(args)
    else:
        parser.print_help()
    return 0


def credential_main(argv: Sequence[str] | None = None) -> int:
    """Run a shim's interactive sign-in and print its credential record"""
    parser = _command_parser(
        "credential", "Run a shim's interactive sign-in and print its credential record"
    )
    parser.add_argument("provider", help="Configured provider name")
    parser.add_argument(
        "--out",
        default=None,
        help="Write the record to this file (mode 0600) instead of stdout",
    )
    # The sign-in loads the same document the server would, with the same
    # overrides, so these flags exist here too.
    parser.add_argument("--proxy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tls-cert", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tls-key", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(None if argv is None else list(argv))
    _cmd_credential(args)
    return 0


def password_main(argv: Sequence[str] | None = None) -> int:
    """Set or change the admin password (forgot-password recovery)"""
    parser = argparse.ArgumentParser(
        prog="llm-rosetta password",
        description="Set or change admin password (for forgot-password recovery)",
    )
    parser.add_argument(
        "config_path", nargs="?", default=None, help="Path to config file"
    )
    parser.add_argument(
        "--config", "-c", dest="config_flag", default=None, help="Path to config file"
    )
    args = parser.parse_args(None if argv is None else list(argv))
    args.config = args.config_flag or args.config_path
    _cmd_set_password(args)
    return 0


def db_main(argv: Sequence[str] | None = None) -> int:
    """Database maintenance: cleanup and error-dump export"""
    parser = _command_parser("db", "Database maintenance commands")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing gateway.db (overrides config)",
    )
    db_sub = parser.add_subparsers(dest="db_type")
    cleanup_parser = db_sub.add_parser(
        "cleanup", help="Delete all records older than max-age-days and vacuum"
    )
    cleanup_parser.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Delete records older than this many days (default: 90)",
    )
    cl_parser = db_sub.add_parser(
        "cleanup-logs", help="Delete request logs older than max-age-days"
    )
    cl_parser.add_argument(
        "--max-age-days", type=int, default=90, help="Max age in days (default: 90)"
    )
    ce_parser = db_sub.add_parser(
        "cleanup-errors", help="Delete error dumps older than max-age-days"
    )
    ce_parser.add_argument(
        "--max-age-days", type=int, default=90, help="Max age in days (default: 90)"
    )
    export_parser = db_sub.add_parser(
        "export-errors", help="Export error dumps to a tar.gz file"
    )
    export_parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    export_parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    export_parser.add_argument(
        "-o", "--output", default=None, help="Output file (default: error-dumps.tar.gz)"
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if args.db_type == "cleanup":
        _cmd_db_cleanup(args)
    elif args.db_type == "cleanup-logs":
        _cmd_db_cleanup_logs(args)
    elif args.db_type == "cleanup-errors":
        _cmd_db_cleanup_errors(args)
    elif args.db_type == "export-errors":
        _cmd_db_export_errors(args)
    else:
        parser.print_help()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: Server options that take a value, so the token after one is not a command.
_VALUE_OPTIONS = frozenset(
    {
        "--config",
        "-c",
        "--host",
        "--port",
        "--socket",
        "-S",
        "--tls-cert",
        "--tls-key",
        "--proxy",
        "--data-dir",
        "--log-level",
    }
)

#: The options that came before a command and still reach it, per command.
_CARRIED: dict[str, tuple[str, ...]] = {
    "credential": ("config", "proxy", "data_dir", "tls_cert", "tls_key"),
    "db": ("config", "data_dir"),
}


def _first_positional(argv: Sequence[str]) -> int | None:
    """Return the index of the first token that is neither option nor value."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-"):
            i += 2 if token in _VALUE_OPTIONS else 1
            continue
        return i
    return None


def _carry(prefix: Sequence[str], command: str) -> list[str]:
    """Parse the options before a command and forward the ones it reads."""
    parsed = _serve_parser().parse_args(list(prefix))
    forwarded: list[str] = []
    for name in _CARRIED.get(command, ("config",)):
        value = getattr(parsed, name)
        if value:
            forwarded += ["--" + name.replace("_", "-"), value]
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``llm-rosetta-gateway``: a subcommand, or the server when none is named.

    Equivalent to ``llm-rosetta <command>`` and ``llm-rosetta serve``: an
    argument list that is empty or starts with an option starts the server.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns:
        The exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    register_commands()
    at = _first_positional(args)
    if at is None:
        return int(serve_main(args) or 0)
    name = _ALIASES.get(args[at], args[at])
    if name in CORE_COMMAND_NAMES:
        # ``ext`` and ``help`` belong to the core dispatcher.  The gateway
        # binary has no ``llm-rosetta`` script beside it, so this is how an
        # operator without Python or pip installs an extension; the gateway's
        # own commands are registered above, so ``help`` lists them too.
        if at != 0:
            _serve_parser().error(
                f"options before {name!r} are not accepted: {' '.join(args[:at])}"
            )
        return int(run_cli(args) or 0)
    if name not in {spec.name for spec in _COMMANDS}:
        # Not one of the gateway's commands: an installed extension may own
        # the name (``llm-rosetta-gateway argo tunnel``), and the dispatcher
        # also words the refusal when nothing does.
        if at != 0:
            _serve_parser().error(
                f"options before {args[at]!r} are not accepted: {' '.join(args[:at])}"
            )
        return int(run_cli(args) or 0)
    if name == "serve":
        return int(serve_main(args[:at] + args[at + 1 :]) or 0)
    forwarded = _carry(args[:at], name)
    return int(resolve_command(name)(forwarded + args[at + 1 :]) or 0)

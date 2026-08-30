"""CLI entry point and subcommands for llm-rosetta gateway."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any

import asyncio

from llm_rosetta import __version__

from .banner import print_banner
from .config import (
    PATHS_TO_TRY,
    GatewayConfig,
    discover_config,
    load_config,
    load_config_raw,
    write_config,
)
from .logging import get_logger, setup_logging
from .providers import (
    get_default_api_key_env,
    get_default_base_url,
    known_provider_types,
)

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
            "openai_chat": {
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
            },
            "anthropic": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "base_url": "https://api.anthropic.com",
            },
            "google": {
                "api_key": "${GOOGLE_API_KEY}",
                "base_url": "https://generativelanguage.googleapis.com",
            },
        },
        "models": {
            "gpt-4o": "openai_chat",
            "claude-sonnet-4-20250514": "anthropic",
            "gemini-2.0-flash": "google",
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
    default_key = f"${{{get_default_api_key_env(name)}}}"
    default_url = get_default_base_url(name)

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
        return configured
    return os.path.join(os.path.dirname(config_path), "data")


def _cmd_db_cleanup(args: argparse.Namespace) -> None:
    """Run age-based database cleanup."""
    from llm_rosetta.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    pm = PersistenceManager(data_dir)
    result = pm.cleanup_by_age(args.max_age_days)
    pm.close()

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
    from llm_rosetta.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    pm = PersistenceManager(data_dir)
    result = pm.cleanup_logs_by_age(args.max_age_days)
    pm.close()

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
    from llm_rosetta.observability import PersistenceManager

    config_path = discover_config(args.config)
    if config_path is None:
        print("No config file found. Use --config to specify one.", file=sys.stderr)
        sys.exit(1)

    data_dir = _resolve_data_dir(config_path, args)
    db_path = os.path.join(data_dir, "gateway.db")
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    pm = PersistenceManager(data_dir)
    result = pm.cleanup_error_dumps_by_age(args.max_age_days)
    pm.close()

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
    from llm_rosetta.observability import PersistenceManager

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

    pm = PersistenceManager(data_dir)
    data = pm.export_error_dumps(start=start, end=end)
    pm.close()

    output = args.output or "error-dumps.tar.gz"
    with open(output, "wb") as f:
        f.write(data)
    print(f"Exported {len(data)} bytes to {output}")


def _dispatch_subcommand(args: argparse.Namespace, sub: Any) -> bool:
    """Handle subcommands; return True if one matched."""
    if args.command == "init":
        _cmd_init(args)
        return True
    if args.command == "add":
        if args.add_type == "provider":
            _cmd_add_provider(args)
        elif args.add_type == "model":
            _cmd_add_model(args)
        else:
            sub.choices["add"].print_help()
        return True
    if args.command == "set-password":
        _cmd_set_password(args)
        return True
    if args.command == "db":
        if args.db_type == "cleanup":
            _cmd_db_cleanup(args)
        elif args.db_type == "cleanup-logs":
            _cmd_db_cleanup_logs(args)
        elif args.db_type == "cleanup-errors":
            _cmd_db_cleanup_errors(args)
        elif args.db_type == "export-errors":
            _cmd_db_export_errors(args)
        else:
            sub.choices["db"].print_help()
        return True
    return False


def main() -> None:
    """Parse CLI arguments and either run a subcommand or start the server."""
    from .app import create_app

    parser = argparse.ArgumentParser(
        description="llm-rosetta Gateway — cross-provider LLM proxy",
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
        version=f"%(prog)s {__version__}",
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

    # ``init`` subcommand
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "init", help="Create a template config.jsonc at ~/.config/llm-rosetta-gateway/"
    )

    # ``add`` subcommands
    add_parser = sub.add_parser("add", help="Add a provider or model to the config")
    add_sub = add_parser.add_subparsers(dest="add_type")

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

    # ``set-password`` subcommand
    pw_parser = sub.add_parser(
        "set-password",
        help="Set or change admin password (for forgot-password recovery)",
    )
    pw_parser.add_argument(
        "config", nargs="?", default=None, help="Path to config file"
    )

    # ``db`` subcommands
    db_parser = sub.add_parser("db", help="Database maintenance commands")
    db_parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing gateway.db (overrides config)",
    )
    db_sub = db_parser.add_subparsers(dest="db_type")
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

    args = parser.parse_args()

    # --- edit mode ---
    if args.edit:
        _open_in_editor(args.config)
        return

    # --- subcommands ---
    if _dispatch_subcommand(args, sub):
        return

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

    raw_config = load_config(config_path)

    # CLI --proxy overrides config-level server.proxy
    if args.proxy:
        raw_config.setdefault("server", {})["proxy"] = args.proxy
    if args.data_dir:
        raw_config.setdefault("server", {})["data_dir"] = args.data_dir

    config = GatewayConfig(raw_config)

    # Resolve verbosity: CLI --verbose wins, then config/env, then --log-level
    verbose = args.verbose or config.verbose

    setup_logging(
        verbose=verbose,
        log_bodies=config.log_bodies,
        log_format=config.log_format,
    )

    host = args.host or config.host
    port = args.port or config.port
    socket_path = args.socket or config.socket

    logger.info("Config loaded from %s", config_path)
    if socket_path:
        logger.info("Starting llm-rosetta gateway on unix:%s", socket_path)
    else:
        logger.info("Starting llm-rosetta gateway on %s:%d", host, port)
    logger.info("Configured providers: %s", list(config.providers.keys()))
    logger.info("Configured models: %s", list(config.models.keys()))
    if verbose:
        logger.info("Verbose logging enabled (DEBUG level)")
    if config.log_bodies:
        logger.info("Request/response body logging enabled")

    app = create_app(config, config_path=config_path)

    from .app import run_gateway

    try:
        asyncio.run(run_gateway(app, host, port, socket=socket_path))
    except KeyboardInterrupt:
        pass

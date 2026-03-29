"""Logging utilities for llm-rosetta gateway.

Provides colorized, loguru-style output with configurable request/response body
logging, truncation, and sanitization.  Ported from argo-proxy's logger module.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------


class Colors:
    """ANSI colour codes for terminal colourisation."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# Level-specific colours (matching loguru style)
LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Colors.BLUE,
    logging.INFO: Colors.BRIGHT_WHITE,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
}

LEVEL_NAME_COLORS: dict[int, str] = {
    logging.DEBUG: Colors.CYAN,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
}

LEVEL_NAMES: dict[int, str] = {
    logging.DEBUG: "DEBUG   ",
    logging.INFO: "INFO    ",
    logging.WARNING: "WARNING ",
    logging.ERROR: "ERROR   ",
    logging.CRITICAL: "CRITICAL",
}


# ---------------------------------------------------------------------------
# Colour detection
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    """Check if the terminal supports colour output."""
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stderr, "isatty"):
        return False
    if not sys.stderr.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False
    return True


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """Loguru-style coloured formatter: ``YYYY-MM-DD HH:MM:SS.mmm | LEVEL | msg``."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        timestamp = self.formatTime(record, self.datefmt)
        level_name = LEVEL_NAMES.get(record.levelno, "UNKNOWN ")
        level_name_color = LEVEL_NAME_COLORS.get(record.levelno, Colors.WHITE)
        message_color = LEVEL_COLORS.get(record.levelno, Colors.WHITE)

        if self.use_colors:
            formatted = (
                f"{Colors.GREEN}{timestamp}{Colors.RESET} | "
                f"{level_name_color}{Colors.BOLD}{level_name}{Colors.RESET} | "
                f"{message_color}{record.getMessage()}{Colors.RESET}"
            )
        else:
            formatted = f"{timestamp} | {level_name} | {record.getMessage()}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                if self.use_colors:
                    formatted += f"\n{Colors.RED}{record.exc_text}{Colors.RESET}"
                else:
                    formatted += f"\n{record.exc_text}"

        return formatted


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

_handler: logging.Handler | None = None
_logger: logging.Logger = logging.getLogger("llm-rosetta-gateway")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

# Whether body logging is enabled (set by ``setup_logging``)
_log_bodies: bool = False


def get_logger() -> logging.Logger:
    """Return the gateway logger instance."""
    return _logger


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def setup_logging(
    verbose: bool = False,
    use_colors: bool = True,
    log_bodies: bool = False,
) -> logging.Logger:
    """Configure the gateway logger.

    Args:
        verbose: If *True*, set handler level to DEBUG; otherwise INFO.
        use_colors: Whether to use ANSI colours in output.
        log_bodies: If *True*, enable request/response body logging at DEBUG level.

    Returns:
        The configured logger.
    """
    global _handler, _log_bodies
    _log_bodies = log_bodies

    logger = get_logger()

    # Remove existing handler if present
    if _handler is not None:
        logger.removeHandler(_handler)

    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = ColoredFormatter(
        datefmt="%Y-%m-%d %H:%M:%S.%f",
        use_colors=use_colors,
    )

    # Override formatTime to include milliseconds
    import datetime
    import types

    def _format_time(
        self: logging.Formatter, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        ct = datetime.datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(record.msecs):03d}"

    formatter.formatTime = types.MethodType(_format_time, formatter)  # type: ignore[assignment]

    _handler.setFormatter(formatter)
    logger.addHandler(_handler)

    return logger


# ---------------------------------------------------------------------------
# String / base64 truncation
# ---------------------------------------------------------------------------


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate *s* to *max_length*, appending a char-count suffix."""
    if len(s) <= max_length:
        return s
    remaining = len(s) - max_length
    return f"{s[:max_length]}{suffix}[{remaining} more chars]"


def truncate_base64(data_url: str, max_length: int = 100) -> str:
    """Truncate base64 data-URLs for cleaner logging."""
    if not data_url.startswith("data:"):
        return data_url
    if ";base64," in data_url:
        header, base64_data = data_url.split(";base64,", 1)
        if len(base64_data) > max_length:
            truncated = base64_data[:max_length]
            remaining_chars = len(base64_data) - max_length
            return f"{header};base64,{truncated}...[{remaining_chars} more chars]"
    return data_url


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------


def sanitize_request_data(
    data: dict[str, Any],
    *,
    max_base64_length: int = 100,
    max_content_length: int = 500,
    max_tool_desc_length: int = 100,
    truncate_tools: bool = True,
    truncate_messages: bool = True,
) -> dict[str, Any]:
    """Deep-copy and truncate long content for logging."""
    sanitized = copy.deepcopy(data)

    # --- messages ---
    if (
        truncate_messages
        and "messages" in sanitized
        and isinstance(sanitized["messages"], list)
    ):
        for message in sanitized["messages"]:
            if not isinstance(message, dict) or "content" not in message:
                continue
            content = message["content"]

            if isinstance(content, str) and len(content) > max_content_length:
                message["content"] = truncate_string(content, max_content_length)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    # Image URLs
                    if (
                        part.get("type") == "image_url"
                        and "image_url" in part
                        and "url" in part["image_url"]
                    ):
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            part["image_url"]["url"] = truncate_base64(
                                url, max_base64_length
                            )
                    # Text content
                    elif (
                        part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        and len(part["text"]) > max_content_length
                    ):
                        part["text"] = truncate_string(part["text"], max_content_length)

    # --- tools ---
    if truncate_tools and "tools" in sanitized and isinstance(sanitized["tools"], list):
        tool_count = len(sanitized["tools"])
        sanitized["tools"] = f"[{tool_count} tools defined - truncated for logging]"

    return sanitized


# ---------------------------------------------------------------------------
# Request summary
# ---------------------------------------------------------------------------


def create_request_summary(data: dict[str, Any]) -> str:
    """One-line summary of a request body."""
    parts: list[str] = []
    if "model" in data:
        parts.append(f"model={data['model']}")
    if "messages" in data and isinstance(data["messages"], list):
        parts.append(f"messages={len(data['messages'])}")
    if "tools" in data and isinstance(data["tools"], list):
        parts.append(f"tools={len(data['tools'])}")
    if "stream" in data:
        parts.append(f"stream={data['stream']}")
    if "max_tokens" in data:
        parts.append(f"max_tokens={data['max_tokens']}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Visual separator
# ---------------------------------------------------------------------------


def _make_bar(message: str = "", bar_length: int = 60) -> str:
    message = message.strip()
    if message:
        message = f" {message} "
    dash_length = max((bar_length - len(message)) // 2, 2)
    return "-" * dash_length + message + "-" * dash_length


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------


def log_request(
    data: dict[str, Any],
    label: str = "REQUEST",
    *,
    show_summary: bool = True,
    show_full: bool | None = None,
    sanitize: bool = True,
    max_content_length: int = 500,
    truncate_tools: bool = True,
) -> None:
    """Log a request with configurable verbosity.

    *show_full* defaults to the module-level ``_log_bodies`` flag when *None*.
    """
    if show_full is None:
        show_full = _log_bodies

    if show_summary:
        summary = create_request_summary(data)
        _logger.info("[%s] %s", label, summary)

    if show_full:
        log_data = (
            sanitize_request_data(
                data,
                max_content_length=max_content_length,
                truncate_tools=truncate_tools,
            )
            if sanitize
            else data
        )
        _logger.debug(_make_bar(f"[{label}]"))
        _logger.debug(json.dumps(log_data, indent=2, ensure_ascii=False))
        _logger.debug(_make_bar())


def log_original_request(
    data: dict[str, Any],
    *,
    max_content_length: int = 500,
) -> None:
    """Log the original (source-format) request."""
    log_request(
        data,
        label="ORIGINAL REQUEST",
        show_summary=True,
        max_content_length=max_content_length,
    )


def log_converted_request(
    data: dict[str, Any],
    *,
    max_content_length: int = 500,
) -> None:
    """Log the converted (target-format) request."""
    log_request(
        data,
        label="CONVERTED REQUEST",
        show_summary=False,
        max_content_length=max_content_length,
    )


def log_response(
    data: dict[str, Any],
    label: str = "RESPONSE",
    *,
    sanitize: bool = True,
    max_content_length: int = 500,
) -> None:
    """Log a response body (sanitized & truncated at DEBUG level)."""
    if not _log_bodies:
        return

    log_data = (
        sanitize_request_data(
            data,
            max_content_length=max_content_length,
            truncate_tools=True,
        )
        if sanitize
        else data
    )
    _logger.debug(_make_bar(f"[{label}]"))
    _logger.debug(json.dumps(log_data, indent=2, ensure_ascii=False))
    _logger.debug(_make_bar())


def log_stream_summary(
    *,
    model: str,
    duration_s: float,
    chunk_count: int,
) -> None:
    """Log a streaming-session summary (no per-chunk spam)."""
    _logger.info(
        "[STREAM COMPLETE] model=%s chunks=%d duration=%.2fs",
        model,
        chunk_count,
        duration_s,
    )


def log_upstream_error(
    status_code: int,
    error_text: str,
    *,
    endpoint: str = "unknown",
    is_streaming: bool = False,
) -> None:
    """Log an upstream API error in a structured format."""
    request_type = "streaming" if is_streaming else "non-streaming"
    _logger.error(
        "[UPSTREAM ERROR] endpoint=%s, type=%s, status=%d, error=%s",
        endpoint,
        request_type,
        status_code,
        error_text,
    )

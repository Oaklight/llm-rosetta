"""Logging utilities for llm-rosetta gateway.

Provides colorized, loguru-style output with configurable request/response body
logging, truncation, and sanitization.  Ported from argo-proxy's logger module.
"""

from __future__ import annotations

import copy
import datetime
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


# Standard LogRecord attributes to exclude from JSON extra fields
_STANDARD_LOG_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter — one JSON object per line.

    Output schema::

        {"timestamp": "...", "level": "INFO", "logger": "...",
         "message": "...", ...extra_fields}

    Any *extra* dict entries on the :class:`~logging.LogRecord` that are
    **not** standard Python logging attributes are promoted to top-level
    JSON keys.  This lets callers do::

        logger.info("handled request", extra={"request_id": rid, "model": m})

    and get ``{"request_id": "...", "model": "...", ...}`` in the output.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()

        ct = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        entry: dict[str, Any] = {
            "timestamp": ct.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Promote non-standard extra fields.
        # We use a denylist of known stdlib LogRecord attributes rather than
        # an allowlist so that callers can attach arbitrary structured fields
        # via ``extra={...}``.  The denylist is exhaustive for Python 3.12+
        # stdlib attrs; any new stdlib attr added in future versions should
        # be added to ``_STANDARD_LOG_ATTRS``.
        for key, value in record.__dict__.items():
            if key.startswith("_"):  # skip private/internal attrs
                continue
            if key not in _STANDARD_LOG_ATTRS and key not in entry:
                entry[key] = value

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            entry["exception"] = record.exc_text
        if record.stack_info:
            entry["stack_info"] = record.stack_info

        return json.dumps(entry, default=str, ensure_ascii=False)


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

    def formatTime(  # noqa: N802
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Format timestamp with millisecond precision."""
        import datetime

        ct = datetime.datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(record.msecs):03d}"

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
# Resolved log format: "json" or "text" (set by ``setup_logging``)
_log_format: str = "text"


def get_logger() -> logging.Logger:
    """Return the gateway logger instance."""
    return _logger


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def _resolve_log_format(log_format: str) -> str:
    """Resolve ``'auto'`` to ``'json'`` or ``'text'`` based on TTY status."""
    if log_format == "auto":
        return (
            "text"
            if (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
            else "json"
        )
    return log_format


def setup_logging(
    verbose: bool = False,
    use_colors: bool = True,
    log_bodies: bool = False,
    log_format: str = "auto",
) -> logging.Logger:
    """Configure the gateway logger.

    Args:
        verbose: If *True*, set handler level to DEBUG; otherwise INFO.
        use_colors: Whether to use ANSI colours in output.
        log_bodies: If *True*, enable request/response body logging at DEBUG level.
        log_format: ``'json'``, ``'text'``, or ``'auto'`` (default).  When
            ``'auto'``, JSON is used for non-TTY stderr, text for interactive.

    Returns:
        The configured logger.
    """
    global _handler, _log_bodies, _log_format
    _log_bodies = log_bodies
    _log_format = _resolve_log_format(log_format)

    logger = get_logger()

    # Remove existing handler if present
    if _handler is not None:
        logger.removeHandler(_handler)

    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    if _log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = ColoredFormatter(
            datefmt="%Y-%m-%d %H:%M:%S.%f",
            use_colors=use_colors,
        )

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


def _sanitize_content_part(
    part: dict[str, Any],
    *,
    max_base64_length: int,
    max_content_length: int,
) -> None:
    """Truncate a single content part in-place for logging.

    Handles ``image_url`` parts (truncating base64 data-URLs) and
    ``text`` parts (truncating long text content).
    """
    part_type = part.get("type")
    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
            if url.startswith("data:"):
                image_url["url"] = truncate_base64(url, max_base64_length)
    elif part_type == "text":
        text = part.get("text")
        if isinstance(text, str) and len(text) > max_content_length:
            part["text"] = truncate_string(text, max_content_length)


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

    if truncate_messages and isinstance(sanitized.get("messages"), list):
        for message in sanitized["messages"]:
            if not isinstance(message, dict) or "content" not in message:
                continue
            content = message["content"]
            if isinstance(content, str) and len(content) > max_content_length:
                message["content"] = truncate_string(content, max_content_length)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        _sanitize_content_part(
                            part,
                            max_base64_length=max_base64_length,
                            max_content_length=max_content_length,
                        )

    if truncate_tools and isinstance(sanitized.get("tools"), list):
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


def _structured_extra(**kwargs: Any) -> dict[str, Any]:
    """Build an ``extra`` dict for :meth:`logging.Logger.info` & friends.

    Only non-``None`` values are included so JSON output stays clean.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


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
    request_id: str | None = None,
    model: str | None = None,
    source_provider: str | None = None,
    target_provider: str | None = None,
) -> None:
    """Log a request with configurable verbosity.

    *show_full* defaults to the module-level ``_log_bodies`` flag when *None*.
    Structured fields (*request_id*, *model*, etc.) are attached as ``extra``
    for JSON log consumers.
    """
    if show_full is None:
        show_full = _log_bodies

    extra = _structured_extra(
        request_id=request_id,
        model=model,
        source_provider=source_provider,
        target_provider=target_provider,
        label=label,
    )

    if show_summary:
        summary = create_request_summary(data)
        _logger.info("[%s] %s", label, summary, extra=extra)

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
        _logger.debug(_make_bar(f"[{label}]"), extra=extra)
        _logger.debug(json.dumps(log_data, indent=2, ensure_ascii=False), extra=extra)
        _logger.debug(_make_bar(), extra=extra)


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
    request_id: str | None = None,
    model: str | None = None,
    duration_ms: int | None = None,
    status: str | None = None,
) -> None:
    """Log a response body (sanitized & truncated at DEBUG level)."""
    if not _log_bodies:
        return

    extra = _structured_extra(
        request_id=request_id,
        model=model,
        duration_ms=duration_ms,
        status=status,
        label=label,
    )

    log_data = (
        sanitize_request_data(
            data,
            max_content_length=max_content_length,
            truncate_tools=True,
        )
        if sanitize
        else data
    )
    _logger.debug(_make_bar(f"[{label}]"), extra=extra)
    _logger.debug(json.dumps(log_data, indent=2, ensure_ascii=False), extra=extra)
    _logger.debug(_make_bar(), extra=extra)


def log_stream_summary(
    *,
    model: str,
    duration_s: float,
    chunk_count: int,
    request_id: str | None = None,
    source_provider: str | None = None,
    target_provider: str | None = None,
    status: str = "success",
) -> None:
    """Log a streaming-session summary (no per-chunk spam)."""
    extra = _structured_extra(
        request_id=request_id,
        model=model,
        source_provider=source_provider,
        target_provider=target_provider,
        duration_ms=round(duration_s * 1000),
        chunk_count=chunk_count,
        status=status,
    )
    _logger.info(
        "[STREAM COMPLETE] model=%s chunks=%d duration=%.2fs",
        model,
        chunk_count,
        duration_s,
        extra=extra,
    )


def log_upstream_error(
    status_code: int,
    error_text: str,
    *,
    endpoint: str = "unknown",
    is_streaming: bool = False,
    request_id: str | None = None,
    model: str | None = None,
) -> None:
    """Log an upstream API error in a structured format."""
    request_type = "streaming" if is_streaming else "non-streaming"
    extra = _structured_extra(
        request_id=request_id,
        model=model,
        status=status_code,
        endpoint=endpoint,
        request_type=request_type,
    )
    _logger.error(
        "[UPSTREAM ERROR] endpoint=%s, type=%s, status=%d, error=%s",
        endpoint,
        request_type,
        status_code,
        error_text,
        extra=extra,
    )

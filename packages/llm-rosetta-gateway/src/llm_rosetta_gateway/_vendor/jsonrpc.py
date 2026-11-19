# /// zerodep
# version = "0.4.0"
# deps = []
# tier = "medium"
# category = "protocol"
# note = "Install/update via `zerodep add jsonrpc`"
# ///

"""JSON-RPC 2.0 -- Zero-dependency Python implementation.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

A pure-stdlib implementation of the JSON-RPC 2.0 specification
(https://www.jsonrpc.org/specification).  Provides core data types, an
exception hierarchy, an auto-incrementing ID generator, a method dispatcher,
and an async transport for newline-delimited JSON streams.

Requires:
    Python >= 3.10, no external packages.

Sections:
    1. Constants          - protocol version and standard error codes
    2. Core Types         - JSONRPCError, JSONRPCRequest, JSONRPCResponse
    3. Exception          - JSONRPCException wrapping JSONRPCError
    4. ID Generation      - thread-safe auto-incrementing counter
    5. Dispatcher         - method routing with streaming support
    6. Async Transport    - newline-delimited JSON-RPC over asyncio streams
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterator,
    Union,
)

__all__ = [
    # constants
    "JSONRPC_VERSION",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    # core types
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    # exception
    "JSONRPCException",
    # id generation
    "next_id",
    # dispatcher
    "JSONRPCDispatcher",
    # transport
    "JSONRPCTransport",
]

__version__ = "0.3.0"

logger = logging.getLogger("jsonrpc")

# ===================================================================
# 1. CONSTANTS
# ===================================================================

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ===================================================================
# 2. CORE TYPES
# ===================================================================


@dataclass
class JSONRPCError:
    """A JSON-RPC 2.0 error object.

    Attributes:
        code: Numeric error code.
        message: Human-readable error message.
        data: Optional additional error data.
    """

    code: int = INTERNAL_ERROR
    message: str = "Internal error"
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JSONRPCError:
        """Deserialize from a dictionary."""
        return cls(
            code=raw["code"],
            message=raw["message"],
            data=raw.get("data"),
        )


@dataclass
class JSONRPCRequest:
    """A JSON-RPC 2.0 request object.

    Attributes:
        method: The RPC method name.
        params: Method parameters.
        id: Request identifier (``None`` for notifications).
        jsonrpc: Protocol version (always ``"2.0"``).
    """

    method: str = ""
    params: dict[str, Any] | None = None
    id: Union[str, int, None] = None
    jsonrpc: str = JSONRPC_VERSION

    @property
    def is_notification(self) -> bool:
        """Whether this request is a notification (no ``id``)."""
        return self.id is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.id is not None:
            d["id"] = self.id
        if self.params is not None:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JSONRPCRequest:
        """Deserialize from a dictionary."""
        return cls(
            method=d.get("method", ""),
            params=d.get("params"),
            id=d.get("id"),
            jsonrpc=d.get("jsonrpc", JSONRPC_VERSION),
        )


@dataclass
class JSONRPCResponse:
    """A JSON-RPC 2.0 response object.

    Exactly one of ``result`` or ``error`` should be set.

    Attributes:
        id: Matching request identifier.
        result: Successful result payload.
        error: Error details on failure.
        jsonrpc: Protocol version (always ``"2.0"``).
    """

    id: Union[str, int, None] = None
    result: Any = None
    error: JSONRPCError | None = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JSONRPCResponse:
        """Deserialize from a dictionary."""
        error_raw = d.get("error")
        return cls(
            id=d.get("id"),
            result=d.get("result"),
            error=JSONRPCError.from_dict(error_raw) if error_raw else None,
            jsonrpc=d.get("jsonrpc", JSONRPC_VERSION),
        )

    @classmethod
    def success(cls, request_id: Any, result: Any) -> JSONRPCResponse:
        """Create a successful response."""
        return cls(id=request_id, result=result)

    @classmethod
    def from_error(cls, request_id: Any, error: Any) -> JSONRPCResponse:
        """Create an error response.

        Args:
            request_id: The id of the original request.
            error: A ``JSONRPCError`` dataclass, a ``JSONRPCException``
                (via its ``.error`` attribute), or any object with
                ``.code``, ``.rpc_message``, and ``.data`` attributes.
        """
        if isinstance(error, JSONRPCError):
            return cls(id=request_id, error=error)
        if isinstance(error, JSONRPCException):
            return cls(id=request_id, error=error.error)
        # Duck-typed protocol error (e.g. A2AError with code/rpc_message/data)
        return cls(
            id=request_id,
            error=JSONRPCError(
                code=getattr(error, "code", INTERNAL_ERROR),
                message=getattr(error, "rpc_message", str(error)),
                data=getattr(error, "data", None),
            ),
        )


# ===================================================================
# 3. EXCEPTION
# ===================================================================


class JSONRPCException(Exception):
    """Exception wrapper around a ``JSONRPCError`` data object.

    Attributes:
        error: The underlying ``JSONRPCError`` instance.
    """

    def __init__(self, error: JSONRPCError) -> None:
        super().__init__(error.message)
        self.error = error


# ===================================================================
# 4. ID GENERATION
# ===================================================================

_counter = itertools.count(1)


def next_id() -> int:
    """Return the next auto-incrementing request id (thread-safe)."""
    return next(_counter)


# ===================================================================
# 5. DISPATCHER
# ===================================================================

MethodHandler = Callable[..., Any]


class JSONRPCDispatcher:
    """Routes JSON-RPC method calls to registered handler functions.

    Example::

        dispatcher = JSONRPCDispatcher()

        @dispatcher.register("SendMessage")
        def handle_send(params):
            ...
            return result_dict
    """

    def __init__(self) -> None:
        self._handlers: dict[str, MethodHandler] = {}

    def register(self, method: str) -> Callable[[MethodHandler], MethodHandler]:
        """Decorator to register a handler for *method*.

        Args:
            method: JSON-RPC method name (e.g. ``"SendMessage"``).

        Returns:
            The original handler function, unmodified.
        """

        def decorator(fn: MethodHandler) -> MethodHandler:
            self._handlers[method] = fn
            return fn

        return decorator

    def dispatch(
        self, request: JSONRPCRequest
    ) -> Union[JSONRPCResponse, Iterator[JSONRPCResponse]]:
        """Dispatch a parsed JSON-RPC request to the appropriate handler.

        Catches ``JSONRPCException`` (and subclasses) raised by handlers and
        converts them to error responses.

        Args:
            request: The parsed JSON-RPC request.

        Returns:
            A single ``JSONRPCResponse`` or, for streaming methods, a
            generator yielding ``JSONRPCResponse`` objects.
        """
        handler = self._handlers.get(request.method)
        if handler is None:
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(
                    code=METHOD_NOT_FOUND,
                    message=f"Method not found: {request.method}",
                ),
            )
        try:
            result = handler(request.params or {})
            if hasattr(result, "__next__"):
                return self._stream_wrap(request.id, result)
            return JSONRPCResponse.success(request.id, result)
        except JSONRPCException as exc:
            return JSONRPCResponse.from_error(request.id, exc)
        except Exception as exc:
            logger.exception("Unhandled error in handler %s", request.method)
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError(code=INTERNAL_ERROR, message=str(exc)),
            )

    @staticmethod
    def _stream_wrap(request_id: Any, gen: Iterator[Any]) -> Iterator[JSONRPCResponse]:
        """Wrap a generator so each yielded value becomes a JSONRPCResponse."""
        try:
            for item in gen:
                yield JSONRPCResponse.success(request_id, item)
        except JSONRPCException as exc:
            yield JSONRPCResponse.from_error(request_id, exc)
        except Exception as exc:
            logger.exception("Unhandled error in streaming handler")
            yield JSONRPCResponse(
                id=request_id,
                error=JSONRPCError(code=INTERNAL_ERROR, message=str(exc)),
            )


# ===================================================================
# 6. ASYNC TRANSPORT
# ===================================================================


class JSONRPCTransport:
    """Async JSON-RPC 2.0 transport over newline-delimited JSON streams.

    Each message is a single JSON object terminated by ``\\n``.
    Messages must not contain embedded newlines.

    Attributes:
        reader: Async stream to read incoming messages from.
        writer: Async stream to write outgoing messages to.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self._closed = False

    async def read_message(self) -> dict[str, Any] | None:
        """Read the next JSON-RPC message.

        Returns:
            Parsed JSON object, or ``None`` on EOF.
        """
        while True:
            line = await self.reader.readline()
            if not line:
                return None
            text = line.decode("utf-8").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed JSON line: %s", text[:120])

    async def write_message(self, msg: dict[str, Any]) -> None:
        """Write a JSON-RPC message followed by a newline.

        Args:
            msg: JSON-serializable dictionary to send.

        Raises:
            ValueError: If *msg* carries a non-finite float. ``NaN`` and
                ``Infinity`` are not valid JSON, so the peer could not parse
                the message; nothing is written in that case.
        """
        raw = json.dumps(
            msg, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        self.writer.write((raw + "\n").encode("utf-8"))
        await self.writer.drain()

    async def send_request(
        self,
        method: str,
        params: Any = None,
        req_id: Union[int, str, None] = None,
    ) -> dict[str, Any]:
        """Build and send a JSON-RPC request.

        Args:
            method: The RPC method name.
            params: Parameters for the method.
            req_id: Optional explicit request id.

        Returns:
            The message dictionary that was sent.
        """
        msg: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id if req_id is not None else next_id(),
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        await self.write_message(msg)
        return msg

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Build and send a JSON-RPC notification (no ``id``).

        Args:
            method: The RPC method name.
            params: Parameters for the notification.
        """
        msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params is not None:
            msg["params"] = params
        await self.write_message(msg)

    async def send_result(self, req_id: Union[int, str], result: Any) -> None:
        """Send a JSON-RPC success response.

        Args:
            req_id: The id of the original request.
            result: The result payload.
        """
        await self.write_message(
            {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}
        )

    async def send_error(
        self, req_id: Union[int, str, None], error: JSONRPCError
    ) -> None:
        """Send a JSON-RPC error response.

        Args:
            req_id: The id of the original request (may be ``None``).
            error: The error object.
        """
        await self.write_message(
            {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": error.to_dict()}
        )

    async def close(self) -> None:
        """Close the writer stream."""
        if not self._closed:
            self._closed = True
            self.writer.close()

    @property
    def is_closed(self) -> bool:
        """Whether the writer has been closed."""
        return self._closed

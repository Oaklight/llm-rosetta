"""
LLM-Rosetta - Conversion Context

Provides the context hierarchy for conversion pipelines:

- ``ConversionContext``: Base context for non-streaming conversions.
  Carries warnings, structured options, and opaque metadata through
  the conversion pipeline.
- ``StreamContext``: Extended context for streaming conversions.
  Adds session-level metadata, tool call tracking, lifecycle flags,
  and deferred event payloads on top of ConversionContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversionContext:
    """Shared context for a conversion pipeline.

    Created once per conversion operation (request or response cycle)
    and threaded through converter methods to carry shared state.

    Attributes:
        warnings: Accumulated warnings from conversion steps.
        options: Structured conversion options (e.g., ``output_format``).
        metadata: Opaque store for debugging and provider-specific state.
    """

    warnings: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamContext(ConversionContext):
    """Maintains state across stream chunk conversions.

    Extends :class:`ConversionContext` with session-level metadata and
    per-block state to enable stateful stream transformations in
    Man-in-the-Middle scenarios.

    Attributes:
        response_id: Provider response ID (e.g., chatcmpl-xxx, msg_xxx).
        model: Model name from the provider response.
        created: Unix timestamp of the response creation.
        current_block_index: Current 0-based content block index.
        tool_call_id_map: Mapping from tool_call_id to tool_name.
        tool_call_item_id_map: Mapping from tool_call_id to item_id.
        pending_usage: Usage info stored by UsageEvent for later merging
            into a FinishEvent (prevents duplicate terminal events).
        pending_finish: Deferred finish event payload.
        pending_response: Deferred response.completed payload stored by
            FinishEvent, emitted by StreamEndEvent after usage is merged.
    """

    # Session-level metadata
    response_id: str = ""
    model: str = ""
    created: int = 0
    current_block_index: int = -1

    # Tool call tracking
    tool_call_id_map: dict[str, str] = field(default_factory=dict)
    tool_call_item_id_map: dict[str, str] = field(default_factory=dict)

    # Deferred event payloads
    pending_usage: dict | None = None
    pending_finish: dict | None = None
    pending_response: dict | None = None

    # Lifecycle flags
    _started: bool = field(default=False, repr=False)
    _ended: bool = field(default=False, repr=False)

    # Tool call accumulation for streaming
    _tool_call_args: dict[str, str] = field(default_factory=dict, repr=False)
    _tool_call_order: list[str] = field(default_factory=list, repr=False)

    def next_block_index(self) -> int:
        """Increment and return the next block index.

        Returns:
            The next 0-based block index.
        """
        self.current_block_index += 1
        return self.current_block_index

    def register_tool_call(self, tool_call_id: str, tool_name: str) -> None:
        """Register a tool call ID to name mapping.

        Args:
            tool_call_id: The unique identifier for the tool call.
            tool_name: The name of the tool being called.
        """
        self.tool_call_id_map[tool_call_id] = tool_name
        self._tool_call_args[tool_call_id] = ""
        if tool_call_id not in self._tool_call_order:
            self._tool_call_order.append(tool_call_id)

    def register_tool_call_item(self, tool_call_id: str, item_id: str) -> None:
        """Register the Responses output item ID for a tool call.

        Args:
            tool_call_id: The stable tool correlation identifier.
            item_id: The Responses output item identifier for the function call.
        """
        if tool_call_id and item_id:
            self.tool_call_item_id_map[tool_call_id] = item_id

    def get_tool_call_item_id(self, tool_call_id: str) -> str:
        """Get the Responses output item ID for a tool call.

        Args:
            tool_call_id: The stable tool correlation identifier.

        Returns:
            The output item ID, or empty string if not found.
        """
        return self.tool_call_item_id_map.get(tool_call_id, "")

    def append_tool_call_args(self, tool_call_id: str, delta: str) -> None:
        """Append argument delta to accumulated tool call arguments.

        Args:
            tool_call_id: The tool call identifier.
            delta: The argument text delta to append.
        """
        if tool_call_id not in self._tool_call_args:
            self._tool_call_args[tool_call_id] = ""
            if tool_call_id not in self._tool_call_order:
                self._tool_call_order.append(tool_call_id)
        self._tool_call_args[tool_call_id] += delta

    def set_tool_call_args(self, tool_call_id: str, arguments: str) -> None:
        """Set the final arguments for a tool call.

        Args:
            tool_call_id: The tool call identifier.
            arguments: The complete arguments string.
        """
        self._tool_call_args[tool_call_id] = arguments

    def get_tool_name(self, tool_call_id: str) -> str:
        """Get tool name by tool call ID.

        Args:
            tool_call_id: The unique identifier for the tool call.

        Returns:
            The tool name, or empty string if not found.
        """
        return self.tool_call_id_map.get(tool_call_id, "")

    def get_tool_call_args(self, tool_call_id: str) -> str:
        """Get accumulated arguments for a tool call.

        Args:
            tool_call_id: The tool call identifier.

        Returns:
            The accumulated arguments string, or empty string if not found.
        """
        return self._tool_call_args.get(tool_call_id, "")

    def get_pending_tool_calls(self) -> list[tuple[str, str, str]]:
        """Get all registered tool calls with their accumulated arguments.

        Returns:
            List of (tool_call_id, tool_name, accumulated_args) tuples
            in the order they were registered.
        """
        result = []
        for call_id in self._tool_call_order:
            name = self.tool_call_id_map.get(call_id, "")
            args = self._tool_call_args.get(call_id, "")
            result.append((call_id, name, args))
        return result

    def mark_started(self) -> None:
        """Mark the stream as started."""
        self._started = True

    def mark_ended(self) -> None:
        """Mark the stream as ended."""
        self._ended = True

    @property
    def is_started(self) -> bool:
        """Whether the stream has been started."""
        return self._started

    @property
    def is_ended(self) -> bool:
        """Whether the stream has been ended."""
        return self._ended

"""
LLM-Rosetta - Stream Context

Maintains state across stream chunk conversions for stateful stream
transformations in Man-in-the-Middle scenarios.
"""


class StreamContext:
    """Maintains state across stream chunk conversions.

    Tracks session-level metadata and per-block state to enable
    stateful stream transformations in Man-in-the-Middle scenarios.

    Attributes:
        response_id: Provider response ID (e.g., chatcmpl-xxx, msg_xxx).
        model: Model name from the provider response.
        created: Unix timestamp of the response creation.
        current_block_index: Current 0-based content block index.
        tool_call_id_map: Mapping from tool_call_id to tool_name.
        pending_usage: Usage info stored by UsageEvent for later merging
            into a FinishEvent (prevents duplicate terminal events).
    """

    def __init__(self) -> None:
        self.response_id: str = ""
        self.model: str = ""
        self.created: int = 0
        self.current_block_index: int = -1
        self.tool_call_id_map: dict[str, str] = {}  # tool_call_id -> tool_name
        self.pending_usage: dict | None = None
        self._started: bool = False
        self._ended: bool = False

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

    def get_tool_name(self, tool_call_id: str) -> str:
        """Get tool name by tool call ID.

        Args:
            tool_call_id: The unique identifier for the tool call.

        Returns:
            The tool name, or empty string if not found.
        """
        return self.tool_call_id_map.get(tool_call_id, "")

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

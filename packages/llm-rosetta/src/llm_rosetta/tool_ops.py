"""
LLM-Rosetta - Tool Conversion Convenience API

Lightweight top-level API for converting tool artifacts between IR and
provider formats without instantiating full converter pipelines.

Covers the full tool lifecycle: definition, choice, call, result, config.

All imports are lazy to avoid pulling provider dependencies at import time.

Example usage::

    from llm_rosetta import tool_ops

    ir_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather info",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }

    # Per-provider shortcuts (definitions only)
    openai_tool = tool_ops.to_chat_completions(ir_tool)
    messages_tool = tool_ops.to_messages(ir_tool)

    # Unified dispatch (definitions)
    gemini_tool = tool_ops.to_provider(ir_tool, provider="google_generate")

    # Reverse direction
    recovered = tool_ops.from_provider(openai_tool, provider="chat_completions")

    # Full lifecycle dispatch (choice, call, result, config)
    provider_choice = tool_ops.choice_to_provider(ir_choice, provider="messages")
    provider_call = tool_ops.call_to_provider(ir_call, provider="chat_completions")
"""

from typing import Any, Literal

from .retired_names import PROVIDER_TYPE, retired_name_message
from .types.ir.tools import ToolCallConfig, ToolChoice, ToolDefinition
from .types.ir.parts import ToolCallPart, ToolResultPart

__all__ = [
    # Unified dispatch
    "to_provider",
    "from_provider",
    "choice_to_provider",
    "choice_from_provider",
    "call_to_provider",
    "call_from_provider",
    "result_to_provider",
    "result_from_provider",
    "config_to_provider",
    "config_from_provider",
    # Per-format shortcuts
    "to_chat_completions",
    "to_open_responses",
    "to_messages",
    "to_google_generate",
    "to_google_interactions",
    "from_chat_completions",
    "from_open_responses",
    "from_messages",
    "from_google_generate",
    "from_google_interactions",
]

ToolProvider = Literal[
    # Canonical names — the base converter types (underscore spelling).
    "messages",
    "google_generate",
    "google_interactions",
    "chat_completions",
    "open_responses",
]

# Canonical tool_ops provider names, one per base converter type.
_CANONICAL_TOOL_PROVIDERS = frozenset(
    {
        "messages",
        "google_generate",
        "google_interactions",
        "chat_completions",
        "open_responses",
    }
)


def _get_tool_ops(provider: str) -> Any:
    """Return the ToolOps class bound to *provider*'s converter.

    Resolved through the converter registry — the format's own package names
    its ToolOps class, so a new format needs no branch here.
    """
    from .converters.registry import CONVERTER_REGISTRY

    converter_cls = CONVERTER_REGISTRY.get("llm", {}).get(provider)
    tool_ops_cls = getattr(converter_cls, "tool_ops_class", None)
    if tool_ops_cls is None:
        message = (
            f"Unknown provider: {provider!r}. "
            f"Supported: {', '.join(sorted(_CANONICAL_TOOL_PROVIDERS))}."
        )
        hint = retired_name_message(provider, PROVIDER_TYPE)
        if hint is not None:
            message += f" {hint[0].upper()}{hint[1:]}."
        raise ValueError(message)
    return tool_ops_cls


# ==================== Definition dispatch ====================


def to_provider(ir_tool: ToolDefinition, provider: ToolProvider, **kwargs: Any) -> Any:
    """Convert an IR tool definition to provider-native format.

    Args:
        ir_tool: IR ToolDefinition dict.
        provider: Target provider name or alias.
        **kwargs: Forwarded to the underlying ToolOps method.

    Returns:
        Provider-native tool definition dict.

    Raises:
        ValueError: If *provider* is not recognized.
    """
    return _get_tool_ops(provider).ir_tool_definition_to_p(ir_tool, **kwargs)


def from_provider(
    provider_tool: Any, provider: ToolProvider, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert a provider-native tool definition to IR format.

    Args:
        provider_tool: Provider-native tool definition.
        provider: Source provider name or alias.
        **kwargs: Forwarded to the underlying ToolOps method.

    Returns:
        IR ToolDefinition, a list of them (Google may return multiple),
        or None if the tool type is not supported.

    Raises:
        ValueError: If *provider* is not recognized.
    """
    return _get_tool_ops(provider).p_tool_definition_to_ir(provider_tool, **kwargs)


# ==================== Choice dispatch ====================


def choice_to_provider(
    ir_choice: ToolChoice, provider: ToolProvider, **kwargs: Any
) -> Any:
    """Convert an IR tool choice to provider-native format."""
    return _get_tool_ops(provider).ir_tool_choice_to_p(ir_choice, **kwargs)


def choice_from_provider(
    provider_choice: Any, provider: ToolProvider, **kwargs: Any
) -> ToolChoice:
    """Convert a provider-native tool choice to IR format."""
    return _get_tool_ops(provider).p_tool_choice_to_ir(provider_choice, **kwargs)


# ==================== Call dispatch ====================


def call_to_provider(
    ir_call: ToolCallPart, provider: ToolProvider, **kwargs: Any
) -> Any:
    """Convert an IR tool call to provider-native format."""
    return _get_tool_ops(provider).ir_tool_call_to_p(ir_call, **kwargs)


def call_from_provider(
    provider_call: Any, provider: ToolProvider, **kwargs: Any
) -> ToolCallPart:
    """Convert a provider-native tool call to IR format."""
    return _get_tool_ops(provider).p_tool_call_to_ir(provider_call, **kwargs)


# ==================== Result dispatch ====================


def result_to_provider(
    ir_result: ToolResultPart, provider: ToolProvider, **kwargs: Any
) -> Any:
    """Convert an IR tool result to provider-native format."""
    return _get_tool_ops(provider).ir_tool_result_to_p(ir_result, **kwargs)


def result_from_provider(
    provider_result: Any, provider: ToolProvider, **kwargs: Any
) -> ToolResultPart:
    """Convert a provider-native tool result to IR format."""
    return _get_tool_ops(provider).p_tool_result_to_ir(provider_result, **kwargs)


# ==================== Config dispatch ====================


def config_to_provider(
    ir_config: ToolCallConfig, provider: ToolProvider, **kwargs: Any
) -> Any:
    """Convert an IR tool call config to provider-native format."""
    return _get_tool_ops(provider).ir_tool_config_to_p(ir_config, **kwargs)


def config_from_provider(
    provider_config: Any, provider: ToolProvider, **kwargs: Any
) -> ToolCallConfig:
    """Convert a provider-native tool call config to IR format."""
    return _get_tool_ops(provider).p_tool_config_to_ir(provider_config, **kwargs)


# ==================== Per-provider definition shortcuts ====================


def to_chat_completions(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to OpenAI Chat Completions format."""
    from .converters.llm.chat_completions import ChatCompletionsToolOps

    return ChatCompletionsToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_open_responses(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to OpenAI Responses format."""
    from .converters.llm.open_responses import OpenResponsesToolOps

    return OpenResponsesToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_messages(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Messages format."""
    from .converters.llm.messages import MessagesToolOps

    return MessagesToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_google_generate(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Google generateContent format."""
    from .converters.llm.google_generate import GoogleGenerateToolOps

    return GoogleGenerateToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_google_interactions(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Google Interactions format."""
    from .converters.llm.google_interactions import GoogleInteractionsToolOps

    return GoogleInteractionsToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def from_chat_completions(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert OpenAI Chat Completions tool definition to IR format."""
    from .converters.llm.chat_completions import ChatCompletionsToolOps

    return ChatCompletionsToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_open_responses(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert OpenAI Responses tool definition to IR format."""
    from .converters.llm.open_responses import OpenResponsesToolOps

    return OpenResponsesToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_messages(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert Messages tool definition to IR format."""
    from .converters.llm.messages import MessagesToolOps

    return MessagesToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_google_generate(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert Google generateContent tool definition to IR format."""
    from .converters.llm.google_generate import GoogleGenerateToolOps

    return GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_google_interactions(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert Google Interactions tool definition to IR format."""
    from .converters.llm.google_interactions import GoogleInteractionsToolOps

    return GoogleInteractionsToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)

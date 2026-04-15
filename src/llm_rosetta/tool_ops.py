"""
LLM-Rosetta - Tool Definition Convenience API

Lightweight top-level API for converting tool definitions between IR and
provider formats without instantiating full converter pipelines.

All imports are lazy to avoid pulling provider dependencies at import time.

Example usage::

    from llm_rosetta import tool_ops

    ir_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather info",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }

    # Per-provider shortcuts
    openai_tool = tool_ops.to_openai_chat(ir_tool)
    anthropic_tool = tool_ops.to_anthropic(ir_tool)

    # Unified dispatch
    gemini_tool = tool_ops.to_provider(ir_tool, provider="google")

    # Reverse direction
    recovered = tool_ops.from_provider(openai_tool, provider="openai_chat")
"""

from typing import Any, Literal

from .types.ir.tools import ToolDefinition

ToolProvider = Literal[
    # Canonical names (match ProviderType / internal module names)
    "openai_chat",
    "openai_responses",
    "anthropic",
    "google",
    # Hyphenated aliases for ergonomic use
    "openai-chat",
    "openai-responses",
    "open_responses",
    "open-responses",
    "google-genai",
]

_PROVIDER_ALIASES: dict[str, str] = {
    "openai-chat": "openai_chat",
    "openai-responses": "openai_responses",
    "open_responses": "openai_responses",
    "open-responses": "openai_responses",
    "google-genai": "google",
}


def _resolve(provider: str) -> str:
    """Normalize provider name to canonical form."""
    return _PROVIDER_ALIASES.get(provider, provider)


def _get_tool_ops(provider: str) -> Any:
    """Lazy-import and return the ToolOps class for *provider*."""
    canonical = _resolve(provider)
    if canonical == "openai_chat":
        from .converters.openai_chat import OpenAIChatToolOps

        return OpenAIChatToolOps
    if canonical == "openai_responses":
        from .converters.openai_responses import OpenAIResponsesToolOps

        return OpenAIResponsesToolOps
    if canonical == "anthropic":
        from .converters.anthropic import AnthropicToolOps

        return AnthropicToolOps
    if canonical == "google":
        from .converters.google_genai import GoogleGenAIToolOps

        return GoogleGenAIToolOps
    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Supported: openai_chat, openai_responses, anthropic, google "
        f"(aliases: openai-chat, openai-responses, google-genai)"
    )


# ==================== Unified dispatch ====================


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


# ==================== Per-provider shortcuts ====================


def to_openai_chat(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to OpenAI Chat format."""
    from .converters.openai_chat import OpenAIChatToolOps

    return OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_openai_responses(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to OpenAI Responses format."""
    from .converters.openai_responses import OpenAIResponsesToolOps

    return OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_anthropic(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Anthropic format."""
    from .converters.anthropic import AnthropicToolOps

    return AnthropicToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_google_genai(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Google GenAI format."""
    from .converters.google_genai import GoogleGenAIToolOps

    return GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def from_openai_chat(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert OpenAI Chat tool definition to IR format."""
    from .converters.openai_chat import OpenAIChatToolOps

    return OpenAIChatToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_openai_responses(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert OpenAI Responses tool definition to IR format."""
    from .converters.openai_responses import OpenAIResponsesToolOps

    return OpenAIResponsesToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_anthropic(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert Anthropic tool definition to IR format."""
    from .converters.anthropic import AnthropicToolOps

    return AnthropicToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_google_genai(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert Google GenAI tool definition to IR format."""
    from .converters.google_genai import GoogleGenAIToolOps

    return GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)

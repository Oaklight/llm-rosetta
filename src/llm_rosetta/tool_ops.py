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
    openai_tool = tool_ops.to_openai_chat(ir_tool)
    anthropic_tool = tool_ops.to_anthropic(ir_tool)

    # Unified dispatch (definitions)
    gemini_tool = tool_ops.to_provider(ir_tool, provider="google")

    # Reverse direction
    recovered = tool_ops.from_provider(openai_tool, provider="openai_chat")

    # Full lifecycle dispatch (choice, call, result, config)
    provider_choice = tool_ops.choice_to_provider(ir_choice, provider="anthropic")
    provider_call = tool_ops.call_to_provider(ir_call, provider="openai_chat")
"""

from typing import Any, Literal

from .types.ir.tools import ToolCallConfig, ToolChoice, ToolDefinition
from .types.ir.parts import ToolCallPart, ToolResultPart

ToolProvider = Literal[
    # Canonical names (match ProviderType / internal module names)
    "openai_chat",
    "openai_responses",
    "anthropic",
    "google",
    "google_interactions",
    # Hyphenated aliases for ergonomic use
    "openai-chat",
    "openai-responses",
    "open_responses",
    "open-responses",
    "google-genai",
    "google-interactions",
]

_PROVIDER_ALIASES: dict[str, str] = {
    "openai-chat": "openai_chat",
    "openai-responses": "openai_responses",
    "open_responses": "openai_responses",
    "open-responses": "openai_responses",
    "google-genai": "google",
    "google-interactions": "google_interactions",
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
        from .converters.google_generate import GoogleGenerateToolOps

        return GoogleGenerateToolOps
    if canonical == "google_interactions":
        from .converters.google_interactions import GoogleInteractionsToolOps

        return GoogleInteractionsToolOps
    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Supported: openai_chat, openai_responses, anthropic, google, "
        f"google_interactions "
        f"(aliases: openai-chat, openai-responses, google-genai, google-interactions)"
    )


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


def to_google_generate(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Google generateContent format."""
    from .converters.google_generate import GoogleGenerateToolOps

    return GoogleGenerateToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def to_google_interactions(ir_tool: ToolDefinition, **kwargs: Any) -> dict[str, Any]:
    """Convert IR tool definition to Google Interactions format."""
    from .converters.google_interactions import GoogleInteractionsToolOps

    return GoogleInteractionsToolOps.ir_tool_definition_to_p(ir_tool, **kwargs)


def from_openai_chat(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert OpenAI Chat tool definition to IR format."""
    from .converters.openai_chat import OpenAIChatToolOps

    return OpenAIChatToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_openai_responses(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert OpenAI Responses tool definition to IR format."""
    from .converters.openai_responses import OpenAIResponsesToolOps

    return OpenAIResponsesToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_anthropic(provider_tool: Any, **kwargs: Any) -> ToolDefinition | None:
    """Convert Anthropic tool definition to IR format."""
    from .converters.anthropic import AnthropicToolOps

    return AnthropicToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_google_generate(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert Google generateContent tool definition to IR format."""
    from .converters.google_generate import GoogleGenerateToolOps

    return GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)


def from_google_interactions(
    provider_tool: Any, **kwargs: Any
) -> ToolDefinition | list[ToolDefinition] | None:
    """Convert Google Interactions tool definition to IR format."""
    from .converters.google_interactions import GoogleInteractionsToolOps

    return GoogleInteractionsToolOps.p_tool_definition_to_ir(provider_tool, **kwargs)

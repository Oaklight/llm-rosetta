"""Integration tests for tool conversion caching across all converters.

Verifies that the process-level LRU cache correctly caches tool
conversion results, skips validation on cache hits, and produces
identical output to uncached conversion.
"""

import copy

import pytest

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.base.cache import (
    cache_info,
    clear_all_caches,
)
from llm_rosetta.converters.google_genai import GoogleConverter
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter

# ---------------------------------------------------------------------------
# Fixtures — provider-specific tool definitions and minimal requests
# ---------------------------------------------------------------------------

ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
]

OPENAI_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    },
]

OPENAI_RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
]

GOOGLE_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]
    },
    {
        "function_declarations": [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
    },
]


def _anthropic_request(tools):
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": tools,
    }


def _openai_chat_request(tools):
    return {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }


def _openai_responses_request(tools):
    return {
        "model": "gpt-4",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "tools": tools,
    }


def _google_request(tools):
    return {
        "model": "gemini-2.0-flash",
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
        ],
        "config": {"tools": tools},
    }


# (converter_class, tools, request_builder)
CONVERTER_CONFIGS = [
    pytest.param(
        AnthropicConverter, ANTHROPIC_TOOLS, _anthropic_request, id="anthropic"
    ),
    pytest.param(
        OpenAIChatConverter, OPENAI_CHAT_TOOLS, _openai_chat_request, id="openai_chat"
    ),
    pytest.param(
        OpenAIResponsesConverter,
        OPENAI_RESPONSES_TOOLS,
        _openai_responses_request,
        id="openai_responses",
    ),
    pytest.param(GoogleConverter, GOOGLE_TOOLS, _google_request, id="google_genai"),
]


# ---------------------------------------------------------------------------
# Tests — request_from_provider (Provider → IR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conv_cls,tools,make_req", CONVERTER_CONFIGS)
class TestFromProviderCache:
    """Tests for _convert_tools_from_p caching in request_from_provider."""

    def test_cache_hit_on_second_call(self, conv_cls, tools, make_req):
        """Second call with same tools should hit the per-entry cache."""
        clear_all_caches()
        conv = conv_cls()
        req = make_req(tools)
        n_tools = len(tools)

        # First call — cold (each tool is a cache miss)
        conv.request_from_provider(copy.deepcopy(req))
        info1 = cache_info()["tool_entry"]
        assert info1["misses"] >= n_tools
        assert info1["hits"] == 0

        # Second call — warm (each tool is a hit)
        conv2 = conv_cls()
        conv2.request_from_provider(copy.deepcopy(req))
        info2 = cache_info()["tool_entry"]
        assert info2["hits"] >= n_tools

    def test_cache_miss_on_different_tools(self, conv_cls, tools, make_req):
        """Different tools should not hit the cache."""
        clear_all_caches()
        conv = conv_cls()
        conv.request_from_provider(copy.deepcopy(make_req(tools)))

        # Modify one tool — it should miss, the other should still hit
        modified_tools = copy.deepcopy(tools)
        modified_tools[0]["description"] = "MODIFIED"
        conv2 = conv_cls()
        conv2.request_from_provider(copy.deepcopy(make_req(modified_tools)))

        info = cache_info()["tool_entry"]
        # First call: all misses. Second call: 1 miss (modified) + rest hits
        assert info["hits"] >= 1  # at least the unmodified tool(s) hit

    def test_cached_matches_uncached(self, conv_cls, tools, make_req):
        """Cached result should be identical to a fresh conversion."""
        clear_all_caches()
        conv = conv_cls()
        req = make_req(tools)

        # Cold path
        ir1 = conv.request_from_provider(copy.deepcopy(req))

        # Warm path
        conv2 = conv_cls()
        ir2 = conv2.request_from_provider(copy.deepcopy(req))

        assert ir1.get("tools") == ir2.get("tools")
        assert ir1["model"] == ir2["model"]


# ---------------------------------------------------------------------------
# Tests — request_to_provider (IR → Provider)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conv_cls,tools,make_req", CONVERTER_CONFIGS)
class TestToProviderCache:
    """Tests for _apply_tool_config caching in request_to_provider."""

    def test_cache_hit_on_second_call(self, conv_cls, tools, make_req):
        """Second roundtrip should hit the per-entry to_p cache."""
        clear_all_caches()
        conv = conv_cls()
        req = make_req(tools)

        # First roundtrip (from_p misses + to_p misses)
        ir = conv.request_from_provider(copy.deepcopy(req))
        conv.request_to_provider(ir)

        # Second roundtrip (from_p hits + to_p hits)
        conv2 = conv_cls()
        ir2 = conv2.request_from_provider(copy.deepcopy(req))
        conv2.request_to_provider(ir2)
        info2 = cache_info()["tool_entry"]
        # All tools should hit on second roundtrip (both directions)
        assert info2["hits"] >= len(tools) * 2


# ---------------------------------------------------------------------------
# Cross-converter isolation
# ---------------------------------------------------------------------------


def test_cross_converter_no_pollution():
    """Cache entries from one converter should not leak to another."""
    clear_all_caches()

    anth = AnthropicConverter()
    anth.request_from_provider(copy.deepcopy(_anthropic_request(ANTHROPIC_TOOLS)))

    oai = OpenAIChatConverter()
    oai.request_from_provider(copy.deepcopy(_openai_chat_request(OPENAI_CHAT_TOOLS)))

    # Per-entry: both converters have 2 tools each → 4 misses total
    # (different converter tags ensure no cross-pollution)
    info = cache_info()["tool_entry"]
    assert info["misses"] == 4  # 2 tools × 2 converters
    assert info["hits"] == 0


# ---------------------------------------------------------------------------
# Cache survives converter instance recreation
# ---------------------------------------------------------------------------


def test_cache_survives_new_instance():
    """Module-level cache persists across converter instances."""
    clear_all_caches()

    conv1 = AnthropicConverter()
    req = _anthropic_request(ANTHROPIC_TOOLS)
    conv1.request_from_provider(copy.deepcopy(req))

    # Brand new instance
    conv2 = AnthropicConverter()
    conv2.request_from_provider(copy.deepcopy(req))

    info = cache_info()["tool_entry"]
    # 2 tools × first call miss + 2 tools × second call hit
    assert info["hits"] == 2


# ---------------------------------------------------------------------------
# Validation still catches bad tools on cold path
# ---------------------------------------------------------------------------


def test_invalid_tools_not_cached():
    """Invalid tools should raise on the first call and not be cached.

    Uses a non-dict tool entry which triggers ValueError in _convert_tools_from_p.
    """
    clear_all_caches()

    conv = AnthropicConverter()
    # A non-dict entry will fail in p_tool_definition_to_ir (tries .get on a string)
    broken_tools = ["not_a_tool_dict"]
    broken_req = {
        "model": "test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": broken_tools,
    }

    with pytest.raises((ValueError, TypeError, AttributeError)):
        conv.request_from_provider(broken_req)

    # Cache should be empty — error during conversion means no cache entry
    info = cache_info()["tool_entry"]
    assert info["currsize"] == 0

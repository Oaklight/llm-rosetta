"""Integration tests for tool conversion caching across all converters.

Verifies that the process-level LRU cache correctly caches tool
conversion results, skips validation on cache hits, and produces
identical output to uncached conversion.
"""

import copy

import pytest

from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.base.helpers.cache import (
    cache_info,
    clear_all_caches,
)
from llm_rosetta.converters.llm.google_generate import GoogleConverter
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.open_responses import OpenAIResponsesConverter

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
    pytest.param(MessagesConverter, ANTHROPIC_TOOLS, _anthropic_request, id="messages"),
    pytest.param(
        ChatCompletionsConverter,
        OPENAI_CHAT_TOOLS,
        _openai_chat_request,
        id="openai_chat",
    ),
    pytest.param(
        OpenAIResponsesConverter,
        OPENAI_RESPONSES_TOOLS,
        _openai_responses_request,
        id="openai_responses",
    ),
    pytest.param(GoogleConverter, GOOGLE_TOOLS, _google_request, id="google_generate"),
]


# ---------------------------------------------------------------------------
# Tests — request_from_provider (Provider → IR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conv_cls,tools,make_req", CONVERTER_CONFIGS)
class TestFromProviderUncached:
    """The provider→IR tool reader is deliberately uncached (#1410).

    A per-tool key is a canonical-JSON hash of the declaration and costs
    more than the read it would skip in every chat dialect, so
    ``request_from_provider`` must leave the entry cache untouched.
    """

    def test_first_call_populates_nothing(self, conv_cls, tools, make_req):
        """The reader never consults or fills the per-entry cache."""
        clear_all_caches()
        conv = conv_cls()
        conv.request_from_provider(copy.deepcopy(make_req(tools)))

        info = cache_info()["tool_entry"]
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["currsize"] == 0

    def test_repeated_calls_stay_out_of_the_cache(self, conv_cls, tools, make_req):
        """A second identical read still does not touch the cache."""
        clear_all_caches()
        req = make_req(tools)
        conv_cls().request_from_provider(copy.deepcopy(req))
        conv_cls().request_from_provider(copy.deepcopy(req))

        info = cache_info()["tool_entry"]
        assert info["hits"] == 0
        assert info["misses"] == 0

    def test_repeated_requests_match(self, conv_cls, tools, make_req):
        """Two identical reads produce the same IR tools."""
        clear_all_caches()
        conv = conv_cls()
        req = make_req(tools)

        ir1 = conv.request_from_provider(copy.deepcopy(req))
        ir2 = conv_cls().request_from_provider(copy.deepcopy(req))

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
        # Every tool hits on the second roundtrip; the provider→IR direction
        # is uncached, so the IR→provider hits are the whole count (#1410).
        assert info2["hits"] >= len(tools)


# ---------------------------------------------------------------------------
# Cross-converter isolation
# ---------------------------------------------------------------------------


def test_cross_converter_no_pollution():
    """Cache entries from one converter should not leak to another.

    No chat dialect caches provider→IR tools (#1410), so the per-entry cache
    stays empty; the tag isolation itself is covered in
    ``tests/converters/base/test_cache.py``.
    """
    clear_all_caches()

    anth = MessagesConverter()
    anth.request_from_provider(copy.deepcopy(_anthropic_request(ANTHROPIC_TOOLS)))

    oai = ChatCompletionsConverter()
    oai.request_from_provider(copy.deepcopy(_openai_chat_request(OPENAI_CHAT_TOOLS)))

    info = cache_info()["tool_entry"]
    assert info["misses"] == 0
    assert info["hits"] == 0


# ---------------------------------------------------------------------------
# Cache survives converter instance recreation
# ---------------------------------------------------------------------------


def test_cache_survives_new_instance():
    """Module-level cache persists across converter instances."""
    clear_all_caches()

    conv1 = GoogleConverter()
    req = _google_request(GOOGLE_TOOLS)
    conv1.request_to_provider(conv1.request_from_provider(copy.deepcopy(req)))

    # Brand new instance
    conv2 = GoogleConverter()
    conv2.request_to_provider(conv2.request_from_provider(copy.deepcopy(req)))

    info = cache_info()["tool_entry"]
    # 2 declarations × first roundtrip miss + 2 declarations × second hit
    assert info["hits"] == 2


# ---------------------------------------------------------------------------
# Validation still catches bad tools on cold path
# ---------------------------------------------------------------------------


def test_invalid_tools_not_cached():
    """Invalid tools should raise on the first call and not be cached.

    Uses a non-dict tool entry which triggers ValueError in _convert_p_tools_to_ir.
    """
    clear_all_caches()

    conv = MessagesConverter()
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

"""Tests for ConversionContext and StreamContext inheritance."""

from llm_rosetta.converters.base import BaseConverter
from llm_rosetta.converters.base.context import ConversionContext, StreamContext
from llm_rosetta.converters.openai_responses.stream_context import (
    OpenAIResponsesStreamContext,
)


class TestConversionContext:
    """Test ConversionContext dataclass."""

    def test_defaults(self):
        ctx = ConversionContext()
        assert ctx.warnings == []
        assert ctx.options == {}
        assert ctx.metadata == {}

    def test_warnings_accumulation(self):
        ctx = ConversionContext()
        ctx.warnings.append("warn1")
        ctx.warnings.extend(["warn2", "warn3"])
        assert ctx.warnings == ["warn1", "warn2", "warn3"]

    def test_options(self):
        ctx = ConversionContext(options={"output_format": "rest"})
        assert ctx.options["output_format"] == "rest"

    def test_metadata(self):
        ctx = ConversionContext()
        ctx.metadata["debug_info"] = {"step": "request_to_provider"}
        assert ctx.metadata["debug_info"]["step"] == "request_to_provider"

    def test_instances_isolated(self):
        ctx1 = ConversionContext()
        ctx2 = ConversionContext()
        ctx1.warnings.append("only-in-ctx1")
        assert ctx2.warnings == []


class TestStreamContextInheritance:
    """Test that StreamContext IS-A ConversionContext."""

    def test_isinstance(self):
        sc = StreamContext()
        assert isinstance(sc, ConversionContext)

    def test_inherits_fields(self):
        sc = StreamContext()
        assert hasattr(sc, "warnings")
        assert hasattr(sc, "options")
        assert hasattr(sc, "metadata")
        assert sc.warnings == []
        assert sc.options == {}
        assert sc.metadata == {}

    def test_stream_fields_intact(self):
        sc = StreamContext()
        assert sc.response_id == ""
        assert sc.model == ""
        assert sc.created == 0
        assert sc.current_block_index == -1
        assert sc.tool_call_id_map == {}

    def test_stream_methods_work(self):
        sc = StreamContext()
        sc.register_tool_call("tc_1", "get_weather")
        assert sc.get_tool_name("tc_1") == "get_weather"
        sc.append_tool_call_args("tc_1", '{"city":')
        sc.append_tool_call_args("tc_1", '"NYC"}')
        assert sc.get_tool_call_args("tc_1") == '{"city":"NYC"}'

    def test_warnings_in_stream_context(self):
        sc = StreamContext()
        sc.warnings.append("stream warning")
        assert sc.warnings == ["stream warning"]

    def test_options_in_stream_context(self):
        sc = StreamContext(options={"output_format": "rest"})
        assert sc.options["output_format"] == "rest"
        assert sc.response_id == ""  # stream fields still default


class TestOpenAIResponsesStreamContextInheritance:
    """Test the full inheritance chain: OpenAIResponsesStreamContext -> StreamContext -> ConversionContext."""

    def test_isinstance_chain(self):
        ctx = OpenAIResponsesStreamContext()
        assert isinstance(ctx, StreamContext)
        assert isinstance(ctx, ConversionContext)

    def test_has_all_fields(self):
        ctx = OpenAIResponsesStreamContext()
        # ConversionContext fields
        assert ctx.warnings == []
        assert ctx.options == {}
        # StreamContext fields
        assert ctx.response_id == ""
        assert ctx.tool_call_id_map == {}
        # OpenAIResponsesStreamContext fields
        assert ctx.item_id_to_call_id == {}
        assert ctx.output_item_emitted is False


class TestFactoryMethods:
    """Test create_conversion_context and create_stream_context."""

    def test_create_conversion_context(self):
        ctx = BaseConverter.create_conversion_context()
        assert isinstance(ctx, ConversionContext)
        assert ctx.warnings == []
        assert ctx.options == {}

    def test_create_conversion_context_with_options(self):
        ctx = BaseConverter.create_conversion_context(output_format="rest")
        assert ctx.options["output_format"] == "rest"

    def test_create_stream_context_is_conversion_context(self):
        sc = BaseConverter.create_stream_context()
        assert isinstance(sc, StreamContext)
        assert isinstance(sc, ConversionContext)

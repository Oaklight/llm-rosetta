"""Deprecated: moved to :mod:`llm_rosetta.ir.model.type_guards`.

``llm_rosetta.types.ir.type_guards`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.type_guards import (
    TYPE_CLASS_MAP,
    get_part_type,
    is_audio_part,
    is_citation_part,
    is_content_block_end_event,
    is_content_block_start_event,
    is_file_part,
    is_finish_event,
    is_image_part,
    is_part_type,
    is_provider_passthrough_event,
    is_provider_passthrough_item,
    is_reasoning_delta_event,
    is_reasoning_part,
    is_refusal_delta_event,
    is_refusal_part,
    is_stream_end_event,
    is_stream_start_event,
    is_text_delta_event,
    is_text_part,
    is_tool_call_delta_event,
    is_tool_call_part,
    is_tool_call_start_event,
    is_tool_result_part,
    is_usage_event,
    isinstance_part,
)

__all__ = [
    "TYPE_CLASS_MAP",
    "get_part_type",
    "is_audio_part",
    "is_citation_part",
    "is_content_block_end_event",
    "is_content_block_start_event",
    "is_file_part",
    "is_finish_event",
    "is_image_part",
    "is_part_type",
    "is_provider_passthrough_event",
    "is_provider_passthrough_item",
    "is_reasoning_delta_event",
    "is_reasoning_part",
    "is_refusal_delta_event",
    "is_refusal_part",
    "is_stream_end_event",
    "is_stream_start_event",
    "is_text_delta_event",
    "is_text_part",
    "is_tool_call_delta_event",
    "is_tool_call_part",
    "is_tool_call_start_event",
    "is_tool_result_part",
    "is_usage_event",
    "isinstance_part",
]

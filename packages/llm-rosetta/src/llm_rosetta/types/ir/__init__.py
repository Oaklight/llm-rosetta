"""Deprecated: the IR types moved to :mod:`llm_rosetta.ir.model`.

``llm_rosetta.types.ir`` shipped in 0.13.0.  The IR data classes, type guards
and helpers now live in :mod:`llm_rosetta.ir.model`, beside the IR logic in
:mod:`llm_rosetta.ir`.  This package forwards the names 0.13.0 exported, and its
submodules forward the names 0.13.0 had, as the same objects for one minor release.  Importing it
(or one of those submodules) warns once.  It will be removed in 0.15.0.  Names
and submodules that first appeared in 0.14.0 are not forwarded: import them from
:mod:`llm_rosetta.ir.model`.
"""

from __future__ import annotations

import warnings

from llm_rosetta.ir import model as _model

#: The names ``llm_rosetta.types.ir.__all__`` held on the v0.13.0 tag.
__all__ = [
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "RefusalPart",
    "CitationPart",
    "AudioPart",
    "AudioData",
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    "Message",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "LegacyMessage",
    "MessageMetadata",
    "StreamingMetadata",
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "ReasoningEffortLevel",
    "CacheConfig",
    "IRInputItem",
    "IRRequest",
    "IRResponse",
    "ExtensionItem",
    "ProviderPassthroughItem",
    "ProviderPassthroughEvent",
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    "IRStreamEvent",
    "is_provider_passthrough_event",
    "is_provider_passthrough_item",
    "StreamStartEvent",
    "StreamEndEvent",
    "ContentBlockStartEvent",
    "ContentBlockEndEvent",
    "TextDeltaEvent",
    "ReasoningDeltaEvent",
    "RefusalDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    "EmbeddingTaskType",
    "EmbeddingEncodingFormat",
    "EmbeddingUsageInfo",
    "EmbeddingItem",
    "IREmbeddingRequest",
    "IREmbeddingResponse",
    "RerankDocument",
    "RerankUsageInfo",
    "RerankResultItem",
    "IRRerankRequest",
    "IRRerankResponse",
    "IRInput",
    "IRInputSimple",
    "experimental",
]

for _name in __all__:
    globals()[_name] = getattr(_model, _name)

# Deprecated 0.13.0 path, removed in 0.15.0; the replacement is
# llm_rosetta.ir.model.
warnings.warn(
    "llm_rosetta.types.ir is deprecated and will be removed in 0.15.0; "
    "import from llm_rosetta.ir.model",
    DeprecationWarning,
    stacklevel=2,
)

---
title: API Layers & Import Guide
---

# API Layers & Import Guide

LLM-Rosetta organizes its public API into three stability tiers. This guide helps you choose the right import path and understand what each tier promises.

## Stability Tiers

### Stable API

Symbols you can rely on across minor versions. These are the recommended imports for most users.

| Category | Symbols | Import from |
|----------|---------|-------------|
| Converters | `OpenAIChatConverter`, `AnthropicConverter`, `GoogleGenAIConverter`, `GoogleConverter`, `OpenAIResponsesConverter`, `BaseConverter` | `llm_rosetta` |
| Context | `ConversionContext`, `StreamContext` | `llm_rosetta` |
| Convenience | `detect_provider`, `get_converter_for_provider`, `convert`, `ProviderType` | `llm_rosetta` |
| Core IR types | `Message`, `IRRequest`, `IRResponse`, `ContentPart`, `TextPart`, `ToolCallPart`, stream events, etc. | `llm_rosetta.types.ir` |

```python
# Recommended: stable imports
from llm_rosetta import OpenAIChatConverter, AnthropicConverter
from llm_rosetta import ConversionContext
from llm_rosetta.types.ir import Message, IRRequest, IRResponse, TextPart
```

### Advanced API

Useful for power users who need fine-grained control. These may change in minor versions, but will be documented in the changelog.

| Category | Examples | Import from |
|----------|----------|-------------|
| Type guards | `is_text_part`, `is_tool_call_part`, `is_stream_start_event`, `is_message`, etc. | `llm_rosetta.types.ir.type_guards`, `llm_rosetta.types.ir.messages` |
| Message factories | `create_system_message`, `create_user_message`, `create_assistant_message`, `create_tool_message` | `llm_rosetta.types.ir.messages` |
| Content helpers | `extract_text_content`, `extract_all_text`, `extract_tool_calls`, `create_tool_result_message` | `llm_rosetta.types.ir.helpers` |
| Extension guards | `is_extension_item` | `llm_rosetta.types.ir.extensions` |

```python
# Advanced: import from submodules
from llm_rosetta.types.ir.type_guards import is_text_part, is_tool_call_part
from llm_rosetta.types.ir.messages import create_user_message
from llm_rosetta.types.ir.helpers import extract_text_content
```

!!! note
    These symbols are also importable from `llm_rosetta.types.ir` for convenience, but they are not part of the stable `__all__` surface.

### Internal API

Implementation details used within the library. Import at your own risk — these may change without notice.

| Category | Examples | Import from |
|----------|----------|-------------|
| Type mappings | `TYPE_CLASS_MAP`, `get_part_type`, `isinstance_part` | `llm_rosetta.types.ir.type_guards` |
| Validation | `ValidationError`, `validate_ir_request`, `validate_ir_response` | `llm_rosetta.types.ir.validation` |
| Ops base classes | `BaseContentOps`, `BaseToolOps`, `BaseMessageOps`, `BaseConfigOps` | `llm_rosetta.converters.base.content`, etc. |
| Schema utilities | `sanitize_schema` | `llm_rosetta.converters.base.tools` |
| Content helpers | `convert_content_blocks_to_ir`, `convert_ir_content_blocks_to_p` | `llm_rosetta.converters.base.tool_content` |

```python
# Internal: use deep submodule imports
from llm_rosetta.types.ir.validation import validate_ir_request
from llm_rosetta.converters.base.tools import sanitize_schema
```

## Quick Reference

```text
llm_rosetta                          # Stable: converters, context, convenience
llm_rosetta.types.ir                 # Stable: core IR data types
llm_rosetta.types.ir.type_guards     # Advanced: is_*_part, is_*_event guards
llm_rosetta.types.ir.messages        # Advanced: create_*, is_*_message
llm_rosetta.types.ir.helpers         # Advanced: extract_*, create_tool_result_message
llm_rosetta.types.ir.validation      # Internal: validation utilities
llm_rosetta.converters.base          # Stable: BaseConverter, context classes
llm_rosetta.converters.base.content  # Internal: BaseContentOps
llm_rosetta.converters.base.tools    # Internal: BaseToolOps, sanitize_schema
```

## CI Enforcement

The export surface is enforced by `tests/test_public_api.py`. If you add a new symbol to any `__all__`, the test will fail unless you update the expected export list. This prevents accidental surface growth.

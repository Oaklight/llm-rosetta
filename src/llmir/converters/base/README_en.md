# Base Converter Module - Lightweight Modular Architecture

[中文版](./README_zh.md) | [English Version](./README_en.md)

## Overview

The Base Converter Module provides abstract base classes and lightweight modular architecture for converters, adopting the **composition over inheritance** design pattern to establish unified and flexible interface specifications for all provider converters.

## File Structure

```
src/llmir/converters/base/
├── __init__.py          # Module exports
├── converter.py         # BaseConverter main converter abstract base class
├── atomic_ops.py        # BaseAtomicOps atomic conversion operations abstract base class
├── complex_ops.py       # BaseComplexOps complex conversion operations abstract base class
├── README_en.md         # English documentation
└── README_zh.md         # Chinese documentation
```

## Module Description

### `converter.py` - BaseConverter Main Converter Abstract Base Class (Lightweight)

- **Responsibility**: Define core converter interfaces and common logic
- **Key Features**:
  - `to_provider()`: Smart conversion interface (IR → Provider)
  - `from_provider()`: Smart conversion interface (Provider → IR)
  - `validate_ir_input()`: IR input validation
  - `atomic_ops_class`: Class attribute to specify atomic operations class
  - `complex_ops_class`: Class attribute to specify complex operations class

**Important Change**: All atomic and complex level abstract methods have been removed, replaced with composition pattern using class attributes to specify ops classes.

### `atomic_ops.py` - BaseAtomicOps Atomic Conversion Operations Abstract Base Class

- **Responsibility**: Define unified interfaces for basic content type conversions
- **Abstract Methods**:
  - Text conversion (`ir_text_to_p()`, `p_text_to_ir()`)
  - Image conversion (`ir_image_to_p()`, `p_image_to_ir()`)
  - File conversion (`ir_file_to_p()`, `p_file_to_ir()`)
  - Tool call conversion (`ir_tool_call_to_p()`, `p_tool_call_to_ir()`)
  - Tool result conversion (`ir_tool_result_to_p()`, `p_tool_result_to_ir()`)
  - Tool definition conversion (`ir_tool_to_p()`, `p_tool_to_ir()`)
  - Tool choice conversion (`ir_tool_choice_to_p()`, `p_tool_choice_to_ir()`)
  - Content part conversion (`p_content_part_to_ir()`)

### `complex_ops.py` - BaseComplexOps Complex Conversion Operations Abstract Base Class

- **Responsibility**: Define unified interfaces for message, request, and response level conversions
- **Abstract Methods**:
  - Message level conversion (`ir_message_to_p()`, `p_message_to_ir()`)
  - Content part conversion (`ir_content_part_to_p()`)
  - Request level conversion (`ir_request_to_p()`, `p_request_to_ir()`)
  - Response level conversion (`ir_response_to_p()`, `p_response_to_ir()`)
  - Helper methods (`p_user_message_to_ir()`, `p_assistant_message_to_ir()`)

## Design Advantages

1. **Lightweight Design**: BaseConverter reduced from 298 lines to ~100 lines (66% reduction)
2. **Composition Over Inheritance**: Specify ops classes via class attributes instead of forcing abstract method implementation
3. **Reduced Boilerplate**: Each converter saves ~60-80 lines of delegation code
4. **Clear Separation of Concerns**: Atomic operations, complex operations, and main converters each have distinct responsibilities
5. **Strong Type Constraints**: Ensures consistency across all implementations through abstract base classes
6. **Extensibility**: Adding new providers only requires implementing ops classes and setting class attributes
7. **Maintainability**: Modular design facilitates understanding, testing, and maintenance
8. **Flexibility**: Easy to switch or combine different ops implementations

## Implementation Guide

### Creating a New Provider Converter

1. **Create provider directory**:

   ```
   src/llmir/converters/your_provider/
   ├── __init__.py
   ├── converter.py
   ├── atomic_ops.py
   └── complex_ops.py
   ```

2. **Implement atomic operations**:

   ```python
   from ..base import BaseAtomicOps

   class YourProviderAtomicOps(BaseAtomicOps):
       @staticmethod
       def ir_text_to_p(text_part, **kwargs):
           # Implement text conversion logic
           pass

       # Implement other abstract methods...
   ```

3. **Implement complex operations**:

   ```python
   from ..base import BaseComplexOps

   class YourProviderComplexOps(BaseComplexOps):
       @staticmethod
       def ir_message_to_p(message, ir_input, **kwargs):
           # Implement message conversion logic
           pass

       # Implement other abstract methods...
   ```

4. **Implement main converter (lightweight approach)**:

   ```python
   from ..base import BaseConverter
   from .atomic_ops import YourProviderAtomicOps
   from .complex_ops import YourProviderComplexOps

   class YourProviderConverter(BaseConverter):
       # Set ops class attributes
       atomic_ops_class = YourProviderAtomicOps
       complex_ops_class = YourProviderComplexOps

       def to_provider(self, ir_data, **kwargs):
           # Directly call ops class static methods
           converted, warnings = self.complex_ops_class.ir_message_to_p(message, **kwargs)
           # ... implement conversion logic
           pass

       def from_provider(self, provider_data, **kwargs):
           # Directly call ops class static methods
           ir_message = self.complex_ops_class.p_message_to_ir(provider_message)
           # ... implement conversion logic
           pass
   ```

   **Note**: No need to implement numerous delegation methods anymore. Simply use `self.atomic_ops_class` and `self.complex_ops_class` to call ops methods directly.

## Existing Implementations

### Refactored (Lightweight Architecture)

- ✅ **OpenAI Chat Converter**: Adopted lightweight architecture
  - [`OpenAIChatConverter`](../openai_chat/converter.py): Main converter (236 lines, 28% reduction)
  - [`OpenAIChatAtomicOps`](../openai_chat/atomic_ops.py): Atomic operations
  - [`OpenAIChatComplexOps`](../openai_chat/complex_ops.py): Complex operations

### Pending Refactoring (Traditional Architecture)

- ⏳ **Anthropic Converter**: Uses traditional monolithic architecture
- ⏳ **Google Converter**: Uses traditional monolithic architecture
- ⏳ **OpenAI Responses Converter**: Uses traditional monolithic architecture

**Note**: Traditional architecture converters still work normally and can be gradually migrated to lightweight architecture as needed.

## Bidirectional Conversion Support

The new architecture fully supports bidirectional conversion:

1. **IR → Provider**: `ir_*_to_p()` series methods
2. **Provider → IR**: `p_*_to_ir()` series methods

This enables LLMIR to be used for building AI API bridge services, performing bidirectional conversions between different formats.

## Method Naming Convention

- **`ir_*_to_p()`**: Convert from IR format to Provider format
- **`p_*_to_ir()`**: Convert from Provider format to IR format
- **`*_message_*`**: Message-level operations
- **`*_request_*`**: Request-level operations
- **`*_response_*`**: Response-level operations
- **`*_content_part_*`**: Content part operations
- **`*_tool_*`**: Tool-related operations

## Type Safety

All abstract methods include proper type hints to ensure:

- Input parameter types are clearly defined
- Return types are explicitly specified
- Optional parameters are properly marked
- Generic types are used where appropriate

## Error Handling

Base classes provide common error handling patterns:

- Input validation for IR data
- Type checking for provider data
- Graceful handling of unsupported features
- Warning collection for lossy conversions

## Testing Strategy

The modular architecture facilitates comprehensive testing:

- **Unit tests**: Test individual atomic operations
- **Integration tests**: Test complex operations and workflows
- **End-to-end tests**: Test complete conversion pipelines
- **Compatibility tests**: Ensure backward compatibility

## Refactoring Results

Through adopting the lightweight architecture, we achieved:

- **Code Reduction**: BaseConverter reduced by 66%, OpenAIChatConverter by 28%
- **Clearer Architecture**: Composition over inheritance, clearer responsibilities
- **100% Backward Compatible**: All 137 tests passed
- **Easier Maintenance**: Reduced boilerplate code, improved code quality

For detailed refactoring information, please refer to the project documentation.

## Future Roadmap

The lightweight modular architecture enables systematic refactoring of existing converters:

1. **Anthropic Converter**: Can be refactored following the same pattern
2. **Google Converter**: Can be refactored following the same pattern
3. **OpenAI Responses Converter**: Can be refactored following the same pattern
4. **New Providers**: Only need to implement ops classes and set class attributes

This creates a consistent, maintainable, and extensible foundation for the entire LLMIR converter ecosystem.

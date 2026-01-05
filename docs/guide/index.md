# User Guide

This guide provides detailed usage instructions and best practices for LLMIR.

## Core Concepts

### Intermediate Representation (IR)

LLMIR uses a unified intermediate representation to handle message formats from different providers. This design allows:

- Lossless conversion: Maintain message integrity
- Extensibility: Easily add support for new providers
- Consistency: Unified data structures and processing logic

### Converters

Converters are the core components of LLMIR, responsible for conversion between specific provider formats and intermediate representation:

- `to_ir()`: Convert provider format to intermediate representation
- `from_ir()`: Convert intermediate representation to provider format

## Main Features

### [Converters](converters/)

Learn how to use different converters:

- [OpenAI Chat](converters/openai-chat.md) - OpenAI Chat Completions API
- [OpenAI Responses](converters/openai-responses.md) - OpenAI Responses API
- [Anthropic](converters/anthropic.md) - Claude message format
- [Google](converters/google.md) - Gemini/PaLM message format

### [Intermediate Representation](intermediate-representation.md)

Deep dive into LLMIR's intermediate representation design and data structures.

### [Auto Detection](auto-detection.md)

Learn how to use the auto-detection feature to identify message formats.

## Best Practices

### Performance Optimization

- Reuse converter instances to avoid repeated initialization
- Consider batch processing for large volumes of messages
- Use appropriate configuration options to balance performance and functionality

### Error Handling

- Always handle exceptions that may occur during conversion
- Use validation features to ensure input data correctness
- Log warnings and errors during the conversion process

### Type Safety

- Use type annotations to improve code quality
- Leverage IDE type checking capabilities
- Follow LLMIR's type definitions

## Common Use Cases

1. **Multi-provider Support**: Support multiple LLM providers in applications
2. **Format Standardization**: Unify different format messages into standard format
3. **Data Migration**: Migrate conversation data between different providers
4. **Testing and Development**: Use unified format for testing and development

## Next Steps

- [View specific converter documentation](converters/)
- [Learn about API reference](../api/)
- [Check out real examples](../examples/)
# Development Guide

Welcome to LLMIR development! This guide will help you understand how to contribute to the project.

## Development Environment Setup

### Clone Repository

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

## Project Structure

```
llmir/
├── src/llmir/           # Main source code
│   ├── converters/      # Converter implementations
│   ├── types/          # Type definitions
│   └── utils/          # Utility functions
├── tests/              # Test code
├── docs/               # Documentation
└── examples/           # Example code
```

## Contributing Guidelines

### Reporting Issues

If you find bugs or have feature requests, please create an issue on GitHub.

### Submitting Code

1. Fork the repository
2. Create a feature branch
3. Write code and tests
4. Submit a Pull Request

### Code Standards

- Follow PEP 8 code style
- Add type annotations
- Write test cases
- Update documentation

## Architecture Design

Learn about LLMIR's core architecture and design principles:

- [Architecture Design](architecture.md)
- [Contributing Guide](contributing.md)

## Next Steps

- [View contributing guide](contributing.md)
- [Learn about architecture design](architecture.md)
- [Join community discussions](https://github.com/Oaklight/llmir/discussions)
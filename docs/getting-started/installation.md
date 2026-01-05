# Installation

This page will guide you through installing LLMIR.

## System Requirements

- Python 3.8 or higher
- pip package manager

## Install with pip

The simplest way to install is using pip:

```bash
pip install llmir
```

## Install from Source

If you want the latest development version, you can install from the GitHub repository:

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir
pip install -e .
```

## Verify Installation

After installation, you can verify that the installation was successful:

```python
import llmir
print(llmir.__version__)
```

## Optional Dependencies

LLMIR's core functionality doesn't require additional dependencies, but some advanced features may require installing extra packages:

```bash
# For development and testing
pip install llmir[dev]

# For documentation building
pip install llmir[docs]

# Install all optional dependencies
pip install llmir[all]
```

## Troubleshooting

If you encounter issues during installation, check these common solutions:

### Python Version Issues

Make sure you're using Python 3.8 or higher:

```bash
python --version
```

### Permission Issues

If you encounter permission errors, you can use user installation mode:

```bash
pip install --user llmir
```

### Network Issues

If you have network connectivity issues, you can use a mirror:

```bash
pip install -i https://pypi.org/simple llmir
```

## Next Steps

After installation, you can:

- [Learn basic usage](basic-usage.md)
- [View example code](../examples/)
- [Read API documentation](../api/)
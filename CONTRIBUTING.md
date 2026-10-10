# Contributing to LLM-Rosetta

Thanks for your interest in contributing! Full contributing documentation lives on ReadTheDocs:

- [Contributing Guide](https://llm-rosetta.readthedocs.io/en/latest/contributing/guide/) — getting started, branch naming, commit messages, PR workflow
- [Style Guide](https://llm-rosetta.readthedocs.io/en/latest/contributing/style-guide/) — code style, docstrings, naming conventions, tooling
- [Architecture Guide](https://llm-rosetta.readthedocs.io/en/latest/contributing/architecture/) — converter structure, ops modules, round-trip compatibility

## Design decisions

Standing choices that should not be reversed without a new decision record:

- **The gateway always converts through IR — including same-format routes.** Every request still converts through IR; no path skips it. `baseline` is the pipeline's diagnostic mode for surfacing translation problems **synchronously** (it shadows the IR path, and the fidelity checker diffs the two) — not a gateway or production path. `prefer_same_format` chooses only *which* upstream provider is used — it never bypasses IR. Rationale: [#577 (comment)](https://github.com/Oaklight/llm-rosetta/issues/577#issuecomment-5461182245).

## Quick Start

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta
pip install -e ".[all]"
pre-commit install
make test
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

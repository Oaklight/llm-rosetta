---
title: Contributing
---

# Contributing to LLM-Rosetta

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Install dev dependencies: `pip install -e ".[all]"`
4. Set up pre-commit hooks: `pre-commit install`
5. Make your changes
6. Run `make lint` (or let pre-commit catch issues on commit)
7. Run `make test` to verify nothing is broken
8. Commit and push
9. Open a Pull Request

## Branch Naming

Use descriptive prefixes:

- `feature/xxx` — new functionality
- `fix/xxx` — bug fix
- `refactor/xxx` — code restructuring
- `docs/xxx` — documentation updates
- `test/xxx` — test additions or changes

## Commit Messages

Keep commit messages concise and focused on *why*, not *what*. One logical change per commit.

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Include a brief description of what changed and why
- Mention any breaking changes explicitly
- Ensure `make lint` passes before submitting
- Merge strategy: rebase (use `scripts/merge-pr.sh` for local rebase merges)

## Adding a New Converter

LLM-Rosetta uses a hub-and-spoke architecture. To add support for a new API standard:

1. Create a converter directory under `src/llm_rosetta/converters/<name>/`
2. Implement bidirectional conversion (request/response) to/from IR
3. Add a shim in `src/llm_rosetta/shims/builtins.py` if needed
4. Add tests under `tests/converters/`
5. Submit a PR

See existing converters under `src/llm_rosetta/converters/` for reference.

## AI-Assisted Contributions

Using AI tools (e.g. Claude, Cursor, Copilot) to assist with development is welcome. However:

- **No AI co-author tags in commits.** Do not add `Co-authored-by` lines for AI tools in git commit messages. This keeps the git history clean and readable.
- **Disclose in PR description.** If AI tools were used significantly in your contribution, add a brief note in the PR description (e.g. "AI was used to assist with implementation").
- **You own the code.** Contributors are fully responsible for any AI-generated code they submit — review it, test it, understand it.

## Code Style

- Python code follows `ruff` defaults
- Docstrings use Google style
- Comments and docstrings in English
- Type hints are encouraged
- Do not edit files under `src/llm_rosetta/_vendor/` — those are managed externally

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE).

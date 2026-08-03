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

## Converter Architecture

LLM-Rosetta uses a hub-and-spoke architecture where each converter handles bidirectional conversion between a specific API format and the shared IR.

### Structure

Every converter lives under `src/llm_rosetta/converters/<name>/` and extends `BaseConverter`. The base class uses a **composition pattern** — subclasses declare ops classes for each concern:

```
converters/<name>/
├── converter.py      # Main converter class (extends BaseConverter)
├── content_ops.py    # Content part conversion (text, images, refusal, etc.)
├── message_ops.py    # Message-level conversion (roles, multi-turn)
├── tool_ops.py       # Tool definitions, tool calls, tool results
├── config_ops.py     # Request config (temperature, top_p, stream options)
└── _constants.py     # Format-specific constants
```

New features for an existing format (e.g. a new content type, a new field) should be implemented in the **corresponding ops module** of that converter, not in ad-hoc standalone code. Reuse shared logic from `converters/base/` wherever possible — the base modules (`content.py`, `messages.py`, `tools.py`, `reasoning.py`, `schema.py`, etc.) provide common building blocks that all converters share.

### Adding a New Converter

To add support for a new API standard:

1. Create a converter directory under `src/llm_rosetta/converters/<name>/`
2. Subclass `BaseConverter` and implement all abstract methods
3. Create ops classes following the pattern above
4. Add a shim under `src/llm_rosetta/shims/providers/<name>/`
5. Add tests under `tests/converters/`
6. Submit a PR

See existing converters for reference.

### Round-Trip Compatibility

All conversion paths must maintain **round-trip compatibility**. Every change must be tested against these scenarios:

- **A → IR → A** (same-format round-trip): converting to IR and back to the same format must produce a valid, semantically equivalent result. No fields should be silently dropped.
- **A → IR → B** (cross-format): converting from one format to another must produce valid output for the target format, even if the source format has fields with no direct equivalent.
- **A → IR → B → IR → A** (full round-trip): a message that goes through two conversions and back must remain usable. This is the gateway's actual execution path — the request converts inbound, the response converts outbound.

When adding or modifying converter logic, write tests that cover at least the first two scenarios. The gateway's cross-format routing depends on all converters agreeing on IR semantics — a change that breaks one converter's IR output can cascade into failures for every other converter.

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

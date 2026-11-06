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
make test-js   # admin UI: node --test over tests/js/ (skipped when node is absent)
make check-public-api  # public surface vs the latest stable tag
```

## Running the full suite on another machine

`make test` pins a core on your dev box for a couple of minutes. When you want
the full suite without that, `scripts/dev/rtest.sh` copies a worktree to a
remote host and runs pytest there:

```bash
RTEST_HOST=me@build-host scripts/dev/rtest.sh   # tests/, integration excluded
RTEST_HOST=me@build-host scripts/dev/rtest.sh tests/gateway -q --no-cov
```

| Variable | Meaning |
| --- | --- |
| `RTEST_HOST` | ssh destination. Required. |
| `RTEST_ROOT` | remote directory holding the trees (default `$HOME/.cache/rtest`) |
| `RTEST_PYTHON` | interpreter on that host (default `<RTEST_ROOT>/venv/bin/python`) |
| `RTEST_SSH_OPTS` | extra ssh options, passed through verbatim |

The file list comes from `git ls-files -co --exclude-standard`, so ignored
files stay home and a worktree's own ignore rules apply. Each tree syncs to
`<RTEST_ROOT>/<basename>-<path-hash>/`, so parallel sessions never share a
`--delete` target. The suite runs *inside that copy* — the `cd` in the command
below is what makes it exercise your working tree rather than whatever is
installed on the remote — and `--dry-run` shows exactly which tree that is:

```console
$ RTEST_HOST=me@build-host scripts/dev/rtest.sh --dry-run
host:    me@build-host
tree:    /src/llm-rosetta-myfeature
files:   1091
dest:    .cache/rtest/llm-rosetta-myfeature-1a2b3c4d
rsync:   git ls-files -z -co --exclude-standard | rsync -a --delete --from0 --files-from=- /src/llm-rosetta-myfeature/ me@build-host:.cache/rtest/llm-rosetta-myfeature-1a2b3c4d/
command: ssh me@build-host 'mkdir -p .cache/rtest/llm-rosetta-myfeature-1a2b3c4d && cd .cache/rtest/llm-rosetta-myfeature-1a2b3c4d && PYTHONPATH=packages/llm-rosetta-gateway/src:packages/llm-rosetta/src $HOME/.cache/rtest/venv/bin/python -m pytest tests/ --ignore=tests/integration -q'
```

Limits:

- The remote host needs the project's dependencies installed in
  `RTEST_PYTHON`; the script installs nothing.
- The docs worktrees are separate checkouts, so they only travel if they sit
  inside the synced tree.
- **Permissions are only as faithful as the remote account.** A host that runs
  the job as root (uid 0) ignores read-only directory modes, so a test that
  asserts a permission is refused can fail there and still pass locally —
  `tests/gateway/test_config_load_migrations.py::TestThePublicReader::test_a_read_only_directory_still_serves_the_migrated_document`
  is the known case. Treat a lone permission failure as environmental and
  re-check it on your own machine before chasing it.

## Public API changes

CI compares the public surface (`__all__` names and callable signatures) of the
latest stable `vX.Y.Z` tag with the working tree. Removing a name or parameter
fails the check.

Keep the old spelling as a deprecated alias until its removal window. If a
removal is intentional, record it with a reason in
`scripts/dev/public_api_allowlist.json`. Additions and optional parameters are
reported for information only.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

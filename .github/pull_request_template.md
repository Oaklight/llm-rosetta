<!-- Read the "Pull requests" section of CONTRIBUTING.md first. -->

## What and why

<!-- One concern per PR. Link the issue. -->

## Risk tier (tick one)

- [ ] Pure move: code moves, no logic edit (a mechanical check proves it)
- [ ] Boundary or contract change: imports, public API, config schema, wire format
- [ ] Runtime-behaviour change: logic that runs differently after this PR
- [ ] Security-relevant: auth, keys, credentials, sanitizing, network exposure

## Verification

- [ ] `make lint` and `make test` pass. Result:
- [ ] `ty` clean (`python scripts/dev/run_ty.py`; new diagnostics: 0):
- [ ] Pure move only: move-check summary (tool: `check_move.py`, added by a sibling PR):
- [ ] Overlap check done: `python scripts/dev/pr_overlap.py` shows no open PR on the same files, or the merge order is stated here:
- [ ] CI is fully green before review is requested

## AI disclosure

- [ ] AI tools were used for: <!-- describe, or write "none" -->. No `Co-authored-by` AI tags are in the commits.

## Skipped hooks

<!-- If a hook was skipped (for example a CI-only check you could not run locally), name it and say why. Write "none" otherwise. -->

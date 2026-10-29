# Conformance probes

Developer-only probe harness. It lives outside the `packages/` trees so the wheel
does not ship it.

```bash
python -m tools.conformance run --suite drift --out /tmp/sdk-report.json
```

A report is schema-versioned (`schema_version: 1`), carries the environment
and one result per case, and uses four statuses:

| status | meaning |
| --- | --- |
| `pass` | the probe ran and the expectation held |
| `fail` | the probe ran and the result is wrong |
| `error` | the probe could not run (missing module, network, timeout) |
| `skip` | the probe is not comparable here (no baseline, changed SDKs) |

One case never aborts the others; transient errors can be retried with
`--retries`. Every string that reaches the report passes three redaction
layers: the exact credential values a suite resolved (`--secret-env NAME`,
repeatable), values under sensitive mapping keys (`x-api-key`, `authorization`,
`token`, …), and the credential patterns shared with the gateway plus
`user:pass@` and `key=value` parameters. A credential the harness knows about
cannot survive in any position, including a bare mention in an error message.

## Suites

### `drift`

Wraps [`scripts/dev/scan_sdk_types.py`](../../scripts/dev/scan_sdk_types.py).
Each provider is a case; a case fails when a field absent from
`baselines/sdk-types.json` appears in a replica. A different SDK version skips
the case, because the baseline is only comparable against the SDKs that
produced it.

Refresh the baseline after reviewing the new drift:

```bash
python -c "from tools.conformance.suites.drift import write_baseline; write_baseline()"
```

### `fields`

Every response field of each chat format, taken from the provider SDK type
(`openai.types.chat.ChatCompletion`, `openai.types.responses.Response`,
`anthropic.types.Message`, `google.genai.types.GenerateContentResponse`) or, for
the Interactions API, from a hand-authored schema. Each field path is recorded in
`baselines/response-fields.json` with a classification:

| class | meaning |
| --- | --- |
| `content` | part of the comparable turn content |
| `envelope` | id/model/type/index metadata |
| `terminal` | why the turn ended |
| `usage` | token accounting |
| `provider_only` | provider metadata that is not compared (needs a reason) |
| `echoed_request` | Responses echoes the request configuration (needs a reason) |
| `not_on_wire` | SDK-only helper (needs a reason) |
| `not_compared` | a union variant the harness does not exercise (needs a reason) |

A field the SDK declares and the ledger does not have fails the case; a field the
ledger has and the SDK no longer declares is reported, not failed. Union variants
are classified the same way, so a new output item or content block is visible
without enumerating every deep field of a type the harness does not exercise.

```bash
python -m tools.conformance run --suite fields --out /tmp/fields-report.json
```

Refresh the ledger after reviewing a change:

```bash
python -c "from tools.conformance.suites.fields import write_baseline; write_baseline()"
```

New fields are added as `unclassified`, so the suite keeps failing until they are
classified. `write_baseline(seed=True)` fills them with a heuristic classification
instead; use it only when seeding the ledger for the first time.

The ledger records the SDK versions it was generated against, but they are
provenance rather than a gate: the field set is compared regardless of version, so
an upstream release that adds a field is a named failure with a reviewable diff.
A format whose SDK is not installed skips; a missing or renamed ledger fails,
because that is a repository state rather than an environment property. The suite
runs in the `conformance` CI job, which installs `[all]`.

### `wire`

One canonical request, round-tripped through each of the five chat formats
(`IR → target body → IR`), in two cases per format:

```bash
python -m tools.conformance run --suite wire --out /tmp/wire-report.json
```

| case | request |
| --- | --- |
| `request/<format>` | instructions, a user turn, a function tool, `tool_choice: auto`, `temperature`/`top_p`/`max_tokens` |
| `sampling/<format>` | the same, plus every other `GenerationConfig` knob — `top_k`, `stop_sequences`, `seed`, `logprobs`, `top_logprobs` |

These are **identity** round trips (format → its own body → back), not the 25
ordered format pairs; cross-format translation is what the converter test
suites cover.  Each case reports what the round trip lost and which
conversion warnings named it.

A case **fails** only when it loses a field no conversion warning accounts
for.  "Accounts for" is decided on the warning's `subject`: it must name the
field (or the container that held it).  Prose is a fallback only for
reporters with no subject yet, and only for a value that vanished — a
*changed* value needs a warning that names it.  That keeps the suite to the
one assertion it can make without knowing each vendor's surface: a loss the
caller is told about is data, a silent one is a red case.

The canonical request is deliberately inside the shared core, so a
difference means the format is not carrying something it should, rather than
that the request asked for an extension.  The suite is offline and
deterministic; upstream wire behaviour (what a vendor accepts and echoes) is
the next family of cases and needs the scheduled, credentialled run the epic
plans.

`baselines/wire.json` records what each case loses today.  A loss that appears
and is not in the baseline fails the case with the path that changed; a loss
that disappears is reported as `restored` and does not.  After reviewing a
change deliberately:

```bash
python -c "from tools.conformance.suites.wire import write_baseline; write_baseline()"
```

Adding a case is a two-step change: with no entry in the baseline, every loss
it reports is "new" and the case fails on its first run — which is the point,
since a case has to be reviewed rather than trusted. Run it once, decide the
losses are the ones you meant, then re-baseline.

CI runs this suite on every PR that touches the core, and uploads the report.

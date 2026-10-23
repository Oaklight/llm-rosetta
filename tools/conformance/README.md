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

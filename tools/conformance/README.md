# Conformance probes

Developer-only probe harness. It lives outside `src/llm_rosetta` so the wheel
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

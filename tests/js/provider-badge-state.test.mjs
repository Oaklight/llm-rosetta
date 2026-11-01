/**
 * How a provider's capability badge reports a non-routing endpoint.
 *
 * A badge is greyed and carries an `!` overlay in two cases: the endpoint
 * failed to parse (degraded, #1106) or the operator turned it off
 * (`<type>_enabled: false`).  The state has to be derived from the entry
 * alone — the card renders from the same config payload the editor saves —
 * and an entry that says nothing must stay `ok`, or every badge would grey
 * out.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { endpointState, degradedReason, providerStatus, providerOrder } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/provider-endpoints.js';

test('an entry that says nothing is ok', () => {
  assert.equal(endpointState({}, 'embeddings'), 'ok');
  assert.equal(endpointState(null, 'embeddings'), 'ok');
  assert.equal(endpointState(undefined, 'embeddings'), 'ok');
  assert.equal(endpointState({ embeddings_enabled: true }, 'embeddings'), 'ok');
  assert.equal(endpointState({ embeddings_enabled: '' }, 'embeddings'), 'ok');
  assert.equal(endpointState({}, ''), 'ok');
});

test('the flag disables exactly the type it names', () => {
  const cfg = { embeddings_enabled: false };
  assert.equal(endpointState(cfg, 'embeddings'), 'disabled');
  assert.equal(endpointState(cfg, 'rerank'), 'ok');
  assert.equal(endpointState(cfg, 'llm'), 'ok');
});

test('a degraded endpoint wins over a disabled one', () => {
  const cfg = {
    embeddings_enabled: false,
    _auto_disabled_endpoints: {
      embeddings: { reason: 'embeddings_shim is unknown', category: 'unknown_shim' },
    },
  };
  assert.equal(endpointState(cfg, 'embeddings'), 'degraded');
  assert.equal(degradedReason(cfg, 'embeddings'), 'embeddings_shim is unknown');
});

test('the reason is empty for anything that is not degraded', () => {
  assert.equal(degradedReason({ embeddings_enabled: false }, 'embeddings'), '');
  assert.equal(degradedReason({}, 'embeddings'), '');
  assert.equal(degradedReason(null, 'embeddings'), '');
});

test('a provider is ok, off, or needs attention', () => {
  assert.equal(providerStatus({}), 'ok');
  assert.equal(providerStatus({ enabled: true }), 'ok');
  assert.equal(providerStatus({ enabled: false }), 'off');
  // A degraded endpoint is a problem even while the provider serves other
  // types; the whole-provider failure marker is a problem too.
  assert.equal(providerStatus({ embeddings_enabled: false }), 'ok');
  assert.equal(
    providerStatus({
      _auto_disabled_endpoints: { rerank: { reason: 'x', category: 'unknown_shim' } },
    }),
    'problem',
  );
  assert.equal(providerStatus({ enabled: false, _auto_disabled: { reason: 'x' } }), 'problem');
  assert.equal(providerStatus(null), 'ok');
});

test('a provider that failed whole degrades every capability', () => {
  // `_auto_disabled` is the provider's own shim failing, which drops every
  // endpoint it serves — not just one — so no badge may read as healthy.
  const cfg = {
    enabled: false,
    _auto_disabled: { reason: 'unknown shim', category: 'unknown_shim' },
  };
  assert.equal(endpointState(cfg, 'llm'), 'degraded');
  assert.equal(endpointState(cfg, 'embeddings'), 'degraded');
  assert.equal(degradedReason(cfg, 'llm'), 'unknown shim');
});

test('a provider whose every capability is off reads off to the facet', () => {
  // Without the capability list only the provider-level switch is known, so
  // the state stays `ok`; the page always passes the caps it renders.
  assert.equal(providerStatus({ embeddings_enabled: false }), 'ok');
  assert.equal(providerStatus({ embeddings_enabled: false }, ['embeddings']), 'off');
  // One endpoint still serving keeps the provider ok.
  assert.equal(
    providerStatus({ embeddings_enabled: false, rerank_enabled: true }, ['embeddings', 'rerank']),
    'ok',
  );
  // A degraded endpoint is a problem, not an off: it is at the top of the list.
  assert.equal(
    providerStatus(
      { embeddings_enabled: false, _auto_disabled_endpoints: { rerank: { reason: 'x', category: 'unknown_shim' } } },
      ['embeddings', 'rerank'],
    ),
    'problem',
  );
  assert.equal(providerOrder({ embeddings_enabled: false }, ['embeddings']), 2);
});

test('problems sort first, switched-off providers last', () => {
  const problem = { _auto_disabled_endpoints: { rerank: { reason: 'x', category: 'unknown_shim' } } };
  const ok = { base_url: 'https://x' };
  const off = { enabled: false };

  assert.ok(providerOrder(problem) < providerOrder(ok));
  assert.ok(providerOrder(ok) < providerOrder(off));
  assert.equal(providerOrder(problem), 0);
  assert.equal(providerOrder(off), 2);
});

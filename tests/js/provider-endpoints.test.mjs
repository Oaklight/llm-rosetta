/**
 * The provider card's degraded-endpoint marker (#1106).
 *
 * A provider that serves several model types keeps the endpoints it parsed
 * when one of them fails; the failure travels to the panel as
 * `_auto_disabled_endpoints`.  The card must show it, and it must not invent
 * one: a marker that says nothing, or a shape the payload would never send,
 * has to be dropped rather than rendered as an empty chip.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { degradedEndpoints, categoryKey } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/provider-endpoints.js';

test('a provider with no degraded endpoint yields nothing', () => {
  assert.deepEqual(degradedEndpoints({}), []);
  assert.deepEqual(degradedEndpoints({ base_url: 'https://x', _auto_disabled: { reason: 'r' } }), []);
  assert.deepEqual(degradedEndpoints(null), []);
  assert.deepEqual(degradedEndpoints(undefined), []);
});

test('degraded endpoints come back sorted by model type', () => {
  const cfg = {
    _auto_disabled_endpoints: {
      rerank: { reason: 'rerank_shim is unknown', category: 'unknown_shim' },
      embeddings: { reason: 'embeddings_shim serves llm, not embeddings', category: 'wrong_model_type' },
    },
  };
  assert.deepEqual(degradedEndpoints(cfg), [
    { type: 'embeddings', reason: 'embeddings_shim serves llm, not embeddings', category: 'wrong_model_type' },
    { type: 'rerank', reason: 'rerank_shim is unknown', category: 'unknown_shim' },
  ]);
});

test('a marker with no reason and no known category is dropped', () => {
  const cfg = {
    _auto_disabled_endpoints: {
      embeddings: { category: 'unknown_shim' },
      rerank: {},
      decision: { reason: '', category: '' },
      llm: null,
      other: 'not-a-marker',
    },
  };
  assert.deepEqual(degradedEndpoints(cfg), [
    { type: 'embeddings', reason: '', category: 'unknown_shim' },
  ]);
});

test('an unrecognised category keeps its reason and loses the category', () => {
  const cfg = { _auto_disabled_endpoints: { embeddings: { reason: 'something new', category: 'brand_new' } } };
  const [entry] = degradedEndpoints(cfg);
  assert.equal(entry.reason, 'something new');
  assert.equal(categoryKey(entry.category), '');
  // ...and the card then shows the reason, because the category has no label.
  assert.equal(categoryKey('unknown_shim'), 'provider.endpointCategory.unknown_shim');
});

test('a malformed marker container is ignored rather than thrown on', () => {
  assert.deepEqual(degradedEndpoints({ _auto_disabled_endpoints: 'nope' }), []);
  assert.deepEqual(degradedEndpoints({ _auto_disabled_endpoints: ['nope'] }), []);
  assert.deepEqual(degradedEndpoints({ _auto_disabled_endpoints: null }), []);
});

test('the whole-provider marker does not become an endpoint marker', () => {
  const cfg = {
    enabled: false,
    _auto_disabled: { reason: 'unknown shim nope--llm', category: 'unknown_shim', shim: 'nope--llm' },
  };
  assert.deepEqual(degradedEndpoints(cfg), []);
});

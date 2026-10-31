/**
 * The typed endpoint's single dropdown: the tree it shows and what its value
 * means in the config.
 *
 * A typed endpoint lists two kinds of choice in one place — a bare format
 * (a custom base URL on an OpenAI-dialect endpoint) and a vendor shim (the
 * hosted default).  The two must not collide even when the vendor and the
 * format share a name (`openai`), and saving a bare format must clear the shim
 * key so a stale shim cannot keep supplying another vendor's path.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildEndpointTypeModel,
  endpointChoiceConfig,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-tree.js';

const SHIMS = [
  {
    name: 'openai--embedding',
    model_type: 'embeddings',
    format: 'openai',
    group: 'openai',
    group_label: 'OpenAI',
    display_name: 'Embeddings',
  },
  {
    name: 'cohere--embedding',
    model_type: 'embeddings',
    format: 'cohere',
    group: 'cohere',
    group_label: 'Cohere',
    display_name: 'Embeddings',
  },
];

const FORMATS = ['openai', 'cohere', 'jina'];

test('formats and vendor shims are two sections in one list', () => {
  const { roots } = buildEndpointTypeModel(SHIMS, FORMATS);
  const formats = roots.filter((r) => r.kind === 'format').map((r) => r.name);
  const vendors = roots.filter((r) => r.kind !== 'format').map((r) => r.name);
  assert.deepEqual(formats, ['openai', 'cohere', 'jina']);
  assert.deepEqual(vendors, ['vendor:cohere', 'vendor:openai']);
});

test('a bare format survives a vendor of the same name', () => {
  const { roots } = buildEndpointTypeModel(SHIMS, FORMATS);
  const byName = new Map(roots.map((r) => [r.name, r]));
  assert.equal(byName.get('openai').kind, 'format');
  assert.equal(byName.get('openai').defaultValue, 'openai');
  assert.equal(byName.get('vendor:openai').kind, 'shim');
  assert.equal(byName.get('vendor:openai').defaultValue, 'openai--embedding');
});

test('a vendor row commits its first shim', () => {
  const { roots } = buildEndpointTypeModel(SHIMS, []);
  const vendor = roots.find((r) => r.name === 'vendor:cohere');
  assert.equal(vendor.label, 'Cohere');
  assert.deepEqual(vendor.children.map((c) => c.value), ['cohere--embedding']);
});

test('a shim value names the shim and the format it serves', () => {
  assert.deepEqual(
    endpointChoiceConfig('cohere--embedding', SHIMS, FORMATS),
    { shim: 'cohere--embedding', format: 'cohere' },
  );
});

test('a bare format clears the shim key', () => {
  // The vendor shim `openai--embedding` exists, and must not win: the operator
  // picked the bare format.
  assert.deepEqual(
    endpointChoiceConfig('openai', SHIMS, FORMATS),
    { shim: '', format: 'openai' },
  );
});

test('no choice falls back to the type default', () => {
  assert.deepEqual(
    endpointChoiceConfig('', SHIMS, FORMATS),
    { shim: '', format: 'openai' },
  );
  assert.deepEqual(endpointChoiceConfig('', SHIMS, []), { shim: '', format: '' });
});

test('a dialect the tree does not list survives as the format', () => {
  // A config written by a newer release (or an external shim that did not
  // load) names a format this build has never heard of; reopening and saving
  // must not rewrite it to the type default.
  assert.deepEqual(
    endpointChoiceConfig('fixed', SHIMS, FORMATS),
    { shim: '', format: 'fixed' },
  );
});

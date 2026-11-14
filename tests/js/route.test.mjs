/**
 * The address bar's grammar (#1521).
 *
 * Every tab had one URL and one title, filters lived only in the DOM, and the
 * model drawer could not be linked.  The grammar — `#<tab>`, `#<tab>/<id>`,
 * `?q=…` — is pinned here so a tab cannot invent its own spelling.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildHash,
  documentTitleOf,
  parseHash,
  sameRoute,
} from './route-helpers.mjs';

test('a bare hash is the front door', () => {
  assert.deepEqual(parseHash(''), { tab: 'dashboard', id: '', params: {} });
  assert.deepEqual(parseHash('#'), { tab: 'dashboard', id: '', params: {} });
});

test('a tab alone names that tab', () => {
  assert.deepEqual(parseHash('#models'), { tab: 'models', id: '', params: {} });
});

test('the old #logs spelling still opens observability', () => {
  assert.equal(parseHash('#logs').tab, 'observability');
  assert.equal(buildHash({ tab: 'logs' }), '#observability');
});

test('an id after the tab is the page inside it', () => {
  assert.deepEqual(parseHash('#models/claude%2Fhaiku'), {
    tab: 'models', id: 'claude/haiku', params: {},
  });
  assert.equal(buildHash({ tab: 'models', id: 'claude/haiku' }), '#models/claude%2Fhaiku');
});

test('filters travel as params, and only the known ones', () => {
  assert.deepEqual(parseHash('#providers?q=attn&provider=openrouter').params, {
    q: 'attn', provider: 'openrouter',
  });
  assert.deepEqual(parseHash('#providers?nonsense=1').params, {});
});

test('an empty param is not written', () => {
  assert.equal(buildHash({ tab: 'keys', params: { q: '' } }), '#keys');
});

test('parse and build round-trip', () => {
  const route = { tab: 'models', id: 'm one', params: { q: 'claude', provider: 'anthropic' } };
  assert.deepEqual(parseHash(buildHash(route)), route);
});

test('a hash that cannot be decoded keeps its text instead of throwing', () => {
  // Parsed during module evaluation: a throw here is a dead console, not a
  // bad URL.  A mangled bookmark is the whole threat model (#1521 review).
  assert.equal(parseHash('#models/%E0%A4%A').id, '%E0%A4%A');
  assert.equal(parseHash('#models/%').tab, 'models');
  assert.deepEqual(parseHash('#models/%E0%A4%A').params, {});
});

test('sameRoute ignores the spelling of the same page', () => {
  assert.equal(sameRoute('#logs', '#observability'), true);
  assert.equal(sameRoute('#models?q=a', '#models'), false);
});

test('a page title names the tab, not only the product', () => {
  assert.equal(documentTitleOf('Models'), 'Models · llm-rosetta Gateway Console');
  assert.equal(documentTitleOf('  '), 'llm-rosetta Gateway Console');
});

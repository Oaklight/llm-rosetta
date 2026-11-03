/**
 * The provider list's order, and what a switch flip does to it.
 *
 * Problems sort first and switched-off providers last, but that sort must not
 * run on the row the operator just flipped: with many providers the row left
 * the viewport, so the entry they meant to disable was the one they could no
 * longer find.  A pinned order — the order already on screen — keeps it in
 * place; a fresh view (filter change, tab visit) sorts again.
 *
 * This file covers the rule alone.  The wiring that calls it — the pin on a
 * toggle, the four clears, the tab-entry reset — is held by
 * `tests/js/provider-list-wiring.test.mjs`; 119 green here does not mean the
 * page is covered.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { orderedProviders } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/provider-endpoints.js';

const names = (entries) => entries.map(([name]) => name);

test('problems come first and switched-off providers last', () => {
  const entries = [
    ['off', { enabled: false }],
    ['ok', { enabled: true }],
    ['broken', { enabled: false, _auto_disabled: { reason: 'x' } }],
  ];

  assert.deepEqual(names(orderedProviders(entries)), ['broken', 'ok', 'off']);
});

test('the configured order holds inside a group', () => {
  const entries = [
    ['a', { enabled: true }],
    ['b', { enabled: true }],
    ['c', { enabled: true }],
  ];

  assert.deepEqual(names(orderedProviders(entries)), ['a', 'b', 'c']);
});

test('a pinned order keeps a freshly disabled row in place', () => {
  const entries = [
    ['a', { enabled: true }],
    ['b', { enabled: false }],
    ['c', { enabled: true }],
  ];

  // Without the pin, disabling b moves it behind both healthy providers.
  assert.deepEqual(names(orderedProviders(entries)), ['a', 'c', 'b']);
  // With the pin, the row the operator flipped stays where it was.
  assert.deepEqual(names(orderedProviders(entries, null, ['a', 'b', 'c'])), [
    'a',
    'b',
    'c',
  ]);
});

test('a name the pin does not list still sorts into place', () => {
  const entries = [
    ['new', { enabled: false }],
    ['a', { enabled: true }],
    ['b', { enabled: true }],
  ];

  assert.deepEqual(names(orderedProviders(entries, null, ['b', 'a'])), [
    'b',
    'a',
    'new',
  ]);
});

test('the pin survives a status change of another row', () => {
  const entries = [
    ['a', { enabled: true }],
    ['b', { enabled: false }],
    ['c', { enabled: true, _auto_disabled: { reason: 'x' } }],
  ];

  assert.deepEqual(names(orderedProviders(entries, null, ['a', 'b', 'c'])), [
    'a',
    'b',
    'c',
  ]);
});

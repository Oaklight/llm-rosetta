/**
 * The preload-preset chips (#1324).
 *
 * The Shims pane used to take a comma-separated list of package keys, which
 * made the operator remember the keys and spell them correctly.  The value on
 * the wire is still that list; these helpers are the bridge between it and the
 * chips, and they are what decides the order and the duplicates.
 *
 * Run with `make test-js` (or `node --test tests/js/*.test.mjs`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  formatPresets,
  parsePresets,
  presetChips,
  presetGroups,
  togglePreset,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-presets.js';

test('a declared value splits on commas and whitespace, without duplicates', () => {
  assert.deepEqual(parsePresets('alcf, argo'), ['alcf', 'argo']);
  assert.deepEqual(parsePresets(' alcf ,, argo\nasksage '), ['alcf', 'argo', 'asksage']);
  assert.deepEqual(parsePresets('alcf alcf'), ['alcf']);
  assert.deepEqual(parsePresets(''), []);
  assert.deepEqual(parsePresets(undefined), []);
});

test('selecting appends, deselecting removes, and order is preserved', () => {
  assert.deepEqual(togglePreset(['alcf'], 'argo', true), ['alcf', 'argo']);
  assert.deepEqual(togglePreset(['alcf', 'argo'], 'alcf', false), ['argo']);
  // Toggling on a key that is already on must not duplicate it.
  assert.deepEqual(togglePreset(['argo'], 'argo', true), ['argo']);
  // Toggling off a key that is not there is a no-op.
  assert.deepEqual(togglePreset(['argo'], 'alcf', false), ['argo']);
});

test('the formatted value is what Save reads back', () => {
  assert.equal(formatPresets(['alcf', 'argo']), 'alcf, argo');
  assert.equal(formatPresets([]), '');
  // Duplicates that reached the list by hand collapse on the way out.
  assert.equal(formatPresets(['alcf', 'alcf']), 'alcf');
});

test('the chips mark the declared packages and carry their shim counts', () => {
  const presets = [
    { key: 'alcf', label: 'ALCF', shims: ['a', 'b', 'c'], source_repo: 'https://example/alcf' },
    { key: 'argo', label: 'Argo', shims: ['x'], source_repo: 'https://example/argo' },
    { key: 'asksage', shims: [] },
  ];
  const chips = presetChips(presets, 'argo');
  assert.deepEqual(chips.map(c => c.key), ['alcf', 'argo', 'asksage']);
  assert.deepEqual(chips.map(c => c.selected), [false, true, false]);
  assert.deepEqual(chips.map(c => c.count), [3, 1, 0]);
  // A package without a label falls back to its key.
  assert.equal(chips[2].label, 'asksage');
});

test('a declared key the catalog does not know is kept, not rendered', () => {
  const chips = presetChips([{ key: 'alcf', label: 'ALCF', shims: [] }], 'alcf, future_pkg');
  assert.deepEqual(chips.map(c => c.selected), [true]);
  assert.equal(chips.length, 1);
  // Rendering does not drop it...
  assert.deepEqual(parsePresets('alcf, future_pkg'), ['alcf', 'future_pkg']);
  // ...and a toggle keeps it, because the toggle starts from the declaration.
  assert.deepEqual(togglePreset(['alcf', 'future_pkg'], 'alcf', false), ['future_pkg']);
});

test('free packages are grouped apart from the paid ones', () => {
  const presets = [
    { key: 'alcf', label: 'ALCF', shims: ['a'], free: false },
    { key: 'opencode_zen_free', label: 'OpenCode Zen free lane', shims: ['b', 'c'], free: true },
    { key: 'argo', label: 'Argo', shims: [] },
  ];
  const groups = presetGroups(presets, 'opencode_zen_free');
  assert.deepEqual(groups.map(g => g.key), ['free', 'paid']);
  assert.deepEqual(groups[0].items.map(c => c.key), ['opencode_zen_free']);
  assert.deepEqual(groups[1].items.map(c => c.key), ['alcf', 'argo']);
  assert.equal(groups[0].items[0].selected, true);
  // A package with no `free` flag is paid: only an explicit true is free.
  assert.equal(groups[1].items[1].free, false);
});

test('an empty group is not drawn', () => {
  const paidOnly = presetGroups([{ key: 'argo', label: 'Argo', shims: [] }], '');
  assert.deepEqual(paidOnly.map(g => g.key), ['paid']);
  const freeOnly = presetGroups([{ key: 'f', label: 'F', shims: [], free: true }], '');
  assert.deepEqual(freeOnly.map(g => g.key), ['free']);
});

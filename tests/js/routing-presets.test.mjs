/**
 * Routing presets: the block each one writes, and which one a block is.
 *
 * A preset is a literal block in the config's own vocabulary, so these pin the
 * exact key set — a preset that wrote an extra key would be a value the
 * operator could not have typed.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ROUTING_PRESETS,
  matchPreset,
  presetBlock,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/routing-presets.js';

test('every preset writes a fresh block', () => {
  for (const preset of ROUTING_PRESETS) {
    const block = presetBlock(preset.id);
    assert.deepEqual(block, preset.block);
    // A copy, so a caller that edits the block cannot edit the preset.
    assert.notEqual(block, preset.block);
  }
});

test('the five presets cover the strategy vocabulary', () => {
  const strategies = ROUTING_PRESETS.map((entry) => entry.block.strategy);
  assert.deepEqual(
    [...new Set(strategies)].sort(),
    ['ordered', 'pace', 'rotate', 'weighted']
  );
});

test('sticky sessions is the one preset that asks for affinity', () => {
  const withAffinity = ROUTING_PRESETS.filter(
    (entry) => entry.block.affinity !== 'off'
  );
  assert.deepEqual(
    withAffinity.map((entry) => entry.id),
    ['sticky']
  );
});

test('a known block matches its preset', () => {
  assert.equal(matchPreset({ strategy: 'weighted', affinity: 'off', sink: true }), 'balanced');
  assert.equal(matchPreset({ strategy: 'rotate', affinity: 'off', sink: true }), 'spread');
});

test('a missing key takes the preset default, an extra key does not', () => {
  // The block the panel reports for a route with no stored block.
  assert.equal(matchPreset({ strategy: 'weighted' }), 'balanced');
  // `first_token_ms` and `rules` are carried by every block; they do not
  // disqualify a match.
  assert.equal(
    matchPreset({ strategy: 'pace', affinity: 'off', sink: true, rules: [] }),
    'quota_saver'
  );
  // An edited strategy is no preset at all, and the panel says so.
  assert.equal(matchPreset({ strategy: 'ordered', affinity: 'session' }), null);
});

test('no block is no preset', () => {
  assert.equal(matchPreset(null), null);
  assert.equal(matchPreset(undefined), null);
});

test('an unknown preset id writes nothing', () => {
  assert.equal(presetBlock('nope'), null);
});

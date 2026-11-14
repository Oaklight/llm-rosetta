/**
 * A full retention cap is steady state, not an incident (§13 of #1502).
 *
 * The footer colours the success-log and dump counters by how full they are.
 * Both stores are trimmed to their cap, so `n === cap` is what a healthy
 * gateway spends its life at — red there trains the operator to ignore the
 * bar.  Red belongs to a store *over* its cap, where the trimmer is behind.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { capClass } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/status-ink.js';

test('a full cap reads as normal', () => {
  assert.equal(capClass(50000, 50000), 'warn');
  // ...and only because 100% is also "on its way" — but never crit.
  assert.notEqual(capClass(50000, 50000), 'crit');
  assert.equal(capClass(0, 50000), '');
});

test('over the cap is the critical case', () => {
  assert.equal(capClass(50001, 50000), 'crit');
});

test('three quarters is the amber line, and a cap of zero is no cap', () => {
  assert.equal(capClass(37499, 50000), '');
  assert.equal(capClass(37500, 50000), 'warn');
  assert.equal(capClass(10, 0), '');
});

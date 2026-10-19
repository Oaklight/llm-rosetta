/**
 * Segmented-control keyboard model.
 *
 * `seg.js` is import-free so it runs under node, which has no DOM.  Two things
 * regress silently if untested: the wrap-around at the ends of a group (the
 * arrow keys must cycle, not dead-end) and Home/End landing on the first/last
 * item rather than being ignored.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { isSegActivationKey, segKeyTargetIndex } from '../../src/llm_rosetta/gateway/admin/js/core/seg.js';

test('Arrow keys move one step and wrap at both ends', () => {
  assert.equal(segKeyTargetIndex(3, 0, 'ArrowRight'), 1);
  assert.equal(segKeyTargetIndex(3, 2, 'ArrowRight'), 0);  // wraps forward
  assert.equal(segKeyTargetIndex(3, 0, 'ArrowLeft'), 2);   // wraps backward
  assert.equal(segKeyTargetIndex(3, 1, 'ArrowLeft'), 0);
});

test('Home and End jump to the first and last item', () => {
  assert.equal(segKeyTargetIndex(4, 2, 'Home'), 0);
  assert.equal(segKeyTargetIndex(4, 1, 'End'), 3);
});

test('unknown keys and out-of-group indices leave the event alone', () => {
  assert.equal(segKeyTargetIndex(3, 1, 'ArrowDown'), -1);
  assert.equal(segKeyTargetIndex(3, 1, 'a'), -1);
  assert.equal(segKeyTargetIndex(3, -1, 'ArrowRight'), -1);
  assert.equal(segKeyTargetIndex(3, 3, 'ArrowRight'), -1);
  assert.equal(segKeyTargetIndex(0, 0, 'Home'), -1);
});

test('Enter and both spellings of Space activate; other keys do not', () => {
  assert.equal(isSegActivationKey('Enter'), true);
  assert.equal(isSegActivationKey(' '), true);
  assert.equal(isSegActivationKey('Spacebar'), true);
  assert.equal(isSegActivationKey('ArrowRight'), false);
  assert.equal(isSegActivationKey('Tab'), false);
});

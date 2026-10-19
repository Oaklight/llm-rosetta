/**
 * Chip state mirroring: the handler must read the checkbox, never write it.
 *
 * The bug this pins is a double toggle.  A label wrapping a checkbox
 * dispatches a click for the label and a synthesised one for the input, and
 * that second click bubbles back through the label, so the handler runs
 * twice per interaction.  Reading `checked` makes the second run a no-op;
 * writing it would cancel the browser's toggle and the chip would never
 * change — exactly the regression this fix removed.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { syncChipState } from '../../src/llm_rosetta/gateway/admin/js/core/chips.js';

function makeChip() {
  const classes = new Set();
  return {
    classList: {
      toggle(name, on) { if (on) classes.add(name); else classes.delete(name); },
    },
    has: name => classes.has(name),
  };
}

test('syncChipState follows the checkbox on both transitions', () => {
  const chip = makeChip();
  const cb = {checked: false};
  syncChipState(chip, cb);
  assert.equal(chip.has('active'), false);
  cb.checked = true;
  syncChipState(chip, cb);
  assert.equal(chip.has('active'), true);
  cb.checked = false;
  syncChipState(chip, cb);
  assert.equal(chip.has('active'), false);
});

test('syncChipState is idempotent, so the label double-fire is harmless', () => {
  const chip = makeChip();
  const cb = {checked: true};
  syncChipState(chip, cb);
  syncChipState(chip, cb);  // the second click of one interaction
  assert.equal(chip.has('active'), true);
  assert.equal(cb.checked, true);
});

test('syncChipState never writes the checkbox', () => {
  const chip = makeChip();
  const cb = {
    get checked() { return true; },
    set checked(_) { throw new Error('handler wrote checked'); },
  };
  assert.doesNotThrow(() => syncChipState(chip, cb));
});

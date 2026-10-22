/**
 * The model card's reasoning override: what a save keeps, changes and clears.
 *
 * The card renders four controls, but a stored override can carry fields it
 * cannot express (`effort_field`, `effort_range`, …) and the card *shows* those
 * as inherited values.  Two failures are silent, so they are pinned here: a
 * control set back to "inherit" must delete its key rather than omit it (an
 * absent key means "leave what is stored"), and an unrelated save must not
 * delete a field the form never rendered.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { mergeReasoningOverride } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/reasoning-override.js';

const INHERIT = { thinking_type: '', budget_ratio: null, disabled: '', replay_missing: '' };

test('nothing set and nothing loaded is an empty override', () => {
  assert.deepEqual(mergeReasoningOverride({}, INHERIT), {});
  assert.deepEqual(mergeReasoningOverride(null, INHERIT), {});
});

test('setting a control writes its key', () => {
  assert.deepEqual(mergeReasoningOverride({}, { ...INHERIT, replay_missing: 'fill_empty' }), {
    replay_missing: 'fill_empty',
  });
  assert.deepEqual(mergeReasoningOverride({}, { ...INHERIT, budget_ratio: 0.5 }), {
    budget_ratio: 0.5,
  });
});

test('setting a control back to inherit deletes the key', () => {
  const loaded = { replay_missing: 'fill_empty' };

  assert.deepEqual(mergeReasoningOverride(loaded, INHERIT), {});
});

test('an override the form cannot express survives an unrelated save', () => {
  const loaded = { effort_field: 'output_config.effort', replay_missing: 'fill_empty' };

  assert.deepEqual(mergeReasoningOverride(loaded, INHERIT), {
    effort_field: 'output_config.effort',
  });
});

test('editing one control leaves the others as they were', () => {
  const loaded = { disabled: 'omit', effort_field: 'reasoning_effort' };

  assert.deepEqual(
    mergeReasoningOverride(loaded, { ...INHERIT, disabled: 'omit', replay_missing: 'as_is' }),
    { disabled: 'omit', effort_field: 'reasoning_effort', replay_missing: 'as_is' },
  );
});

test('a loaded override is not mutated', () => {
  const loaded = { replay_missing: 'as_is' };

  mergeReasoningOverride(loaded, { ...INHERIT, replay_missing: '' });

  assert.deepEqual(loaded, { replay_missing: 'as_is' });
});

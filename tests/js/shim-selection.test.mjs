/**
 * The install-time selection: what the Available card writes, and what it shows.
 *
 * The two halves are tested together because they must agree — a save that
 * writes one set and a repaint that shows another would silently re-install a
 * shim the operator unticked.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  hasPattern,
  keptShims,
  selectionMaps,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-selection.js';

const SHIMS = ['kilo_free--chat_completions', 'llm7--chat_completions', 'ovhcloud--chat_completions'];

test('everything ticked writes no entry at all', () => {
  const { include, exclude } = selectionMaps({}, 'free', SHIMS, 3);

  assert.deepEqual(include, {});
  assert.deepEqual(exclude, {});
});

test('a partial tick is a closed include list', () => {
  const { include, exclude } = selectionMaps({}, 'free', [SHIMS[0]], 3);

  assert.deepEqual(include, { free: [SHIMS[0]] });
  assert.deepEqual(exclude, {});
});

test('another package entry is carried over untouched', () => {
  const section = { include: { argo: ['argo--chat_completions'] }, exclude: {} };

  const { include } = selectionMaps(section, 'free', [SHIMS[1]], 3);

  assert.deepEqual(include, {
    argo: ['argo--chat_completions'],
    free: [SHIMS[1]],
  });
});

test('re-ticking everything drops the entry instead of writing the full list', () => {
  const section = { include: { free: [SHIMS[0]] } };

  const { include } = selectionMaps(section, 'free', [...SHIMS], 3);

  assert.equal('free' in include, false);
});

test('a stale exclude for the same package is dropped', () => {
  const section = { exclude: { free: [SHIMS[2]] } };

  const { include, exclude } = selectionMaps(section, 'free', [SHIMS[2]], 3);

  // One of three ticked is still an include; the exclude no longer speaks for
  // the key, so the two maps cannot contradict each other.
  assert.deepEqual(include, { free: [SHIMS[2]] });
  assert.equal('free' in exclude, false);
});

test('what is kept follows include and exclude the way the loader does', () => {
  assert.deepEqual(keptShims(SHIMS, {}), SHIMS);
  assert.deepEqual(keptShims(SHIMS, { include: [SHIMS[1]] }), [SHIMS[1]]);
  assert.deepEqual(keptShims(SHIMS, { exclude: [SHIMS[1]] }), [SHIMS[0], SHIMS[2]]);
  assert.deepEqual(
    keptShims(SHIMS, { include: [SHIMS[0], SHIMS[1]], exclude: [SHIMS[1]] }),
    [SHIMS[0]],
  );
});

test('a pattern is recognised, so the card can say the list is not its own', () => {
  assert.equal(hasPattern(['llm7--*']), true);
  assert.equal(hasPattern([SHIMS[0]]), false);
  assert.equal(hasPattern([]), false);
});

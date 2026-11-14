/**
 * The switch's knob is the track's contrast colour, not white (#1502 §13).
 *
 * The knob used to be a literal `#fff`.  In dark mode a checked track is the
 * accent — `#ededed` in minimal — so the knob measured 1.15:1 against it and
 * the switch read as one pale blob.  Checked, the knob takes `--accent-on`
 * (the ink every accented fill already uses); unchecked, the track is the
 * hairline and the knob takes `--text-dim`.
 *
 * Text-level on purpose: the measurements live in the theme files, and this
 * pins the rule that keeps them true in every scheme and both modes.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const CSS = readFileSync(
  join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/css/components.css'),
  'utf8',
).replace(/\/\*[\s\S]*?\*\//g, '');

/** The declarations of one rule, by selector. */
function rule(selector) {
  const at = CSS.indexOf(selector + ' {');
  assert.ok(at >= 0, `rule not found: ${selector}`);
  const end = CSS.indexOf('}', at);
  return CSS.slice(at, end);
}

test('a checked knob is the accent ink, never white', () => {
  const checked = rule('.toggle input:checked + .slider::before');
  assert.match(checked, /background: var\(--accent-on\)/);
  assert.ok(!/#fff\b|\bwhite\b/.test(checked), 'the knob is a literal white again');
});

test('an unchecked knob follows the track', () => {
  const base = rule('.toggle .slider::before');
  assert.match(base, /background: var\(--text-dim\)/);
});

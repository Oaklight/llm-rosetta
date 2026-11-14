/**
 * A segmented option owns no box of its own (#1502 §13).
 *
 * The component styled spacing and separators but left the option's own fill
 * and frame alone, so a `<button>` child kept the browser's grey button skin
 * while a `<label>` child was transparent — the Samples switch read as four
 * separate buttons beside the shims switch that had patched the reset in
 * locally.  The reset belongs to the component; this pins it there.
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

test('the option reset lives in the component, not a call site', () => {
  const at = CSS.indexOf('.seg-control > * {');
  assert.ok(at >= 0, '.seg-control > * is gone');
  const body = CSS.slice(at, CSS.indexOf('}', at));
  for (const decl of ['appearance:none', 'border:0', 'background:transparent']) {
    assert.ok(body.includes(decl), `the option no longer resets ${decl}`);
  }
});

test('no call site re-patches the reset', () => {
  assert.ok(!/\.shims-seg > \[data-view\]\s*\{[^}]*appearance/.test(CSS),
    'the shims switch patched the option again instead of using the component');
});

/**
 * The fixed bars' clearance stays inside the content column, not on `body`.
 *
 * `body { padding-bottom: 26px }` sat outside `.app-shell`, so the document
 * was always 26px taller than the viewport and the sticky rail was lifted by
 * that much at maximum scroll (measured: sidebar bottom 800 → 774).  The
 * clearance already existed on `.content` in base.css; the fix is deleting the
 * body rule, not adding a second value that lowers the 56px base reserves.
 *
 * Text-level, and over every stylesheet: the rule this protects lives in
 * base.css, the one that broke it lived in components.css, and a future sheet
 * could bring either back.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const CSS_DIR = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/css',
);
const SHEETS = readdirSync(CSS_DIR)
  .filter((name) => name.endsWith('.css'))
  .map((name) => ({ name, text: readFileSync(join(CSS_DIR, name), 'utf8') }));
const base = SHEETS.find((sheet) => sheet.name === 'base.css');
const components = SHEETS.find((sheet) => sheet.name === 'components.css');

test('no stylesheet reserves the bars on body', () => {
  for (const sheet of SHEETS) {
    assert.doesNotMatch(
      sheet.text,
      /body\s*\{[^}]*padding-bottom/,
      `${sheet.name}: padding on body inflates the document and bounds the sticky rail`,
    );
  }
});

test('the content column keeps the clearance base.css gives it', () => {
  assert.ok(base, 'base.css is still there');
  assert.match(
    base.text,
    /\.content\s*\{[^}]*padding:\s*24px 24px 56px/,
    '56px below the last row, so it scrolls clear of the status bar',
  );
});

test('components.css does not lower that clearance', () => {
  assert.ok(components, 'components.css is still there');
  assert.doesNotMatch(
    components.text,
    /\.content\s*\{[^}]*padding-bottom:\s*26px/,
    'a 26px override would leave ~4px of air under the last row',
  );
});

test('the phone layout reserves room for the tab bar and footer on content', () => {
  assert.ok(components, 'components.css is still there');
  const start = components.text.indexOf('@media (max-width: 768px)');
  assert.notEqual(start, -1, 'the phone layout still exists');
  const block = components.text.slice(start);
  assert.doesNotMatch(block, /body\s*\{[^}]*padding-bottom/, 'still not on body');
  assert.match(
    block,
    /\.content\s*\{[^}]*padding-bottom:\s*calc\(84px \+ env\(safe-area-inset-bottom/,
    'tab bar (56px) + status footer (23px) + margin',
  );
});

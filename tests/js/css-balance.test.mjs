/**
 * Every stylesheet parses (found by the #1551 review round).
 *
 * A single stray `{` — a rule left open by a merge, a block deleted from the
 * middle of another — makes the browser drop *every rule after it*, silently.
 * It is not a syntax error in the console and not a 500: the page just loses
 * its skin from that point, which is how a broken merge hid `.section-toggle`
 * and every rule below it while the file still looked fine in a diff.
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

/** Walk a sheet, ignoring comments, and report the depth after the last byte. */
function depthOf(text) {
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    if (text.startsWith('/*', i)) {
      const end = text.indexOf('*/', i + 2);
      i = end < 0 ? text.length : end + 2;
      continue;
    }
    if (text[i] === '{') depth += 1;
    else if (text[i] === '}') depth -= 1;
    if (depth < 0) return depth;
    i += 1;
  }
  return depth;
}

function sheets(dir = CSS_DIR) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...sheets(path));
    else if (entry.name.endsWith('.css')) out.push({ name: path.slice(CSS_DIR.length + 1), path });
  }
  return out;
}

test('every stylesheet closes what it opens', () => {
  const files = sheets();
  assert.ok(files.length >= 5, `expected the panel sheets, found ${files.length}`);
  for (const file of files) {
    assert.equal(depthOf(readFileSync(file.path, 'utf8')), 0, `${file.name} does not balance`);
  }
});

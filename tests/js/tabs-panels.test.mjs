/**
 * A tab id the branding may list as disabled needs somewhere to go.
 *
 * `disabled_tabs` is validated against `_VALID_TABS`, and the client hides a
 * tab by its rail button.  A new id added to one side and not the other makes
 * `disabled_tabs: [<id>]` either rejected or silently ineffective, so the two
 * lists are checked against the markup here.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..');
const HTML = readFileSync(
  join(ROOT, 'packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/admin.html'),
  'utf8',
);
const ADMIN_INIT = readFileSync(
  join(ROOT, 'packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/__init__.py'),
  'utf8',
);

/** `logs` is the pre-rename spelling of the observability tab. */
const ALIAS = { logs: 'observability' };

function validTabs(text) {
  const block = /_VALID_TABS = frozenset\(\s*\{([^}]*)\}\s*\)/.exec(text);
  assert.ok(block, '_VALID_TABS not found');
  return [...block[1].matchAll(/"([a-z_]+)"/g)].map((m) => m[1]);
}

/** Every id that has no rail button or no panel, as `id: missing-what`. */
function missingFor(ids, html) {
  const missing = [];
  for (const id of ids) {
    const panel = ALIAS[id] || id;
    if (!html.includes(`id="tab-${panel}"`)) missing.push(`${id}: #tab-${panel}`);
    if (!html.includes(`id="tabBtn-${panel}"`)) missing.push(`${id}: #tabBtn-${panel}`);
  }
  return missing;
}

test('the id list parses to the panel ids the markup has', () => {
  const ids = validTabs(ADMIN_INIT);
  assert.ok(ids.length >= 7, `expected the tab ids, found ${ids.length}`);
  assert.deepEqual(missingFor(ids, HTML), []);
});

test('the check fails on an id with no page', () => {
  // Guard the guard: an id the markup does not know must be reported.
  assert.deepEqual(missingFor(['sentiment'], HTML), [
    'sentiment: #tab-sentiment',
    'sentiment: #tabBtn-sentiment',
  ]);
});

test('the alias is what lets the older spelling pass', () => {
  assert.deepEqual(missingFor(['logs'], HTML), []);
  assert.ok(HTML.includes('id="tab-observability"'));
});

/**
 * The rail's wiring: the one ordering the settings marker depends on.
 *
 * The marker is CSS (`theme-contract` polices its shape), and the toggle is
 * one line in `_syncRailPrefs`.  What neither covers is *when* that line
 * runs: the gear reads `S.currentTab`, so `activateTab` has to record the tab
 * before it repaints the rail.  Move the call above the assignment and the
 * marker lags one click behind, which no browser-free test would notice.
 *
 * Text-level on purpose: the call order is the contract, and reading the
 * source is enough to hold it without a DOM.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const INIT_JS = join(
  dirname(fileURLToPath(import.meta.url)),
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/init.js',
);
const SOURCE = readFileSync(INIT_JS, 'utf8');

/** The body of `activateTab`, up to the settings-group section that follows it. */
function activateTabBody(text) {
  const start = text.indexOf('function activateTab(');
  assert.notEqual(start, -1, 'activateTab is still a function');
  const end = text.indexOf('// ---- Settings groups ----', start);
  assert.notEqual(end, -1, 'activateTab still ends before the settings groups');
  return text.slice(start, end);
}

test('activateTab records the tab before it repaints the rail', () => {
  const body = activateTabBody(SOURCE);
  const assigned = body.indexOf('S.currentTab = id');
  const repaint = body.indexOf('_syncRailPrefs');
  assert.notEqual(assigned, -1, 'activateTab records the current tab');
  assert.notEqual(repaint, -1, 'activateTab repaints the rail');
  assert.ok(assigned < repaint, 'the gear reads the new tab, not the previous one');
});

test('the rail repaint toggles the settings gear', () => {
  assert.match(SOURCE, /getElementById\('railSettingsBtn'\)/);
  assert.match(SOURCE, /classList\.toggle\('is-active', S\.currentTab === 'settings'\)/);
});

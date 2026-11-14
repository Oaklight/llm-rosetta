/**
 * One component per role holds (issue #1535).
 *
 * The tracking issue's remaining checklist items were each a small control
 * that had grown its own copy.  Closing them means a rule lives in exactly
 * one place; these assertions are the part a later edit can quietly undo:
 *
 * 1. the scope badge's pip is the shared `.dot.is-sm` element, not a
 *    `::before` drawn by the badge — and it sits outside the `data-i18n`
 *    span, because `applyI18n()` rewrites textContent and would erase a child
 *    placed inside it;
 * 2. the fetch dialog's row is the shared `.toolbar`; the `.fetch-toolbar`
 *    copy is gone from both markup and styles;
 * 3. the argv row's remove control is `.icon-btn.is-bordered`; the
 *    `.prov-icon-btn` copy is gone;
 * 4. a menu anchor is recognised by `aria-haspopup="true"`, so a plain button
 *    can be one without a class of its own;
 * 5. the three chips that stay bespoke carry their decision in the sheet, so
 *    an unexplained copy cannot reappear beside them.
 *
 * Each check was mutated once to prove it can fail: a `::before` restored, a
 * `fetch-toolbar` class restored, a `prov-icon-btn` class restored, the ARIA
 * selector reverted, and a `Decision (#1535)` comment deleted.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);
const CSS_DIR = join(ADMIN, 'css');
const SHEETS = readdirSync(CSS_DIR)
  .filter((name) => name.endsWith('.css'))
  .map((name) => ({ name, text: readFileSync(join(CSS_DIR, name), 'utf8') }));
const HTML = readFileSync(join(ADMIN, 'admin.html'), 'utf8');
const INIT = readFileSync(join(ADMIN, 'js/core/init.js'), 'utf8');
const PROVIDERS = readFileSync(join(ADMIN, 'js/tabs/providers.js'), 'utf8');

test('the scope badge wears the shared dot, outside the translated span', () => {
  const badges = HTML.split('\n').filter((line) => line.includes('class="scope-badge'));
  assert.ok(badges.length >= 9, `expected the scope badges, found ${badges.length}`);
  for (const line of badges) {
    assert.match(
      line,
      /<span class="scope-badge [a-z-]+"><span class="dot is-sm" aria-hidden="true"><\/span><span data-i18n="/,
      `a scope badge draws its own pip or translates over the dot: ${line.trim()}`,
    );
  }
  for (const sheet of SHEETS) {
    assert.ok(
      !/\.scope-badge::before/.test(sheet.text),
      `${sheet.name}: the badge has a ::before pip again`,
    );
  }
});

test('the fetch dialog uses the shared toolbar', () => {
  for (const sheet of SHEETS) {
    assert.ok(
      !/\.fetch-toolbar\b/.test(sheet.text),
      `${sheet.name}: the .fetch-toolbar copy is back`,
    );
  }
  assert.ok(!HTML.includes('fetch-toolbar'), 'admin.html still carries .fetch-toolbar');
});

test('the argv remove control is the shared icon button', () => {
  for (const sheet of SHEETS) {
    assert.ok(
      !/\.prov-icon-btn\b/.test(sheet.text),
      `${sheet.name}: the .prov-icon-btn copy is back`,
    );
  }
  assert.ok(!PROVIDERS.includes('prov-icon-btn'), 'providers.js still sets prov-icon-btn');
  assert.ok(
    PROVIDERS.includes("del.className = 'icon-btn is-bordered'"),
    'the argv remove control no longer wears .icon-btn.is-bordered',
  );
});

test('a menu anchor is recognised by its ARIA contract, not a class', () => {
  assert.ok(
    INIT.includes("closest('[aria-haspopup=\"true\"]')"),
    'the outside-click handler no longer recognises aria-haspopup anchors',
  );
  const free = HTML.split('\n').find((line) => line.includes('id="freeProviderBtn"'));
  assert.ok(free, 'the free-resource button is gone');
  assert.ok(!free.includes('more-menu-toggle'), 'the free-resource button carries the row hook again');
});

test('a bespoke chip states why it is not a tag', () => {
  const all = SHEETS.map((s) => s.text).join('\n');
  for (const selector of ['.provider-multi-count', '.settings-token-chip', '.update-badge']) {
    assert.ok(
      new RegExp(`Decision \\(#1535\\)[\\s\\S]{0,240}?\\${selector} \\{`).test(all),
      `${selector} has no recorded decision`,
    );
  }
});

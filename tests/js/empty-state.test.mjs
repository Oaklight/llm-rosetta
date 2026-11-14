/**
 * One wording for "the filter matched nothing" (#1519).
 *
 * Two pages used to claim the config held nothing while it held dozens, and
 * none of the four offered a way back.  The rule — rows exist but none match
 * means *filtered*, not *empty* — is pinned by `emptyChoice`, and the four
 * pages are pinned to the same builder so the wording cannot drift again.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

// `core/state.js` reads localStorage at module scope, so the page's minimum
// environment comes first — the same stub `modules-load.test.mjs` uses.
globalThis.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
};
globalThis.window = globalThis;
globalThis.document = {
  documentElement: { getAttribute: () => null, setAttribute: () => {} },
  body: { classList: { add: () => {}, remove: () => {}, contains: () => false } },
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
};

const {
  emptyChoice,
  FILTERED_EMPTY_KEY,
  filteredEmptyHtml,
  filteredEmptyRowHtml,
} = await import('../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/empty.js');
const { I18N } = await import('../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/i18n.js');

const here = dirname(fileURLToPath(import.meta.url));
const js = (rel) => readFileSync(join(here, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js', rel), 'utf8');

test('a store with rows and no matches is filtered, not empty', () => {
  assert.equal(emptyChoice(12, 0), 'filtered');
  assert.equal(emptyChoice(0, 0), 'none');
  assert.equal(emptyChoice(12, 3), 'shown');
});

test('the message names the filter and carries the way back', () => {
  const html = filteredEmptyHtml('resetModelFilters()');
  assert.equal(html.includes(I18N.en[FILTERED_EMPTY_KEY]), true);
  assert.match(html, /onclick="resetModelFilters\(\)"/);
  assert.match(html, /class="empty-clear"/);
});

test('a table message is one spanning row', () => {
  const html = filteredEmptyRowHtml('clearKeySearch()', 9);
  assert.match(html, /^<tr role="row"><td colspan="9" role="cell"/);
  assert.match(html, /clearKeySearch\(\)/);
});

test('the reset call cannot carry anything but a function call', () => {
  // The call is interpolated into an onclick; it comes from our own modules,
  // but the builder refuses anything that is not one.
  const html = filteredEmptyHtml('x(); alert(1)');
  assert.doesNotMatch(html, /alert/);
});

test('both dictionaries carry the shared wording', () => {
  for (const lang of ['en', 'zh']) {
    assert.ok(I18N[lang][FILTERED_EMPTY_KEY], `${lang} ${FILTERED_EMPTY_KEY}`);
    assert.ok(I18N[lang]['empty.clearFilter'], `${lang} empty.clearFilter`);
  }
});

test('the four pages use the one builder, each with its own reset', () => {
  // A page that answers with its own sentence is the defect this guards.
  const pages = {
    'tabs/models.js': 'resetModelFilters()',
    'tabs/providers.js': 'clearProviderSearch()',
    'tabs/keys.js': 'clearKeySearch()',
    'tabs/routing.js': 'clearRoutingSearch()',
  };
  for (const [file, call] of Object.entries(pages)) {
    const src = js(file);
    assert.match(src, /from '\.\.\/core\/empty\.js'/, `${file} imports the builder`);
    assert.ok(src.includes(call), `${file} passes ${call}`);
    assert.doesNotMatch(src, /empty\.searchResults/, `${file} no longer has its own sentence`);
  }
});

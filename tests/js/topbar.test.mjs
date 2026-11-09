/**
 * The shell's topbar.
 *
 * The title is copied from the active panel's own heading, which is why the
 * language switch has to re-run the fill: `applyI18n()` rewrites the heading,
 * and without a second pass the topbar would keep the old language above a
 * translated page.  That call site is pinned by reading the source, because
 * the modules that hold it need a DOM to import.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  indexLabel,
  sectionIndexOf,
  topbarTitle,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/topbar.js';

const ADMIN = join(dirname(fileURLToPath(import.meta.url)), '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin');

// ---- the pure helpers ---------------------------------------------------

test('indexLabel brackets and pads the section number', () => {
  assert.equal(indexLabel(1), '[01]');
  assert.equal(indexLabel(7), '[07]');
  assert.equal(indexLabel(12), '[12]');
  // A tab outside the rail has no number rather than a wrong one.
  assert.equal(indexLabel(0), '');
  assert.equal(indexLabel(-1), '');
  assert.equal(indexLabel(1.5), '');
  assert.equal(indexLabel('3'), '');
});

test('sectionIndexOf counts from one and reports absence as zero', () => {
  const rail = ['dashboard', 'providers', 'models', 'settings'];
  assert.equal(sectionIndexOf('dashboard', rail), 1);
  assert.equal(sectionIndexOf('settings', rail), 4);
  assert.equal(sectionIndexOf('nowhere', rail), 0);
  assert.equal(sectionIndexOf(undefined, rail), 0);
});

test('topbarTitle prefers the panel heading, then the rail label', () => {
  assert.equal(topbarTitle('  Providers  ', 'Providers'), 'Providers');
  assert.equal(topbarTitle('', 'API Keys'), 'API Keys');
  assert.equal(topbarTitle(null, 'Models'), 'Models');
  assert.equal(topbarTitle('   ', '  '), '');
  // The settings pane has no panel heading and no rail label.
  assert.equal(topbarTitle('', '', 'Settings'), 'Settings');
  assert.equal(topbarTitle('Routing', '', 'Settings'), 'Routing');
});

// ---- the call sites that make it correct --------------------------------

const initSource = readFileSync(join(ADMIN, 'js/core/init.js'), 'utf8');
const i18nSource = readFileSync(join(ADMIN, 'js/core/i18n.js'), 'utf8');
const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');

/** Strip line and block comments, leaving string contents alone enough for
 *  the statement checks below (no `//` or `/*` appears inside a URL here). */
export function stripComments(code) {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n')
    .map(line => line.replace(/(^|[^:])\/\/.*$/, '$1'))
    .join('\n');
}

/** The body of the top-level function whose signature contains *signature*. */
function functionBody(source, signature) {
  const at = source.indexOf(signature);
  assert.notEqual(at, -1, `expected to find ${signature}`);
  const open = source.indexOf('{', at);
  const close = source.indexOf('\n}', open);
  assert.notEqual(close, -1, `expected the body of ${signature} to end`);
  return stripComments(source.slice(open + 1, close));
}

/** The last non-empty statement of *body*, or ''. */
function lastStatement(body) {
  const statements = body
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean);
  return statements.length ? statements[statements.length - 1] : '';
}

test('the topbar is filled when a tab is activated, after the state records it', () => {
  const body = functionBody(initSource, 'function activateTab');
  const state = body.indexOf('S.currentTab = id;');
  const fill = body.indexOf('syncTopbarTitle();');
  assert.notEqual(state, -1);
  assert.notEqual(fill, -1);
  assert.ok(state < fill, 'the fill must run after the tab id is recorded');
});

test('the topbar refill is the last thing applyI18n does', () => {
  // Position matters: applyI18n rewrites the panel headings first, and the
  // fill copies them.  A call anywhere else reads stale text.
  const body = functionBody(i18nSource, 'export function applyI18n()');
  assert.equal(lastStatement(body), 'window.syncTopbarTitle?.();');
});

test('the statement checks cannot be satisfied by a comment or by order', () => {
  // Baseline: the real shape passes.
  assert.equal(
    lastStatement(stripComments('  window.updateClockLabel?.();\n  window.syncTopbarTitle?.();\n')),
    'window.syncTopbarTitle?.();',
  );
  // A comment naming the call is not the call.
  assert.equal(
    lastStatement(stripComments('  doSomething();\n  // window.syncTopbarTitle?.();\n')),
    'doSomething();',
  );
  assert.equal(
    lastStatement(stripComments('  doSomething();\n  /* window.syncTopbarTitle?.(); */\n')),
    'doSomething();',
  );
  // First instead of last fails, which the old substring check allowed.
  assert.notEqual(
    lastStatement(stripComments('  window.syncTopbarTitle?.();\n  doSomething();\n')),
    'window.syncTopbarTitle?.();',
  );
});

test('the page has one heading level, and the brand is not one', () => {
  const topbar = html.slice(html.indexOf('<header class="topbar">'), html.indexOf('</header>'));
  assert.match(topbar, /<h1 class="topbar-title"/);
  // The brand is a link, not the page's heading.
  assert.doesNotMatch(html, /<h1 id="brandTitle"/);
  assert.match(html, /<div id="brandTitle"/);
  // Still exactly one topbar, with no live region duplicating the tab switch.
  assert.equal((html.match(/class="topbar"/g) || []).length, 1);
  assert.doesNotMatch(topbar, /aria-live/);
});

test('the hidden heading is the shape the panels actually use', () => {
  const css = readFileSync(join(ADMIN, 'css/base.css'), 'utf8');
  assert.match(css, /\.tab-panel\.active > \.section > \.section-header > h2 \{ display: none; \}/);
  // Hiding the title must not move the controls: with one flow child left,
  // `space-between` would push it to the left edge.
  assert.match(css, /\.tab-panel\.active \.section-header > h2 \+ \* \{ margin-left: auto; \}/);
});

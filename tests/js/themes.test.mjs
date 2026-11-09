/**
 * The admin theme registry, as the page sees it.
 *
 * `admin/themes.py` is the single list of themes; it reaches the page as the
 * `#adminThemes` JSON the server injects.  These tests cover the page-side
 * helpers in `js/core/themes.js` without a DOM, plus the two facts that tie the
 * injected registry and the inline first-frame seed to the files on disk:
 *
 *  - the page's boot script and `themes.js` agree on the element ids, and
 *  - the seed colours are exactly the default theme's, so the first frame
 *    cannot drift from the sheet that replaces it.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parseRegistry, pickFallback, resolveSchemeIn } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/themes.js';

const ADMIN = join(dirname(fileURLToPath(import.meta.url)), '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin');

const REGISTRY = {
  default: 'minimal',
  themes: [
    { name: 'minimal', label: 'Minimal', url: '/admin/static/css/themes/minimal.css' },
    { name: 'emerald', label: 'Emerald', url: '/admin/static/css/themes/emerald.css' },
  ],
};

// ---- parseRegistry ------------------------------------------------------

test('parseRegistry keeps a well-formed registry', () => {
  assert.deepEqual(parseRegistry(REGISTRY), REGISTRY);
});

test('parseRegistry fills what the page needs and drops what it cannot use', () => {
  const parsed = parseRegistry({
    default: 'missing',
    themes: [
      { name: 'minimal', label: 'Minimal', url: '/a.css' },
      { name: 'bare', url: '/b.css' },
      { label: 'no name' },
      null,
      'string',
    ],
  });
  // A default the registry does not ship resolves to the first theme, so the
  // page is never pointed at a file that does not exist.
  assert.equal(parsed.default, 'minimal');
  assert.deepEqual(parsed.themes.map(t => t.name), ['minimal', 'bare']);
  // A theme without a label is named by its own name rather than an empty
  // option; the label is what the Settings picker shows.
  assert.equal(parsed.themes[1].label, 'bare');
  assert.equal(parsed.themes[1].url, '/b.css');
});

test('parseRegistry is total', () => {
  for (const raw of [null, undefined, {}, [], { themes: 'nope' }]) {
    assert.deepEqual(parseRegistry(raw), { default: '', themes: [] });
  }
});

// ---- resolveSchemeIn ----------------------------------------------------

test('resolveSchemeIn keeps a shipped name and replaces anything else', () => {
  assert.equal(resolveSchemeIn(REGISTRY, 'emerald'), 'emerald');
  // A theme removed in an upgrade, a hand-edited value, and the absent case
  // all land on the registry's default.
  assert.equal(resolveSchemeIn(REGISTRY, 'neon'), 'minimal');
  assert.equal(resolveSchemeIn(REGISTRY, null), 'minimal');
  assert.equal(resolveSchemeIn(REGISTRY, undefined), 'minimal');
  assert.equal(resolveSchemeIn(REGISTRY, ''), 'minimal');
  // An empty registry has no default to fall back to; callers turn that into
  // a no-op rather than an invalid scheme attribute.
  assert.equal(resolveSchemeIn(parseRegistry(null), 'emerald'), '');
});

// ---- pickFallback -------------------------------------------------------

test('pickFallback tries the default exactly once', () => {
  assert.equal(pickFallback(new Set(), 'minimal'), 'minimal');
  // The second failure has nothing left to try: re-requesting the same file
  // would spin.
  assert.equal(pickFallback(new Set(['minimal']), 'minimal'), null);
  // A different theme having failed first does not consume the default.
  assert.equal(pickFallback(new Set(['emerald']), 'minimal'), 'minimal');
  assert.equal(pickFallback(new Set(), ''), null);
  assert.equal(pickFallback(undefined, 'minimal'), 'minimal');
});

// ---- the page and the files agree --------------------------------------

const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');

test('the boot script and themes.js name the same elements', () => {
  // The server replaces this placeholder with the registry payload.
  assert.equal(html.match(/<!--admin-themes-->/g)?.length, 1);
  // The boot script reads the registry from this id before any module runs.
  assert.match(html, /getElementById\('adminThemes'\)/);
  // The theme sheet is the link both the boot script and setThemeSheet move.
  assert.match(html, /<link[^>]+id="themeSheet"/);
});

/** Parse a flat `a: b; c: d` declaration list into a Map. */
function declarations(body) {
  const out = new Map();
  for (const decl of body.split(';')) {
    const colon = decl.indexOf(':');
    if (colon === -1) continue;
    const prop = decl.slice(0, colon).trim();
    if (prop) out.set(prop, decl.slice(colon + 1).trim());
  }
  return out;
}

/** The declarations of the first rule with exactly *selector* in *css*. */
function ruleDeclarations(css, selector) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const at = text.indexOf(selector + ' {');
  assert.ok(at !== -1, `expected a rule for ${selector}`);
  const open = text.indexOf('{', at);
  const close = text.indexOf('}', open);
  return declarations(text.slice(open + 1, close));
}

test('the inline seed is exactly the default theme’s first frame', () => {
  const seedTag = html.match(/<style id="themeSeed">([\s\S]*?)<\/style>/);
  assert.ok(seedTag, 'expected the inline theme seed');
  const seed = seedTag[1];
  const minimal = readFileSync(join(ADMIN, 'css/themes/minimal.css'), 'utf8');

  // Exactly these keys: the seed exists so the first frame has a background,
  // text and accent colour before the sheet arrives.  Anything else belongs in
  // the theme file.
  const KEYS = ['--bg', '--text', '--accent'];
  const light = ruleDeclarations(seed, 'html');
  const dark = ruleDeclarations(seed, 'html[data-mode="dark"]');
  assert.deepEqual([...light.keys()].sort(), [...KEYS].sort());
  assert.deepEqual([...dark.keys()].sort(), [...KEYS].sort());

  const themeLight = ruleDeclarations(minimal, '[data-scheme="minimal"][data-mode="light"]');
  const themeDark = ruleDeclarations(minimal, '[data-scheme="minimal"][data-mode="dark"]');
  for (const key of KEYS) {
    assert.equal(light.get(key), themeLight.get(key), `seed light ${key}`);
    assert.equal(dark.get(key), themeDark.get(key), `seed dark ${key}`);
  }
});

/**
 * A dialog is a frame, not a floating card (issue #1472 P5).
 *
 * The rules this file pins, each of which a later edit can quietly undo:
 *
 * 1. there is exactly one scrim, it is `var(--overlay)`, and the token is a
 *    literal rgba in each theme block — light and dark — so the value a
 *    reviewer reads in the sheet is the value `getComputedStyle` returns;
 * 2. no `backdrop-filter`: "no shadow" excludes frosted glass, which is a
 *    depth effect and fights the hairline grid;
 * 3. a panel's corner is `var(--card-radius)` — the panel corner token, not
 *    the control token and not a literal — and its edge is a 1px border;
 * 4. a panel carries no `box-shadow`; depth is the scrim's job;
 * 5. a drawer is `var(--drawer-w)` wide and rounds only the corners that
 *    stand free;
 * 6. every overlay has exactly one panel child, and that panel — not the
 *    overlay — carries `role="dialog"`, `aria-modal="true"` and a label;
 * 7. the model side panel is a dialog surface too: not inside a `.modal-overlay`
 *    (it is an `<aside>` with its own scrim), but with the same role, the same
 *    frame and the same behaviour, so the panel has 15 dialog surfaces in all:
 *    twelve centred dialogs, two overlay drawers and one side panel.
 *
 * Modal behaviour (focus, scroll lock, Escape, backdrop) is JS and is checked
 * in the browser by the P5 measurement run, not here.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin');
const CSS_DIR = join(ADMIN, 'css');
const THEME_DIR = join(CSS_DIR, 'themes');

const strip = (text) => text.replace(/\/\*[\s\S]*?\*\//g, '');
const read = (path) => strip(readFileSync(path, 'utf8'));

const BASE = read(join(CSS_DIR, 'base.css'));
const COMPONENTS = read(join(CSS_DIR, 'components.css'));
const THEMES = readdirSync(THEME_DIR)
  .filter((name) => name.endsWith('.css'))
  .map((name) => ({ name, text: read(join(THEME_DIR, name)) }));

/** Every `selector { declarations }` block in a sheet. */
function rules(text) {
  const out = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(text)) !== null) out.push({ selector: m[1].trim(), body: m[2] });
  return out;
}

const declarations = (body) => body
  .split(';')
  .map((d) => d.trim())
  .filter(Boolean)
  .map((d) => {
    const at = d.indexOf(':');
    return [d.slice(0, at).trim(), d.slice(at + 1).trim()];
  });

const valueOf = (body, prop) => (declarations(body).find(([p]) => p === prop) || [])[1];

test('one scrim, and it is the theme token', () => {
  // The token: one declaration per theme, in its light block and its dark
  // block, and a literal rgba in both.
  for (const { name, text } of THEMES) {
    const values = rules(text).flatMap((r) => declarations(r.body)
      .filter(([p]) => p === '--overlay')
      .map(([, v]) => v));
    assert.equal(values.length, 2, `${name}: expected one --overlay per mode, found ${values.length}`);
    for (const v of values) {
      assert.match(v, /^rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*(\.\d+|0?\.\d+)\)$/,
        `${name}: --overlay must be a literal black rgba, got "${v}"`);
    }
  }
  // The consumer: the overlay rule reads the token and nothing else.
  const overlay = rules(COMPONENTS).find((r) => r.selector === '.modal-overlay');
  assert.ok(overlay, 'expected a .modal-overlay rule');
  assert.equal(valueOf(overlay.body, 'background'), 'var(--overlay)');
  // A dialog surface in a component sheet must not paint its own scrim.  The
  // check is scoped to the dialog selectors: a popover's shadow may be any
  // colour, and is P6's business.
  const dialogSelectors = /modal|drawer|overlay/;
  for (const rule of rules(COMPONENTS)) {
    if (!dialogSelectors.test(rule.selector)) continue;
    for (const [prop, value] of declarations(rule.body)) {
      assert.ok(!/^rgba\(\s*0\s*,\s*0\s*,\s*0/.test(value),
        `${rule.selector} { ${prop}: ${value} } paints a scrim colour in a component sheet`);
    }
  }
});

test('no frosted glass', () => {
  for (const [name, text] of [['base.css', BASE], ['components.css', COMPONENTS]]) {
    assert.ok(!/backdrop-filter/.test(text), `${name}: backdrop-filter is out of scope for a hairline panel`);
  }
});

test('a panel keeps the panel corner, a 1px edge and no shadow', () => {
  const modal = rules(COMPONENTS).find((r) => r.selector === '.modal');
  assert.ok(modal, 'expected a .modal rule');
  assert.equal(valueOf(modal.body, 'border-radius'), 'var(--card-radius)');
  assert.equal(valueOf(modal.body, 'box-shadow'), undefined, '.modal must not carry a shadow');
  assert.match(valueOf(modal.body, 'border') || '', /^1px solid var\(--border\)$/);

  const drawerRules = rules(COMPONENTS).filter((r) => /#providerModal > \.modal/.test(r.selector));
  const body = drawerRules.find((r) => valueOf(r.body, 'width') === 'var(--drawer-w)');
  assert.ok(body, 'expected the drawer width to be var(--drawer-w)');
  // The phone breakpoint takes the whole width, and with it the last corner.
  const phone = drawerRules.find((r) => valueOf(r.body, 'width') === '100vw');
  assert.ok(phone, 'expected a full-width rule for the phone');
  assert.equal(valueOf(phone.body, 'border-radius'), '0');
  // Only the corners that stand free: left two round, right two square.
  assert.equal(valueOf(body.body, 'border-radius'), 'var(--card-radius) 0 0 var(--card-radius)');
  assert.equal(valueOf(body.body, 'box-shadow'), undefined, 'a drawer must not carry a shadow');
  assert.equal(valueOf(body.body, 'border-right'), '0');
  assert.equal(valueOf(body.body, 'border-top'), '0');
  assert.equal(valueOf(body.body, 'border-bottom'), '0');
});

test('the drawer width is geometry, declared once at the root', () => {
  const decls = rules(BASE).flatMap((r) => declarations(r.body).filter(([p]) => p === '--drawer-w'));
  assert.equal(decls.length, 1, `expected one --drawer-w, found ${decls.length}`);
  assert.match(decls[0][1], /^\d+px$/);
  assert.ok(!THEMES.some(({ text }) => text.includes('--drawer-w')),
    '--drawer-w is geometry; a theme may not override it');
});

test('the model side panel takes the same frame and the same role', () => {
  const drawer = rules(COMPONENTS).find((r) => r.selector === '.model-drawer' && /position/.test(r.body));
  assert.ok(drawer, 'expected a .model-drawer rule');
  assert.equal(valueOf(drawer.body, 'width'), 'var(--drawer-w)');
  assert.equal(valueOf(drawer.body, 'border-radius'), 'var(--card-radius) 0 0 var(--card-radius)');
  assert.equal(valueOf(drawer.body, 'box-shadow'), undefined, 'the side panel must not carry a shadow');
  assert.equal(valueOf(drawer.body, 'border-left'), '1px solid var(--border)');
  const scrim = rules(COMPONENTS).find((r) => r.selector === '.drawer-scrim');
  assert.ok(scrim, 'expected the side panel to have a scrim');
  assert.equal(valueOf(scrim.body, 'background'), 'var(--overlay)');

  const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');
  const tag = /<aside class="model-drawer"[^>]*>/.exec(html);
  assert.ok(tag, 'expected the model drawer element');
  for (const attr of ['role="dialog"', 'aria-modal="true"', 'tabindex="-1"', 'aria-labelledby="drawerTitle"']) {
    assert.ok(tag[0].includes(attr), `the model drawer is missing ${attr}`);
  }
  assert.ok(!/aria-label=/.test(tag[0]), 'the panel is labelled by its heading, not a frozen English label');
  assert.ok(/<div class="drawer-scrim" id="modelDrawerScrim"/.test(html), 'expected the side panel scrim element');
});

test('the role is on the panel, the overlay is presentational', () => {
  const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');
  const overlays = [...html.matchAll(/<div class="modal-overlay"([^>]*)>/g)];
  assert.equal(overlays.length, 14, `expected the 14 overlays, found ${overlays.length}`);
  for (const [, attrs] of overlays) {
    assert.ok(!/role=|aria-modal/.test(attrs),
      `overlay "${attrs.trim()}" carries dialog semantics; they belong on the panel`);
  }
  const panels = [...html.matchAll(/<div class="modal(?: modal-(?:wide|yaml))?"([^>]*)>/g)];
  assert.equal(panels.length, 14, 'expected one panel per overlay');
  for (const [, attrs] of panels) {
    assert.match(attrs, /role="dialog"/);
    assert.match(attrs, /aria-modal="true"/);
    assert.match(attrs, /tabindex="-1"/, 'the panel takes focus on open');
    assert.match(attrs, /aria-labelledby="[^"]+"/, 'the panel is labelled by its heading');
  }
  // Every label points at an id that exists.
  for (const [, attrs] of panels) {
    const id = /aria-labelledby="([^"]+)"/.exec(attrs)[1];
    assert.ok(html.includes(`id="${id}"`), `aria-labelledby="${id}" names no element`);
  }
  // The total the docs quote: 12 centred dialogs + 2 overlay drawers inside
  // the 14 overlays, plus the model side panel, which is its own element.
  const sidePanels = [...html.matchAll(/<aside class="model-drawer"[^>]*>/g)];
  assert.equal(sidePanels.length, 1, 'expected the one model side panel');
  assert.equal(overlays.length + sidePanels.length, 15,
    '15 dialog surfaces: 14 overlays and the model side panel');
});

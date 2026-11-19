/**
 * Markup built for innerHTML escapes what it interpolates.
 *
 * An interpolation inside an innerHTML / outerHTML / insertAdjacentHTML
 * assignment must go through esc(), jsArg(), raw() or an html`` template
 * (core/html.js).  Offenders that predate this lint sit in the LEGACY map
 * (innerhtml-legacy.mjs).  The map can only shrink: a new offender fails the
 * test, and so does an entry that no longer matches the source, so a fix has
 * to delete its line.
 *
 * Scope: the scanner reads what the assignment spells out.  It does not follow
 * a variable assigned earlier (`el.innerHTML = markup`), and it does not check
 * that an escaper suits its position (that is admin-handlers.test.mjs).
 *
 * Two more positions are outside it, because html protects HTML quoting only:
 * `style="...${v}"` (eight occurrences across seven files), where `esc` does
 * not stop `;` or `)` — providers.js `background:${dotColor}` takes
 * `desc.color` from the shim descriptor — and `<script>` context (none today).
 * The one position the lint does enforce inside html`` is the unquoted
 * attribute value (`title=${v}`), which it rejects.
 */

import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

import { isSafeExpr, keyCounts, scanSource, scanUnquoted } from './innerhtml-scan.mjs';
import { LEGACY } from './innerhtml-legacy.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);

function adminFiles(dir = ADMIN_JS) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...adminFiles(path));
    else if (entry.endsWith('.js')) out.push(path);
  }
  return out.sort();
}

test('scanner: escaped interpolations pass, raw ones are reported', () => {
  const found = (src) => scanSource(src).map((f) => f.expr);
  assert.deepEqual(found('el.innerHTML = `<b>${esc(name)}</b>`;'), []);
  assert.deepEqual(found('el.innerHTML = `<b>${name}</b>`;'), ['name']);
  assert.deepEqual(found('el.innerHTML += `<i>${a}</i>${jsArg(b)}`;'), ['a']);
  assert.deepEqual(found('el.outerHTML = `${raw(x)}${y}`;'), ['y']);
  assert.deepEqual(found('el.insertAdjacentHTML("beforeend", `<p>${v}</p>`);'), ['v']);
  assert.deepEqual(found('el.innerHTML = html`<p>${v}</p>`;'), []);
  // A nested template is judged by its own interpolations.
  assert.deepEqual(
    found('el.innerHTML = `${xs.map(x => `<li>${esc(x)}</li>`).join("")}`;'),
    [],
  );
  assert.deepEqual(
    found('el.innerHTML = `${xs.map(x => `<li>${x}</li>`).join("")}`;'),
    ['x'],
  );
  // A call that merely starts with esc( does not cover the whole expression.
  assert.equal(isSafeExpr('esc(a) + b'), false);
  assert.equal(isSafeExpr('esc(a)'), true);
  // Assignments that are not markup sinks, and comments, are ignored.
  assert.deepEqual(found('el.textContent = `${x}`;'), []);
  assert.deepEqual(found('// el.innerHTML = `${x}`;'), []);
  assert.deepEqual(found('if (el.innerHTML === `${x}`) {}'), []);
});

test('admin js: no new unescaped innerHTML interpolation, and LEGACY only shrinks', () => {
  const items = [];
  for (const file of adminFiles()) {
    const rel = relative(ADMIN_JS, file);
    for (const { expr } of scanSource(readFileSync(file, 'utf8'))) {
      items.push({ file: rel, expr });
    }
  }
  const actual = keyCounts(items);
  const added = [];
  const stale = [];
  for (const [key, n] of Object.entries(actual)) {
    const allowed = LEGACY[key] || 0;
    if (n > allowed) added.push(`${key} (found ${n}, allowed ${allowed})`);
  }
  for (const [key, n] of Object.entries(LEGACY)) {
    const have = actual[key] || 0;
    if (have < n) stale.push(`${key} (listed ${n}, found ${have})`);
  }
  assert.deepEqual(
    added,
    [],
    'unescaped interpolation in an innerHTML assignment: wrap it in esc()/jsArg()/raw() or use html``',
  );
  assert.deepEqual(
    stale,
    [],
    'LEGACY entry no longer present: delete it from tests/js/innerhtml-legacy.mjs',
  );
});

test('scanner: an unquoted attribute value in html`` is flagged, quoted forms are not', () => {
  const flagged = (src) => scanUnquoted(src).map((f) => f.expr);
  assert.deepEqual(flagged('x = html`<div title=${p}>`;'), ['p']);
  assert.deepEqual(flagged('x = html`<div title = ${p}>`;'), ['p']);
  assert.deepEqual(flagged('x = html`<a href="/u" title=${p}>${q}</a>`;'), ['p']);
  assert.deepEqual(flagged('x = `${xs.map(x => html`<i data-id=${x}>`)}`;'), ['x']);
  assert.deepEqual(flagged('x = html`<div title="${p}">`;'), []);
  assert.deepEqual(flagged("x = html`<div title='${p}'>`;"), []);
  assert.deepEqual(flagged('x = html`<div class="a ${p}">${q}</div>`;'), []);
  assert.deepEqual(flagged('x = html`<div onclick="f(${jsArg(p)})">`;'), []);
  // Plain templates and comments are not html`` templates.
  assert.deepEqual(flagged('x = `<div title=${p}>`;'), []);
  assert.deepEqual(flagged('// html`<div title=${p}>`'), []);
});

test('admin js: no html`` template interpolates an unquoted attribute value', () => {
  const bad = [];
  for (const file of adminFiles()) {
    for (const { line, expr } of scanUnquoted(readFileSync(file, 'utf8'))) {
      bad.push(`${relative(ADMIN_JS, file)}:${line} \${${expr}}`);
    }
  }
  assert.deepEqual(bad, [], 'quote the attribute value: title="${v}", not title=${v}');
});

/**
 * Inline handlers build JS literals with `jsArg`, never `esc`.
 *
 * `esc()` maps `'` to `&#39;`, and the HTML parser decodes that back to a
 * real quote before the handler is compiled — so `fn('${esc(name)}')` with a
 * name like `a'-alert(1)-'b` renders code the browser runs.  `jsArg()` writes
 * the literal with JSON and escapes the attribute's own quotes instead.
 *
 * The sweep behind issue #1394 converted every site; these cases keep the
 * next one from coming back, and prove that `jsArg` output survives the
 * attribute parse.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

import { esc, jsArg } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/html.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);

/** Every .js file under the admin panel, tests included. */
function adminFiles(dir = ADMIN_JS) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...adminFiles(path));
    else if (entry.endsWith('.js')) out.push(path);
  }
  return out;
}

/** A handler handed to `setAttribute` or assigned as a string.
 *
 *  These never pass through the HTML parser, so `jsArg`'s `&quot;` is not the
 *  answer there either — the literal is compiled as written.  The rule flags
 *  any interpolation into a quoted position. */
function handlerStringInJs(text) {
  const offenders = [];
  text.split('\n').forEach((line, i) => {
    const at = line.search(/setAttribute\(\s*['"]on\w+['"]|\.on\w+\s*=\s*[`'"]/);
    if (at === -1) return;
    const rest = line.slice(at);
    if (/'\$\{|`\$\{/.test(rest)) offenders.push(i + 1);
  });
  return offenders;
}

/** A handler attribute that builds a quoted argument the parser can undo.
 *
 *  Two shapes: `esc()` called inside the attribute, and a single-quoted
 *  literal that opens with an interpolation (`runTest('${n}',…)`) — the
 *  second catches the aliasing case, where the escaping happened in a
 *  variable and the attribute itself says nothing about it. */
function escInHandler(text) {
  // `esc(` anywhere in the attribute, or an *opening* quote followed by an
  // interpolation — the closing quote of `'text'${arg}` is legitimate.
  const quoted = /on\w+="[^"]*(?:esc\(|[(,;\s]'\$\{)|on\w+='[^']*esc\(/;
  const offenders = [];
  text.split('\n').forEach((line, i) => {
    if (quoted.test(line)) offenders.push(i + 1);
  });
  return offenders;
}

/** What the HTML parser hands the JS compiler: one pass, entities decoded. */
function parseAttribute(value) {
  return value.replace(/&(amp|quot|lt|gt);/g, (_, name) => ({
    amp: '&',
    quot: '"',
    lt: '<',
    gt: '>',
  })[name]);
}

test('the scan notices the shape it exists for', () => {
  // Guard the guard: a scan that matches nothing would pass on a broken tree.
  assert.deepEqual(
    escInHandler(`<button onclick="del('\${esc(name)}')">x</button>`),
    [1],
  );
  assert.deepEqual(
    escInHandler(`' onchange="toggle(\\'' + esc(name) + '\\',this)">' +`),
    [1],
  );
  assert.deepEqual(escInHandler(`<button onclick="del(\${jsArg(name)})">x</button>`), []);
  assert.deepEqual(
    escInHandler(`<button onclick="runTest('\${n}','text')">x</button>`),
    [1],
    'an aliased literal is the same bug',
  );
  assert.deepEqual(escInHandler(`<button onclick="go(\${jsArg(name)}, this)">x</button>`), []);
  assert.deepEqual(
    handlerStringInJs("div.setAttribute('onclick', `go('\${desc.name}')`);"),
    [1],
    'a handler string set in JS is compiled as written',
  );
  assert.deepEqual(handlerStringInJs("div.addEventListener('click', () => go(div, desc.name));"), []);
  assert.deepEqual(escInHandler(`<td title="\${esc(name)}">x</td>`), []);
});

test('no handler in the admin builds its argument with esc()', () => {
  const offenders = [];
  for (const file of adminFiles()) {
    for (const line of escInHandler(readFileSync(file, 'utf8'))) {
      offenders.push(`${relative(ADMIN_JS, file)}:${line}`);
    }
  }
  assert.deepEqual(offenders, [], 'use jsArg() (core/html.js), or a data-* attribute');
});

test('no handler string handed to setAttribute interpolates a quoted value', () => {
  const offenders = [];
  for (const file of adminFiles()) {
    for (const line of handlerStringInJs(readFileSync(file, 'utf8'))) {
      offenders.push(`${relative(ADMIN_JS, file)}:${line}`);
    }
  }
  assert.deepEqual(offenders, [], 'use a listener, or JSON.stringify for the literal');
});

test('every module that builds a handler argument imports jsArg', () => {
  for (const file of adminFiles()) {
    const text = readFileSync(file, 'utf8');
    if (!text.includes('jsArg(')) continue;
    if (relative(ADMIN_JS, file) === 'core/html.js') continue;
    assert.match(
      text,
      /import \{[^}]*jsArg[^}]*\} from '[^']*html\.js';/,
      `${relative(ADMIN_JS, file)} uses jsArg without importing it`,
    );
  }
});

test('a name survives the attribute parse whatever it contains', () => {
  const names = [
    'plain',
    "a'-alert(1)-'b",
    'say "hi"',
    "both ' and \"",
    '<b>&amp;</b>',
    'back\\slash',
    'line\nbreak',
    '${notAnInterpolation}',
    '日本語のプロバイダ',
  ];
  for (const name of names) {
    const attribute = jsArg(name);
    assert.deepEqual(
      JSON.parse(parseAttribute(attribute)),
      name,
      `jsArg(${JSON.stringify(name)}) must come back as the same string`,
    );
  }
});

test('the shape it replaced does not survive the same parse', () => {
  // The old literal: a single-quoted string built from esc().  The parser
  // turns &#39; back into a quote and the literal ends early.
  const old = `'${esc("a'-alert(1)-'b")}'`;
  assert.throws(() => JSON.parse(parseAttribute(old)));
});

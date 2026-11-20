/**
 * `esc(t(key))` assumes the translation carries no markup.
 *
 * `esc` turns `<strong>` into literal text, so wrapping a key whose sentence
 * holds tags (as `confirm.cleanupMsg` does) prints them on screen.  Today no
 * wrapped key has markup; this test makes that a checked contract: every
 * literal key inside `esc(t(...))` must be free of `<` and `>` in the English
 * AND the Chinese dictionary.  A key that needs markup belongs in `raw(t(...))`
 * and out of `esc(...)`.
 */

import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { test } from 'node:test';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { scanEscT } from './innerhtml-scan.mjs';

// core/state.js reads localStorage at module scope (see strict-i18n.test.mjs).
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {}, clear() {} };
globalThis.window = globalThis;
globalThis.document = {
  documentElement: { getAttribute: () => null, setAttribute() {} },
  body: { classList: { add() {}, remove() {}, contains: () => false } },
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
};

const ADMIN_JS = join(
  dirname(fileURLToPath(import.meta.url)),
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const { I18N } = await import(join(ADMIN_JS, 'core/i18n.js'));

function adminFiles(dir = ADMIN_JS) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...adminFiles(path));
    else if (entry.endsWith('.js')) out.push(path);
  }
  return out.sort();
}

/** The sites whose wrapped translation holds markup, in either language.
 *  A zh gap falls back to English, as t() does. */
function markupSites(sites, dict) {
  const bad = [];
  for (const s of sites) {
    for (const lang of ['en', 'zh']) {
      const text = dict[lang]?.[s.key] ?? dict.en[s.key];
      if (text == null) bad.push(`${s.where}: ${s.key} (${lang}) is missing`);
      else if (/[<>]/.test(text)) bad.push(`${s.where}: ${s.key} (${lang}) holds markup`);
    }
  }
  return bad;
}

test('scanner: literal keys, ternary arms and dynamic keys are told apart', () => {
  const r = scanEscT(
    "a(`${esc(t('a.b'))}`);\n" +
      '`${_esc(t("c.d", {x: 1}))}` `${esc(t(on ? \'e.f\' : \'g.h\'))}`\n' +
      "`${esc(t('jobs.' + action))}` `${esc(t(key))}` `${esc(label)}` `${raw(t('x.y'))}`",
  );
  assert.deepEqual(
    r.keys.map((k) => k.key),
    ['a.b', 'c.d', 'e.f', 'g.h'],
  );
  assert.equal(r.dynamic.length, 2);
});

test('checker: a wrapped key whose translation gains markup is reported', () => {
  const sites = [{ key: 'k.one', where: 'f.js:1' }];
  const dict = { en: { 'k.one': 'Plain' }, zh: { 'k.one': '纯文本' } };
  assert.deepEqual(markupSites(sites, dict), []);
  assert.equal(markupSites(sites, { ...dict, en: { 'k.one': 'A <b>B</b>' } }).length, 1);
  assert.equal(markupSites(sites, { ...dict, zh: { 'k.one': '甲 <b>乙</b>' } }).length, 1);
  assert.equal(markupSites(sites, { en: { 'k.one': '1 > 0' }, zh: {} }).length, 2);
});

test('every esc(t(literal key)) in admin/js has markup-free en and zh text', () => {
  const sites = [];
  let dynamic = 0;
  for (const file of adminFiles()) {
    const r = scanEscT(readFileSync(file, 'utf8'));
    dynamic += r.dynamic.length;
    for (const k of r.keys) {
      sites.push({ key: k.key, where: `${relative(ADMIN_JS, file)}:${k.line}` });
    }
  }
  // A scanner that matches nothing would pass vacuously.
  assert.ok(sites.length >= 20, `scanner found only ${sites.length} esc(t('key')) sites`);
  assert.ok(dynamic > 0, 'scanner found no run-time-built keys; its dynamic branch is untested');
  assert.deepEqual(markupSites(sites, I18N), []);
});

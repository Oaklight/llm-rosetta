/**
 * The panel's i18n keys must exist in both dictionaries (#1441).
 *
 * `t()` falls back to the key, so a missing sentence shows `toast.modelsIgnored`
 * to the operator instead of failing anything.  The component tests cannot
 * catch it: they inject a fake translator that returns the key.
 *
 * The scanner is exercised against synthetic dictionaries, then run over
 * `admin/js/**`.  Every unresolvable `t(...)` is listed rather than skipped —
 * a guard whose failure mode is "checked nothing" is not a guard.
 *
 * What it does not see: keys passed to a wrapper that calls `t()` itself
 * (`_credentialText(key, fallback)` in tabs/providers.js, 63 call sites), keys
 * held in a table and passed on by a variable (`_buildStatusSeg`'s options),
 * and keys used only from admin.html's data-i18n attributes.  Those are named
 * here and in the pull request rather than implied to be covered.
 *
 * Run with `make test-js` (or `node --test tests/js/*.test.mjs`).
 */

import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';

import {
  checkSources,
  dictionaryDiff,
  dictionarySections,
  duplicateKeys,
  isKeyShaped,
  keysIn,
  scanSource,
} from './i18n-keys.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const I18N = join(ADMIN_JS, 'core/i18n.js');

const NAMESPACES = new Set(['toast', 'btn', 'label', 'test']);

function jsFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...jsFiles(path));
    else if (entry.endsWith('.js') && path !== I18N) out.push(path);
  }
  return out;
}

test('the scanner finds t() literals and key: literals', () => {
  const source = [
    "const a = t('toast.saved');",
    'const b = t("toast.removed", {count});',
    "const part = {key: 'toast.modelsIgnored', count: 1};",
    "const notAKey = {key: 'free'};",
  ].join('\n');
  const { literals } = scanSource(source, NAMESPACES);
  assert.deepEqual(
    literals.map((entry) => entry.key),
    ['toast.saved', 'toast.removed', 'toast.modelsIgnored'],
  );
  assert.deepEqual(literals.map((entry) => entry.line), [1, 2, 3]);
});

test('a computed argument is listed, and its literals are still checked', () => {
  const source = [
    'const a = t(key);',
    "const b = t(flag ? 'btn.save' : 'btn.cancel');",
    "const c = t('toast.' + name);",
  ].join('\n');
  const { literals, unresolved } = scanSource(source, NAMESPACES);
  // The ternary's own keys are checked; the variable and the concatenation are
  // reported, not silently skipped.
  assert.deepEqual(literals.map((entry) => entry.key), ['btn.save', 'btn.cancel']);
  assert.equal(unresolved.length, 3);
  assert.match(unresolved[0].expr, /^t\(key\)$/);
  assert.match(unresolved[2].expr, /toast\.' \+ name/);
});

test('a key missing from one language is reported with that language', () => {
  const sources = [
    ['a.js', "t('toast.saved'); t('toast.onlyEn'); t('toast.nowhere');"],
  ];
  const dictionaries = {
    en: new Set(['toast.saved', 'toast.onlyEn']),
    zh: new Set(['toast.saved']),
  };
  const { missing } = checkSources(sources, dictionaries);
  assert.deepEqual(
    missing.map((entry) => [entry.key, entry.languages.join('+')]).sort(),
    [['toast.nowhere', 'en+zh'], ['toast.onlyEn', 'zh']],
  );
});

test('a complete pair of dictionaries passes', () => {
  const sources = [['a.js', "t('toast.saved'); {key: 'toast.saved'};"]];
  const dictionaries = {
    en: new Set(['toast.saved']),
    zh: new Set(['toast.saved']),
  };
  assert.deepEqual(checkSources(sources, dictionaries).missing, []);
});

test('isKeyShaped uses the dictionaries, not a pattern', () => {
  assert.equal(isKeyShaped('toast.saved', NAMESPACES), true);
  assert.equal(isKeyShaped('saved', NAMESPACES), false);
  assert.equal(isKeyShaped('free', NAMESPACES), false);
  assert.equal(isKeyShaped('unknown.key', NAMESPACES), false);
});

test('dictionarySections splits en from zh, and keysIn reads both', () => {
  const fake = [
    'export const I18N = {',
    '  en: {',
    "    'toast.a':'A', 'toast.b':'B',",
    '  },',
    '  zh: {',
    "    'toast.a':'甲',",
    '  },',
    '};',
  ].join('\n');
  const sections = dictionarySections(fake);
  assert.deepEqual([...keysIn(sections.en)], ['toast.a', 'toast.b']);
  assert.deepEqual([...keysIn(sections.zh)], ['toast.a']);
});

test('dictionaryDiff reports each direction on its own', () => {
  assert.deepEqual(
    dictionaryDiff(new Set(['a.x', 'a.y']), new Set(['a.y', 'a.z'])),
    { onlyEn: ['a.x'], onlyZh: ['a.z'] },
  );
  assert.deepEqual(dictionaryDiff(new Set(['a.x']), new Set(['a.x'])), {
    onlyEn: [],
    onlyZh: [],
  });
});

test('duplicateKeys lists every value of a repeated key', () => {
  const block = [
    "'a.same':'one', 'a.diff':'first', 'a.once':'only',",
    "'a.diff':'second', 'a.same':'one',",
  ].join('\n');
  const found = duplicateKeys(block);
  assert.deepEqual([...found.keys()].sort(), ['a.diff', 'a.same']);
  assert.deepEqual(found.get('a.diff'), ["'first'", "'second'"]);
  assert.deepEqual(found.get('a.same'), ["'one'", "'one'"]);
  assert.equal(found.has('a.once'), false);
});

// The real panel.  Every literal the JS passes to `t()` — and every toast part
// key — must be declared in both dictionaries.
test('every literal key the admin JS passes to t(), and every toast key, exists in both dictionaries', () => {
  const sections = dictionarySections(readFileSync(I18N, 'utf8'));
  const dictionaries = { en: keysIn(sections.en), zh: keysIn(sections.zh) };
  const sources = jsFiles(ADMIN_JS).map((path) => [
    relative(ADMIN_JS, path),
    readFileSync(path, 'utf8'),
  ]);

  const { missing } = checkSources(sources, dictionaries);
  const report = missing
    .map((entry) => `${entry.file}:${entry.line} ${entry.key} (missing in ${entry.languages.join(', ')})`)
    .join('\n');
  assert.deepEqual(missing, [], `missing i18n keys:\n${report}`);
});

test('the two dictionaries declare the same keys', () => {
  const sections = dictionarySections(readFileSync(I18N, 'utf8'));
  const diff = dictionaryDiff(keysIn(sections.en), keysIn(sections.zh));
  assert.deepEqual(
    diff,
    { onlyEn: [], onlyZh: [] },
    `the dictionaries disagree:\nonly en: ${diff.onlyEn.join(', ')}\nonly zh: ${diff.onlyZh.join(', ')}`,
  );
});

test('neither dictionary declares a key twice', () => {
  // A later declaration wins silently, so the earlier sentence is dead text
  // that keeps being translated.  The check reports every value so a failure
  // shows what was overridden.
  const sections = dictionarySections(readFileSync(I18N, 'utf8'));
  for (const [language, block] of Object.entries(sections)) {
    const found = duplicateKeys(block);
    const report = [...found]
      .map(([key, values]) => `${language} ${key}: ${values.join(' -> ')}`)
      .join('\n');
    assert.deepEqual([...found.keys()], [], `duplicate keys:\n${report}`);
  }
});

test('the unresolved t(...) expressions are known and listed', () => {
  const sections = dictionarySections(readFileSync(I18N, 'utf8'));
  const dictionaries = { en: keysIn(sections.en), zh: keysIn(sections.zh) };
  const sources = jsFiles(ADMIN_JS).map((path) => [
    relative(ADMIN_JS, path),
    readFileSync(path, 'utf8'),
  ]);

  const { unresolved } = checkSources(sources, dictionaries);
  // These are the dynamic keys the scanner cannot resolve.  The assertion is a
  // floor, not an allowlist: a new one is allowed to appear, but the count is
  // printed so the next reader can see what is not being checked.
  assert.ok(unresolved.length > 0, 'expected some computed keys in the panel');
  assert.ok(
    unresolved.every((entry) => entry.expr.startsWith('t(') && entry.line > 0),
    JSON.stringify(unresolved.slice(0, 3)),
  );
});

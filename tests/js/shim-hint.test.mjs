/**
 * The shim hint's language pick (#1311).
 *
 * The server normalises a `hint_msg` declaration to `{language: text}`, so the
 * panel's only job is the fallback order: the interface language, then
 * English, then the first declared language.  Nothing readable means no hint,
 * not an empty box.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { shimHintText } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-hint.js';

test('the interface language wins when it is declared', () => {
  const hint = { en: 'English note', zh: '中文说明' };
  assert.equal(shimHintText(hint, 'zh'), '中文说明');
  assert.equal(shimHintText(hint, 'en'), 'English note');
});

test('a missing language falls back to English', () => {
  assert.equal(shimHintText({ en: 'Only english' }, 'zh'), 'Only english');
});

test('a declaration with no English falls back to the first language', () => {
  assert.equal(shimHintText({ zh: '只有中文' }, 'en'), '只有中文');
});

test('no declaration yields no hint', () => {
  assert.equal(shimHintText(null, 'en'), '');
  assert.equal(shimHintText(undefined, 'zh'), '');
  assert.equal(shimHintText({}, 'en'), '');
});

test('a value that is not a language mapping yields no hint', () => {
  // The loader turns a bare YAML string into `{en: ...}` before the payload,
  // so a raw string here would be a contract violation, not a translation.
  assert.equal(shimHintText('bare string', 'en'), '');
});

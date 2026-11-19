/**
 * strictI18n: t() throws on a missing key instead of guessing a label.
 *
 * Production keeps the guess (label.fooEndpoint -> "Foo Endpoint").  The
 * strict mode is for tests and debugging, so a key that no dictionary holds
 * fails where it is used.
 */

import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

// core/state.js reads localStorage at module scope: give it the page's minimum
// environment first, as empty-state.test.mjs does.
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

const { setStrictI18n, strictI18nEnabled, t } = await import(
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/i18n.js'
);

afterEach(() => {
  setStrictI18n(null);
  delete globalThis.__ROSETTA_STRICT_I18N__;
  delete globalThis.location;
});

test('default: a missing label.* key is guessed, as in production', () => {
  assert.equal(strictI18nEnabled(), false);
  assert.equal(t('label.noSuchThing'), 'No Such Thing');
  assert.equal(t('label.noSuchEndpoint'), 'No Such Endpoint');
  assert.equal(t('toast.noSuchKey'), 'toast.noSuchKey');
});

test('strict: a missing key throws and names the key', () => {
  setStrictI18n(true);
  assert.throws(() => t('label.noSuchThing'), /Missing i18n key: label\.noSuchThing/);
  assert.throws(() => t('toast.noSuchKey'), /Missing i18n key: toast\.noSuchKey/);
  assert.throws(() => t('test.noSuch'), /Missing i18n key: test\.noSuch/);
});

test('strict: a key the dictionary holds still resolves, with params', () => {
  setStrictI18n(true);
  assert.equal(t('btn.edit'), 'Edit');
  assert.equal(t('keys.count', { count: 3 }), '3 keys');
});

test('the window flag and the query flag turn it on', () => {
  globalThis.__ROSETTA_STRICT_I18N__ = true;
  assert.equal(strictI18nEnabled(), true);
  delete globalThis.__ROSETTA_STRICT_I18N__;
  assert.equal(strictI18nEnabled(), false);
  globalThis.location = { search: '?tab=keys&strictI18n=1' };
  assert.equal(strictI18nEnabled(), true);
  globalThis.location = { search: '?strictI18n=0' };
  assert.equal(strictI18nEnabled(), false);
});

test('an explicit setting beats the environment flags', () => {
  globalThis.__ROSETTA_STRICT_I18N__ = true;
  setStrictI18n(false);
  assert.equal(t('label.noSuchThing'), 'No Such Thing');
});

/**
 * The vendor manifest editor's inheritance view.
 *
 * A manifest sets the connection defaults every leaf below it inherits, field
 * by field (`headers` per key).  These are the pure builders the panel paints
 * from: which field rows exist, which of them an unsaved edit touches, and
 * which leaves that edit changes.  No DOM, so node runs them directly.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  inheritanceRows, annotateBlast, changedLeaves, nameList,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/manifest-inherit.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const I18N = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/i18n.js',
);

const FIELDS = {
  base_url: { inherited_by: ['a--x', 'b--x'], overridden_by: ['c--x'] },
  api_key_env: { inherited_by: ['a--x', 'b--x', 'c--x'], overridden_by: [] },
  headers: {
    'x-api-version': { inherited_by: ['a--x'], overridden_by: ['b--x', 'c--x'] },
    'x-client': { inherited_by: ['a--x', 'b--x', 'c--x'], overridden_by: [] },
  },
};

test('each field is one row, and each header is its own row', () => {
  const rows = inheritanceRows(FIELDS);

  assert.deepEqual(rows.map(r => r.id), [
    'base_url',
    'api_key_env',
    'headers.x-api-version',
    'headers.x-client',
  ]);
  assert.deepEqual(rows[0].inherited, ['a--x', 'b--x']);
  assert.deepEqual(rows[0].overridden, ['c--x']);
  assert.equal(rows[2].field, 'headers');
  assert.equal(rows[2].header, 'x-api-version');
});

test('an empty or missing map has no rows', () => {
  assert.deepEqual(inheritanceRows({}), []);
  assert.deepEqual(inheritanceRows(null), []);
  assert.deepEqual(inheritanceRows(undefined), []);
});

test('a malformed entry does not throw', () => {
  const rows = inheritanceRows({ base_url: null, headers: { 'x-a': {} } });
  assert.deepEqual(rows[0].inherited, []);
  assert.deepEqual(rows[1].inherited, []);
});

test('only the touched field is marked, and it carries the leaves it changes', () => {
  const rows = annotateBlast(inheritanceRows(FIELDS), [
    { field: 'base_url', header: null, removed: false, leaves: ['b--x'] },
  ]);

  const [base, key, header] = rows;
  assert.equal(base.touched, true);
  assert.equal(base.removed, false);
  assert.deepEqual(base.affected, ['b--x']);
  assert.equal(key.touched, false);
  assert.deepEqual(key.affected, []);
  assert.equal(header.touched, false);
});

test('a header change marks the header row, not the whole headers field', () => {
  const rows = annotateBlast(inheritanceRows(FIELDS), [
    { field: 'headers', header: 'x-client', removed: false, leaves: ['a--x'] },
  ]);

  assert.equal(rows[2].touched, false);
  assert.equal(rows[3].touched, true);
  assert.deepEqual(rows[3].affected, ['a--x']);
});

test('removing a field is marked as emptying its inheritors', () => {
  const rows = annotateBlast(inheritanceRows(FIELDS), [
    { field: 'api_key_env', header: null, removed: true, leaves: ['a--x', 'b--x', 'c--x'] },
  ]);

  assert.equal(rows[1].touched, true);
  assert.equal(rows[1].removed, true);
});

test('the changed leaves are the union over every touched field', () => {
  const names = changedLeaves([
    { field: 'base_url', header: null, leaves: ['b--x'] },
    { field: 'headers', header: 'x-client', leaves: ['b--x', 'a--x'] },
  ]);

  assert.deepEqual(names, ['a--x', 'b--x']);
  assert.deepEqual(changedLeaves([]), []);
});

test('a long name list is truncated, not dropped', () => {
  assert.equal(nameList(['a', 'b']), 'a, b');
  assert.equal(nameList(['a', 'b', 'c'], 2), 'a, b +1');
  assert.equal(nameList([]), '');
  assert.equal(nameList(null), '');
});

test('every manifest key is translated in both languages', () => {
  // i18n.js imports state.js, which reads localStorage; the panel's own tests
  // read the dictionary as text instead of loading it.
  const text = readFileSync(I18N, 'utf8');
  const keys = [...text.matchAll(/'(shim\.manifest\.[A-Za-z]+)'/g)].map(m => m[1]);
  const counts = new Map();
  for (const key of keys) counts.set(key, (counts.get(key) || 0) + 1);

  assert.ok(counts.size >= 10, `expected the manifest block, saw ${counts.size} keys`);
  for (const [key, count] of counts) {
    assert.equal(count, 2, `${key} is in ${count} dictionary/ies, not 2`);
  }
});

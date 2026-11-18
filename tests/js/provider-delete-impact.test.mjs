/**
 * The delete-provider dialog names what each model will lose (#1593).
 *
 * It used to list every model that named the provider on any lane as "will
 * also be deleted", and the backend then made that true.  The two lists now
 * come from the same rule the cascade applies: an entry is deleted when no
 * lane of it survives, and trimmed when the provider was one of several.
 *
 * The helper is a module private and this file runs without a DOM, so the
 * case extracts it from the source and evaluates it — the point is the rule,
 * not the rendering next to it.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PROVIDERS = readFileSync(
  join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/providers.js'),
  'utf8',
);

/** The two helpers, run against a stub `S`. */
function impact(models, provider) {
  const start = PROVIDERS.indexOf('/** The lanes a model entry lists');
  const end = PROVIDERS.indexOf('function deleteProvider(');
  assert.ok(start !== -1 && end > start, 'the impact helpers are still there');
  const src = PROVIDERS.slice(start, end);
  // eslint-disable-next-line no-new-func
  const build = new Function('S', `${src}; return { _entryLanes, _providerModelImpact };`);
  return build({ configData: { models } });
}

test('a model served only by the deleted provider is deleted', () => {
  const { _providerModelImpact } = impact({ m: { provider: 'gone' } }, 'gone');

  assert.deepEqual(_providerModelImpact('gone'), { deleted: ['m'], trimmed: [] });
});

test('a model that keeps other lanes is trimmed, not deleted', () => {
  const models = {
    m: {
      providers: [
        { name: 'a', weight: 3 },
        { name: 'gone', weight: 2 },
        { name: 'b', weight: 4 },
      ],
    },
  };
  const { _providerModelImpact } = impact(models, 'gone');

  assert.deepEqual(_providerModelImpact('gone'), { deleted: [], trimmed: ['m'] });
});

test('the same provider twice on one entry still leaves the entry alive', () => {
  const models = {
    m: {
      providers: [
        { name: 'gone', upstream_model: 'one' },
        { name: 'a' },
        { name: 'gone', upstream_model: 'two' },
      ],
    },
  };
  const { _providerModelImpact } = impact(models, 'gone');

  assert.deepEqual(_providerModelImpact('gone'), { deleted: [], trimmed: ['m'] });
});

test('an entry where every lane names the provider is a deletion', () => {
  const models = { m: { providers: [{ name: 'gone' }, { name: 'gone', upstream_model: 'x' }] } };
  const { _providerModelImpact } = impact(models, 'gone');

  assert.deepEqual(_providerModelImpact('gone'), { deleted: ['m'], trimmed: [] });
});

test('an unrelated provider touches nothing', () => {
  const models = { m: { provider: 'a' }, n: { providers: ['a', 'b'] }, s: 'a' };
  const { _providerModelImpact } = impact(models, 'gone');

  assert.deepEqual(_providerModelImpact('gone'), { deleted: [], trimmed: [] });
});

test('a string entry is the same single lane as a dict one', () => {
  const models = { m: 'gone', n: 'a' };
  const { _entryLanes } = impact(models, 'gone');

  assert.deepEqual(_entryLanes('gone'), [['gone', null]]);
  assert.deepEqual(_entryLanes({ provider: 'a', upstream_model: 'u' }), [['a', 'u']]);
  assert.deepEqual(_entryLanes({ providers: ['a', { name: 'b' }] }), [['a', null], ['b', null]]);
});

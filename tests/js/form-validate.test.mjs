/**
 * Which field the provider form reports, and in what order (#1519).
 *
 * Every wrong form used to answer "Select at least one capability" — the
 * capability gate ran first, so an empty name and a duplicate name were never
 * reached and the message pointed at a field that was not the problem.  The
 * order is the order the operator fills the form, and it is a rule, so it is
 * pinned here rather than in a browser session.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { providerFormError } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/form-validate.js';

const OK = {
  name: 'openrouter',
  baseUrl: 'https://openrouter.ai/api/v1',
  type: 'openrouter--chat_completions',
  isLlm: true,
  anyCap: true,
  existing: [],
  editing: null,
};

const bad = (over) => providerFormError({ ...OK, ...over });

test('a filled form has no problem', () => {
  assert.equal(providerFormError(OK), null);
});

test('the name is reported before anything else', () => {
  // Every other field is also wrong; the name is still the answer.
  assert.deepEqual(
    bad({ name: '  ', baseUrl: '', type: '', anyCap: false }),
    { field: 'name', key: 'error.nameRequired' },
  );
});

test('a name already configured is refused, and before the capability gate', () => {
  // The defect: the capability message hid this one.
  assert.deepEqual(
    bad({ name: 'openrouter', existing: ['openrouter'], anyCap: false }),
    { field: 'name', key: 'error.nameTaken' },
  );
});

test('the existing-name check compares the way the config loader does', () => {
  assert.equal(
    bad({ name: 'OpenRouter', existing: ['openrouter'] }).key,
    'error.nameTaken',
  );
});

test('editing an entry keeps its own name', () => {
  assert.equal(
    providerFormError({ ...OK, existing: ['openrouter'], editing: 'openrouter' }),
    null,
  );
});

test('renaming one entry onto another is refused', () => {
  assert.equal(
    providerFormError({ ...OK, existing: ['a', 'b'], editing: 'a', name: 'b' }).key,
    'error.nameTaken',
  );
});

test('the base URL is checked after the name', () => {
  assert.deepEqual(
    bad({ baseUrl: ' ', type: '', anyCap: false }),
    { field: 'base_url', key: 'error.baseUrlRequired' },
  );
});

test('an LLM capability needs a top level before the capability gate', () => {
  assert.deepEqual(
    bad({ type: '', anyCap: false }),
    { field: 'type', key: 'error.typeRequired' },
  );
});

test('the capability gate is last', () => {
  assert.deepEqual(
    bad({ anyCap: false }),
    { field: 'capability', key: 'error.noCapability' },
  );
});

test('a non-LLM provider needs no top level', () => {
  assert.equal(bad({ isLlm: false, type: '' }), null);
});

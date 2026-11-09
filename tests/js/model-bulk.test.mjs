/**
 * The bulk-add toast reads every outcome the server reports (#1436).
 *
 * `POST /admin/api/config/models` answers with `added`, `appended`, `skipped`,
 * `ignored` and, when nothing changed, a `message`.  The dialog used to read
 * only the first two, so a fully refused batch said "All selected models
 * already exist" and a refused entry in a mixed batch vanished.
 *
 * The three payloads below are copied from a running gateway (0.14 dev), so
 * the assertions describe the wire, not an assumption about it:
 *
 *   all added   200 {"ok":true,"added":["new-a","new-b"],"appended":[],
 *                    "skipped":[],"ignored":[],"models":[…]}
 *   all ignored 200 {"ok":true,"added":[],"appended":[],"skipped":[],
 *                    "ignored":["weird-model"]}
 *   all exist   200 {"ok":true,"added":[],"appended":[],"skipped":[…],
 *                    "ignored":[],"message":"All models already exist"}
 *
 * Run with `make test-js` (or `node --test tests/js/*.test.mjs`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ALL_EXIST_MESSAGE,
  bulkAddOutcome,
  bulkAddResult,
  fallbackMessage,
  isRefusalOnly,
  toastFor,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/model-bulk.js';

test('a fully added batch reports the added count and nothing else', () => {
  const outcome = bulkAddOutcome({
    ok: true,
    added: ['new-a', 'new-b'],
    appended: [],
    skipped: [],
    ignored: [],
    models: ['existing-model', 'new-a', 'new-b'],
  });
  assert.deepEqual(outcome.parts, [{ key: 'toast.modelsAdded', count: 2 }]);
  assert.deepEqual(outcome.ignored, []);
  assert.equal(outcome.message, '');
  assert.equal(isRefusalOnly(outcome.parts), false);
});

test('a fully ignored batch is refused, never "already exists"', () => {
  // The measured all-ignored response: `ok` is true and `message` is absent,
  // so the client cannot lean on either to tell it apart from "all exist".
  const outcome = bulkAddOutcome({
    ok: true,
    added: [],
    appended: [],
    skipped: [],
    ignored: ['weird-model'],
  });
  assert.deepEqual(outcome.parts, [
    { key: 'toast.modelsIgnored', count: 1, names: 'weird-model' },
  ]);
  // Parts is non-empty, so the dialog shows this toast and never reaches the
  // `toast.modelsAllExist` fallback.
  assert.notEqual(outcome.parts.length, 0);
  assert.equal(outcome.message, '');
  assert.equal(isRefusalOnly(outcome.parts), true);
});

test('an all-exists batch keeps the server message', () => {
  const outcome = bulkAddOutcome({
    ok: true,
    added: [],
    appended: [],
    skipped: ['existing-model'],
    ignored: [],
    message: 'All models already exist',
  });
  assert.deepEqual(outcome.parts, []);
  assert.deepEqual(outcome.skipped, ['existing-model']);
  assert.equal(outcome.message, 'All models already exist');
  assert.equal(isRefusalOnly(outcome.parts), false);
});

test('a mixed batch names both what was added and what was refused', () => {
  const outcome = bulkAddOutcome({
    ok: true,
    added: ['new-a'],
    appended: [],
    skipped: [],
    ignored: ['weird-model', 'odd-model'],
    models: ['existing-model', 'new-a'],
  });
  assert.deepEqual(outcome.parts, [
    { key: 'toast.modelsAdded', count: 1 },
    { key: 'toast.modelsIgnored', count: 2, names: 'weird-model, odd-model' },
  ]);
  // Something did change, so the refusal is named in a success toast rather
  // than replacing it.
  assert.equal(isRefusalOnly(outcome.parts), false);
  assert.equal(isRefusalOnly(outcome.parts, 1), false);
});

test('the refused names are capped so the toast stays readable', () => {
  const ignored = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
  const outcome = bulkAddOutcome({ ok: true, ignored });
  const part = outcome.parts.find((p) => p.key === 'toast.modelsIgnored');
  assert.equal(part.count, 7);
  assert.equal(part.names, 'a, b, c, d, e (+2)');
});

test('a body without the new fields still reads as before', () => {
  // An older server, or an error envelope: no lists, no message.
  const outcome = bulkAddOutcome({ ok: true, added: ['only-one'] });
  assert.deepEqual(outcome.parts, [{ key: 'toast.modelsAdded', count: 1 }]);
  assert.deepEqual(outcome.ignored, []);
  assert.equal(bulkAddOutcome(null).parts.length, 0);
  assert.equal(isRefusalOnly([], 0), false);
});

test('a 500 envelope is a failure, not an empty batch', () => {
  // The measured mixed-batch answer: `api.post` resolves with this body and no
  // `ok`, so a caller asking only `if (res.ok)` falls through to the success
  // fallback and paints it over the error toast (#1436 review).
  const result = bulkAddResult({
    error: "Config saved but reload failed: config: invalid model entry for 'weird-model'",
    saved: true,
    reloaded: false,
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /reload failed/);
  // The success path is not reachable from a failure: no parts to render.
  assert.equal(result.parts, undefined);
});

test('a body that does not say ok is a failure too', () => {
  for (const body of [{}, null, undefined, { added: ['x'] }]) {
    assert.equal(bulkAddResult(body).ok, false, JSON.stringify(body));
  }
  const empty = bulkAddResult({ ok: true });
  assert.equal(empty.ok, true);
  assert.deepEqual(empty.parts, []);
});

test('the default "already exist" wording is localized, other messages are shown', () => {
  // The server's default reaches a Chinese panel as English; the panel has its
  // own key for exactly this sentence.
  assert.deepEqual(
    fallbackMessage({ message: ALL_EXIST_MESSAGE }),
    { key: 'toast.modelsAllExist' },
  );
  assert.deepEqual(fallbackMessage({ message: '' }), { key: 'toast.modelsAllExist' });
  assert.deepEqual(fallbackMessage({}), { key: 'toast.modelsAllExist' });
  // A message the panel does not have is passed through unchanged.
  assert.deepEqual(
    fallbackMessage({ message: 'Nothing to change: 3 models were skipped' }),
    { text: 'Nothing to change: 3 models were skipped' },
  );
});

// The tests read keys where `t()` would read sentences, so a translation change
// cannot make them pass for the wrong reason.
const tr = (key, params) => {
  const shown = { names: params?.names, count: params?.count };
  return `${key}(${shown.count ?? ''}${shown.names ? `|${shown.names}` : ''})`;
};

test('toastFor renders each measured outcome, with its kind', () => {
  const added = toastFor(bulkAddResult({ ok: true, added: ['a', 'b'] }), { tr });
  assert.deepEqual(added, { text: 'toast.modelsAdded(2)', kind: 'success' });

  const ignored = toastFor(bulkAddResult({ ok: true, ignored: ['weird'] }), { tr });
  assert.deepEqual(ignored, {
    text: 'toast.modelsIgnored(1|weird)',
    kind: 'error',
  });

  const allExist = toastFor(bulkAddResult({ ok: true, message: ALL_EXIST_MESSAGE }), { tr });
  assert.deepEqual(allExist, { text: 'toast.modelsAllExist()', kind: 'success' });

  const custom = toastFor(bulkAddResult({ ok: true, message: 'Nothing changed' }), { tr });
  assert.deepEqual(custom, { text: 'Nothing changed', kind: 'success' });
});

test('toastFor turns a failure into the error toast and nothing else', () => {
  const failure = toastFor(
    bulkAddResult({ error: 'Config saved but reload failed: bad entry' }),
    { tr },
  );
  assert.deepEqual(failure, {
    text: 'Config saved but reload failed: bad entry',
    kind: 'error',
  });
  // No error text still yields an error, not a success with an empty message.
  assert.deepEqual(toastFor(bulkAddResult({}), { tr }), {
    text: 'toast.error()',
    kind: 'error',
  });
});

test('toastFor folds removals into the summary and un-refuses a mixed batch', () => {
  const ignoredOnly = bulkAddResult({ ok: true, ignored: ['weird'] });
  assert.equal(toastFor(ignoredOnly, { tr }).kind, 'error');
  assert.equal(toastFor(ignoredOnly, { tr, removedCount: 1 }).kind, 'success');
  assert.equal(
    toastFor(ignoredOnly, { tr, removedCount: 1 }).text,
    'toast.modelsIgnored(1|weird), toast.modelsRemoved(1)',
  );

  const mixed = bulkAddResult({ ok: true, added: ['a'], ignored: ['weird'] });
  assert.deepEqual(toastFor(mixed, { tr }), {
    text: 'toast.modelsAdded(1), toast.modelsIgnored(1|weird)',
    kind: 'success',
  });
});

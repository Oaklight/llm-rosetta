/**
 * The Routing board keeps a lane the route dropped.
 *
 * `GET /admin/api/routing/<model>` now appends the lanes the document
 * declares but the selector cannot offer (state `off`).  The board must draw
 * them without letting them take part in drag, weight or share — while still
 * offering the switch that brings the provider back.  Text-level, like
 * `provider-list-wiring`: the connections are the contract.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROUTING = readFileSync(
  join(
    HERE,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/routing.js',
  ),
  'utf8',
);

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

test('an excluded lane says why instead of where it goes', () => {
  const body = bodyOf(ROUTING, 'function _laneSub(');
  assert.match(body, /provider\.excluded/, '_laneSub branches on the flag');
  assert.match(body, /routing\.reason\.off/, 'a provider switch is named as one');
  assert.match(body, /routing\.reason\.entry/, "a model entry's switch is not called the operator's");
  assert.match(body, /reason_kind === 'auto'/, 'the gateway diagnostic is shown as written');
});

test('a reorder posts to the route that takes an order', () => {
  const body = bodyOf(ROUTING, 'async function _reorderLane(');
  assert.match(body, /\/reorder-providers/, 'the endpoint is reorder-providers');
  assert.doesNotMatch(body, /\/models\/[^`']*\/providers`/, 'not the update-provider route');
});

test('the switch writes whichever switch is off', () => {
  const body = bodyOf(ROUTING, 'async function _toggleLane(');
  assert.match(body, /reason_kind === 'entry'/, 'an entry-off lane is recognised');
  assert.match(body, /update-provider/, 'it writes the model entry');
  assert.match(body, /\/toggle/, 'every other lane writes the provider switch');
});

test('the board excludes an off lane from drag and weight', () => {
  const body = bodyOf(ROUTING, 'function _renderBoard(');
  assert.match(body, /provider\.excluded/, 'the row reads the flag');
  assert.match(
    body,
    /!excluded && editable/,
    'drag and the weight stepper take the same guard',
  );
  assert.match(body, /routing-switch/, 'every lane still carries its switch');
  assert.match(
    body,
    /_toggleLane\(input\.dataset\.lane\)/,
    'the switch writes the provider entry',
  );
});

test('a reorder names only the lanes the route carries', () => {
  const body = bodyOf(ROUTING, 'async function _reorderLane(');
  assert.match(body, /filter\(\(entry\) => !entry\.excluded\)/);
});

test('the switch writes through the provider toggle endpoint', () => {
  const body = bodyOf(ROUTING, 'async function _toggleLane(');
  assert.match(body, /\/toggle/);
  assert.match(body, /loadRouting\(_model\)/, 'the board repaints after the write');
});

test('a toggled lane keeps the row it was clicked on', () => {
  const helper = bodyOf(ROUTING, 'function _withPinnedLane(');
  assert.match(helper, /_pinnedLane/, 'the helper reads the pin');
  const board = bodyOf(ROUTING, 'function _renderBoard(');
  assert.match(
    board,
    /_withPinnedLane\(_data\.providers\)/,
    'the board renders the pinned order, not the payload order',
  );
  const toggle = bodyOf(ROUTING, 'async function _toggleLane(');
  const read = toggle.indexOf('findIndex');
  const request = toggle.indexOf('await api.post(');
  assert.notEqual(read, -1, 'the row index is read');
  assert.notEqual(request, -1, 'the switch still writes');
  assert.ok(read < request, 'the index is read before the write, not after');
  assert.match(
    toggle,
    /_pinnedLane = index === -1 \? null : \{ name, index \}/,
    'a successful flip pins that row',
  );
});

test('the pin is dropped when the tab opens or the model changes', () => {
  const body = bodyOf(ROUTING, 'export async function loadRouting(');
  assert.match(
    body,
    /if \(!model \|\| asked !== _model\) _pinnedLane = null;/,
    'a fresh visit or another model re-sorts the board',
  );
});

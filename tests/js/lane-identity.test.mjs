/**
 * A routing lane is (provider, upstream_model), and the Models page must say so.
 *
 * A model may route through the same provider twice, once per upstream model
 * (#1430).  Every control on a lane row — switch, weight, capabilities, remove
 * — used to send the provider name alone, which either hit the wrong lane or
 * was refused once the endpoints learned to tell them apart.  These cases pin
 * the shared key helpers and the call sites that must pass the second half.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  entryHasLane,
  findLane,
  laneBody,
  laneKey,
  laneLabel,
  laneOf,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/lanes.js';
import { laneWeightInputHtml } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/components/provider-row.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const MODELS = readFileSync(join(ADMIN_JS, 'tabs/models.js'), 'utf8');

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

const DETAILS = [
  { name: 'p', upstream_model: 'fast' },
  { name: 'p', upstream_model: 'eco' },
  { name: 'q' },
];

// ── the identity ──

test('two lanes of one provider are two keys', () => {
  assert.notEqual(laneKey('p', 'fast'), laneKey('p', 'eco'));
  assert.equal(laneKey('p', 'fast'), laneKey('p', 'fast'));
});

test('an absent upstream is a lane, not a wildcard', () => {
  const absent = laneKey('q', undefined);
  assert.equal(absent, laneKey('q', null));
  assert.equal(absent, laneKey('q', ''));
  assert.notEqual(absent, laneKey('q', 'fast'), 'the named lane is a different lane');
});

test('laneOf normalises a detail row', () => {
  assert.deepEqual(laneOf({ name: 'p', upstream_model: 'fast' }), {
    name: 'p',
    upstream: 'fast',
  });
  assert.deepEqual(laneOf({ name: 'q' }), { name: 'q', upstream: '' });
  assert.deepEqual(laneOf(null), { name: '', upstream: '' });
});

test('findLane returns the row the pair names, or null', () => {
  assert.equal(findLane(DETAILS, 'p', 'eco').upstream_model, 'eco');
  assert.equal(findLane(DETAILS, 'q', '').name, 'q');
  assert.equal(findLane(DETAILS, 'p', 'turbo'), null, 'no such lane is not the first one');
  assert.equal(findLane(undefined, 'p', 'fast'), null);
});

test('the label shows the upstream only when the lane declares one', () => {
  assert.equal(laneLabel('p', 'fast'), 'p → fast');
  assert.equal(laneLabel('q', ''), 'q');
});

// ── the request body ──

test('laneBody names the lane and keeps the fields it was given', () => {
  assert.deepEqual(laneBody('p', 'fast', { weight: 3 }), {
    provider: 'p',
    upstream_model: 'fast',
    weight: 3,
  });
});

test('laneBody says "declares none" instead of saying nothing', () => {
  // An absent field is "not specified", which the endpoints refuse with a 409
  // when the provider appears twice; '' is the lane that declares no upstream.
  assert.deepEqual(laneBody('q', '', { enabled: true }), {
    provider: 'q',
    upstream_model: '',
    enabled: true,
  });
});

// ── the call sites ──

test('every control of the Models lane row hands over the upstream', () => {
  const body = bodyOf(MODELS, 'function _providerMenuRows(');
  assert.match(body, /const upstream = p\.upstream_model \|\| ''/);
  assert.ok(
    body.includes(
      'toggleModelProvider(${jsArg(name)},${jsArg(p.name)},${jsArg(upstream)})',
    ),
    'the switch names the lane',
  );
  assert.match(body, /laneWeightInputHtml\(\{[^}]*upstream,/s);
  assert.ok(
    body.includes(
      'removeProviderFromModel(${jsArg(name)},${jsArg(p.name)},${jsArg(upstream)})',
    ),
    'the remove button names the lane',
  );
});

test('the capability picker names the lane too', () => {
  const body = bodyOf(MODELS, 'function _capPicker(');
  assert.match(body, /jsArg\(provider\.upstream_model \|\| ''\)/);
});

test('the handlers look the lane up instead of the first row of that name', () => {
  for (const signature of [
    'async function toggleModelProvider(',
    'async function updateProviderWeight(',
    'async function removeProviderFromModel(',
  ]) {
    const body = bodyOf(MODELS, signature);
    assert.match(body, /findLane\(|laneBody\(|laneLabel\(/, `${signature} uses the lane`);
    assert.doesNotMatch(
      body,
      /\{provider: providerName/,
      `${signature} cannot send the name alone`,
    );
  }
});

test('the weight field passes the upstream through the shared builder', () => {
  const html = laneWeightInputHtml({
    model: 'm1',
    provider: 'p',
    upstream: 'fast',
    weight: 2,
    title: 'Weight',
  });
  assert.match(
    html,
    /onchange="updateProviderWeight\(&quot;m1&quot;,&quot;p&quot;,&quot;fast&quot;,this\)"/,
  );
});

test('the drawer can add a lane without preselecting a provider', () => {
  const body = bodyOf(MODELS, 'function addModelLane(');
  assert.match(
    body,
    /openModelModal\(name, '', caps, '', undefined, modelType, \{ addLane: true \}\)/,
    'an edit of the first provider would read as the way to add a lane',
  );
  const detail = bodyOf(MODELS, 'function _modelDetailHtml(');
  assert.match(detail, /_addLaneHtml\(name\)/, 'the drawer offers the entry point');
  assert.match(
    detail,
    /_fallbackRowHtml\(name, info\)/,
    'the value in effect is shown, not only the lanes',
  );
});

test('the entry-level fallback is reported, not edited in place', () => {
  const body = bodyOf(MODELS, 'function _fallbackRowHtml(');
  assert.doesNotMatch(body, /onchange=/, 'a lane\'s upstream belongs on the lane');
  assert.match(body, /clearModelUpstreamFallback\(/, 'deleting it is its own act');
});

test('clearing the fallback is an explicit request with its own flag', () => {
  const body = bodyOf(MODELS, 'async function clearModelUpstreamFallback(');
  assert.match(body, /clear_upstream_fallback: true/);
  assert.match(body, /confirm\(/, 'deleting changes what those lanes request');
  assert.doesNotMatch(
    body,
    /upstream_model:/,
    'the clear must not also post a lane upstream',
  );
});

test('rows are not draggable when one provider holds two lanes', () => {
  const rows = bodyOf(MODELS, 'function _providerMenuRows(');
  assert.match(
    rows,
    /new Set\(details\.map\(\(d\) => d\.name\)\)\.size === details\.length/,
    'the order endpoint addresses rows by name, which is not unique here',
  );
  const reorder = bodyOf(MODELS, 'async function _reorderModelProviders(');
  assert.match(
    reorder,
    /new Set\(names\)\.size !== names\.length/,
    'a duplicated name would make the order say two things at once',
  );
});

test('the lane switch reads the lane flag, not the provider flag', () => {
  // A lane closed for this model and a provider closed for every model are
  // two different switches.  The row shows the lane's own state, says the
  // provider's beside the name, and only the lane control appears here.
  const rows = bodyOf(MODELS, 'function _providerMenuRows(');
  assert.match(rows, /const laneOff = p\.enabled === false/);
  assert.match(rows, /const accountOff = disabledProviders\.has\(p\.name\)/);
  assert.match(rows, /on: !laneOff/, 'the switch shows the lane, not the account');
  assert.match(rows, /t\('model\.laneAccountOff'\)/, 'the account off is stated');
  assert.doesNotMatch(
    rows,
    /toggleModel\(/,
    'the provider-wide switch lives on the Providers page',
  );
});

test('the drawer closes and deletes the model from its own actions', () => {
  const detail = bodyOf(MODELS, 'function _modelDetailHtml(');
  assert.match(detail, /toggleModel\(\$\{jsArg\(name\)\}\)/, 'the enable/disable action');
  assert.match(detail, /btn\.disable/, 'a model that is on offers Disable');
  assert.match(detail, /btn\.enable/, 'a model that is off offers Enable');
  assert.match(detail, /deleteModel\(\$\{jsArg\(name\)\}/, 'delete stays the model delete');
});

test('a one-element lane list keeps its own switch in the drawer', () => {
  // `providers: [x]` carries a lane flag; `provider: x` does not, and the
  // model switch writes the entry flag instead.  Offering the lane switch for
  // the shorthand shape would write a key the entry does not have a place for.
  const body = bodyOf(MODELS, 'function _detailProviderRows(');
  assert.match(body, /const lanes = info && Array\.isArray\(info\.providers\)/);
  assert.match(body, /const switchHtml = lanes\s*\n?\s*\? laneSwitchHtml\(/);
  assert.match(body, /toggleModelProvider\(/, 'the switch names the lane');
  assert.match(body, /model\.laneOff/, 'a lane that is off says so');
});

// ── the save that says "saved" ──

test('entryHasLane reads both entry shapes and the lane with no upstream', () => {
  const list = {
    providers: [{ name: 'p', upstream_model: 'fast' }, { name: 'q' }],
  };
  assert.equal(entryHasLane(list, 'p', 'fast'), true);
  assert.equal(entryHasLane(list, 'q', undefined), true, 'the absent upstream is a lane');
  assert.equal(entryHasLane(list, 'p', 'eco'), false);
  assert.equal(entryHasLane(list, 'r'), false, 'a dropped provider is not found');

  const short = { provider: 'p', upstream_model: 'fast' };
  assert.equal(entryHasLane(short, 'p', 'fast'), true);
  assert.equal(entryHasLane(short, 'p'), false);
  assert.equal(entryHasLane('p', 'p', ''), true, 'the legacy string entry');

  assert.equal(entryHasLane(undefined, 'p', 'fast'), false, 'no entry says nothing');
});

test('the lane add is sent as an intent and checked against the stored entry', () => {
  const body = bodyOf(MODELS, 'async function _saveModel(');
  assert.match(body, /if \(addingLane\) body\.add_lane = true;/, 'the request says so');
  assert.match(
    body,
    /entryHasLane\(res\.entry, provider, body\.upstream_model\)/,
    'the toast describes what the server kept, not what was sent',
  );
  assert.match(
    body,
    /t\('toast\.laneNotSaved', \{lane: laneLabel\(provider, body\.upstream_model\)\}\)/,
    'a dropped lane is reported',
  );
  assert.match(
    bodyOf(MODELS, 'function openModelModal('),
    /dataset\.addLane = addLane \? '1' : ''/,
    'every other entry point clears the intent',
  );
});

test('the response that dropped the lane is the one the editor now refuses', () => {
  // The measured answer of the bug: `ok` with the lane absent from the entry.
  const dropped = {
    ok: true,
    model: 'deepseek-v4.1-flash',
    provider: 'other',
    entry: {
      providers: [{ name: 'deepseek', upstream_model: 'deepseek-v4.1-flash' }],
      capabilities: ['text', 'tools'],
    },
  };
  assert.equal(entryHasLane(dropped.entry, 'other', undefined), false);

  // The same save once the endpoint keeps the lane it was sent.
  const kept = {
    ...dropped,
    merged: true,
    entry: {
      ...dropped.entry,
      providers: [...dropped.entry.providers, { name: 'other' }],
    },
  };
  assert.equal(entryHasLane(kept.entry, 'other', undefined), true);
});

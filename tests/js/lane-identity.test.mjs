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
    /openModelModal\(name, '', caps, '', undefined, modelType\)/,
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

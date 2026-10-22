/**
 * The router's verdict on a provider, as the card renders it.
 *
 * The precedence is the contract: a provider can have a healthy reading and
 * still be out of rotation, and a row claiming a share on a skipped provider
 * tells the operator the opposite of what the router is doing.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  blockRow,
  routingRows,
  routingWeight,
  sharePercent,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/quota-lines.js';

/** A translator that records what it was asked for. */
function translator() {
  const calls = [];
  const t = (key, vars) => {
    calls.push([key, vars]);
    return vars ? `${key}:${JSON.stringify(vars)}` : key;
  };
  return { t, calls };
}

test('a live block wins: no share row next to it', () => {
  const { t } = translator();
  const rows = routingRows({ blocked: true, excluded: false, headroom: 0.9 }, t);
  assert.deepEqual(rows, []);
});

test('excluded is failsafe and carries a translated reason', () => {
  const { t, calls } = translator();
  const rows = routingRows({ blocked: false, excluded: true, headroom: 0, reason: 'spent' }, t);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].tone, 'fail');
  assert.ok(calls.some(([key]) => key === 'quota.reason.spent'));
  assert.ok(calls.some(([key]) => key === 'quota.routingExcluded'));
});

test('an excluded provider without a reason still renders one row', () => {
  const { t } = translator();
  const rows = routingRows({ blocked: false, excluded: true, headroom: 0 }, t);
  assert.equal(rows.length, 1);
  assert.match(rows[0].text, /^—|—/);
});

test('a measured provider reports the weight it serves at', () => {
  const { t } = translator();
  const rows = routingRows({ blocked: false, excluded: false, headroom: 0.62 }, t);
  assert.equal(rows[0].tone, 'ok');
  // 0.25 + 0.75 * 0.62 = 0.715 -> 72
  assert.match(rows[0].text, /"share":72/);
  assert.match(rows[0].text, /"pct":62/);
});

test('an unmeasured provider keeps its weight, and says so', () => {
  const { t, calls } = translator();
  const rows = routingRows({ blocked: false, excluded: false, headroom: null }, t);
  assert.equal(rows[0].tone, 'warn');
  assert.ok(calls.some(([key]) => key === 'quota.routingUnmeasured'));
});

test('no routing block means no rows', () => {
  const { t } = translator();
  assert.deepEqual(routingRows(undefined, t), []);
  assert.deepEqual(routingRows(null, t), []);
});

test('a spent share still shows the quarter the router applies', () => {
  assert.equal(routingWeight(0), 0.25);
  assert.equal(sharePercent(routingWeight(0)), 25);
  // The credential share itself is 0, and the row already says "25% weight".
  assert.equal(sharePercent(0), 0);
});

test('the server weight wins over a local derivation', () => {
  const { t } = translator();
  // A payload whose weight disagrees with the local formula must render the
  // server's number: the rule lives in one place.
  const rows = routingRows({ blocked: false, excluded: false, headroom: 0.5, weight: 0.1 }, t);
  assert.match(rows[0].text, /"share":10/);
  const fallback = routingRows({ blocked: false, excluded: false, headroom: 0.5 }, t);
  assert.match(fallback[0].text, /"share":63/);
});

test('a gateway-owned block reason is translated, a vendor one passes through', () => {
  const { t, calls } = translator();
  const keyless = blockRow({ until: '2026-10-23T00:01:02Z', reason: 'keyless lane rate limited' }, t);
  assert.ok(calls.some(([key]) => key === 'quota.blockReason.keyless'));
  const vendor = blockRow({ until: '2026-10-23T00:01:02Z', reason: 'usage limit exceeded' }, t);
  assert.match(vendor.text, /usage limit exceeded/);
  assert.ok(keyless.text.startsWith('quota.blocked:'));
});

test('a nearly full window is not rounded up to a hundred', () => {
  assert.equal(sharePercent(0.996), 99);
  assert.equal(sharePercent(1), 100);
  assert.equal(sharePercent(routingWeight(1)), 100);
});

test('a block row carries the cooldown, or nothing', () => {
  const { t, calls } = translator();
  assert.equal(blockRow(null, t), null);
  const row = blockRow({ until: '2026-10-23T00:01:02Z', reason: 'usage limit' }, t);
  assert.equal(row.tone, 'warn');
  assert.ok(calls.some(([key]) => key === 'quota.blocked'));
  assert.match(row.text, /usage limit/);
});

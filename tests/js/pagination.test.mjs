/**
 * Page-jump math shared by the Request Log, Server Ops Log and Error Dumps
 * pagers.
 *
 * `pagination.js` is deliberately DOM-free so it can run under node.  The two
 * contracts worth pinning are the ones a jump box gets wrong silently: an
 * out-of-range page clamps instead of producing a negative offset, and a box
 * that holds no number leaves the current page alone rather than jumping to 1.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { pageCount, clampPage, offsetForPage, shouldResync } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/pagination.js';

test('pageCount spans the rows and never drops below one page', () => {
  assert.equal(pageCount(0, 30), 1);
  assert.equal(pageCount(1, 30), 1);
  assert.equal(pageCount(30, 30), 1);
  assert.equal(pageCount(31, 30), 2);
  assert.equal(pageCount(268275, 30), 8943);
  assert.equal(pageCount(41, 20), 3);
});

test('pageCount tolerates a missing or nonsense limit', () => {
  assert.equal(pageCount(100, 0), 1);
  assert.equal(pageCount(100, undefined), 1);
  assert.equal(pageCount(undefined, 30), 1);
  assert.equal(pageCount('120', '30'), 4);
});

test('clampPage keeps a typed page inside the list', () => {
  assert.equal(clampPage('1', 25), 1);
  assert.equal(clampPage('7', 25), 7);
  assert.equal(clampPage('25', 25), 25);
  assert.equal(clampPage('26', 25), 25);
  assert.equal(clampPage('0', 25), 1);
  assert.equal(clampPage('-4', 25), 1);
});

test('clampPage returns null for a box holding no page number', () => {
  assert.equal(clampPage('', 25), null);
  assert.equal(clampPage('   ', 25), null);
  assert.equal(clampPage('abc', 25), null);
  assert.equal(clampPage(null, 25), null);
  assert.equal(clampPage(undefined, 25), null);
  assert.equal(clampPage('2.9', 25), 2);
});

test('clampPage with a single page pins to that page', () => {
  assert.equal(clampPage('1', 1), 1);
  assert.equal(clampPage('9', 1), 1);
  assert.equal(clampPage('9', 0), 1);
});

test('offsetForPage converts a 1-based page to the endpoint offset', () => {
  assert.equal(offsetForPage(1, 30), 0);
  assert.equal(offsetForPage(2, 30), 30);
  assert.equal(offsetForPage(10, 20), 180);
  assert.equal(offsetForPage(0, 30), 0);
});

test('offsetForPage never yields NaN from an unusable page size', () => {
  for (const bad of [undefined, null, 0, -30, NaN, 'many']) {
    const offset = offsetForPage(3, bad);
    assert.ok(Number.isFinite(offset), `limit=${String(bad)} gave ${offset}`);
    assert.equal(offset, 0);
  }
  assert.equal(offsetForPage('2', '30'), 30);
});

test('shouldResync leaves a box the user has edited alone', () => {
  // Mid-edit, whatever the value looks like: this is the window that made the
  // first keystroke of "12" on page 1 vulnerable to a refresh.
  assert.equal(shouldResync('1', 1, { focused: true, dirty: true }), false);
  assert.equal(shouldResync('1', 1, { focused: false, dirty: true }), false);
  assert.equal(shouldResync('3', 1, { focused: true, dirty: true }), false);
});

test('shouldResync follows the page for a box the user has not touched', () => {
  assert.equal(shouldResync('1', 1, {}), true);
  assert.equal(shouldResync('3', 1, { focused: false }), true);
  // Focused but still holding the page it was synced to: the page moved under
  // the user (a filter or a rows-per-page change), so the box follows.
  assert.equal(shouldResync('1', 1, { focused: true }), true);
});

test('shouldResync holds a diverged value even without the dirty flag', () => {
  assert.equal(shouldResync('12', 1, { focused: true }), false);
  assert.equal(shouldResync('12', 1, { focused: false }), true);
});

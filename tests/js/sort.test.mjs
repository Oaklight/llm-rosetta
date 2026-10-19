/**
 * Sortable-header support: `aria-sort` bookkeeping and keyboard wiring.
 *
 * `sort.js` is deliberately import-free so it can run under node, which has no
 * DOM.  The two contracts pinned here are the ones that silently regress: the
 * attribute is *removed* from inactive columns (so it cannot accumulate on a
 * header that was sorted once), and wiring twice does not attach a second key
 * handler.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { syncAriaSort, wireSortableHeaders } from '../../src/llm_rosetta/gateway/admin/js/core/sort.js';

function makeTh() {
  const attrs = new Map();
  const listeners = new Map();
  let clicks = 0;
  return {
    dataset: {},
    tabIndex: -1,
    setAttribute(k, v) { attrs.set(k, v); },
    removeAttribute(k) { attrs.delete(k); },
    getAttribute(k) { return attrs.has(k) ? attrs.get(k) : null; },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    listenerCount(type) { return (listeners.get(type) || []).length; },
    fire(type, event) { for (const fn of listeners.get(type) || []) fn(event); },
    click() { clicks += 1; },
    clicks() { return clicks; },
  };
}

function keyEvent(key) {
  return {
    key,
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
}

test('syncAriaSort puts the state on the active header only', () => {
  const name = makeTh();
  const provider = makeTh();
  const headers = [[name, 'name'], [provider, 'provider']];

  syncAriaSort(headers, 'name', 'asc');
  assert.equal(name.getAttribute('aria-sort'), 'ascending');
  assert.equal(provider.getAttribute('aria-sort'), null);

  // Flipping the direction updates the active header...
  syncAriaSort(headers, 'name', 'desc');
  assert.equal(name.getAttribute('aria-sort'), 'descending');

  // ...and moving the sort removes the attribute from the old one rather than
  // leaving a stale value behind.
  syncAriaSort(headers, 'provider', 'asc');
  assert.equal(name.getAttribute('aria-sort'), null);
  assert.equal(provider.getAttribute('aria-sort'), 'ascending');
});

test('syncAriaSort tolerates a missing header element', () => {
  const total = makeTh();
  syncAriaSort([[null, 'label'], [total, 'total']], 'total', 'desc');
  assert.equal(total.getAttribute('aria-sort'), 'descending');
});

test('wireSortableHeaders makes a header a tab stop with one key handler', () => {
  const th = makeTh();
  const root = { querySelectorAll: () => [th] };

  wireSortableHeaders(root);
  assert.equal(th.tabIndex, 0);
  assert.equal(th.dataset.kbdSort, '1');

  // Calling it again (a renderer re-wiring the same subtree) must not stack a
  // second handler, or one Enter press would sort twice.
  wireSortableHeaders(root);
  assert.equal(th.listenerCount('keydown'), 1);
});

test('Enter and Space run the click handler; Space does not scroll', () => {
  const th = makeTh();
  wireSortableHeaders({ querySelectorAll: () => [th] });

  const enter = keyEvent('Enter');
  th.fire('keydown', enter);
  assert.equal(th.clicks(), 1);
  assert.equal(enter.prevented, true);

  const space = keyEvent(' ');
  th.fire('keydown', space);
  assert.equal(th.clicks(), 2);
  assert.equal(space.prevented, true);

  // Every other key passes through untouched.
  const other = keyEvent('ArrowDown');
  th.fire('keydown', other);
  assert.equal(th.clicks(), 2);
  assert.equal(other.prevented, false);
});

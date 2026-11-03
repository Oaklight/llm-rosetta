/**
 * The wiring around the provider list's pinned order.
 *
 * `orderedProviders()` is tested on its own; this file holds the lines that
 * connect it to the page, which no DOM-free test would notice.  A review of
 * the fix mutated each of them and the suite stayed green: dropping the pin in
 * `toggleProvider`, pinning nothing (the pre-fix behaviour), keeping the pin
 * through a new search, and dropping the clear on tab entry.
 *
 * Text-level on purpose, like `rail-wiring`: the call order is the contract,
 * and reading the source is enough to hold it without a DOM.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const PROVIDERS = readFileSync(join(ADMIN_JS, 'tabs/providers.js'), 'utf8');
const INIT = readFileSync(join(ADMIN_JS, 'core/init.js'), 'utf8');

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

/** `activateTab`, up to the settings-group section that follows it. */
function activateTabBody(text) {
  const start = text.indexOf('function activateTab(');
  assert.notEqual(start, -1, 'activateTab is still a function');
  const end = text.indexOf('// ---- Settings groups ----', start);
  assert.notEqual(end, -1, 'activateTab still ends before the settings groups');
  return text.slice(start, end);
}

test('a toggle pins the order it was clicked on, before the request', () => {
  const body = bodyOf(PROVIDERS, 'async function toggleProvider(');
  const capture = body.indexOf('const order = _currentProviderOrder()');
  const request = body.indexOf('await api.post(');
  assert.notEqual(capture, -1, 'toggleProvider reads the on-screen order');
  assert.notEqual(request, -1, 'toggleProvider still calls the toggle API');
  assert.ok(
    capture < request,
    'the order is read at click time, not when the response lands',
  );

  const pin = body.indexOf('_pinnedOrder = order');
  const reload = body.indexOf('loadConfig()');
  assert.notEqual(pin, -1, 'a successful toggle pins the captured order');
  assert.notEqual(reload, -1, 'a successful toggle reloads the config');
  assert.ok(pin < reload, 'the pin is set before the list re-renders');
});

test('the render hands the pinned order to the sort', () => {
  const body = bodyOf(PROVIDERS, 'function renderProviders(');
  assert.match(
    body,
    /orderedProviders\(\s*entries,[\s\S]*?_pinnedOrder/,
    'renderProviders sorts through the pin',
  );
});

test('a new query, a filter or a status change clears the pin', () => {
  for (const signature of [
    'function onProviderSearch(',
    'function switchProviderFilter(',
    'function setProviderStatus(',
    'function goToProviderFromModel(',
  ]) {
    const body = bodyOf(PROVIDERS, signature);
    assert.match(body, /_pinnedOrder = null/, `${signature} clears the pin`);
  }
});

test('entering the providers tab clears the pin before the reload', () => {
  const body = activateTabBody(INIT);
  const reset = body.indexOf('window.resetProviderListOrder?.()');
  const reload = body.indexOf('loadConfig()');
  assert.notEqual(reset, -1, 'activateTab drops the provider-list pin');
  assert.notEqual(reload, -1, 'activateTab reloads the config');
  assert.ok(reset < reload, 'the tab entry sorts afresh, not the previous pin');
});

test('a view switch keeps the pin', () => {
  // The two views show the same rows in another layout; re-sorting while
  // switching would move the row under the pointer in the other direction.
  const body = bodyOf(PROVIDERS, 'function setProviderView(');
  assert.match(body, /renderProviders\(\)/);
  assert.doesNotMatch(body, /_pinnedOrder/);
});

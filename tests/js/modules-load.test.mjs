/**
 * Every admin module parses and evaluates.
 *
 * The suite imports modules one at a time, and each test names the file it
 * exercises, so nothing ever imported `tabs/providers.js` — and a committed
 * merge marker in a 1,400-line tab passed 330 green tests while the console
 * refused to boot.  This walks the tree instead of a list, so a new file is
 * covered the moment it exists.
 *
 * `core/state.js` reads `localStorage` at module scope and the tabs reach for
 * `document`, so the page's minimum environment is stubbed first.  That is a
 * test harness, not a claim that these modules are DOM-free.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const adminRoot = join(
  here,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);

/** Every `.js` module under `admin/js`, tests and fixtures excluded. */
function moduleFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...moduleFiles(path));
    else if (entry.name.endsWith('.js')) out.push(path);
  }
  return out;
}

function stubDom() {
  const noop = () => {};
  const element = {
    style: {},
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    dataset: {},
    appendChild: noop,
    addEventListener: noop,
    removeEventListener: noop,
    setAttribute: noop,
    getAttribute: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    focus: noop,
    click: noop,
    closest: () => null,
    contains: () => false,
    matches: () => false,
    getBoundingClientRect: () => ({ top: 0, left: 0, width: 0, height: 0, bottom: 0, right: 0 }),
    insertAdjacentHTML: noop,
    remove: noop,
  };
  const storage = () => {
    const map = new Map();
    return {
      getItem: (k) => (map.has(k) ? map.get(k) : null),
      setItem: (k, v) => map.set(k, String(v)),
      removeItem: (k) => map.delete(k),
      clear: () => map.clear(),
    };
  };
  globalThis.localStorage = storage();
  globalThis.sessionStorage = storage();
  globalThis.document = {
    documentElement: element,
    body: element,
    head: element,
    getElementById: () => element,
    querySelector: () => element,
    querySelectorAll: () => [],
    createElement: () => element,
    createTextNode: () => element,
    addEventListener: noop,
    removeEventListener: noop,
  };
  globalThis.window = globalThis;
  globalThis.location = { hash: '', pathname: '/', search: '', href: 'http://localhost/' };
  globalThis.history = { replaceState: noop, pushState: noop };
  globalThis.addEventListener = noop;
  globalThis.removeEventListener = noop;
  for (const name of ['MutationObserver', 'ResizeObserver', 'IntersectionObserver']) {
    globalThis[name] = class {
      observe() {}
      unobserve() {}
      disconnect() {}
      takeRecords() { return []; }
    };
  }
  globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
  globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
  globalThis.matchMedia = () => ({
    matches: false,
    addEventListener: noop,
    removeEventListener: noop,
  });
  globalThis.fetch = async () => ({ ok: true, json: async () => ({}), text: async () => '' });
  // `core/init.js` starts a clock at module scope.  Let the timer exist but
  // not hold the process open once the assertions are done.
  const timer = (fn) => (...args) => {
    const handle = fn(...args);
    handle?.unref?.();
    return handle;
  };
  globalThis.setInterval = timer(setInterval);
  globalThis.setTimeout = timer(setTimeout);
}

test('the tree holds modules, so a broken walk cannot pass silently', () => {
  assert.ok(moduleFiles(adminRoot).length > 30, 'expected dozens of admin modules');
});

test('every admin module parses and evaluates', async () => {
  stubDom();
  const failures = [];
  for (const path of moduleFiles(adminRoot)) {
    const rel = relative(adminRoot, path);
    // `node --check` first: a syntax error otherwise surfaces as an import
    // rejection that the DOM stubs can mask under a misleading stack.
    try {
      new Function(readFileSync(path, 'utf8'));
    } catch (err) {
      // ESM syntax (`import`/`export`) is not valid inside Function; only a
      // parse error in a module without those statements is a real finding.
      if (!/import|export/.test(String(err.message))) {
        failures.push(`${rel}: ${err.message}`);
        continue;
      }
    }
    try {
      await import(pathToFileURL(path).href);
    } catch (err) {
      failures.push(`${rel}: ${err.message}`);
    }
  }
  assert.deepEqual(failures, [], failures.join('\n'));
});

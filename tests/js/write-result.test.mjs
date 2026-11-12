/**
 * Every mutating admin request resolves to one shape.
 *
 * `api.put` / `api.post` / `api.del` used to return `r.json()` for every
 * status, so a refusal was visible only when the server happened to answer
 * with JSON *and* the call site remembered to check `ok`.  A reverse proxy's
 * 502 HTML page and a dropped connection reached nobody: the promise rejected
 * and the dialog sat unchanged (#1502 §1, both measured in a browser).
 *
 * These tests pin the four outcomes a call site can now branch on, and keep
 * the three helpers from growing a second code path again.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { writeResult, networkResult, httpError } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/write-result.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const read = (name) => readFileSync(join(JS, name), 'utf8');

/** A `Response` stand-in: status, reason phrase and a body (or a body that fails). */
function response(status, { body, statusText = '', throws = false } = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: async () => {
      if (throws) throw new SyntaxError('Unexpected token <');
      return body;
    },
  };
}

test('a success carries ok and the server fields', async () => {
  const res = await writeResult(response(200, { body: { ok: true, key: { key: 'rsk-x' } } }));
  assert.equal(res.ok, true);
  assert.equal(res.key.key, 'rsk-x');
});

test('a success without ok gets it, so one `if (res.ok)` works everywhere', async () => {
  const res = await writeResult(response(200, { body: { task_id: 't1' } }));
  assert.equal(res.ok, true);
  assert.equal(res.task_id, 't1');
});

test('a 204 with no body is a success', async () => {
  const res = await writeResult(response(204, { body: null }));
  assert.deepEqual(res, { ok: true });
});

test('a refusal keeps the server error and its other fields', async () => {
  const res = await writeResult(response(400, {
    body: { error: 'name already taken', merge_possible: true, existing_provider: 'argo' },
  }));
  assert.equal(res.ok, false);
  assert.equal(res.status, 400);
  assert.equal(res.error, 'name already taken');
  assert.equal(res.merge_possible, true);
  assert.equal(res.existing_provider, 'argo');
});

test('a refusal with no error text still says something readable', async () => {
  const res = await writeResult(response(400, { body: { changed: 0 }, statusText: 'Bad Request' }));
  assert.equal(res.ok, false);
  assert.equal(res.error, 'HTTP 400 Bad Request');
  assert.equal(res.changed, 0);
});

test('a 502 HTML page is a refusal, not an exception', async () => {
  const res = await writeResult(response(502, { statusText: 'Bad Gateway', throws: true }));
  assert.deepEqual(res, { ok: false, status: 502, error: 'HTTP 502 Bad Gateway' });
});

test('a success with an unreadable body is still a success', async () => {
  const res = await writeResult(response(200, { throws: true }));
  assert.deepEqual(res, { ok: true });
});

test('an array body keeps ok without losing the array', async () => {
  const res = await writeResult(response(200, { body: [{ id: 1 }] }));
  assert.equal(res.length, 1);
  assert.equal(res.ok, true);
});

test('a 504 says what it usually means', async () => {
  assert.equal(
    httpError(504, 'Gateway Timeout'),
    'Request timed out — the upstream service may be unreachable',
  );
});

test('a transport failure is a refusal with status 0', async () => {
  assert.deepEqual(networkResult('Error'), { ok: false, status: 0, error: 'Error' });
  assert.equal(networkResult().error, 'Network error');
});

test('the three mutating helpers share one code path', () => {
  const text = read('core/utils.js');
  for (const method of ['PUT', 'DELETE', 'POST']) {
    assert.match(
      text,
      new RegExp(`return _send\\('${method}', url`),
      `${method} goes through _send`,
    );
  }
  const body = text.slice(text.indexOf('async function _send('));
  const fn = body.slice(0, body.indexOf('\n}\n'));
  assert.match(fn, /catch \(e\)/, 'a dropped connection is caught');
  assert.match(fn, /networkResult\(/, 'and becomes a result');
  assert.match(fn, /return writeResult\(r\)/, 'every other answer goes through writeResult');
});

test('the busy guard drops a second click instead of queueing it', () => {
  const text = read('core/utils.js');
  assert.match(text, /if \(btn\.dataset\.busy === '1'\) return undefined;/);
  assert.match(text, /btn\.disabled = true;/);
  assert.match(text, /btn\.disabled = wasDisabled;/);
});

/** The body of a top-level function, from its signature to the first `}` at column 0. */
function functionBody(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

test('a failed shim update check leaves the badges alone', () => {
  const body = functionBody(read('tabs/shims.js'), 'async function checkShimUpdates(');
  const guard = body.indexOf('if (!d || !d.ok) return;');
  assert.notEqual(guard, -1, 'a failed answer returns instead of being read');
  assert.ok(
    guard < body.indexOf('_shimUpdates = new Map('),
    'the guard runs before the map is replaced',
  );
});

test('a manifest save does not report success over a failed reload', () => {
  const body = functionBody(read('tabs/shims.js'), 'async function saveManifestYaml(');
  assert.match(
    body,
    /const reload = await api\.post\('\/admin\/api\/config\/shims\/reload'/,
    'the reload answer is kept',
  );
  assert.match(body, /if \(!reload\.ok\) throw new Error\(reload\.error \|\| t\('toast\.error'\)\);/);
  assert.ok(
    body.indexOf('if (!reload.ok)') < body.indexOf("closeModal('shimModal')"),
    'the check runs before the dialog closes',
  );
});

test('the busy guard puts the button back the way it found it', () => {
  const text = read('core/utils.js');
  assert.match(text, /const wasDisabled = btn\.disabled;/);
  assert.match(text, /btn\.disabled = wasDisabled;/);
});

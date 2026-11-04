/**
 * "Stay signed in" must not be cut short by the panel's own idle timer.
 *
 * The server exempts a remembered session from the idle window, but the panel
 * armed `setTimeout(doLogout, idle_minutes)` for every session, and `doLogout`
 * posts `/admin/api/logout` — so after 30 quiet minutes the client destroyed
 * the session the cookie promised to keep (issue #1381).  Text-level, like
 * `provider-list-wiring`: the guard and its two feeds are the contract.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);

const read = (name) => readFileSync(join(JS, name), 'utf8');

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

test('a remembered session arms no idle timer', () => {
  const body = bodyOf(read('core/utils.js'), 'function _idleTimeoutMs(');
  const guard = body.indexOf('if (S.rememberedSession) return 0;');
  const idle = body.indexOf('S.idleMinutes');
  assert.notEqual(guard, -1, 'the timer reads the remembered flag');
  assert.ok(guard < idle, 'the guard answers before the idle window is computed');
});

test('auth-check feeds the flag from the server, not from the checkbox', () => {
  const body = bodyOf(read('core/auth.js'), 'async function checkAuthAndInit(');
  assert.match(body, /S\.rememberedSession = !!authData\.remembered/);
});

test('a successful login records what the server actually granted', () => {
  const body = bodyOf(read('core/auth.js'), 'async function doLogin(');
  const set = body.indexOf('S.rememberedSession = !!(data && data.remember)');
  const arm = body.indexOf('_startInactivityTracking()');
  assert.notEqual(set, -1, 'the login response decides the flag');
  assert.notEqual(arm, -1, 'login still arms the timer path');
  assert.ok(set < arm, 'the flag is set before the timer is armed');
});

test('the flag has a default so a page that never loads auth-check is safe', () => {
  assert.match(read('core/state.js'), /rememberedSession:\s*false/);
});

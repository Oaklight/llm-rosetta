import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  availableCredentialSources, credentialFallbackSource, credentialSource,
  credentialUiSource, hostLoginActive, hostLoginLines, hostLoginState,
  hostLoginUrl,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/credential.js';

const ADMIN = join(
  dirname(fileURLToPath(import.meta.url)),
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);

// What the payload says about the antigravity--acp shim once it ships auth.py.
const AGY = {
  name: 'antigravity--acp', mints_credential: true, credential_kind: 'agent',
  host_login: true, keyless: false,
};
const PLAN = {
  name: 'openai_chatgpt--open_responses', mints_credential: true,
  credential_kind: 'oauth', host_login: false, keyless: false,
};
const FREE = { name: 'free--chat_completions', mints_credential: false, keyless: true };

test('the agy shim offers the sign-in card and nothing else', () => {
  const avail = availableCredentialSources(AGY, null);
  assert.deepEqual(avail, { static: false, config: false, shim: false, exchange: true });
  assert.equal(credentialUiSource(credentialSource(AGY, null)), 'exchange');
});

test('there is no static key and no override for it, so a stale choice falls to sign-in', () => {
  const avail = availableCredentialSources(AGY, null);
  assert.equal(credentialFallbackSource(avail, 'static'), 'exchange');
  assert.equal(credentialFallbackSource(avail, 'config'), 'exchange');
});

test('another sign-in shim still offers the command override', () => {
  assert.equal(availableCredentialSources(PLAN, null).config, true);
});

test('a keyless shim without a sign-in keeps the static card for a paid key', () => {
  const avail = availableCredentialSources(FREE, null);
  assert.equal(avail.static, true);
  assert.equal(avail.exchange, false);
});

test('a stored entry of the agy shim whose shim failed to load still shows sign-in', () => {
  const avail = availableCredentialSources(null, {
    mints_credential: true, credential_source: 'auth_exchange',
  });
  assert.equal(avail.exchange, true);
  assert.equal(avail.static, false);
});

test('only known sign-in states count, and only two are in progress', () => {
  for (const state of ['running', 'verifying']) {
    assert.equal(hostLoginActive({ state }), true, state);
  }
  for (const state of ['idle', 'succeeded', 'failed', 'cancelled', '<script>', undefined]) {
    assert.equal(hostLoginActive({ state }), false, String(state));
  }
  assert.equal(hostLoginState({ state: 'nonsense' }), 'idle');
  assert.equal(hostLoginState(null), 'idle');
});

test('the printed URL is a link only when it is https', () => {
  assert.equal(
    hostLoginUrl({ url: 'https://accounts.example/o/oauth2/auth?state=a' }),
    'https://accounts.example/o/oauth2/auth?state=a',
  );
  for (const url of ['javascript:alert(1)', 'http://x.example/', '', null, undefined, 7]) {
    assert.equal(hostLoginUrl({ url }), null, String(url));
  }
  assert.equal(hostLoginUrl(null), null);
});

test('output lines are strings only, bounded, newest last', () => {
  const lines = Array.from({ length: 30 }, (_, i) => `line ${i}`);
  const shown = hostLoginLines({ lines: [...lines, 7, null, ''] }, 5);
  assert.deepEqual(shown, lines.slice(-5));
  assert.deepEqual(hostLoginLines(null), []);
  assert.deepEqual(hostLoginLines({ lines: 'nope' }), []);
});

test('the panel wires the host sign-in without unescaped interpolation', () => {
  const source = readFileSync(join(ADMIN, 'js/tabs/providers.js'), 'utf8');
  const start = source.indexOf('// ── Host sign-in');
  const end = source.indexOf('/** Whether a refresh or test would run');
  const block = source.slice(start, end);
  assert.ok(start > 0 && end > start);
  // Text reaches the DOM through textContent and href through the https
  // check; nothing from the program's output goes through innerHTML.
  assert.equal(/innerHTML|insertAdjacentHTML|outerHTML/.test(block), false);
  for (const handler of ['beginProviderHostLogin', 'cancelProviderHostLogin']) {
    assert.match(source, new RegExp(`${handler},?\\s`), `${handler} is exported`);
    const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8')
      + source;
    assert.ok(html.includes(handler + '()'), `${handler} is invoked`);
  }
});

test('every new sentence exists in both languages', () => {
  const i18n = readFileSync(join(ADMIN, 'js/core/i18n.js'), 'utf8');
  const keys = [
    'btn.hostLogin', 'btn.hostLoginAgain', 'btn.checkSignIn',
    'prov.hostLoginCancel', 'prov.hostLoginOpen', 'prov.hostLoginDone',
    'prov.hostLoginWhere', 'prov.hostLogin.idle', 'prov.hostLogin.running',
    'prov.hostLogin.verifying', 'prov.hostLogin.succeeded',
    'prov.hostLogin.failed', 'prov.hostLogin.cancelled',
    'prov.sourceHostLogin', 'prov.sourceHostLoginDesc', 'prov.noteHostLogin',
    'label.credentialHintHost', 'label.credentialLastCheck',
    'label.credentialNotChecked', 'label.confirmSignOutHost',
  ];
  for (const key of keys) {
    const count = i18n.split(`'${key}':`).length - 1;
    assert.equal(count, 2, `${key} is defined ${count} times, expected en and zh`);
  }
});

test('the override control is not offered where the command card is refused', () => {
  const source = readFileSync(join(ADMIN, 'js/tabs/providers.js'), 'utf8');
  assert.match(source, /btn\.hidden = alternatives\.length === 0 \|\| \(!avail\.config && ui !== 'config'\)/);
});

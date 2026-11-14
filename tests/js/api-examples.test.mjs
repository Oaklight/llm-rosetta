/**
 * The sample snippets a key page shows.
 *
 * The panel is only as honest as the text it puts on the clipboard: a
 * snippet with the wrong path, the wrong auth header or a body that drifted
 * from the converter sends the operator to a 401 or a 400.  These cases run
 * the builders and read the output, rather than pattern-match the source.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildShell,
  buildSnippets,
  envNameFor,
  DEFAULT_ENV,
  snippetLangs,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/api-examples.js';

const BEARER = { name: 'Authorization', value: 'Bearer {key}' };
const API_KEY = { name: 'x-api-key', value: '{key}' };

const chat = {
  path: '/v1/chat/completions',
  auth: BEARER,
  content_type: 'application/json',
  body: { model: 'gpt-5-nano', messages: [{ role: 'user', content: 'Hello' }] },
};

function curlBody(snippet) {
  // The `-d` argument is one single-quoted shell word.  The JSON inside it is
  // pretty-printed, so the word spans the rest of the snippet.
  const lines = snippet.split('\n');
  const start = lines.findIndex(l => l.trim().startsWith('-d '));
  assert.notEqual(start, -1, 'the snippet passes a body');
  const word = [lines[start].trim().slice(3), ...lines.slice(start + 1)].join('\n');
  assert.equal(word[0], "'");
  assert.equal(word[word.length - 1], "'");
  return JSON.parse(word.slice(1, -1).replace(/'\\''/g, "'"));
}

test('the default env var names no key', () => {
  assert.equal(envNameFor(null), DEFAULT_ENV);
  assert.equal(envNameFor({}), DEFAULT_ENV);
});

test('a selected key names a variable after its label', () => {
  assert.equal(envNameFor({ label: 'ci-runner' }), 'LLM_ROSETTA_KEY_CI_RUNNER');
  assert.equal(envNameFor({ label: 'prod / web' }), 'LLM_ROSETTA_KEY_PROD_WEB');
});

test('a key with no label falls back to its id, then to the default', () => {
  assert.equal(envNameFor({ id: 'ab12cd34' }), 'LLM_ROSETTA_KEY_AB12CD34');
  assert.equal(envNameFor({ id: '' }), DEFAULT_ENV);
});

test('every language reads the same path and env var', () => {
  const s = buildSnippets(chat, { base: 'http://gw.test', env: 'MY_KEY' });
  for (const lang of snippetLangs()) {
    assert.match(s[lang], /http:\/\/gw\.test\/v1\/chat\/completions/, lang);
    assert.match(s[lang], /MY_KEY/, lang);
  }
});

test('the curl body is the example body, byte for byte', () => {
  const s = buildSnippets(chat, { base: 'http://gw.test', env: 'K' });
  assert.deepEqual(curlBody(s.curl), chat.body);
});

test('an apostrophe in the body does not end the shell word early', () => {
  const tricky = { ...chat, body: { model: 'x', messages: [{ role: 'user', content: "it's fine" }] } };
  const s = buildSnippets(tricky, { base: 'http://gw.test', env: 'K' });
  assert.deepEqual(curlBody(s.curl), tricky.body);
});

test('a bearer header keeps its scheme in front of the secret', () => {
  const s = buildSnippets(chat, { base: 'http://gw.test', env: 'K' });
  assert.match(s.curl, /-H "Authorization: Bearer \$K"/);
  assert.match(s.python, /"Authorization": "Bearer " \+ os\.environ\["K"\]/);
  assert.match(s.node, /"Authorization": `Bearer \$\{process\.env\.K\}`/);
});

test('a bare key header has no scheme added', () => {
  const ex = { ...chat, path: '/v1/messages', auth: API_KEY };
  const s = buildSnippets(ex, { base: 'http://gw.test', env: 'K' });
  assert.match(s.curl, /-H "x-api-key: \$K"/);
  assert.match(s.python, /"x-api-key": os\.environ\["K"\]/);
  assert.match(s.node, /"x-api-key": process\.env\.K\b/);
});

test('the python body is parsed from the exact JSON', () => {
  const s = buildSnippets(chat, { base: 'http://gw.test', env: 'K' });
  const m = s.python.match(/json\.loads\('''([\s\S]*?)'''\)/);
  assert.ok(m, 'the payload is a triple-quoted JSON string');
  assert.deepEqual(JSON.parse(m[1]), chat.body);
});

test('the node payload is valid JavaScript for a JSON body', () => {
  const ex = { ...chat, body: { on: true, off: false, none: null, n: 1.5 } };
  const s = buildSnippets(ex, { base: 'http://gw.test', env: 'K' });
  const m = s.node.match(/const payload = ([\s\S]*?);\n\nconst resp/);
  assert.ok(m, 'the payload is a literal');
  // eslint-disable-next-line no-new-func
  const value = new Function(`return (${m[1]});`)();
  assert.deepEqual(value, ex.body);
});


test('the shell sample exports the base URL and the key', () => {
  const sh = buildShell({
    base: 'http://gw.test/',
    basePath: '/v1',
    baseEnv: 'OPENAI_BASE_URL',
    keyEnv: 'OPENAI_API_KEY',
  });
  assert.equal(
    sh,
    'export OPENAI_BASE_URL=http://gw.test/v1\nexport OPENAI_API_KEY=<your gateway key>'
  );
});

test('the shell sample drops the trailing slash and keeps an empty path', () => {
  const sh = buildShell({
    base: 'http://gw.test',
    basePath: '',
    baseEnv: 'ANTHROPIC_BASE_URL',
    keyEnv: 'ANTHROPIC_API_KEY',
  });
  assert.match(sh, /^export ANTHROPIC_BASE_URL=http:\/\/gw\.test\n/);
});

test('the windows shell sample uses the PowerShell form', () => {
  const sh = buildShell({
    base: 'http://gw.test',
    basePath: '/v1',
    baseEnv: 'OPENAI_BASE_URL',
    keyEnv: 'OPENAI_API_KEY',
    platform: 'windows',
  });
  assert.match(sh, /^\$env:OPENAI_BASE_URL = "http:\/\/gw\.test\/v1"\n/);
  assert.match(sh, /\n\$env:OPENAI_API_KEY = "<your gateway key>"$/);
});

test('the shell sample never invents a key value', () => {
  const sh = buildShell({
    base: 'http://gw.test',
    baseEnv: 'X_URL',
    keyEnv: 'X_KEY',
    keyPlaceholder: '<the key you saved>',
  });
  assert.match(sh, /export X_KEY=<the key you saved>/);
});

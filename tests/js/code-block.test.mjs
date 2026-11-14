/**
 * The code-block component: colour per language, and nothing executable.
 *
 * The panel had no highlighter, so a shell line and its own flags read as one
 * string; there is no build step and no third-party script allowed, so the
 * lexer is the panel's own.  What matters for safety is that every piece goes
 * through the escaper — a sample is text about code, never markup.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { highlight, escapeCode, CODE_LANGS } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/code-block.js';

test('markup in a sample is text, never markup', () => {
  assert.equal(escapeCode('<script>'), '&lt;script&gt;');
  const html = highlight('<script>alert(1)</script>', 'json');
  assert.ok(html.startsWith('&lt;script&gt;'), html);
  assert.ok(!html.includes('<script'), 'raw markup survived the lexer');
  assert.ok(!highlight('"<img src=x>"', 'json').includes('<img'));
});

test('a shell line tells its keyword, comment, flag and variable apart', () => {
  const html = highlight('export TOKEN=$KEY --flag # note', 'shell');
  assert.match(html, /tok-kw">export</);
  assert.match(html, /tok-var">\$KEY</);
  assert.match(html, /tok-flag">--flag</);
  assert.match(html, /tok-com"># note</);
});

test('JSON keys, literals and numbers take their own ink', () => {
  const html = highlight('{"a": 1, "b": [true, null]}', 'json');
  assert.match(html, /tok-key">&quot;a&quot;</);
  assert.match(html, /tok-num">1</);
  assert.match(html, /tok-lit">true</);
});

test('a # inside a quoted URL is not a comment', () => {
  const html = highlight('curl "http://h/p#frag"', 'shell');
  assert.ok(!/tok-com/.test(html), html);
});

test('an unknown language is escaped and left plain', () => {
  assert.equal(highlight('<b>x</b>', 'brainfuck'), '&lt;b&gt;x&lt;/b&gt;');
  assert.deepEqual(CODE_LANGS, ['json', 'shell', 'python', 'js']);
});

/**
 * The auto-escaping html`` template, raw() and render() in core/html.js.
 *
 * The attribute cases parse the produced markup the way a browser does, by
 * reading the attribute value back, so a "looks escaped" string cannot pass
 * for an escaped one.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  cssValue,
  html,
  isSafeHtml,
  jsArg,
  raw,
  logoUrl,
  render,
  safeUrl,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/html.js';

/** Decode the five entities esc() writes, as an attribute parse would. */
function decode(s) {
  return s
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

/** The value of the first double-quoted attribute `name` in `markup`. */
function attr(markup, name) {
  const m = new RegExp(`${name}="([^"]*)"`).exec(markup);
  return m && decode(m[1]);
}

test('html`` escapes interpolations and returns SafeHtml', () => {
  const out = html`<p>${'a < b & c'}</p>`;
  assert.equal(isSafeHtml(out), true);
  assert.equal(String(out), '<p>a &lt; b &amp; c</p>');
});

test('quotes and angle brackets are escaped in text', () => {
  const out = String(html`<i>${`"x" 'y' <script>`}</i>`);
  assert.equal(out, '<i>&quot;x&quot; &#39;y&#39; &lt;script&gt;</i>');
});

test('attribute-value context: a quote cannot end the attribute', () => {
  const evil = '" onmouseover="alert(1)" x="';
  const out = String(html`<input value="${evil}" data-a="1">`);
  assert.equal(attr(out, 'value'), evil);
  assert.equal(attr(out, 'data-a'), '1');
  assert.equal(out.includes('onmouseover="alert'), false);
});

test('null, undefined and false render as nothing; numbers as text', () => {
  assert.equal(String(html`[${null}${undefined}${false}${0}${7}]`), '[07]');
});

test('nested SafeHtml and arrays of SafeHtml pass through unescaped', () => {
  const li = (x) => html`<li>${x}</li>`;
  const out = String(html`<ul>${['a&b', '<c>'].map(li)}</ul>`);
  assert.equal(out, '<ul><li>a&amp;b</li><li>&lt;c&gt;</li></ul>');
  assert.equal(String(html`<b>${html`<i>${'<'}</i>`}</b>`), '<b><i>&lt;</i></b>');
});

test('an array of plain strings is escaped item by item', () => {
  assert.equal(String(html`${['<a>', '<b>']}`), '&lt;a&gt;&lt;b&gt;');
});

test('raw() marks trusted markup and skips escaping', () => {
  assert.equal(String(html`<b>${raw('<i>x</i>')}</b>`), '<b><i>x</i></b>');
  assert.equal(isSafeHtml(raw('x')), true);
  assert.equal(String(raw(null)), '');
});

test('a plain string that looks like markup is still escaped', () => {
  assert.equal(String(html`${'<b>x</b>'}`), '&lt;b&gt;x&lt;/b&gt;');
});

test('the injection case of commit 62dbdb387c: jsArg in a handler stays one string', () => {
  const name = "a'-alert(1)-'b";
  const out = String(html`<button onclick="rotateKey(${raw(jsArg(name))})">x</button>`);
  const handler = attr(out, 'onclick');
  // The parsed handler is a call with a single, intact string literal.
  assert.equal(handler, `rotateKey(${JSON.stringify(name)})`);
  assert.deepEqual(JSON.parse(handler.slice('rotateKey('.length, -1)), name);
  // Interpolating it through esc() inside quotes is the bug: the apostrophe
  // survives the attribute parse and closes the literal.
  const bad = String(html`<button onclick="rotateKey('${name}')">x</button>`);
  assert.equal(attr(bad, 'onclick'), `rotateKey('${name}')`);
});

test('render() sets innerHTML from SafeHtml only', () => {
  const el = { innerHTML: '' };
  assert.equal(render(el, html`<p>${'<x>'}</p>`), el);
  assert.equal(el.innerHTML, '<p>&lt;x&gt;</p>');
  render(el, raw('<hr>'));
  assert.equal(el.innerHTML, '<hr>');
});

test('render() throws on a plain string and leaves the element alone', () => {
  const el = { innerHTML: 'keep' };
  assert.throws(() => render(el, '<b>x</b>'), TypeError);
  assert.throws(() => render(el, undefined), TypeError);
  assert.throws(() => render(el, { value: '<b>' }), TypeError);
  assert.equal(el.innerHTML, 'keep');
});

test('cssValue keeps colours, var() and lengths, and drops anything that can add a declaration', () => {
  for (const ok of ['#fff', '#A1B2C3', '#a1b2c3d4', 'rgb(1, 2, 3)', 'rgba(1 2 3 / 0.5)',
    'hsl(120 45% 90%)', 'hsl(120deg 45% 90%)', 'var(--accent)', '8px', '0', '42%', '-1.5rem']) {
    assert.equal(cssValue(ok), ok, ok);
  }
  assert.equal(cssValue('  #fff '), '#fff');
  for (const bad of ['red;position:fixed', '#fff;x:y', 'red', 'url(https://x/y)', 'rgb(1,2,3);a:b',
    'rgb(1,2,calc(3))', 'var(--a);b:c', 'var(--a, red)', 'expression(1)', '#ggg', '8px 0', '',
    'hsl(1 2% 3%) !important', '"; background:url(//e)']) {
    assert.equal(cssValue(bad), '', bad);
  }
  assert.equal(cssValue(null), '');
  assert.equal(cssValue(undefined), '');
  assert.equal(cssValue(12), '12');
});

test('safeUrl keeps http(s), paths and image data URLs, and drops script schemes', () => {
  for (const ok of ['https://a.example/x?y=1', 'HTTP://a.example', '/admin/logo.svg', './a.svg',
    '#frag', 'logo.svg', 'data:image/png;base64,AAAA', '']) {
    assert.equal(safeUrl(ok), ok, ok);
  }
  for (const bad of ['javascript:alert(1)', 'JavaScript:alert(1)', ' javascript:alert(1)',
    'java\tscript:alert(1)', '\u0001javascript:alert(1)', 'vbscript:x', 'data:text/html,<b>',
    'file:///etc/passwd', 'blob:https://a/b']) {
    assert.equal(safeUrl(bad), '', JSON.stringify(bad));
  }
  assert.equal(safeUrl(null), '');
});

test('logoUrl proxies http logos, keeps paths, and drops a script scheme', () => {
  assert.equal(logoUrl('https://a.example/l.svg'), '/admin/api/logo?src=https%3A%2F%2Fa.example%2Fl.svg');
  assert.equal(logoUrl('/static/l.svg'), '/static/l.svg');
  assert.equal(logoUrl('javascript:alert(1)'), '');
});

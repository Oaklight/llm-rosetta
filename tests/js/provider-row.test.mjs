/**
 * One builder per lane control, shared by every surface.
 *
 * The same switch was hand-assembled three times — the Models menu, the
 * drawer and the Routing board — and the copies drifted: one lost its
 * provider pin, another blamed the wrong switch for being off.  These cases
 * pin the builders' actual output, and pin the three call sites to them, so
 * the next fix lands on all three at once.
 *
 * The builders escape through `core/html.js`, which does that without a DOM,
 * so this file runs them rather than reading them.  (It used to be text-level:
 * an `esc` built on a detached `<div>` made every one of these cases
 * unobservable outside a browser, and three mutations of the switch survived.)
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  laneGripHtml,
  laneSwitchHtml,
  laneWeightInputHtml,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/components/provider-row.js';
import { esc, jsArg } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/html.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN_JS = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js',
);
const MODELS = readFileSync(join(ADMIN_JS, 'tabs/models.js'), 'utf8');
const ROUTING = readFileSync(join(ADMIN_JS, 'tabs/routing.js'), 'utf8');

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

// ── the escaping the builders stand on ──

test('esc covers a text node and a quoted attribute', () => {
  assert.equal(esc(`<&>"'`), '&lt;&amp;&gt;&quot;&#39;');
  assert.equal(esc(0), '0', 'a number is a value, not an absence');
  assert.equal(esc(null), '', 'an absent value renders as nothing');
});

test('jsArg builds a literal that survives the attribute parse', () => {
  assert.equal(jsArg('plain'), '&quot;plain&quot;');
  assert.equal(
    jsArg("a'-alert(1)-'b"),
    '&quot;a\'-alert(1)-\'b&quot;',
    'the apostrophe stays inside the double-quoted literal',
  );
  assert.equal(jsArg('say "hi"'), '&quot;say \\&quot;hi\\&quot;&quot;',
    'a quote in the value is JSON-escaped first, then made safe for the attribute');
});

// ── the switch ──

test('the switch draws its own state, on and off', () => {
  const on = laneSwitchHtml({ name: 'alpha', on: true, title: 'enabled' });
  assert.match(on, /role="switch" checked/, 'an on lane starts checked');
  assert.match(on, /aria-checked="true"/);

  const off = laneSwitchHtml({ name: 'alpha', on: false, title: 'disabled' });
  assert.doesNotMatch(off, /role="switch" checked/, 'an off lane starts unchecked');
  assert.match(off, /aria-checked="false"/);
});

test('the switch keeps the surface hook as a separate class', () => {
  const html = laneSwitchHtml({
    name: 'alpha',
    on: true,
    title: 'on',
    extraClass: 'routing-switch',
  });
  assert.match(html, /class="toggle row-toggle routing-switch"/,
    'the space is what makes the board\'s own selector match');
});

test('the switch escapes what it interpolates', () => {
  const html = laneSwitchHtml({ name: 'a"<b>', on: true, title: "it's & more" });
  assert.match(html, /aria-label="a&quot;&lt;b&gt;"/);
  assert.match(html, /title="it&#39;s &amp; more"/);
});

test("the switch passes the surface's data attribute through untouched", () => {
  const html = laneSwitchHtml({
    name: 'alpha',
    on: true,
    title: 'on',
    data: 'data-lane="alpha"',
  });
  assert.match(html, /data-lane="alpha"/, 'the board reads the lane from here');
});

// ── the other two controls ──

test('the grip keeps each surface\'s class', () => {
  assert.match(laneGripHtml({ title: 'drag' }), /class="pm-grip"/);
  assert.match(
    laneGripHtml({ title: 'drag', cls: 'routing-grip' }),
    /class="routing-grip"/,
  );
});

test('the weight field hands the handler names, not escaped text', () => {
  const html = laneWeightInputHtml({
    model: 'm1',
    provider: "a'-alert(1)-'b",
    weight: 3,
    title: 'Weight',
  });
  assert.match(html, /value="3"/);
  assert.match(
    html,
    /onchange="updateProviderWeight\(&quot;m1&quot;,&quot;a'-alert\(1\)-'b&quot;,&quot;&quot;,this\)"/,
    'jsArg, not esc: the apostrophe cannot close the literal, and the lane ' +
      'without an upstream still passes one argument so the handler signature ' +
      'stays one shape',
  );
});

// ── the call sites ──

test('the Models menu rows draw all three controls from the component', () => {
  const body = bodyOf(MODELS, 'function _providerMenuRows(');
  assert.match(body, /laneGripHtml\(/, 'the drag handle is shared');
  assert.match(body, /laneSwitchHtml\(/, 'the switch is shared');
  assert.match(body, /laneWeightInputHtml\(/, 'the weight field is shared');
  assert.match(body, /jsArg\(name\)/, 'the switch handler builds its arguments with jsArg');
  assert.doesNotMatch(body, /<label class="toggle row-toggle/, 'no hand-assembled switch');
  assert.doesNotMatch(body, /class="pd-weight"/, 'no hand-assembled weight field');
  assert.doesNotMatch(body, /class="pm-grip"/, 'no hand-assembled drag handle');
});

test('the Routing board keeps its own hook through the shared builders', () => {
  const body = bodyOf(ROUTING, 'function _renderBoard(');
  assert.match(body, /laneSwitchHtml\(/);
  assert.match(body, /extraClass: 'routing-switch'/);
  assert.match(body, /data: 'data-lane="/,
    'the key is part of the contract: a renamed key toggles a lane named undefined');
  assert.match(body, /laneGripHtml\(\{[^}]*cls: 'routing-grip'/s);
  assert.doesNotMatch(body, /<label class="toggle row-toggle/);
  assert.match(ROUTING, /querySelectorAll\('\.routing-switch input'\)/,
    'the listener still matches the shared markup');
});

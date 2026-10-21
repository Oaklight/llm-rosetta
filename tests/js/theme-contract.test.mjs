/**
 * The admin panel's theme contract.
 *
 * `base.css` owns the tokens — a scheme declares values, per scheme and per
 * mode — and every other stylesheet consumes them.  A scheme that reaches into
 * a component's box model is how #1149 happened: emerald's decorative
 * `::before` grid carried `pointer-events: auto`, sat above the KPI card's
 * contents, and swallowed the Token card's "details" click in every browser.
 * Nothing forbade it, so this test does.
 *
 * The checker parses the stylesheets as text — no browser, no DOM — which is
 * enough for the three rules that matter, and it is exercised against fixture
 * strings below so a checker that silently matches nothing fails here.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const CSS_DIR = join(dirname(fileURLToPath(import.meta.url)), '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/css');

// ---- a very small CSS reader -------------------------------------------

/** Read a `{ ... }` block starting after the opening brace; returns body + index past it. */
function readBlock(text, start) {
  let depth = 1;
  let quote = null;
  let body = '';
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      body += ch;
      if (ch === quote && text[i - 1] !== '\\') quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") { quote = ch; body += ch; continue; }
    if (ch === '{') { depth++; body += ch; continue; }
    if (ch === '}') {
      depth--;
      if (depth === 0) return { body, next: i + 1 };
      body += ch;
      continue;
    }
    body += ch;
  }
  return { body, next: text.length };
}

/**
 * Split a stylesheet into `{selector, declarations, at}` rules.  At-rules are
 * containers: their bodies are parsed recursively and the inner rules are
 * returned too, tagged with the chain of at-rule selectors that encloses them
 * (`at`), so a scheme rule hidden inside `@media` is still policed.
 */
export function parseRules(css) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, '');
  return parseBody(text, []);
}

/** Parse one block body; *atStack* is the at-rule chain it sits inside. */
function parseBody(text, atStack) {
  const rules = [];
  let buf = '';
  let i = 0;

  while (i < text.length) {
    const ch = text[i];
    if (ch === '{') {
      const selector = buf.trim();
      buf = '';
      const { body, next } = readBlock(text, i + 1);
      if (selector.startsWith('@')) {
        const inner = parseBody(body, [...atStack, selector]);
        if (inner.length) {
          rules.push(...inner);
        } else {
          // A declaration block directly inside an at-rule (e.g. @font-face).
          const declarations = splitDeclarations(body);
          if (declarations.length) {
            rules.push({ selector, body, declarations, at: atStack });
          }
        }
      } else {
        rules.push({
          selector,
          body,
          declarations: splitDeclarations(body),
          at: atStack,
        });
      }
      i = next;
      continue;
    }
    if (ch === '}') {
      i++;
      buf = '';
      continue;
    }
    buf += ch;
    i++;
  }
  return rules;
}

function splitDeclarations(body) {
  const out = [];
  let depth = 0;
  let quote = null;
  let buf = '';
  for (const ch of body) {
    if (quote) {
      buf += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") { quote = ch; buf += ch; continue; }
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ';' && depth === 0) { out.push(buf); buf = ''; continue; }
    buf += ch;
  }
  if (buf.trim()) out.push(buf);
  return out.map(d => d.trim()).filter(Boolean);
}

/** All rules, including those nested inside at-rules. */
function allRules(css) {
  return parseRules(css);
}

function property(declaration) {
  const colon = declaration.indexOf(':');
  return colon === -1 ? declaration.trim() : declaration.slice(0, colon).trim();
}

function value(declaration) {
  const colon = declaration.indexOf(':');
  return colon === -1 ? '' : declaration.slice(colon + 1).trim();
}

const SCHEME_SELECTOR = /^\[data-scheme="[^"]+"\](?:\[data-mode="[^"]+"\])?$/;

// ---- the three rules ----------------------------------------------------

/**
 * Rule 1: a scheme selector may only appear in `base.css`, and only as the
 * bare scheme/mode selector — never with a component after it.
 */
export function checkSchemeSelectors(files) {
  const problems = [];
  for (const { name, css, isTokens } of files) {
    for (const rule of allRules(css)) {
      if (!rule.selector.includes('[data-scheme=')) continue;
      if (!isTokens) {
        problems.push(`${name}: scheme-scoped component rule "${rule.selector}" (schemes belong in base.css)`);
        continue;
      }
      if (!SCHEME_SELECTOR.test(rule.selector)) {
        problems.push(`${name}: scheme selector "${rule.selector}" carries a component (declare tokens only)`);
        continue;
      }
      for (const decl of rule.declarations) {
        const prop = property(decl);
        if (!prop.startsWith('--')) {
          problems.push(`${name}: "${rule.selector}" declares "${prop}" — a scheme block may only declare custom properties`);
        }
      }
    }
  }
  return problems;
}

/**
 * Rule 2: a decorative pseudo-element positioned over its parent must not take
 * pointer events.  This is the #1149 bug, stated as a rule.
 *
 * A decoration on a *control* is exempt: a switch knob's originating element
 * (`input[type="checkbox"]`) is itself the click target, so a click on the
 * knob is a click on the control — the behaviour the control wants.  What must
 * never happen is a decoration on a *container* (a card, a section) covering
 * the clickable things inside it.
 *
 * Anything else that looks like a violation is listed below by hand, with the
 * reason it is safe.  An exemption is a deliberate, reviewable decision; a new
 * overlay on a card cannot slip in unnoticed.
 */
const EXEMPT_DECORATIONS = [
  // The switch knob: its parent span is the label's click surface, so a click
  // on the knob reaches the label and toggles the checkbox.
  '.toggle .slider::before',
];

export function checkDecorationPointerEvents(files) {
  const problems = [];
  for (const { name, css } of files) {
    // One declaration map per selector, across every rule that names it: a
    // decoration may put `position` in one rule and `inset` in another, and
    // checking rule by rule would let that split hide the missing
    // `pointer-events`.  Later rules win, as in the cascade.
    const bySelector = new Map();
    for (const rule of allRules(css)) {
      if (!/::(before|after)\b/.test(rule.selector)) continue;
      const selector = rule.selector.replace(/\s+/g, ' ').trim();
      const props = bySelector.get(selector) ?? new Map();
      for (const decl of rule.declarations) props.set(property(decl), value(decl));
      bySelector.set(selector, props);
    }

    for (const [selector, props] of bySelector) {
      if (EXEMPT_DECORATIONS.includes(selector)) continue;
      const originating = selector.replace(/::(before|after)\b.*$/, '').trim();
      const lastCompound = originating.split(/\s+|>/).filter(Boolean).pop() || '';
      if (/\b(input|button|select|label|textarea)\b/.test(lastCompound)) continue;
      const positioned = /^(absolute|fixed)$/.test(props.get('position') || '');
      if (!positioned) continue;
      const pointer = props.get('pointer-events');
      if (pointer !== 'none') {
        problems.push(`${name}: "${selector}" is positioned over its parent but declares pointer-events: ${pointer ?? '(nothing)'} — a decoration must not take a click`);
      }
    }
  }
  return problems;
}

// ---- the stylesheets ----------------------------------------------------

// Every stylesheet in the directory, not a hand-listed pair: a third sheet
// added later must be policed the day it lands, and base.css is the only one
// allowed to carry scheme selectors.  Read fresh, so a new file cannot be
// missed by a stale list.
const FILES = readdirSync(CSS_DIR)
  .filter(name => name.endsWith('.css'))
  .sort()
  .map(name => ({
    name,
    css: readFileSync(join(CSS_DIR, name), 'utf8'),
    isTokens: name === 'base.css',
  }));

test('the checker sees the stylesheets it is meant to police', () => {
  // A parser that matches nothing would pass every rule below; these are the
  // shapes it must find on the current tree.
  const schemeRules = FILES.flatMap(f => allRules(f.css)).filter(r => r.selector.includes('[data-scheme='));
  assert.ok(schemeRules.length >= 6, `expected the scheme token blocks, found ${schemeRules.length}`);
  const decorations = FILES.flatMap(f => allRules(f.css)).filter(r => /::(before|after)\b/.test(r.selector));
  assert.ok(decorations.length >= 1, 'expected at least one decorative pseudo-element');
  // At-rule bodies must be parsed, not skipped: the panel's responsive tweaks
  // live in @media and are exactly where a scheme rule would be added back.
  const nested = FILES.flatMap(f => allRules(f.css)).filter(r => r.at.length > 0);
  assert.ok(nested.length >= 1, 'expected rules nested inside an at-rule');
  // The token sheet is base.css and only base.css; everything else consumes it.
  assert.deepEqual(FILES.filter(f => f.isTokens).map(f => f.name), ['base.css']);
});

test('no scheme-scoped component rules outside base.css', () => {
  assert.deepEqual(checkSchemeSelectors(FILES), []);
});

test('scheme blocks declare custom properties only', () => {
  const problems = checkSchemeSelectors(FILES);
  assert.deepEqual(problems.filter(p => p.includes('custom properties')), []);
});

test('positioned decorations never take a pointer event', () => {
  assert.deepEqual(checkDecorationPointerEvents(FILES), []);
});

// ---- the checker has teeth ---------------------------------------------

test('the scheme-selector check flags a component rule in a component sheet', () => {
  const fixture = [{ name: 'components.css', css: '[data-scheme="emerald"] .stat-card { position: relative; }', isTokens: false }];
  const problems = checkSchemeSelectors(fixture);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /scheme-scoped component rule/);
});

test('the scheme-selector check flags a non-token declaration in base.css', () => {
  const fixture = [{ name: 'base.css', css: '[data-scheme="emerald"] { --radius: 4px; color: red; }', isTokens: true }];
  const problems = checkSchemeSelectors(fixture);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /custom properties/);
});

test('the scheme-selector check looks inside at-rules', () => {
  const inMedia = checkSchemeSelectors([{
    name: 'components.css',
    css: '@media (max-width: 900px) { [data-scheme="emerald"] .stat-card::before { pointer-events: auto; } }',
    isTokens: false,
  }]);
  assert.equal(inMedia.length, 1);
  assert.match(inMedia[0], /scheme-scoped component rule/);

  // ...including an at-rule nested in an at-rule, and the token rule inside it.
  const inSupports = checkSchemeSelectors([{
    name: 'base.css',
    css: '@supports (display: grid) { @media (min-width: 1px) { [data-scheme="emerald"] { color: red; } } }',
    isTokens: true,
  }]);
  assert.equal(inSupports.length, 1);
  assert.match(inSupports[0], /custom properties/);
});

test('the decoration check cannot be split across two rules', () => {
  // `position` in one rule and the paint in another is still one element, and
  // the missing `pointer-events` must not hide between them.
  const fixture = [{
    name: 'components.css',
    css: '.card::before { content:""; position:absolute; inset:0; }\n.card::before { background: red; }',
  }];
  const problems = checkDecorationPointerEvents(fixture);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /pointer-events/);

  // The same split, with `pointer-events: none` supplied by the second rule,
  // is the fixed shape and passes.
  assert.deepEqual(checkDecorationPointerEvents([{
    name: 'components.css',
    css: '.card::before { content:""; position:absolute; inset:0; }\n.card::before { pointer-events: none; }',
  }]), []);
});

test('the decoration check flags exactly the #1149 shape', () => {
  const fixture = [{
    name: 'components.css',
    css: '[data-scheme="emerald"] .stat-card::before { content:""; position:absolute; inset:0; pointer-events:auto; opacity:.5; }',
  }];
  const problems = checkDecorationPointerEvents(fixture);
  assert.equal(problems.length, 1);
  assert.match(problems[0], /pointer-events: auto/);
  // ...and the fixed shape passes
  assert.deepEqual(checkDecorationPointerEvents([{
    name: 'components.css',
    css: '.stat-card::before { content:""; position:absolute; inset:0; pointer-events:none; }',
  }]), []);
  // ...and an unpositioned decoration is not policed
  assert.deepEqual(checkDecorationPointerEvents([{
    name: 'base.css',
    css: '.settings-popup-panel h3::before { content:""; display:block; width:3px; }',
  }]), []);
});

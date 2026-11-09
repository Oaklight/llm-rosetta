/**
 * The control scale holds (issue #1472 P4).
 *
 * The rules the phase exists to establish, each of which a later edit can
 * quietly undo:
 *
 * 1. no stylesheet removes the focus ring — P4 found fifteen rules that set
 *    `outline: none` on focus, and two controls that never had a ring at all;
 * 2. the ring is one declaration set, `2px solid var(--accent)`, drawn at one
 *    of three offsets — not a recipe per context;
 * 3. a control's corner is a token: `--radius-control`, `0`, a pill only on a
 *    status tag, or the two-corner form of a split button.  A literal 3px or
 *    14px radius on a control is how the panel ended up with seven corners;
 * 4. a control is 28px tall and a tag 18px — the scale is a height as much as
 *    a corner.  This guard found two deviations, a 26px icon button and a
 *    22px weight stepper, because the first version of this file asserted the
 *    corner and nothing else;
 * 5. a page or panel frame keeps `--radius`; the control corner does not leak
 *    onto a card;
 * 6. `color-scheme` follows the mode (#1470), so UA widgets — select
 *    popups, checkbox glyphs, the caret — follow it, and only the mode
 *    declares it.  The scrollbar is unconfirmed: this Chromium draws overlay
 *    scrollbars, so there are no pixels to compare.
 *
 * Text-level and over every stylesheet: P4 moved the skin out of twelve
 * context rules and twenty-two inline attributes, so a single sheet can no
 * longer be trusted to hold it.
 *
 * Eleven mutations this file has been checked against — each one reds the
 * test named, and a guard that stops reding is not a guard:
 *
 *   1. `outline: none` on a `:focus-visible` rule          → 1
 *   2. `:focus-visible { outline: 1px … }`                  → 2
 *   3. a literal `4px` corner on `.chip`                    → 3
 *   4. `.btn { height: 24px }`                              → 4
 *   5. `button.prov-icon-btn { height: 44px }`              → 4
 *   6. `.cap-badge .fake-btn { height: 40px }`              → 4
 *   7. `.login-remember input { height: 60px }`             → 4
 *   8. `color-scheme: dark` on `.chip`                      → 6
 *   9. `color-scheme` inside a theme file                   → theme-contract
 *  10. `.modal .prov-tab { outline: none }`                  → 1 (P5: the
 *      dialog exception may not be borrowed by a descendant)
 *  11. `.modal.modal-wide:focus-visible { outline: none }`   → 1 (nor by a
 *      variant)
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);
const CSS_DIR = join(ADMIN, 'css');
const SHEETS = readdirSync(CSS_DIR)
  .filter((name) => name.endsWith('.css'))
  .map((name) => ({
    name,
    text: readFileSync(join(CSS_DIR, name), 'utf8').replace(/\/\*[\s\S]*?\*\//g, ''),
  }));

/** Every `selector { declarations }` block in a sheet. */
function rules(text) {
  const out = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push({ selector: m[1].trim(), body: m[2] });
  }
  return out;
}

function declarations(body) {
  return body
    .split(';')
    .map((d) => d.trim())
    .filter(Boolean)
    .map((d) => {
      const at = d.indexOf(':');
      return [d.slice(0, at).trim(), d.slice(at + 1).trim()];
    });
}

// A control *or* its option; a logo image, a dropdown panel and a card are
// not controls and keep their own corner.
const CONTROL = /\b(button|input|select|textarea)\b|\.(btn|chip|badge|cap-badge|seg-control|key-btn|chart-range-btn|pill-toggle|logo-select-trigger|logo-select-option|facet-chip)/;
const NOT_CONTROL = /-logo\b|-img\b|-panel\b/;

const ALLOWED_RADII = new Set([
  'var(--radius-control)',
  '0',
]);

/** A split button spells two corners; both ends must be the token. */
function splitButtonRadius(value) {
  const parts = value.split(/\s+/);
  return parts.length === 4 && parts.every((p) => ALLOWED_RADII.has(p));
}

test('no stylesheet removes the focus ring', () => {
  // A dialog container takes focus on open as a landing point (`tabindex="-1"`
  // is how the browser says where focus went).  It is not a control, and a
  // 2px ring around a whole panel is not an affordance.  The exception is
  // deliberately narrow: the two dialog panel selectors, and only `outline:
  // none` — not a descendant, not a variant, not another value.
  const FOCUS_LANDING = /^\.(?:modal|model-drawer)(:focus-visible)?$/;
  for (const sheet of SHEETS) {
    for (const rule of rules(sheet.text)) {
      for (const [prop, value] of declarations(rule.body)) {
        const landing = FOCUS_LANDING.test(rule.selector) && prop === 'outline' && /^none\b/.test(value);
        assert.ok(
          landing || !(prop === 'outline' && /^none\b/.test(value)),
          `${sheet.name}: "${rule.selector}" sets outline: ${value} — the panel draws one ring for every control`,
        );
      }
    }
  }
});

test('the ring is one declaration set, declared for :focus-visible', () => {
  const base = SHEETS.find((s) => s.name === 'base.css');
  assert.ok(base, 'expected base.css');
  const rule = rules(base.text).find((r) => r.selector === ':focus-visible');
  assert.ok(rule, 'expected a global :focus-visible rule');
  const decls = new Map(declarations(rule.body));
  assert.equal(decls.get('outline'), '2px solid var(--accent)');
  assert.equal(decls.get('outline-offset'), '2px');
});

test('a control corner is a token, not a literal', () => {
  const problems = [];
  for (const sheet of SHEETS) {
    for (const rule of rules(sheet.text)) {
      if (!CONTROL.test(rule.selector) || NOT_CONTROL.test(rule.selector)) continue;
      for (const [prop, value] of declarations(rule.body)) {
        if (!/^border(-[a-z]+)*-radius$/.test(prop)) continue;
        if (/^(50%|999px|var\(--radius-pill\))$/.test(value)) continue; // dots, status pills
        if (splitButtonRadius(value)) continue;
        if (ALLOWED_RADII.has(value)) continue;
        problems.push(`${sheet.name}: "${rule.selector}" { ${prop}: ${value} }`);
      }
    }
  }
  assert.deepEqual(problems, []);
});

test('a control height is 28px, a tag is 18px, and nothing else', () => {
  // The size half of the scale.  A mutation that changed `.btn`'s height to
  // 24px passed every other test in this file; this one exists so it cannot.
  //
  // The allowlist is the complete set of shapes that are not the 28px control,
  // and it is closed: each entry carries a measured reason, a new shape is
  // 28px until it brings one, and no entry is added to make a red test green.
  // A status or capability tag (18), the alert dot on a capability tag (13),
  // the invisible `.sr-select` stand-in (1), the toggle switch's track (20 —
  // a switch is as tall as its track).  The routing lane's -/+ stepper used to
  // sit at 22px beside a 24px number; measured, 28px leaves the lane row at
  // 51px and asks only that the weight column widen 84→92px, which the two
  // `fr` columns absorb, so it takes the scale like every other button.
  //
  // Four things this guard does NOT see, stated so a later reader does not
  // mistake green for complete: a control sized by `padding` rather than
  // `height` (the 26px close button that started this test was one), a native
  // checkbox or radio, a control rendered by JS without a stylesheet rule, and
  // `min-height`.  Three controls are not 28px by measurement, not by
  // oversight: `.chart-range-btn` 19px inside a chart header, `.btn-sm` 30px
  // and the full-width logo picker 36px below the phone breakpoint.  P4's
  // census over the rendered panel is what covered the first and third; this
  // file covers what a stylesheet declares.
  const CASES = [
    [/-alert$/, '13px'],
    [/\.(badge|cap-badge)\b/, '18px'],
    [/\.sr-select$/, '1px'],
    [/toggle-label input\[type="checkbox"\]$/, '20px'],
  ];
  // A native checkbox or radio is not a control of this scale: it keeps the
  // glyph the platform draws.  The selector has to say so — an untyped `input`
  // is a text field and must be 28px.
  const NATIVE = /input\[type="(?:checkbox|radio)"\]$/;
  // Rules that size a *part* of a control, closed and reasoned.  A class name
  // that merely ends in `-icon` is not on this list: `button.prov-icon-btn` is
  // a 28px button, and letting a naming habit pass is how a guard stops
  // guarding.  A new entry needs a measurement and a reason, like the heights.
  const PARTS = new Map([
    ['.cap-badge svg', 'the 11px glyph inside a capability tag'],
    ['#tab-models .mn-badges .cap-badge svg', 'the same glyph, 12px in the model table'],
    ['.logo-select-img', 'the 16px glyph inside the logo trigger'],
  ]);
  const segments = (sel) => sel.trim().split(/[\s>+~]+/).filter(Boolean);

  // `28.0px`, `1.75rem` and `28px` are the same height.  rem and em resolve
  // against the root font size, which base.css sets to 16px on `html`; the
  // assumption is written here rather than left implicit.  A percentage is
  // relative to its parent, so it cannot be compared with 28px — it is
  // skipped like `auto`, and a control that fills its parent is a decision
  // the layout makes, not one this test can see.  Any other unit is an error,
  // not a skip: a guard that silently passes `height: 2rem` is a guard that
  // has stopped.
  const lengthPx = (value, where) => {
    const text = value.trim();
    if (/^-?\d*\.?\d+%$/.test(text)) return null;
    const m = /^(-?\d*\.?\d+)(px|rem|em)?$/.exec(text);
    if (!m) {
      if (/^-?\d/.test(text) && /[a-z]+$/i.test(text)) {
        throw new Error(`${where}: cannot compare "${value}" — declare the height in px`);
      }
      return null;
    }
    return m[2] === undefined || m[2] === 'px' ? Number(m[1]) : Number(m[1]) * 16;
  };

  const problems = [];
  for (const sheet of SHEETS) {
    for (const rule of rules(sheet.text)) {
      const parts = segments(rule.selector);
      // Any compound that names a control brings the rule into scope, not only
      // the last one: `.cap-badge .fake-btn` sizes something inside a control
      // and used to slip through because the subject was not a control.
      if (!parts.some((p) => CONTROL.test(p)) || NOT_CONTROL.test(rule.selector)) continue;
      if (rule.selector.includes('::') || PARTS.has(rule.selector)) continue;
      const subject = parts[parts.length - 1];
      if (NATIVE.test(subject)) continue;
      const allowed = CASES.find(([re]) => re.test(rule.selector))?.[1] ?? '28px';
      for (const [prop, value] of declarations(rule.body)) {
        if (prop !== 'height') continue;
        if (value === 'auto') continue;
        // The invisible input behind a styled label is 0 by design.
        if (/input/.test(subject) && value === '0') continue;
        const got = lengthPx(value, `${sheet.name}: "${rule.selector}"`);
        if (got === null) continue;
        const want = lengthPx(allowed, `${sheet.name}: "${rule.selector}"`);
        if (got === want) continue;
        problems.push(`${sheet.name}: "${rule.selector}" { height: ${value} } — expected ${allowed}`);
      }
    }
  }
  assert.deepEqual(problems, []);
});

test('page and shell frames are not controls and keep their own corners', () => {
  // The guard above must not have swallowed the card/modal radii: those are
  // P5's, and a token they do not use yet is not a P4 defect.
  const components = SHEETS.find((s) => s.name === 'components.css');
  const card = rules(components.text).find((r) => r.selector === '.chart-box');
  assert.ok(card, 'expected .chart-box to still be styled');
  const radius = declarations(card.body).find(([p]) => p === 'border-radius');
  assert.ok(radius && radius[1] === 'var(--radius)', 'a card keeps --radius');
});

test('color-scheme follows the mode, and only the mode', () => {
  // Every component sheet, not just base.css: the declaration is a mode fact,
  // so it may appear exactly twice, on the two :root[data-mode] selectors.
  // The scheme sheets are not read here — they are covered by
  // theme-contract.test.mjs, whose "a scheme block may only declare custom
  // properties" rule refuses `color-scheme` in a theme file.
  const declared = [];
  for (const sheet of SHEETS) {
    for (const rule of rules(sheet.text)) {
      for (const [prop, value] of declarations(rule.body)) {
        if (prop === 'color-scheme') declared.push([sheet.name, rule.selector, value]);
      }
    }
  }
  assert.deepEqual(declared, [
    ['base.css', ':root[data-mode="light"]', 'light'],
    ['base.css', ':root[data-mode="dark"]', 'dark'],
  ]);
});

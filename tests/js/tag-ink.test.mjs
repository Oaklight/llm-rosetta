/**
 * Tag ink contrast: every tag clears WCAG AA, in every scheme and mode.
 *
 * A tag paints its text in a `--tag-*` ink and derives its fill from that same
 * ink — `color-mix(in srgb, currentColor 8%, transparent)` — so the ratio that
 * decides readability is the ink against its own tint, not against the bare
 * page.  The fill is composited over each of its three seats — `--bg` for a
 * status tag, `--surface` for a capability tag, and the `--surface-2` that
 * `tbody tr:hover td` paints when a row is hovered — quantised to 8-bit the
 * way the browser paints it, and measured with WCAG 2.1: linearise each sRGB
 * channel, `L = 0.2126R + 0.7152G + 0.0722B`, `(L1+0.05)/(L2+0.05)`.
 *
 * The hover seat is the strictest of the three in both modes, and it is the
 * one a two-seat measurement misses: `#ac11c3` (TypeSafe light magenta) reads
 * 4.760 on `--surface` and 4.433 on `--surface-2`.
 *
 * Every tag is `--fs-micro` = 10.5 px, so the small-text threshold is 4.5:1.
 *
 * The same measurement covers `--text-muted`, the quiet grey used for group
 * labels and table headers: it must clear 4.5:1 against `--surface-2`, the
 * strictest surface it lands on (issue #1487).
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const THEME_DIR = join(
  dirname(fileURLToPath(import.meta.url)),
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/css/themes',
);

const INKS = ['pos', 'neg', 'warn', 'info', 'accent', 'purple', 'magenta', 'tools', 'lime', 'decision'];
/** Every surface a tag can sit on: the page, a card, and a hovered row. */
const SEATS = ['--bg', '--surface', '--surface-2'];
const THRESHOLD = 4.5;
const FILL_ALPHA = 0.08;

/** Every theme file, by scheme name. */
const THEMES = readdirSync(THEME_DIR)
  .filter((name) => name.endsWith('.css'))
  .sort()
  .map((name) => ({ scheme: name.replace(/\.css$/, ''), text: readFileSync(join(THEME_DIR, name), 'utf8') }));

/** The declarations of `[data-scheme="<scheme>"][data-mode="<mode>"]`. */
export function modeTokens(css, scheme, mode) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const at = text.indexOf(`[data-scheme="${scheme}"][data-mode="${mode}"]`);
  assert.notEqual(at, -1, `no ${scheme}/${mode} block`);
  const open = text.indexOf('{', at);
  const close = text.indexOf('}', open);
  const out = {};
  for (const decl of text.slice(open + 1, close).split(';')) {
    const colon = decl.indexOf(':');
    if (colon === -1) continue;
    out[decl.slice(0, colon).trim()] = decl.slice(colon + 1).trim();
  }
  return out;
}

/** `#rgb` / `#rrggbb` to a channel triple. */
function rgb(hex) {
  const h = hex.trim().replace(/^#/, '');
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h;
  assert.match(full, /^[0-9a-fA-F]{6}$/, `not a hex colour: ${hex}`);
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
}

/** WCAG 2.1 relative luminance. */
function luminance([r, g, b]) {
  const lin = (v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** The painted fill: the ink at 8 % over *base*, 8-bit, as the browser does. */
function fillOver(ink, base) {
  return ink.map((c, i) => Math.round(FILL_ALPHA * c + (1 - FILL_ALPHA) * base[i]));
}

// ---- the checker --------------------------------------------------------

/**
 * Every tag pair, per scheme and mode, on each of its three seats.
 * Returns `{key, ratio}` entries; the caller asserts on the minimum.
 */
export function tagPairs(themes) {
  const out = [];
  for (const { scheme, text } of themes) {
    for (const mode of ['light', 'dark']) {
      const t = modeTokens(text, scheme, mode);
      const seats = Object.fromEntries(SEATS.map((seat) => [seat, rgb(t[seat])]));
      for (const ink of INKS) {
        const value = t[`--tag-${ink}`];
        assert.ok(value, `${scheme}/${mode} is missing --tag-${ink}`);
        for (const [seat, base] of Object.entries(seats)) {
          out.push({
            key: `${scheme}/${mode} --tag-${ink} on ${seat}`,
            ratio: contrast(rgb(value), fillOver(rgb(value), base)),
          });
        }
      }
    }
  }
  return out;
}

/** `--text-muted` against `--surface-2` in every scheme and mode. */
export function mutedPairs(themes) {
  return themes.flatMap(({ scheme, text }) =>
    ['light', 'dark'].map((mode) => {
      const t = modeTokens(text, scheme, mode);
      assert.ok(t['--text-muted'], `${scheme}/${mode} is missing --text-muted`);
      assert.ok(t['--surface-2'], `${scheme}/${mode} is missing --surface-2`);
      return {
        key: `${scheme}/${mode} --text-muted on --surface-2`,
        ratio: contrast(rgb(t['--text-muted']), rgb(t['--surface-2'])),
      };
    }),
  );
}

// ---- the facts ----------------------------------------------------------

test('the checker sees every theme file and its inks', () => {
  assert.ok(THEMES.length >= 3, `expected the theme sheets, found ${THEMES.map((t) => t.scheme).join(', ')}`);
  // A parser that matched nothing would pass every assertion below.
  const pairs = tagPairs(THEMES);
  assert.equal(pairs.length, THEMES.length * 2 * INKS.length * SEATS.length);
  assert.equal(pairs.length, 180, '3 schemes x 2 modes x 10 inks x 3 seats');
});

test('every tag ink clears 4.5:1 on its own fill, on each of its three seats', () => {
  const pairs = tagPairs(THEMES);
  const failing = pairs.filter((p) => p.ratio < THRESHOLD);
  assert.deepEqual(
    failing.map((p) => `${p.key}: ${p.ratio.toFixed(3)}`),
    [],
    'a tag ink reads below 4.5:1 against its own composited fill',
  );
});

test('the quiet grey clears 4.5:1 on --surface-2, in every scheme and mode', () => {
  const pairs = mutedPairs(THEMES);
  assert.equal(pairs.length, THEMES.length * 2);
  const failing = pairs.filter((p) => p.ratio < THRESHOLD);
  assert.deepEqual(failing.map((p) => `${p.key}: ${p.ratio.toFixed(3)}`), []);
});

test('the measurement itself is right', () => {
  // White on black is 21:1 and a colour against itself is 1:1; if these drift,
  // every threshold above is meaningless.
  assert.equal(Math.round(contrast([255, 255, 255], [0, 0, 0])), 21);
  assert.equal(contrast([18, 52, 86], [18, 52, 86]), 1);
  // An 8 % fill of #000 over #fff is (235,235,235), and #000 on it is 17.6:1.
  assert.deepEqual(fillOver([0, 0, 0], [255, 255, 255]), [235, 235, 235]);
  assert.equal(contrast([0, 0, 0], [235, 235, 235]).toFixed(1), '17.6');
});

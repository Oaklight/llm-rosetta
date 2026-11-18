/**
 * Every shim card head paints the same shape, whatever the text is.
 *
 * The head used to be one flex line — mark, name, badges, counts, chevron —
 * so the layout followed the length of the vendor's name: "OpenCode Zen"
 * wrapped to two lines, the counts wrapped under it, and that one card stood
 * taller than its neighbours while its badges slid to wherever the name
 * ended.  Nobody noticed until the catalogue gained a vendor with two words
 * in its name.
 *
 * The shape these cases pin: one grid, the title row (mark, name, badges)
 * over the counts row, the name ellipsizing inside its column, and the counts
 * never wrapping.  The guards are text-level because the builders are module
 * privates and the tests run without a DOM; the two halves that can drift are
 * pinned separately so a fix to one does not have to touch the other.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin');
const CSS = readFileSync(join(ADMIN, 'css/components.css'), 'utf8');
const JS = readFileSync(join(ADMIN, 'js/tabs/shims-page.js'), 'utf8');

/** One rule's body, by selector, from the first stylesheet that states it. */
function rule(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = CSS.match(new RegExp(`(?:^|\\n)${escaped}\\s*\\{([^}]*)\\}`));
  assert.ok(match, `${selector} is still styled`);
  return match[1];
}

/** A function's body, from its signature to the first `}` at column 0. */
function bodyOf(signature) {
  const start = JS.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = JS.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return JS.slice(start, end);
}

const HEADS = {
  vendor: bodyOf('function _vendorCard('),
  package: bodyOf('function _packageCard('),
};

test('both heads are the same two-row grid', () => {
  for (const [which, selector] of [
    ['vendor', '.shim-vendor-head'],
    ['package', '.shim-card-head'],
  ]) {
    const body = rule(selector);
    assert.match(body, /display:\s*grid/, `${which} head is a grid, not a flex line`);
    assert.match(body, /grid-template-rows:\s*auto auto/, `${which} head has two rows`);
    // The middle column is the only one that may shrink, which is what stops
    // the counts from being pushed into a wrap.
    assert.match(body, /minmax\(0,\s*1fr\)/, `${which} head gives the title the shrink`);
  }
});

test('the name shortens instead of reflowing its card', () => {
  const body = rule('.shim-vendor-name');
  assert.match(body, /white-space:\s*nowrap/, 'a name never takes a second line');
  assert.match(body, /text-overflow:\s*ellipsis/, 'a name too long says so with an ellipsis');
  assert.match(body, /overflow:\s*hidden/, 'the ellipsis needs the overflow clipped');
});

test('the counts never wrap', () => {
  assert.match(rule('.shim-vendor-meta'), /white-space:\s*nowrap/);
});

/** The inner markup of the first `<span class="<cls>">` in *markup*.
 *
 *  A tag-depth scan rather than a regex: the title row holds a name span, so
 *  a `.*?` match would stop at the first `</span>` and the badge after it
 *  would read as "outside the row".
 */
function spanInner(markup, cls) {
  const open = markup.indexOf(`<span class="${cls}"`);
  assert.notEqual(open, -1, `the ${cls} span is in the markup`);
  let depth = 0;
  let at = open;
  for (;;) {
    const nextOpen = markup.indexOf('<span', at + 1);
    const nextClose = markup.indexOf('</span>', at + 1);
    assert.notEqual(nextClose, -1, `the ${cls} span closes`);
    if (nextOpen !== -1 && nextOpen < nextClose) {
      depth += 1;
      at = nextOpen;
      continue;
    }
    if (depth === 0) return markup.slice(markup.indexOf('>', open) + 1, nextClose);
    depth -= 1;
    at = nextClose;
  }
}

test('the badges ride with the name, not beside it', () => {
  const heads = { vendor: bodyOf('function _vendorCard('), package: bodyOf('function _packageCard(') };
  assert.match(spanInner(heads.vendor, 'shim-vendor-title'), /\$\{badges\}/,
    'the vendor card draws its badges inside the title row');
  assert.match(spanInner(heads.package, 'shim-vendor-title'), /\$\{tierTag\}/,
    'the package card draws its tier inside the title row');
  for (const [which, body] of Object.entries(heads)) {
    assert.match(spanInner(body, 'shim-vendor-title'), /shim-vendor-name/,
      `${which} card puts the name in the title row`);
  }
});

test('a toolbox keeps the whole name available', () => {
  for (const [which, body] of Object.entries(HEADS)) {
    assert.match(
      body,
      /class="shim-vendor-name" title="\$\{esc\(/,
      `${which} card names the whole vendor in a tooltip, since the cell ellipsizes`,
    );
  }
});

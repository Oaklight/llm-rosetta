/**
 * The provider menu is placed by measuring, and its width is the block's.
 *
 * The two rules this file pins are the ones a browser measurement found and a
 * later edit can quietly undo:
 *
 * 1. a menu wider than the viewport cannot be placed by `right` alone — the
 *    fixed containing block is narrower than `clientWidth` by a classic
 *    scrollbar, and the panel's own width rule has to use the block's unit
 *    (`%`), not `100vw`, or the panel is a scrollbar wider than the box the
 *    clamp measures against;
 * 2. the guard after placement clears `right` before setting `left`.  With
 *    both margins written, the browser drops `right` — but which one it
 *    honours is not something the code should leave to the layout.
 *
 * The geometry itself (a placed box at 370-550 px) is checked in the browser;
 * this file runs the source and the stylesheet, so it survives the next
 * refactor without one.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin');
const UTILS = readFileSync(join(ADMIN, 'js/core/utils.js'), 'utf8');
const COMPONENTS = readFileSync(join(ADMIN, 'css/components.css'), 'utf8')
  .replace(/\/\*[\s\S]*?\*\//g, '');

/** The `positionMenu` body, from its signature to the first `}` at column 0. */
function positionMenuBody() {
  const start = UTILS.indexOf('function positionMenu(');
  assert.notEqual(start, -1, 'positionMenu still exists');
  const end = UTILS.indexOf('\n}\n', start);
  assert.notEqual(end, -1, 'positionMenu closes at the top level');
  return UTILS.slice(start, end);
}

test('the phone width is the containing block\'s, not 100vw', () => {
  const media = COMPONENTS.slice(COMPONENTS.indexOf('@media (max-width: 560px)'));
  const rule = /\.provider-menu\s*\{([^}]*)\}/.exec(media);
  assert.ok(rule, 'the phone width rule is still there');
  assert.match(rule[1], /width:\s*calc\(100%\s*-\s*32px\)/);
  assert.doesNotMatch(
    COMPONENTS,
    /\.provider-menu\s*\{[^}]*width:\s*calc\(100vw/,
    '100vw includes the scrollbar the fixed containing block excludes',
  );
});

test('the placement guard measures the box it placed', () => {
  const body = positionMenuBody();
  assert.match(
    body,
    /menu\.getBoundingClientRect\(\)\.left\s*<\s*EDGE/,
    'the sums cannot see a containing block narrower than clientWidth',
  );
  assert.match(body, /menu\.style\.left\s*=\s*EDGE/, 'the pin is the left margin');
});

test('the pin clears right before it sets left', () => {
  const body = positionMenuBody();
  const guard = body.slice(body.indexOf('if (menu.getBoundingClientRect()'));
  assert.ok(guard.startsWith('if ('), 'the guard is the branch under test');
  const clear = guard.indexOf("menu.style.right = ''");
  const set = guard.indexOf('menu.style.left =');
  assert.ok(clear !== -1, 'the guard clears the opposite margin');
  assert.ok(set !== -1 && clear < set, 'and it clears before it sets');
});

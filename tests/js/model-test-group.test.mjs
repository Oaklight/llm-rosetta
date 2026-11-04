/**
 * One test button for every place a model is tested.
 *
 * The drawer's single-provider row rendered a bare Test button while the list
 * row and the multi-provider dropdown rendered the button with its kind menu.
 * A review of the fix mutated each connection and the suite stayed green, so
 * this file reads the source and holds the three call sites to the shared
 * builder.
 *
 * Text-level on purpose, like `provider-list-wiring`: no DOM is needed to
 * check that a call site exists and passes its provider pin.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const MODELS = readFileSync(
  join(
    HERE,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/models.js',
  ),
  'utf8',
);

/** A top-level function's body, from its signature to the first `}` at column 0. */
function bodyOf(text, signature) {
  const start = text.indexOf(signature);
  assert.notEqual(start, -1, `${signature} still exists`);
  const end = text.indexOf('\n}\n', start);
  assert.notEqual(end, -1, `${signature} closes at the top level`);
  return text.slice(start, end);
}

test('the shared builder renders the button, the caret and the menu', () => {
  const body = bodyOf(MODELS, 'function _testGroup(');
  assert.match(body, /btn-test/, 'the builder renders the primary test button');
  assert.match(body, /toggleTestMenu\(this\)/, 'the builder renders the caret');
  assert.match(body, /_testMenuItems\(/, 'the caret opens the kind menu');
});

test('a pinned test targets the provider it was clicked on', () => {
  const body = bodyOf(MODELS, 'function _testGroup(');
  assert.match(
    body,
    /runTest\(\$\{n\},\$\{kind\},null,\$\{jsArg\(provider\)\}\)/,
    'the provider name is the fourth argument of runTest, escaped for the attribute',
  );
});

test('the drawer row for a single provider uses the shared builder', () => {
  const body = bodyOf(MODELS, 'function _detailProviderRows(');
  assert.match(body, /_testGroup\(/, 'the single-provider row uses _testGroup');
  assert.match(
    body,
    /_testGroup\(name, modelType, opts, p\.name\)/,
    'the test is pinned to that provider',
  );
  assert.doesNotMatch(
    body,
    /<div class="test-group"><button[^>]*>.*?<\/button><\/div>/,
    'no bare Test button without its menu is left behind',
  );
});

test('the multi-provider dropdown uses the shared builder', () => {
  const body = bodyOf(MODELS, 'function _providerMenuRows(');
  assert.match(body, /_testGroup\(name, modelType, opts, p\.name\)/);
  assert.doesNotMatch(
    body,
    /<div class="test-group">/,
    'the dropdown no longer assembles the group inline',
  );
});

test('the model list row uses the shared builder', () => {
  const body = bodyOf(MODELS, 'function renderModels(');
  assert.match(
    body,
    /_testGroup\(name, modelType, \{hasTools, hasVision, hasReasoning\}, null/,
    'the row tests through routing, so it passes no provider pin',
  );
});

test('a lane that narrows its capabilities narrows its test menu', () => {
  const helper = bodyOf(MODELS, 'function _laneOpts(');
  assert.match(helper, /detail\.capabilities/, "the lane's own list wins");
  assert.match(helper, /info\.capabilities/, 'the model list is the fallback');
  assert.match(
    bodyOf(MODELS, 'function _providerMenuRows('),
    /const opts = _laneOpts\(info, p\)/,
    'each dropdown row reads its own lane',
  );
  assert.match(
    bodyOf(MODELS, 'function _detailProviderRows('),
    /const opts = _laneOpts\(info, p\)/,
    'the drawer row reads its own lane',
  );
});

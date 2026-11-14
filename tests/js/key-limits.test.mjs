/**
 * What a key's two limit fields accept, and that the client parser and the
 * gateway's own agree.
 *
 * The dialog refuses a field it cannot read and the save never leaves the
 * browser; the server refuses the same two fields on its side.  The quota
 * syntax has two homes — `core/quota.js` for the admin, the vendored
 * `ratelimit` module for the gateway — so one corpus runs through both.
 *
 * Run with `node --test tests/js/` (or `make test-js`).
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { test } from 'node:test';

import {
  MODEL_NAME_RE,
  badModelName,
  keyLimitTags,
  keyLimitsMarkHtml,
  modelTokens,
  modelsForSave,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/key-limits.js';
import { validateQuotas } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/quota.js';

const VENDOR = new URL(
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/_vendor/ratelimit.py',
  import.meta.url,
);

test('a blank models field allows every model', () => {
  assert.deepEqual(modelTokens(''), []);
  assert.equal(badModelName(''), null);
});

test('a model name may carry the vendor punctuation', () => {
  for (const ok of ['gpt-4o', 'claude-3-5-sonnet', 'openai/gpt-4o', 'ft:gpt-4o:acme:1', 'a_b.c@d']) {
    assert.equal(badModelName(ok), null, ok);
    assert.match(ok, MODEL_NAME_RE);
  }
});

test('the whole-model marker is not a model name', () => {
  assert.equal(badModelName('*'), null);
  assert.equal(badModelName('*, gpt-4o'), null);
});

test('the first bad name is the one reported', () => {
  // A space is a separator, not part of a name: "bad name" is two names.
  assert.equal(badModelName('gpt-4o, bad name'), null);
  assert.equal(badModelName('gpt-4o, name!'), 'name!');
  assert.equal(badModelName('name! gpt-4o'), 'name!');
});

test('a quota field accepts the settings form and refuses the rest', () => {
  assert.equal(validateQuotas('').ok, true);
  assert.equal(validateQuotas('60/m').ok, true);
  assert.equal(validateQuotas('30/m, 500/h').ok, true);
  assert.equal(validateQuotas('60x').ok, false);
  assert.equal(validateQuotas('30/m, nope').ok, false);
  // Zero parses but would refuse every request; the field says so here
  // rather than letting the server's 400 arrive as a toast.
  assert.equal(validateQuotas('0/m').ok, false);
  assert.equal(validateQuotas('5/m, 0/h').ok, false);
});

test('the two unit tables are the same table', () => {
  const source = fs.readFileSync(VENDOR, 'utf8');
  const block = source.match(/_UNIT_MAP[^=]*=\s*\{([^}]*)\}/s);
  assert.ok(block, 'the vendored _UNIT_MAP is where this test expects it');
  const vendor = Object.fromEntries(
    [...block[1].matchAll(/"([a-z]+)":\s*([\d.]+)/g)].map(([, unit, secs]) => [unit, Number(secs)]),
  );
  const js = Object.fromEntries(
    Object.entries(
      // The module's own table, read through a quota of each unit.
      Object.fromEntries(Object.keys(vendor).map((u) => [u, validateQuotas(`1/${u}`).ok])),
    ),
  );
  for (const [unit, secs] of Object.entries(vendor)) {
    assert.equal(js[unit], true, `the client knows unit '${unit}' (${secs}s)`);
  }
  // And nothing the client knows that the gateway does not.
  for (const unit of ['s', 'sec', 'second', 'm', 'min', 'minute', 'h', 'hr', 'hour', 'd', 'day']) {
    assert.ok(unit in vendor, `the gateway knows unit '${unit}'`);
  }
});

test('the corpus parses the same way on both sides', () => {
  const source = fs.readFileSync(VENDOR, 'utf8');
  const re = source.match(/_QUOTA_RE = re\.compile\(\s*([\s\S]*?)\n\)/);
  assert.ok(re, 'the vendored _QUOTA_RE is where this test expects it');
  const python_pattern = [...re[1].matchAll(/"((?:[^"\\]|\\.)*)"/g)]
    .map(([, part]) => part)
    .join('')
    .replace(/\\\\/g, '\\');
  const parity = new RegExp(python_pattern, 'i');
  const units = new Set(
    Object.keys(
      Object.fromEntries(
        [...source.match(/_UNIT_MAP[^=]*=\s*\{([^}]*)\}/s)[1].matchAll(
          /"([a-z]+)":\s*([\d.]+)/g,
        )].map(([, unit]) => [unit, true]),
      ),
    ),
  );
  // `parse_quota`, as the gateway runs it: the pattern, then the unit table.
  const vendorAccepts = (line) => {
    if (line.trim() === '') return true;
    const m = parity.exec(line);
    if (!m) return false;
    let unit = m[2].toLowerCase();
    if (unit.length > 1 && unit.endsWith('s')) unit = unit.slice(0, -1);
    return units.has(unit);
  };

  // The same corpus, graded by the gateway's pattern and by the client's
  // parser: they must agree on every line.
  const corpus = [
    '60/m', '1/s', '500/h', '10000/d', '5/M', '5/mins', '60 per minute',
    '10 per minute burst 20', '0/m', '1.5/m', '1e3/m', '-1/m', '60x', '5/w',
    '5/mo', '', ' / ', '60 / m',
  ];
  for (const line of corpus) {
    const vendor_ok = vendorAccepts(line);
    // The one deliberate difference: the gateway parses "0/m", but a stored
    // zero cannot build a limiter, so the field refuses it up front.
    if (/^(\d+)\s*(?:\/|per)\s*[a-z]+(?:\s+burst\s+\d+)?$/i.test(line) &&
        Number(line.match(/^\d+/)[0]) < 1) {
      assert.equal(validateQuotas(line).ok, false, `${JSON.stringify(line)} has a floor`);
      continue;
    }
    const client_ok = validateQuotas(line).ok;
    assert.equal(client_ok, vendor_ok, `${JSON.stringify(line)}: client ${client_ok}, gateway ${vendor_ok}`);
  }
});

const KEYS_JS = new URL(
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/keys.js',
  import.meta.url,
);

/** The body of one top-level function in keys.js. */
function bodyOf(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} is in keys.js`);
  const end = source.indexOf('\n}', start);
  assert.ok(end > start, `${name} ends where this test expects it`);
  return source.slice(start, end);
}

/**
 * The guard itself: both validators must be called inside the `if` that
 * returns, so a `false` from either stops the request.  A version that called
 * them without the `if` would pass a test that only counted call order.
 */
const GUARD = /if \(!validateKeyModels\('([A-Za-z]+)'\) \|\| !validateKeyRate\('([A-Za-z]+)'\)\) return;/;

test("both save paths refuse on the validators' answer", () => {
  const source = fs.readFileSync(KEYS_JS, 'utf8');
  for (const [fn, models, rate] of [
    ['_generateKey', 'keyModels', 'keyRate'],
    ['_saveKeyLabel', 'keyModelsInput', 'keyRateInput'],
  ]) {
    const body = bodyOf(source, fn);
    const guard = body.match(GUARD);
    assert.ok(guard, `${fn} guards on both validators`);
    assert.equal(guard[1], models, `${fn} validates its models field`);
    assert.equal(guard[2], rate, `${fn} validates its rate field`);
    const post = body.search(/api\.(post|put)/);
    assert.ok(body.indexOf(guard[0]) < post, `${fn} refuses before it sends`);
  }
});

// The row's tags: a small dictionary stands in for i18n, so the shape is what
// is under test rather than the wording.
const dict = {
  'keys.noModel': 'no model',
  'keys.modelUnit': 'models',
  'keys.modelUnitOne': 'model',
  'label.keyModels': 'Allowed models',
};
const t = (key) => dict[key] ?? key;

test('a key with every model shows no model tag', () => {
  assert.deepEqual(keyLimitTags({allowed_models: ['*'], rate_limit: ''}, t), []);
  assert.deepEqual(keyLimitTags({rate_limit: ''}, t), []);
});

test('a key with no model says so, rather than looking like every model', () => {
  assert.deepEqual(keyLimitTags({allowed_models: [], rate_limit: ''}, t), [
    {text: 'no model', title: 'Allowed models'},
  ]);
});

test('a scoped key counts its models, one or many', () => {
  assert.equal(keyLimitTags({allowed_models: ['m-a']}, t)[0].text, '1 model');
  assert.equal(keyLimitTags({allowed_models: ['m-a', 'm-b']}, t)[0].text, '2 models');
  assert.equal(keyLimitTags({allowed_models: ['m-a', 'm-b']}, t)[0].title, 'm-a, m-b');
});

test('a multi-window quota shows its first window and a count', () => {
  const [tag] = keyLimitTags({allowed_models: ['*'], rate_limit: '30/m, 500/h'}, t);
  assert.deepEqual(tag, {text: '30/m +1', title: '30/m, 500/h'});
});

test('the row markup escapes what it puts in an attribute', () => {
  const html = keyLimitsMarkHtml({allowed_models: ['a"b<c>']}, t);
  assert.equal(html, '<span class="tag is-plain" title="a&quot;b&lt;c&gt;">1 model</span>');
  assert.equal(keyLimitsMarkHtml({allowed_models: ['*']}, t), '');
  assert.match(
    keyLimitsMarkHtml({allowed_models: [], rate_limit: '30/m, 500/h'}, t),
    /title="Allowed models">no model<\/span> <span class="tag is-plain" title="30\/m, 500\/h">30\/m \+1<\/span>/,
  );
});

test('an untouched models field sends back what it loaded', () => {
  // Stored [] and stored ["*"] both open as a blank field; only the loaded
  // value tells them apart, and a label-only save must not widen the key.
  assert.deepEqual(modelsForSave('', []), []);
  assert.deepEqual(modelsForSave('', ['*']), ['*']);
  assert.deepEqual(modelsForSave('', undefined), ['*']);
  assert.deepEqual(modelsForSave('m-a', ['m-a']), ['m-a']);
  assert.deepEqual(modelsForSave('m-a, m-b', ['m-b', 'm-a']), ['m-b', 'm-a']);
});

test('an edited models field is read as written', () => {
  assert.deepEqual(modelsForSave('m-b', []), ['m-b']);
  assert.deepEqual(modelsForSave('m-b', ['*']), ['m-b']);
  // Clearing the field is the explicit "every model" the label promises.
  assert.deepEqual(modelsForSave('', ['m-a']), ['*']);
});

test('the dialog hands the loaded list to the save', () => {
  // The pure function is pinned above; what it is given is the other half of
  // the round trip.  Forgetting the loaded value here re-opens the widen with
  // no other test in the suite noticing.
  const source = fs.readFileSync(KEYS_JS, 'utf8');
  const open = bodyOf(source, 'editKeyLabel');
  assert.match(
    open,
    /_editingModels = Array\.isArray\(key\.allowed_models\) \? key\.allowed_models : \['\*'\]/,
    'the dialog keeps the list it loaded',
  );
  const save = bodyOf(source, '_saveKeyLabel');
  assert.match(
    save,
    /modelsForSave\(\s*document\.getElementById\('keyModelsInput'\)\.value,\s*_editingModels,?\s*\)/,
    'the save sends the loaded list back',
  );
});

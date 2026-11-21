/**
 * Manifest sources in the Shims page (#1660 S9): the selector rules the
 * editor writes, the facts a vendor card shows, and both rendered against a
 * stubbed page.
 *
 * The rules are tested directly (`core/shim-sources.js` is pure).  The card
 * and the editor are then driven through their real modules with the DOM and
 * `fetch` stubbed, so what is asserted is the markup the operator sees and the
 * body the Save button sends — not a re-implementation of either.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DEFAULT_SOURCES, addSource, declareSource, deselectVendor, parseSelector, pickerFor,
  removeSource, selectVendor, sharedSelectors, shownSources, sourceSelection, sourceTags,
  vendorFacts, vendorMatches, vendorStateKey, withSourceSelection,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-sources.js';

const ADMIN = '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js';

// ── The selector grammar ───────────────────────────────────────────────

test('the selector grammar matches the gateway', () => {
  assert.deepEqual(parseSelector('opendesign'),
    { raw: 'opendesign', source: null, kind: 'vendor', value: 'opendesign' });
  assert.deepEqual(parseSelector('tag:tier:free'),
    { raw: 'tag:tier:free', source: null, kind: 'tag', value: 'tier:free' });
  assert.deepEqual(parseSelector('contrib:tag:tier:free'),
    { raw: 'contrib:tag:tier:free', source: 'contrib', kind: 'tag', value: 'tier:free' });
  assert.deepEqual(parseSelector('argonne:*'),
    { raw: 'argonne:*', source: 'argonne', kind: 'all', value: '' });
  assert.equal(parseSelector('Bad Vendor'), null);
  assert.equal(parseSelector(''), null);
});

test('the first entry that picks a vendor speaks for it', () => {
  const install = ['argonne:*', 'tag:tier:free', 'contrib:opendesign'];

  assert.equal(pickerFor(install, 'argonne', 'alcf'), 'argonne:*');
  assert.equal(pickerFor(install, 'contrib', 'llm7', ['tier:free']), 'tag:tier:free');
  assert.equal(pickerFor(install, 'contrib', 'opendesign', ['tier:subscription']), 'contrib:opendesign');
  assert.equal(pickerFor(install, 'contrib', 'ovhcloud', []), null);
});

// ── The editor's writes ────────────────────────────────────────────────

test('a source selection reads only the entries qualified with it', () => {
  const install = ['contrib:opendesign', 'contrib:tag:tier:free', 'argonne:alcf', 'kilo_free'];

  assert.deepEqual(sourceSelection(install, 'contrib'),
    { all: false, tags: ['tier:free'], vendors: ['opendesign'] });
  assert.deepEqual(sharedSelectors(install), ['kilo_free']);
});

test('rewriting one source keeps every other entry in place', () => {
  const install = ['kilo_free', 'contrib:opendesign', 'argonne:alcf', 'contrib:llm7'];

  const next = withSourceSelection(install, 'contrib', { all: false, tags: ['tier:free'], vendors: ['opendesign'] });

  assert.deepEqual(next, ['kilo_free', 'contrib:tag:tier:free', 'contrib:opendesign', 'argonne:alcf']);
});

test('"every vendor" replaces the source\'s tag and vendor entries', () => {
  const next = withSourceSelection(['argonne:alcf', 'argonne:argo'], 'argonne',
    { all: true, tags: [], vendors: ['alcf'] });

  assert.deepEqual(next, ['argonne:*']);
});

test('an emptied selection removes the source\'s entries', () => {
  assert.deepEqual(withSourceSelection(['argonne:alcf', 'x'], 'argonne',
    { all: false, tags: [], vendors: [] }), ['x']);
});

test('a duplicate elsewhere in the list survives an edit of another source', () => {
  // #1663 review: only the edited source's entries are rebuilt.
  const next = withSourceSelection(['llm7', 'llm7', 'contrib:a', 'argonne:x', 'argonne:x'], 'contrib',
    { all: false, tags: [], vendors: ['b', 'b'] });

  assert.deepEqual(next, ['llm7', 'llm7', 'contrib:b', 'argonne:x', 'argonne:x']);
});

test('selecting a vendor adds a qualified entry once', () => {
  const install = selectVendor(['argonne:alcf'], 'contrib', 'llm7', ['tier:free']);
  assert.deepEqual(install, ['argonne:alcf', 'contrib:llm7']);

  assert.deepEqual(selectVendor(install, 'contrib', 'llm7', ['tier:free']), install);
});

test('deselecting removes only this source\'s entry for the vendor', () => {
  const out = deselectVendor(['contrib:llm7', 'other:llm7', 'argonne:alcf'], 'contrib', 'llm7');

  assert.equal(out.refuse, null);
  assert.deepEqual(out.install, ['other:llm7', 'argonne:alcf']);
});

test('deselecting refuses while a tag entry still picks the vendor', () => {
  const before = ['contrib:llm7', 'contrib:tag:tier:free'];
  const out = deselectVendor(before, 'contrib', 'llm7', ['tier:free']);

  assert.deepEqual(out.install, before, 'nothing is changed');
  assert.equal(out.refuse.key, 'shimSrc.stillSelected');
  assert.equal(out.refuse.params.selector, 'contrib:tag:tier:free');
});

test('an unqualified entry is shared by every source, so the card refuses it', () => {
  // #1663 review: a bare `llm7` reads every source; removing it from one
  // source's card would also drop the vendor from the others.
  const before = ['llm7', 'contrib:llm7', 'other:x'];
  const out = deselectVendor(before, 'contrib', 'llm7');

  assert.deepEqual(out.install, before);
  assert.equal(out.refuse.key, 'shimSrc.sharedEntry');
  assert.equal(out.refuse.params.selector, 'llm7');
});

test('a preset behind an explicit entry refuses the removal', () => {
  // #1663 review: removing `contrib:opendesign` would report success while
  // `presets: [opendesign]` keeps installing the vendor.
  const out = deselectVendor(['contrib:opendesign'], 'contrib', 'opendesign', [], ['opendesign']);

  assert.deepEqual(out.install, ['contrib:opendesign']);
  assert.equal(out.refuse.key, 'shimSrc.byPreset');
  assert.equal(out.refuse.params.preset, 'opendesign');
});

test('nothing to remove is a refusal, not a silent write', () => {
  const out = deselectVendor(['argonne:alcf'], 'contrib', 'llm7');

  assert.equal(out.refuse.key, 'shimSrc.nothingToRemove');
});

test('selecting from an implied source writes the declared list', () => {
  // #1663 review: judged against the DECLARED list, not the shown defaults.
  assert.deepEqual(declareSource({ install: [] }, 'contrib'),
    { sources: ['contrib', 'argonne'], changed: true });
  assert.deepEqual(declareSource({ sources: ['argonne'] }, 'contrib'),
    { sources: ['argonne', 'contrib'], changed: true });
  assert.deepEqual(declareSource({ sources: ['contrib'] }, 'contrib'),
    { sources: ['contrib'], changed: false });
  const url = 'https://shims.example/dist/manifest.json';
  assert.deepEqual(declareSource({}, url), { sources: ['contrib', 'argonne', url], changed: true });
});

test('a config that relied on the defaults keeps them when a source is added', () => {
  assert.deepEqual(shownSources({}), [...DEFAULT_SOURCES]);

  const url = 'https://shims.example/dist/manifest.json';
  assert.deepEqual(addSource({ install: ['x'] }, url),
    { sources: ['contrib', 'argonne', url], added: true });
  assert.deepEqual(addSource({ sources: ['contrib'] }, 'contrib'),
    { sources: ['contrib'], added: false });
});

test('removing a source drops the entries qualified with its id', () => {
  const out = removeSource({ sources: ['contrib', 'argonne'],
    install: ['argonne:*', 'contrib:llm7', 'tag:tier:free'] }, 'argonne', 'argonne');

  assert.deepEqual(out, { sources: ['contrib'], install: ['contrib:llm7', 'tag:tier:free'] });
});

// ── What a card says ───────────────────────────────────────────────────

const SRC = { id: 'contrib', label: 'Contrib' };

test('an installed vendor names its source, version and requires match', () => {
  const facts = vendorFacts({
    installed: '0.3.0', installable: '0.4.0', newest: '0.5.0', update_available: true,
    requires_ok: true, newest_problem: 'requires surface 2, this core provides surface 1',
  }, SRC);

  assert.deepEqual(facts.map((f) => f.key), [
    'shimSrc.fromSource', 'shimSrc.installedVersion', 'shimSrc.updateTo',
    'shimSrc.newestVersion', 'shimSrc.requiresOk',
  ]);
  assert.equal(facts[0].params.source, 'Contrib');
  assert.equal(facts[3].cls, 'is-unknown');
  assert.match(facts[3].title, /surface 2/);
});

test('a kept install whose requires no longer match is a warning', () => {
  const facts = vendorFacts({ installed: '0.1.0', requires_ok: false, installed_problem: 'requires IR 9' }, SRC);
  const bad = facts.find((f) => f.key === 'shimSrc.requiresBad');

  assert.ok(bad);
  assert.equal(bad.cls, 'is-unknown');
  assert.equal(bad.title, 'requires IR 9');
});

test('a skipped vendor carries the reason; nothing installable says so', () => {
  const skipped = vendorFacts({ skipped: 'not in the manifest', state: 'skipped' }, SRC);
  assert.ok(skipped.some((f) => f.key === 'shimSrc.skipped' && f.params.reason === 'not in the manifest'));
  assert.equal(vendorStateKey({ state: 'skipped' }), 'shimSrc.state.skipped');

  const none = vendorFacts({ newest: '2.0.0', newest_problem: 'requires surface 2' }, SRC);
  assert.ok(none.some((f) => f.key === 'shimSrc.noInstallable'));
});

test('tags and search read the vendor and its source', () => {
  const src = { id: 'contrib', vendors: [{ tags: ['tier:free'] }, { tags: ['auth:device', 'tier:free'] }] };
  assert.deepEqual(sourceTags(src), ['auth:device', 'tier:free']);
  assert.ok(vendorMatches({ vendor: 'llm7', tags: ['tier:free'] }, src, 'free'));
  assert.ok(!vendorMatches({ vendor: 'llm7', tags: [] }, src, 'argo'));
});

// ── The page, rendered ─────────────────────────────────────────────────

/** A minimal element: what the two modules read and write. */
function element(id) {
  return {
    id, innerHTML: '', textContent: '', hidden: false, value: '', checked: false, disabled: false,
    dataset: {}, style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    setAttribute() {}, getAttribute: () => null, removeAttribute() {},
    addEventListener() {}, removeEventListener() {},
    querySelector: () => null, querySelectorAll: () => [], closest: () => null,
    focus() {}, appendChild() {}, insertAdjacentHTML() {},
  };
}

const CATALOGUE = {
  core: { surface: 1, ir: 1, version: '0.14.0' },
  install: ['contrib:opendesign'],
  sources: [
    {
      id: 'contrib', key: 'contrib', label: 'Contrib', url: 'https://cdn.example/contrib/manifest.json',
      repo: 'https://github.com/Oaklight/llm-rosetta-shim-contrib', entry: 'contrib', configured: true,
      ok: true, error: null, stale: false,
      vendors: [
        { vendor: 'llm7', versions: ['0.4.0'], newest: '0.4.0', installable: '0.4.0', installed: null,
          requires_ok: null, maturity: 'beta', since: '2026-11-13', tags: ['tier:free'],
          conformance: { date: null, result: 'none' }, shims: ['llm7--chat_completions'],
          installed_shims: [], core_provides: [], selected_by: null, state: 'available' },
        { vendor: 'opendesign', versions: ['0.3.0'], newest: '0.3.0', installable: '0.3.0',
          installed: '0.3.0', requires_ok: true, maturity: 'beta', since: '2026-11-14',
          tags: ['tier:subscription', 'auth:device'], conformance: { date: null, result: 'none' },
          shims: ['opendesign--messages'], installed_shims: ['opendesign--messages'],
          core_provides: [], selected_by: 'contrib:opendesign', selected_origin: 'install',
          state: 'installed' },
      ],
    },
    {
      id: 'argonne', key: 'argonne', label: 'Argonne', url: 'https://cdn.example/argonne/manifest.json',
      entry: null, configured: false, ok: false, error: 'ConnectionError: down', vendors: [],
    },
  ],
};

let elements = new Map();
const requests = [];

function installPage() {
  elements = new Map();
  const byId = (id) => {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  };
  const storage = () => {
    const map = new Map();
    return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => map.set(k, String(v)),
      removeItem: (k) => map.delete(k), clear: () => map.clear() };
  };
  globalThis.localStorage = storage();
  globalThis.sessionStorage = storage();
  const root = element('root');
  globalThis.document = {
    documentElement: root, body: root, head: root,
    getElementById: byId, querySelector: () => null, querySelectorAll: () => [],
    createElement: () => element('x'), createTextNode: () => element('x'),
    addEventListener() {}, removeEventListener() {},
  };
  globalThis.window = globalThis;
  globalThis.location = { hash: '', pathname: '/', search: '', href: 'http://localhost/' };
  globalThis.history = { replaceState() {}, pushState() {} };
  globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  const config = { shims: { install: ['contrib:opendesign'] }, registered_shims: [], model_types: [] };
  globalThis.fetch = async (url, opts = {}) => {
    requests.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : null });
    let body = {};
    if (url.startsWith('/admin/api/config/shims/source-catalog')) body = CATALOGUE;
    else if (url === '/admin/api/config') body = config;
    else if (url === '/admin/api/config/shims') body = opts.method === 'PUT' ? { ok: true } : { shims: [] };
    else if (url === '/admin/api/config/shims/catalog') body = { packages: [] };
    else if (url === '/admin/api/config/shims/reload') body = { ok: true, status: { failure_count: 0 } };
    return { ok: true, status: 200, headers: { get: () => 'application/json' },
      json: async () => body, text: async () => JSON.stringify(body) };
  };
  return config;
}

test('the Available view gives every vendor of every source a card, grouped by source', async () => {
  installPage();
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;

  await page.loadShimsPage();
  page.setShimsView('available');
  const html = elements.get('shimCatalogGrid').innerHTML;

  // Both sources have a head, in order; the one that did not answer says so.
  assert.ok(html.indexOf('>Contrib<') < html.indexOf('>Argonne<'));
  assert.match(html, /Manifest unavailable: ConnectionError: down/);
  // One card per vendor, keyed by source and vendor.
  assert.match(html, /data-srcvendor="contrib:llm7"/);
  assert.match(html, /data-srcvendor="contrib:opendesign"/);
  // The installed card: source, version, requires, maturity, tags, selector.
  const card = html.slice(html.indexOf('data-srcvendor="contrib:opendesign"'));
  assert.match(card, /from Contrib/);
  assert.match(card, /installed 0\.3\.0/);
  assert.match(card, /requires matches this core/);
  assert.match(card, /beta since 2026-11-14/);
  assert.match(card, />auth:device</);
  assert.match(card, /Selected by contrib:opendesign/);
  // The other one offers to install.
  const other = html.slice(html.indexOf('data-srcvendor="contrib:llm7"'), html.indexOf('data-srcvendor="contrib:opendesign"'));
  assert.match(other, /selectShimSourceVendor\(this, &quot;contrib&quot;, &quot;llm7&quot;\)/);
  assert.match(other, /newest 0\.4\.0/);
});

test('the editor shows each source\'s selection and saves both keys through the config path', async () => {
  const config = installPage();
  const { S } = await import(`${ADMIN}/core/state.js`);
  const editor = await import(`${ADMIN}/tabs/shim-sources-editor.js`);
  S.configData = config;
  globalThis.loadConfig = async () => {};
  globalThis.loadShimsPage = async () => {};
  editor.resetShimManifestEditor();

  await editor.renderShimManifestEditor();
  const host = elements.get('shimManifestEditor');
  // The implied defaults are both listed; the saved selection is ticked.
  assert.match(host.innerHTML, /data-kind="vendor" data-value="opendesign" checked/);
  assert.match(host.innerHTML, /data-kind="vendor" data-value="llm7"(?! checked)/);
  assert.match(host.innerHTML, /data-kind="tag" data-value="tier:free"/);
  assert.match(host.innerHTML, /Manifest unavailable/);
  assert.equal(elements.get('shimManifestSave').disabled, true, 'nothing to save yet');

  // Tick the free tier, untick opendesign.
  editor.shimSourceDraftToggle({ checked: true, dataset: { src: 'contrib', kind: 'tag', value: 'tier:free' } });
  editor.shimSourceDraftToggle({ checked: false, dataset: { src: 'contrib', kind: 'vendor', value: 'opendesign' } });
  assert.equal(elements.get('shimManifestSave').disabled, false);
  assert.equal(elements.get('shimManifestDirty').hidden, false);

  requests.length = 0;
  await editor.saveShimManifestSources(null);
  const put = requests.find((r) => r.method === 'PUT');
  assert.equal(put.url, '/admin/api/config/shims');
  // The config only implied the defaults: the save writes them out.
  assert.deepEqual(put.body, { install: ['contrib:tag:tier:free'], sources: ['contrib', 'argonne'] });
  assert.match(host.innerHTML, /rewrites the config file without its comments/);
  assert.ok(requests.some((r) => r.url === '/admin/api/config/shims/reload' && r.method === 'POST'),
    'the save reloads so the selection installs');
});

test('adding and removing a source edits the draft, and the save writes the list', async () => {
  const config = installPage();
  const { S } = await import(`${ADMIN}/core/state.js`);
  const editor = await import(`${ADMIN}/tabs/shim-sources-editor.js`);
  S.configData = config;
  globalThis.loadConfig = async () => {};
  globalThis.loadShimsPage = async () => {};
  editor.resetShimManifestEditor();
  await editor.renderShimManifestEditor();

  document.getElementById('shimSourceAddInput').value = 'https://shims.example/dist/manifest.json';
  editor.shimSourceDraftAdd();
  assert.match(elements.get('shimManifestEditor').innerHTML, /Save to read this source/);
  editor.shimSourceDraftRemove({ dataset: { key: 'contrib', id: 'contrib' } });

  requests.length = 0;
  await editor.saveShimManifestSources(null);
  const put = requests.find((r) => r.method === 'PUT');
  assert.deepEqual(put.body, {
    install: [],
    sources: ['argonne', 'https://shims.example/dist/manifest.json'],
  });
});

test('Install on a card of an implied default source writes shims.sources', async () => {
  const config = installPage();
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  page.setShimsView('available');
  // The served config has `install` but no `sources`.
  assert.equal(config.shims.sources, undefined);

  requests.length = 0;
  await page.selectShimSourceVendor(null, 'contrib', 'llm7');
  const put = requests.find((r) => r.method === 'PUT');
  assert.deepEqual(put.body, {
    install: ['contrib:opendesign', 'contrib:llm7'],
    sources: ['contrib', 'argonne'],
  });
});

test('Remove refuses, and writes nothing, while a preset still selects the vendor', async () => {
  installPage();
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  const vendor = CATALOGUE.sources[0].vendors[1];
  vendor.selected_presets = ['opendesign'];
  try {
    await page.loadShimsPage();
    requests.length = 0;
    await page.deselectShimSourceVendor(null, 'contrib', 'opendesign');
    assert.equal(requests.filter((r) => r.method === 'PUT').length, 0);
  } finally {
    delete vendor.selected_presets;
  }
});

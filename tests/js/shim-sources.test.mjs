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
  DEFAULT_SOURCES, addSource, declareSource, deselectVendor, excludedBy, followLatest,
  parseSelector, pickerFor, pinVendor,
  notInstalledVendors, removeSource, selectVendor, sharedSelectors, shownSources, sourceSelection,
  sourceTags, vendorFacts, vendorMatches, vendorStateKey, vendorTier, withSourceSelection,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-sources.js';

const ADMIN = '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js';

// ── The selector grammar ───────────────────────────────────────────────

test('the selector grammar matches the gateway', () => {
  const plain = { negate: false, version: null };
  assert.deepEqual(parseSelector('opendesign'),
    { raw: 'opendesign', source: null, kind: 'vendor', value: 'opendesign', ...plain });
  assert.deepEqual(parseSelector('tag:tier:free'),
    { raw: 'tag:tier:free', source: null, kind: 'tag', value: 'tier:free', ...plain });
  assert.deepEqual(parseSelector('contrib:tag:tier:free'),
    { raw: 'contrib:tag:tier:free', source: 'contrib', kind: 'tag', value: 'tier:free', ...plain });
  assert.deepEqual(parseSelector('argonne:*'),
    { raw: 'argonne:*', source: 'argonne', kind: 'all', value: '', ...plain });
  assert.equal(parseSelector('Bad Vendor'), null);
  assert.equal(parseSelector(''), null);
});

test('an exclusion and a pin are vendor entries', () => {
  assert.deepEqual(parseSelector('!contrib:llm7'),
    { raw: '!contrib:llm7', source: 'contrib', kind: 'vendor', value: 'llm7', negate: true, version: null });
  assert.deepEqual(parseSelector('!llm7'),
    { raw: '!llm7', source: null, kind: 'vendor', value: 'llm7', negate: true, version: null });
  assert.deepEqual(parseSelector('contrib:groq@0.5.0'),
    { raw: 'contrib:groq@0.5.0', source: 'contrib', kind: 'vendor', value: 'groq', negate: false, version: '0.5.0' });
  // As the gateway refuses them: only a vendor is excluded or pinned, and an
  // exclusion carries no version.
  for (const bad of ['!tag:tier:free', '!contrib:*', 'contrib:tag:tier:free@1', '!contrib:groq@0.5.0', 'groq@x']) {
    assert.equal(parseSelector(bad), null, bad);
  }
});

test('an explicit entry speaks for its vendor, else the first match', () => {
  const install = ['argonne:*', 'tag:tier:free', 'contrib:opendesign'];

  assert.equal(pickerFor(install, 'argonne', 'alcf'), 'argonne:*');
  assert.equal(pickerFor(install, 'contrib', 'llm7', ['tier:free']), 'tag:tier:free');
  assert.equal(pickerFor(install, 'contrib', 'opendesign', ['tier:subscription']), 'contrib:opendesign');
  assert.equal(pickerFor(install, 'contrib', 'ovhcloud', []), null);
  // A pin listed after a tag still holds the vendor.
  assert.equal(pickerFor(['contrib:tag:tier:free', 'contrib:llm7@0.4.0'], 'contrib', 'llm7', ['tier:free']),
    'contrib:llm7@0.4.0');
});

test('an exclusion wins wherever it is listed', () => {
  for (const install of [
    ['contrib:tag:tier:free', '!contrib:llm7'],
    ['!contrib:llm7', 'contrib:tag:tier:free'],
    ['contrib:llm7', '!llm7'],
  ]) {
    assert.equal(pickerFor(install, 'contrib', 'llm7', ['tier:free']), null, install.join(' '));
  }
  // It names one vendor of one source.
  assert.equal(pickerFor(['contrib:tag:tier:free', '!contrib:llm7'], 'contrib', 'ovhcloud', ['tier:free']),
    'contrib:tag:tier:free');
  assert.equal(pickerFor(['*', '!contrib:llm7'], 'mirror', 'llm7'), '*');
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

test('Install takes the exclusion out, and adds an entry only when nothing else installs it', () => {
  // Removed from a tag, installed back: the list is what it was.
  assert.deepEqual(selectVendor(['contrib:tag:tier:free', '!contrib:llm7'], 'contrib', 'llm7', ['tier:free']),
    ['contrib:tag:tier:free']);
  // An unqualified exclusion is this vendor's too.
  assert.deepEqual(selectVendor(['!llm7'], 'contrib', 'llm7', ['tier:free']), ['contrib:llm7']);
});

test('deselecting removes only this source\'s entry for the vendor', () => {
  const out = deselectVendor(['contrib:llm7', 'other:llm7', 'argonne:alcf'], 'contrib', 'llm7');

  assert.equal(out.refuse, null);
  assert.deepEqual(out.install, ['other:llm7', 'argonne:alcf']);
});

test('a vendor a tag still installs is removed by an exclusion, not refused', () => {
  const out = deselectVendor(['contrib:llm7', 'contrib:tag:tier:free'], 'contrib', 'llm7', ['tier:free']);

  assert.equal(out.refuse, null);
  assert.deepEqual(out.install, ['contrib:tag:tier:free', '!contrib:llm7']);
  assert.equal(pickerFor(out.install, 'contrib', 'llm7', ['tier:free']), null, 'the vendor is out');
});

test('an unqualified entry is excluded for this source only', () => {
  // A bare `llm7` reads every source: the card excludes it here and leaves
  // the entry for the others.
  const out = deselectVendor(['llm7', 'contrib:llm7', 'other:x'], 'contrib', 'llm7');

  assert.deepEqual(out.install, ['llm7', 'other:x', '!contrib:llm7']);
  assert.equal(pickerFor(out.install, 'mirror', 'llm7'), 'llm7');
});

test('a pinned vendor\'s entry goes with it', () => {
  const out = deselectVendor(['contrib:groq@0.5.0', 'argonne:alcf'], 'contrib', 'groq');
  assert.deepEqual(out.install, ['argonne:alcf']);
});

test('a pin replaces the vendor\'s own entries where they sat, and Follow latest undoes it', () => {
  const pinned = pinVendor(['contrib:groq', 'argonne:alcf', '!contrib:groq'], 'contrib', 'groq', '0.5.0');
  assert.deepEqual(pinned, ['contrib:groq@0.5.0', 'argonne:alcf']);
  // Update to a newer one rewrites the pin in place.
  assert.deepEqual(pinVendor(pinned, 'contrib', 'groq', '0.6.0'), ['contrib:groq@0.6.0', 'argonne:alcf']);
  // A tag-selected vendor gets an explicit pin, which beats the tag.
  const tagged = pinVendor(['contrib:tag:tier:free'], 'contrib', 'llm7', '0.4.0');
  assert.deepEqual(tagged, ['contrib:tag:tier:free', 'contrib:llm7@0.4.0']);
  assert.equal(pickerFor(tagged, 'contrib', 'llm7', ['tier:free']), 'contrib:llm7@0.4.0');
  assert.deepEqual(followLatest(pinned, 'contrib', 'groq'), ['contrib:groq', 'argonne:alcf']);
});

test('the editor keeps pins and exclusions of the source it rewrites', () => {
  const install = ['contrib:groq@0.5.0', 'contrib:tag:tier:free', '!contrib:llm7'];
  assert.deepEqual(sourceSelection(install, 'contrib'), { all: false, tags: ['tier:free'], vendors: ['groq'] });
  assert.deepEqual(withSourceSelection(install, 'contrib', { all: false, tags: ['tier:free'], vendors: ['groq', 'zai'] }),
    ['contrib:tag:tier:free', 'contrib:groq@0.5.0', 'contrib:zai', '!contrib:llm7']);
  assert.equal(excludedBy(install, 'contrib', 'llm7'), '!contrib:llm7');
  assert.equal(excludedBy(install, 'contrib', 'groq'), null);
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

test('a provided vendor says so once, as a warning', () => {
  const available = vendorFacts({ core_provides: ['kilo--chat_completions'] }, SRC);
  const fact = available.find((f) => f.key === 'shimSrc.coreProvides');
  assert.ok(fact);
  assert.equal(fact.cls, 'is-warn');

  const skipped = vendorFacts({
    skipped: 'already provided by core: kilo--chat_completions',
    core_provides: ['kilo--chat_completions'],
  }, SRC);
  assert.equal(skipped.filter((f) => f.key === 'shimSrc.coreProvides').length, 0);
});

test('both provided lines open with the ! marker in each language', async () => {
  // i18n.js reads state.js, which reads localStorage at module scope.
  globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {}, clear() {} };
  globalThis.document = {
    documentElement: { getAttribute: () => null, setAttribute() {} },
    body: { classList: { add() {}, remove() {}, contains: () => false } },
    getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  };
  globalThis.window = globalThis;
  const { I18N } = await import(`${ADMIN}/core/i18n.js`);

  for (const lang of ['en', 'zh']) {
    assert.match(I18N[lang]['shimSrc.skipped'], /^! /);
    assert.match(I18N[lang]['shimSrc.coreProvides'], /^! /);
  }
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

function installPage({
  catalogue = CATALOGUE, config: served = null, inventory = [],
} = {}) {
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
  const config = served
    || { shims: { install: ['contrib:opendesign'] }, registered_shims: [], model_types: [] };
  globalThis.fetch = async (url, opts = {}) => {
    requests.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : null });
    let body = {};
    if (url.startsWith('/admin/api/config/shims/source-catalog')) body = catalogue;
    else if (url === '/admin/api/config') body = config;
    else if (url === '/admin/api/config/shims') body = opts.method === 'PUT' ? { ok: true } : { shims: inventory };
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

test('Remove on a tag-selected vendor writes an exclusion and no refusal', async () => {
  const fx = legacyFixture();
  fx.config.shims = { install: ['contrib:tag:tier:free'] };
  fx.catalogue.sources[0].vendors[1].selected_origin = 'install';
  installPage(fx);
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  requests.length = 0;
  await page.deselectShimSourceVendor(null, 'contrib', 'kilo_free');

  const put = requests.find((r) => r.method === 'PUT');
  assert.deepEqual(put.body, { install: ['contrib:tag:tier:free', '!contrib:kilo_free'] });
});

test('an excluded vendor says so and offers Install; a pinned one offers Update and Follow latest', async () => {
  const fx = legacyFixture();
  const [kilo, kiloFree, groq] = fx.catalogue.sources[0].vendors;
  Object.assign(groq, { excluded_by: '!contrib:groq' });
  Object.assign(kilo, { selected_origin: 'install', selected_by: 'contrib:kilo@0.4.0', pinned: '0.4.0',
    installed: '0.4.0', installable: '0.5.0', update_available: true });
  installPage(fx);
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  page.setShimsView('available');
  const html = elements.get('shimCatalogGrid').innerHTML;

  const g = cardIn(html, 'contrib:groq');
  assert.match(g, /!contrib:groq keeps it out/);
  assert.match(g, /selectShimSourceVendor/);
  const k = cardIn(html, 'contrib:kilo');
  assert.match(k, /pinned at 0\.4\.0/);
  assert.match(k, /pinShimSourceVendor\(this, &quot;contrib&quot;, &quot;kilo&quot;, &quot;0\.5\.0&quot;\)">Update to 0\.5\.0/);
  assert.match(k, /followShimSourceVendor/);
  assert.match(k, /deselectShimSourceVendor/, 'a pinned vendor the config names can be removed');
  // An unpinned installed vendor offers the pin at its installed version.
  assert.match(cardIn(html, 'contrib:kilo_free'),
    /pinShimSourceVendor\(this, &quot;contrib&quot;, &quot;kilo_free&quot;, &quot;0\.4\.0&quot;\)">Pin this version/);
});

test('Pin and Follow latest write the pin, then drop it', async () => {
  const fx = legacyFixture();
  fx.config.shims = { install: ['contrib:tag:tier:free'], sources: ['contrib'] };
  installPage(fx);
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();

  requests.length = 0;
  await page.pinShimSourceVendor(null, 'contrib', 'kilo_free', '0.4.0');
  assert.deepEqual(requests.find((r) => r.method === 'PUT').body,
    { install: ['contrib:tag:tier:free', 'contrib:kilo_free@0.4.0'] });

  fx.config.shims.install = ['contrib:tag:tier:free', 'contrib:kilo_free@0.4.0'];
  requests.length = 0;
  await page.followShimSourceVendor(null, 'contrib', 'kilo_free');
  assert.deepEqual(requests.find((r) => r.method === 'PUT').body,
    { install: ['contrib:tag:tier:free', 'contrib:kilo_free'] });
});

test('a vendor card carries one switch per loaded shim, and names the providers that pulled it in', async () => {
  const fx = legacyFixture();
  installPage({ ...fx, inventory: [
    { name: 'kilo--chat_completions', enabled: true, used_by: ['kilo-paid', 'kilo-2'], origin: 'remote' },
    { name: 'kilo_free--chat_completions', enabled: false, used_by: [], origin: 'remote' },
  ] });
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  page.setShimsView('available');
  const html = elements.get('shimCatalogGrid').innerHTML;

  const kilo = cardIn(html, 'contrib:kilo');
  assert.match(kilo, /Installed for the providers that use it: kilo-paid, kilo-2/);
  assert.match(kilo, /role="switch" aria-label="Enable kilo--chat_completions" checked onchange="shimToggleEnabled\(this, &quot;kilo--chat_completions&quot;\)"/);
  const free = cardIn(html, 'contrib:kilo_free');
  assert.match(free, /aria-label="Enable kilo_free--chat_completions" onchange=/, 'a disabled shim is unchecked');
  assert.match(free, /class="shim-preset-shim off"/);
  // Not installed: no switch to show.
  assert.doesNotMatch(cardIn(html, 'contrib:groq'), /role="switch"/);
});

// ── After the move (#1660): one card per vendor, no package cards ──────

test('a vendor states its tier when it is not paid', () => {
  assert.equal(vendorTier({ tags: ['tier:free'] }), 'free');
  assert.equal(vendorTier({ tags: ['auth:device', 'tier:subscription'] }), 'subscription');
  assert.equal(vendorTier({ tags: ['tier:paid'] }), '');
  assert.equal(vendorTier({}), '');
});

test('the search reads the catalogue label, so "z.ai" finds zai', () => {
  const src = { id: 'contrib', label: 'Contrib' };
  assert.ok(vendorMatches({ vendor: 'zai', label: 'Z.ai' }, src, 'z.ai'));
  assert.ok(!vendorMatches({ vendor: 'zai' }, src, 'z.ai'));
});

/** Two sources that both publish `kilo_free`, one installed vendor and two not. */
const SEARCHABLE = {
  sources: [
    { id: 'contrib', label: 'Contrib', vendors: [
      { vendor: 'kilo', installed: '0.5.0', tags: ['tier:paid'] },
      { vendor: 'kilo_free', label: 'Kilo', installed: null, tags: ['tier:free'] },
      { vendor: 'groq', label: 'Groq', installed: null, tags: [] },
    ] },
    { id: 'mirror', label: 'Mirror', vendors: [
      { vendor: 'kilo_free', installed: null, tags: ['tier:free'] },
    ] },
  ],
};
const names = (rows) => rows.map(({ src, v }) => `${src.id}:${v.vendor}`);

test('the Installed search adds the source vendors it is not running', () => {
  // A query is required: without one the section is the whole Available view.
  assert.deepEqual(notInstalledVendors(SEARCHABLE, {}), []);
  assert.deepEqual(notInstalledVendors(SEARCHABLE, { q: '  ' }), []);
  // The installed `kilo` is a row above; the second source's copy of
  // `kilo_free` is the same vendor and is listed once.
  assert.deepEqual(names(notInstalledVendors(SEARCHABLE, { q: 'kilo' })), ['contrib:kilo_free']);
  assert.deepEqual(names(notInstalledVendors(SEARCHABLE, { q: 'groq' })), ['contrib:groq']);
});

test('the Installed search applies the tier and origin filters to source vendors', () => {
  assert.deepEqual(names(notInstalledVendors(SEARCHABLE, { q: 'contrib', tier: 'free' })),
    ['contrib:kilo_free']);
  // Untagged reads as paid, as a loaded shim with no tier does.
  assert.deepEqual(names(notInstalledVendors(SEARCHABLE, { q: 'contrib', tier: 'paid' })),
    ['contrib:groq']);
  // Every source vendor is external.
  assert.deepEqual(notInstalledVendors(SEARCHABLE, { q: 'kilo', origin: 'builtin' }), []);
  assert.deepEqual(notInstalledVendors(SEARCHABLE, { q: 'kilo', origin: 'path' }), []);
  assert.deepEqual(names(notInstalledVendors(SEARCHABLE, { q: 'kilo', origin: 'remote' })),
    ['contrib:kilo_free']);
  assert.deepEqual(notInstalledVendors(null, { q: 'kilo' }), []);
});

/** The served payloads of a gateway with a tag entry and a provider-installed vendor. */
function legacyFixture() {
  const vendor = (over) => ({
    versions: ['0.4.0'], newest: '0.4.0', installable: '0.4.0', requires_ok: true,
    maturity: 'beta', conformance: { date: null, result: 'none' }, core_provides: [],
    ...over,
  });
  const catalogue = {
    core: { surface: 1, ir: 1, version: '0.14.0' },
    install: [],
    sources: [{
      id: 'contrib', key: 'contrib', label: 'Contrib', url: 'https://cdn.example/contrib/manifest.json',
      repo: 'https://github.com/Oaklight/llm-rosetta-shim-contrib', entry: null, configured: false,
      ok: true, error: null, stale: false,
      vendors: [
        vendor({ vendor: 'kilo', label: 'Kilo', logo: 'https://cdn.example/kilocode.svg',
          installed: '0.5.0', tags: ['tier:paid'], shims: ['kilo--chat_completions'],
          installed_shims: ['kilo--chat_completions'], selected_by: 'contrib:kilo',
          selected_origin: 'moved:kilo', state: 'installed' }),
        vendor({ vendor: 'kilo_free', label: 'Kilo', logo: 'https://cdn.example/kilocode.svg',
          installed: '0.4.0', tags: ['tier:free'], shims: ['kilo_free--chat_completions'],
          installed_shims: ['kilo_free--chat_completions'], selected_by: 'contrib:tag:tier:free',
          selected_origin: 'install', state: 'installed' }),
        vendor({ vendor: 'groq', label: 'Groq', logo: 'https://cdn.example/groq.svg',
          installed: null, tags: [], shims: ['groq--chat_completions'], installed_shims: [],
          selected_by: null, state: 'available' }),
      ],
    }],
  };
  const config = { shims: { install: ['contrib:tag:tier:free'] }, registered_shims: [], model_types: [] };
  return { catalogue, config };
}

/** One rendered card, from its article to its close. */
function cardIn(html, key) {
  const at = html.indexOf(`data-srcvendor="${key}"`);
  assert.notEqual(at, -1, `${key} has a card`);
  return html.slice(at, html.indexOf('</article>', at));
}

async function renderLegacy(view) {
  installPage(legacyFixture());
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  page.setShimsView(view);
  return page;
}

test('Available lists vendors only: no package card, no archived repository', async () => {
  await renderLegacy('available');
  const html = elements.get('shimCatalogGrid').innerHTML;

  assert.doesNotMatch(html, /data-package=/, 'a package is not a card of its own');
  assert.doesNotMatch(html, /llm-rosetta-shim-alcf/, 'no link to an archived package repository');
  // The header counts vendors; packages are not a second unit beside them.
  assert.equal(elements.get('shimsCount').textContent, '3 vendors');
});

test('two vendors with one name are told apart by their tier', async () => {
  await renderLegacy('available');
  const html = elements.get('shimCatalogGrid').innerHTML;
  const title = (c) => c.slice(c.indexOf('shim-vendor-title'), c.indexOf('shim-vendor-meta'));

  assert.match(title(cardIn(html, 'contrib:kilo')), />Kilo</);
  assert.match(title(cardIn(html, 'contrib:kilo_free')), />Kilo</);
  assert.match(title(cardIn(html, 'contrib:kilo_free')), /class="tag is-square is-free">Free</);
  assert.doesNotMatch(title(cardIn(html, 'contrib:kilo')), /is-free/);
  // The catalogue's mark, not the initials tile, for a vendor not installed.
  assert.match(cardIn(html, 'contrib:groq'),
    /src="\/admin\/api\/logo\?src=https%3A%2F%2Fcdn\.example%2Fgroq\.svg"/);
});

test('a vendor a provider pulled in says so, and offers no Remove', async () => {
  await renderLegacy('available');
  const html = elements.get('shimCatalogGrid').innerHTML;
  const kilo = cardIn(html, 'contrib:kilo');

  assert.match(kilo, /Installed because a provider uses it/);
  assert.doesNotMatch(kilo, /Selected by contrib:kilo/, 'no config entry says contrib:kilo');
  assert.doesNotMatch(kilo, /deselectShimSourceVendor/);
  // A vendor a config entry selects names that entry and keeps its Remove.
  const free = cardIn(html, 'contrib:kilo_free');
  assert.match(free, /Selected by contrib:tag:tier:free/);
  assert.match(free, /deselectShimSourceVendor/);
});

test('an Installed search shows the source vendors that are not installed', async () => {
  installPage(legacyFixture());
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  elements.get('shimPageSearch').value = 'groq';
  page.setShimsView('installed');
  const html = elements.get('shimGrid').innerHTML;

  assert.match(html, /data-srcvendor="contrib:groq"/);
  assert.doesNotMatch(html, /data-package=/);
  assert.match(elements.get('shimsCount').textContent, /1 more available/);
});

test('a mark that fails to load tries the catalogue copy before the initials', async () => {
  installPage();
  const page = await import(`${ADMIN}/tabs/shims-page.js`);

  const both = page.logoHtml('https://old.example/argo.png', 'argo', 'https://cdn.example/argo.svg');
  assert.match(both, /src="\/admin\/api\/logo\?src=https%3A%2F%2Fold\.example%2Fargo\.png"/);
  assert.match(both, /data-alt="\/admin\/api\/logo\?src=https%3A%2F%2Fcdn\.example%2Fargo\.svg"/);
  assert.match(both, /onerror="if\(this\.dataset\.alt\)/);
  // No own mark: the catalogue copy is the first candidate, with nothing after.
  const only = page.logoHtml(null, 'groq', 'https://cdn.example/groq.svg');
  assert.match(only, /src="\/admin\/api\/logo\?src=https%3A%2F%2Fcdn\.example%2Fgroq\.svg"/);
  assert.doesNotMatch(only, /data-alt=/);
  // Neither: the initials tile.
  assert.match(page.logoHtml(null, 'groq'), /^<span class="shim-logo-fallback"[^>]*>Gr<\/span>$/);
});

// ── Review round (milo, elena on #1671) ────────────────────────────────

test('Pin replaces an existing pin, qualified or not, instead of adding a second', () => {
  assert.deepEqual(pinVendor(['contrib:kilo@0.4.0'], 'contrib', 'kilo', '0.5.0'), ['contrib:kilo@0.5.0']);
  assert.deepEqual(pinVendor(['kilo@0.4.0', 'argonne:alcf'], 'contrib', 'kilo', '0.5.0'),
    ['contrib:kilo@0.5.0', 'argonne:alcf']);
  // Pinning twice in a row leaves one pin.
  const twice = pinVendor(pinVendor(['contrib:tag:tier:free'], 'contrib', 'kilo', '0.4.0'), 'contrib', 'kilo', '0.5.0');
  assert.deepEqual(twice, ['contrib:tag:tier:free', 'contrib:kilo@0.5.0']);
  // Another vendor's pin, or an unqualified unpinned entry, is not this vendor's pin.
  assert.deepEqual(pinVendor(['kilo', 'groq@1.0.0'], 'contrib', 'kilo', '0.5.0'),
    ['kilo', 'groq@1.0.0', 'contrib:kilo@0.5.0']);
});

test('Remove reports the pin it drops, and Install does not bring it back', () => {
  const removed = deselectVendor(['contrib:kilo@0.4.0'], 'contrib', 'kilo');
  assert.equal(removed.droppedPin, '0.4.0');
  assert.deepEqual(removed.install, []);
  assert.deepEqual(selectVendor(removed.install, 'contrib', 'kilo'), ['contrib:kilo']);
  // Behind a tag: the exclusion is written and the pin is still named.
  const tagged = deselectVendor(['contrib:tag:tier:free', 'contrib:kilo@0.4.0'], 'contrib', 'kilo', ['tier:free']);
  assert.equal(tagged.droppedPin, '0.4.0');
  assert.deepEqual(tagged.install, ['contrib:tag:tier:free', '!contrib:kilo']);
  assert.equal(deselectVendor(['contrib:kilo'], 'contrib', 'kilo').droppedPin, null);
});

test('the Remove toast says the pin was dropped', async () => {
  const fx = legacyFixture();
  fx.config.shims = { install: ['contrib:kilo_free@0.4.0'] };
  Object.assign(fx.catalogue.sources[0].vendors[1], {
    selected_by: 'contrib:kilo_free@0.4.0', selected_origin: 'install', pinned: '0.4.0',
  });
  installPage(fx);
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  requests.length = 0;
  await page.deselectShimSourceVendor(null, 'contrib', 'kilo_free');
  const put = requests.find((r) => r.method === 'PUT');
  assert.deepEqual(put.body, { install: [] });
  assert.match(elements.get('toast').textContent, /pin 0\.4\.0 dropped/);
});

test('a Remove that also disables a provider still says what it did', async () => {
  const fx = legacyFixture();
  fx.config.shims = { install: ['contrib:kilo_free@0.4.0'] };
  Object.assign(fx.catalogue.sources[0].vendors[1], {
    selected_by: 'contrib:kilo_free@0.4.0', selected_origin: 'install', pinned: '0.4.0',
  });
  installPage(fx);
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, opts = {}) => {
    if (url === '/admin/api/config/shims/reload') {
      const body = { ok: true, status: { failure_count: 0 }, provider_warnings: [{ provider: 'kilo-free' }] };
      return { ok: true, status: 200, headers: { get: () => 'application/json' },
        json: async () => body, text: async () => JSON.stringify(body) };
    }
    return realFetch(url, opts);
  };
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  await page.deselectShimSourceVendor(null, 'contrib', 'kilo_free');
  const text = elements.get('toast').textContent;
  assert.match(text, /kilo_free removed/);
  assert.match(text, /pin 0\.4\.0 dropped/);
  assert.match(text, /kilo-free/);
});

test('a record with no label or logo falls back to the key titled and the initials', async () => {
  const fx = legacyFixture();
  const groq = fx.catalogue.sources[0].vendors[2];
  delete groq.label;
  delete groq.logo;
  installPage(fx);
  const { S } = await import(`${ADMIN}/core/state.js`);
  const page = await import(`${ADMIN}/tabs/shims-page.js`);
  S.configData = null;
  await page.loadShimsPage();
  page.setShimsView('available');
  const card = cardIn(elements.get('shimCatalogGrid').innerHTML, 'contrib:groq');

  assert.match(card, /class="shim-vendor-name" title="Groq">Groq</);
  assert.match(card, /<span class="shim-logo-fallback"[^>]*>Gr<\/span>/);
  assert.doesNotMatch(card, /<img/);
});

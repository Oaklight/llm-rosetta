/**
 * The catalogue's two search-and-ref helpers (#1526 follow-up).
 *
 * The Installed view lists what is registered, so a package that is available
 * but not installed was invisible to its search box.  The same predicate now
 * answers both lists, and the ref helper is what lets a card load a tag,
 * branch or commit other than the one the catalogue pins.
 *
 * Run with `make test-js` (or `node --test tests/js/*.test.mjs`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  availableNotInstalled,
  catalogueMatches,
  executeRefInstall,
  failedSource,
  pinnedRemote,
  refInstallAction,
  refSourcesWrite,
  removeSourcesWrite,
  sameRepo,
  selectionOf,
  selectionWrite,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/shim-selection.js';

const PKG = {
  key: 'argo',
  label: 'Argo',
  shims: ['argo--chat_completions', 'argo--messages', 'argo--open_responses'],
  formats: ['chat_completions', 'messages', 'open_responses'],
  remote: {
    url: 'https://github.com/Oaklight/llm-rosetta-shim-argo',
    path: 'src/llm_rosetta_shim_argo/providers',
    ref: 'v0.4.0',
    mirror: {
      url_template: 'https://cdn.example/@{ref}/argo.tar.gz',
      sha256: 'deadbeef',
      path: 'providers',
    },
  },
};

test('an empty query matches everything', () => {
  assert.equal(catalogueMatches(PKG, ''), true);
  assert.equal(catalogueMatches(PKG, '   '), true);
  assert.equal(catalogueMatches(PKG, undefined), true);
});

test('the key, the label, a shim name and a format all match', () => {
  for (const q of ['argo', 'Argo', 'ARGO', 'argo--messages', 'messages', 'chat_completions']) {
    assert.equal(catalogueMatches(PKG, q), true, q);
  }
});

test('every term must match, so a query narrows instead of widening', () => {
  assert.equal(catalogueMatches(PKG, 'argo messages'), true);
  assert.equal(catalogueMatches(PKG, 'argo alcf'), false);
  assert.equal(catalogueMatches(PKG, 'alcf'), false);
});

test('a package with no shims or formats still matches by key and label', () => {
  const bare = { key: 'asksage', label: 'AskSage' };
  assert.equal(catalogueMatches(bare, 'asksage'), true);
  assert.equal(catalogueMatches(bare, 'argo'), false);
});

test('a chosen ref replaces the pinned one and drops the mirror', () => {
  const remote = pinnedRemote(PKG, 'abc1234');
  assert.equal(remote.ref, 'abc1234');
  assert.equal(remote.url, PKG.remote.url);
  // The path is the catalogue's: it names the directory inside the archive.
  assert.equal(remote.path, PKG.remote.path);
  // A digest computed for the tagged archive would reject these bytes.
  assert.equal('mirror' in remote, false);
});

test('an empty ref means the repository head', () => {
  const remote = pinnedRemote(PKG, '  ');
  assert.equal('ref' in remote, false);
  assert.equal(remote.url, PKG.remote.url);
});

test('the source package is not mutated', () => {
  pinnedRemote(PKG, 'v9');
  assert.equal(PKG.remote.ref, 'v0.4.0');
  assert.equal(PKG.remote.mirror.sha256, 'deadbeef');
});

test('a package without a remote falls back to its repository URL', () => {
  const remote = pinnedRemote(
    { key: 'k', source_repo: 'https://example.test/repo' }, 'main');
  assert.deepEqual(remote, { url: 'https://example.test/repo', ref: 'main' });
});

test('a failed source is found by its URL, not by the failure count', () => {
  const sources = [
    { location: 'https://example.test/other', ok: false, error: 'HTTP 500' },
    { location: 'https://example.test/mine', ok: false, error: 'HTTP 404' },
  ];
  assert.equal(failedSource(sources, 'https://example.test/mine').error, 'HTTP 404');
  // The other package's failure must not undo a ref that loaded.
  assert.equal(failedSource(sources, 'https://example.test/third'), null);
  // A source that loaded is not a failure even if it is ours.
  assert.equal(
    failedSource([{ location: 'https://example.test/mine', ok: true }],
      'https://example.test/mine'),
    null,
  );
});

test('no report, no URL or a non-list reads as "nothing failed"', () => {
  assert.equal(failedSource(undefined, 'https://example.test/mine'), null);
  assert.equal(failedSource([], null), null);
  assert.equal(failedSource('nope', 'https://example.test/mine'), null);
});

const CATALOGUE = [
  {
    key: 'argo',
    label: 'Argo',
    installed: true,
    tier: 'paid',
    shims: ['argo--messages'],
    remote: { url: 'https://github.com/o/argo', path: 'p', ref: 'v1' },
  },
  {
    key: 'free',
    label: 'Free Endpoints',
    installed: false,
    tier: 'free',
    shims: ['kilo_free--chat_completions'],
    remote: { url: 'https://github.com/o/free', path: 'p', ref: 'v1' },
  },
];

test('the not-installed section needs a query', () => {
  // Without one it would be the whole Available view.
  assert.deepEqual(availableNotInstalled(CATALOGUE, {}), []);
  assert.deepEqual(availableNotInstalled(CATALOGUE, { q: '   ' }), []);
});

test('the section lists only packages that are not installed', () => {
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'argo' }).map((p) => p.key), []);
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'free' }).map((p) => p.key), ['free']);
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'kilo' }).map((p) => p.key), ['free']);
});

test('the tier filter applies and the origin filter excludes catalogue packages', () => {
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'free', tier: 'paid' }), []);
  // Every catalogue package is an external source, so a builtin or local
  // filter matches none of them.
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'free', origin: 'builtin' }), []);
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'free', origin: 'path' }), []);
  assert.deepEqual(
    availableNotInstalled(CATALOGUE, { q: 'free', origin: 'remote' }).map((p) => p.key),
    ['free']);
});

test('a ref write drops the preset and appends one explicit remote', () => {
  const section = { presets: ['argo', 'alcf'], remotes: [] };
  const next = refSourcesWrite(section, CATALOGUE[1], 'main');
  assert.deepEqual(next.presets, ['argo', 'alcf']);
  assert.equal(next.remotes.length, 1);
  assert.equal(next.remotes[0].ref, 'main');
  // The section that was read is not mutated.
  assert.deepEqual(section, { presets: ['argo', 'alcf'], remotes: [] });
});

test('the write removes the package key from presets', () => {
  const next = refSourcesWrite({ presets: ['free', 'argo'], remotes: [] },
    CATALOGUE[1], 'main');
  assert.deepEqual(next.presets, ['argo']);
});

test('an existing remote for the same repository is replaced, not duplicated', () => {
  const section = {
    presets: [],
    remotes: [
      { url: 'https://github.com/o/free.git', path: 'p', ref: 'v1' },
      { url: 'https://github.com/o/other', path: 'p' },
    ],
  };
  const next = refSourcesWrite(section, CATALOGUE[1], 'main');
  assert.equal(next.remotes.length, 2);
  assert.deepEqual(next.remotes.map((r) => r.url),
    ['https://github.com/o/other', 'https://github.com/o/free']);
  assert.equal(next.remotes[1].ref, 'main');
});

test('two spellings of one repository compare equal, as the loader decides', () => {
  assert.equal(sameRepo('https://github.com/O/Free.git/', 'https://github.com/o/free'), true);
  assert.equal(sameRepo('https://github.com/o/free', 'https://github.com/o/argo'), false);
  assert.equal(sameRepo(null, 'https://github.com/o/free'), false);
});

test('the reload verdict rolls back only the source that failed', () => {
  const url = 'https://github.com/o/free';
  const ok = { status: { failure_count: 0, sources: [{ location: url, ok: true }] } };
  assert.deepEqual(refInstallAction(ok, url), { restore: false, error: null });

  // Another package's failure does not undo a ref that loaded.
  const other = {
    status: {
      failure_count: 1,
      sources: [{ location: 'https://github.com/o/other', ok: false, error: 'x' }],
    },
  };
  assert.deepEqual(refInstallAction(other, url), { restore: false, error: null });

  const mine = {
    status: {
      failure_count: 1,
      sources: [{ location: url, ok: false, error: 'HTTP 404' }],
    },
  };
  assert.deepEqual(refInstallAction(mine, url),
    { restore: true, error: 'HTTP 404' });
});

test('a missing report restores nothing', () => {
  assert.deepEqual(refInstallAction(null, 'https://github.com/o/free'),
    { restore: false, error: null });
  assert.deepEqual(refInstallAction({}, 'https://github.com/o/free'),
    { restore: false, error: null });
});

test('a pinned install carries the preset selection onto the remote', () => {
  // The selection is keyed by preset; the preset leaves, so the map would be
  // inert ("selection has no effect") and the package would load everything.
  const section = {
    presets: ['argo'],
    include: { argo: ['argo--messages'], other: ['x'] },
    exclude: { argo: ['argo--open_responses'] },
    remotes: [],
  };
  const next = refSourcesWrite(section, CATALOGUE[0], 'main');
  assert.equal(next.remotes[0].include.length, 1);
  assert.equal(next.remotes[0].include[0], 'argo--messages');
  assert.equal(next.remotes[0].exclude[0], 'argo--open_responses');
  // The package's maps are gone; another package's map is untouched.
  assert.deepEqual(next.include, { other: ['x'] });
  assert.deepEqual(next.exclude, {});
});

test('remove drops whichever carrier installed the package', () => {
  const pinned = {
    presets: ['alcf'],
    remotes: [{ url: 'https://github.com/o/free.git', path: 'p', ref: 'main' }],
    include: { free: ['kilo_free--chat_completions'] },
  };
  const next = removeSourcesWrite(pinned, CATALOGUE[1]);
  assert.deepEqual(next.presets, ['alcf']);
  assert.deepEqual(next.remotes, []);
  assert.deepEqual(next.include, {});

  // The preset shape still removes, and other packages survive both.
  const preset = { presets: ['free', 'argo'], remotes: [{ url: 'https://x/y' }] };
  const after = removeSourcesWrite(preset, CATALOGUE[1]);
  assert.deepEqual(after.presets, ['argo']);
  assert.equal(after.remotes.length, 1);
});

async function _recordingIo({ fail = false } = {}) {
  const calls = [];
  return {
    calls,
    io: {
      writeShims: async (body) => { calls.push(['write', body]); },
      reload: async () => {
        calls.push(['reload']);
        return fail
          ? { status: { failure_count: 1, sources: [
            { location: 'https://github.com/o/free', ok: false, error: 'HTTP 404' }] } }
          : { status: { failure_count: 0, sources: [
            { location: 'https://github.com/o/free', ok: true }] } };
      },
      readConfig: async () => { calls.push(['read']); },
    },
  };
}

test('a successful ref install writes once and restores nothing', async () => {
  const { calls, io } = await _recordingIo();
  const section = { presets: ['free'], remotes: [] };
  const out = await executeRefInstall({ section, pkg: CATALOGUE[1], ref: 'main', io });
  assert.equal(out.ok, true);
  assert.deepEqual(calls.map((c) => c[0]), ['write', 'reload', 'read']);
  assert.deepEqual(calls[0][1].presets, []);
  assert.equal(calls[0][1].remotes[0].ref, 'main');
});

test('a failed ref install writes the snapshot back before reporting', async () => {
  const { calls, io } = await _recordingIo({ fail: true });
  const section = { presets: ['free'], remotes: [{ url: 'https://keep/me' }] };
  const out = await executeRefInstall({ section, pkg: CATALOGUE[1], ref: 'nope', io });
  assert.equal(out.ok, false);
  assert.equal(out.error, 'HTTP 404');
  assert.deepEqual(calls.map((c) => c[0]), ['write', 'reload', 'write', 'reload', 'read']);
  // The second write is the section as it was, not the bad ref.
  assert.deepEqual(calls[2][1], {
    presets: ['free'], remotes: [{ url: 'https://keep/me' }], include: {}, exclude: {},
  });
});

test('a failed ref install keeps a narrowed package narrowed', async () => {
  // The ref write clears the keyed map as it moves the selection onto the
  // entry, so a snapshot of `presets`/`remotes` alone restores the package at
  // full width — every shim the map used to decline loads again.
  const { calls, io } = await _recordingIo({ fail: true });
  const section = {
    presets: ['free'],
    include: { free: ['kilo_free--chat_completions'] },
    remotes: [{ url: 'https://keep/me' }],
  };
  await executeRefInstall({ section, pkg: CATALOGUE[1], ref: 'nope', io });
  assert.deepEqual(calls[0][1].include, {});
  assert.deepEqual(calls[2][1].include, { free: ['kilo_free--chat_completions'] });
  assert.deepEqual(calls[2][1].exclude, {});
});

test('a second pin keeps the selection the entry already carries', () => {
  // After the first pin the selection lives on the entry, not on the keyed
  // map: reading the map alone rebuilds the entry empty and loads everything.
  const section = {
    presets: [],
    remotes: [{
      url: 'https://github.com/o/free',
      path: 'p',
      ref: 'v1.0.0',
      include: ['kilo_free--chat_completions'],
    }],
  };
  const next = refSourcesWrite(section, CATALOGUE[1], 'main');
  assert.equal(next.remotes.length, 1);
  assert.equal(next.remotes[0].ref, 'main');
  assert.deepEqual(next.remotes[0].include, ['kilo_free--chat_completions']);
});

test('Save selection writes the map on a preset package', () => {
  const maps = { include: { free: ['a'] }, exclude: {} };
  const body = selectionWrite({ presets: ['free'], remotes: [] }, CATALOGUE[1], maps);
  assert.deepEqual(body, { include: { free: ['a'] }, exclude: {} });
});

test('Save selection writes the entry on a pinned package', () => {
  // The map is keyed by preset; a pinned package has no preset key for it to
  // bite on, so writing only the map would be a no-op that reports success.
  const section = {
    presets: [],
    remotes: [{ url: 'https://github.com/o/free.git', path: 'p', ref: 'main' }],
  };
  const maps = { include: { free: ['kilo_free--chat_completions'] }, exclude: {} };
  const body = selectionWrite(section, CATALOGUE[1], maps);
  assert.deepEqual(body.include, {});
  assert.deepEqual(body.remotes[0].include, ['kilo_free--chat_completions']);
  assert.equal(body.remotes[0].ref, 'main');
});

test('selecting everything on a pinned package clears the entry selection', () => {
  const section = {
    remotes: [{ url: 'https://github.com/o/free', path: 'p', ref: 'main',
      include: ['a'], exclude: ['b'] }],
  };
  const body = selectionWrite(section, CATALOGUE[1], { include: {}, exclude: {} });
  assert.equal('include' in body.remotes[0], false);
  assert.equal('exclude' in body.remotes[0], false);
});

test('the selection is read from the preset map when a preset holds it', () => {
  const section = { presets: ['free'], include: { free: ['a'] }, exclude: { free: ['b'] } };
  assert.deepEqual(selectionOf(section, CATALOGUE[1]), { include: ['a'], exclude: ['b'] });
});

test('the selection is read from the pinned entry when one holds it', () => {
  // The card paints its checkboxes from this, so reading only the preset maps
  // would show a narrowed pinned package as fully ticked — and Save would
  // then look like it changed nothing.
  const section = {
    presets: [],
    include: {},
    remotes: [{ url: 'https://github.com/o/free.git', path: 'p', ref: 'main',
      include: ['kilo_free--chat_completions'] }],
  };
  assert.deepEqual(selectionOf(section, CATALOGUE[1]),
    { include: ['kilo_free--chat_completions'], exclude: undefined });
});

test('an uninstalled package reads as fully selected', () => {
  assert.deepEqual(selectionOf({}, CATALOGUE[0]),
    { include: undefined, exclude: undefined });
  assert.deepEqual(selectionOf(null, CATALOGUE[0]),
    { include: undefined, exclude: undefined });
});

test('Save selection round-trips through selectionOf', () => {
  const section = {
    remotes: [{ url: 'https://github.com/o/free', path: 'p', ref: 'main' }],
  };
  const body = selectionWrite(section, CATALOGUE[1],
    { include: { free: ['kilo_free--chat_completions'] }, exclude: {} });
  const after = { ...section, ...body };
  assert.deepEqual(selectionOf(after, CATALOGUE[1]),
    { include: ['kilo_free--chat_completions'], exclude: undefined });
});

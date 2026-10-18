/**
 * The admin provider-type picker tree, pinned against the builder it replaced.
 *
 * `shim-tree.js` carries one promise that only a test can hold it to: a shim that
 * declares none of the new grouping fields renders exactly as it did before they
 * existed.  The rendering lives in JavaScript, so the pre-`group` builder is
 * reimplemented here as an oracle and the two are compared — for the shims that
 * declare nothing, for the shims written against the older schema (which declare
 * a `display_name` that meant their *group* title), and for the declared ones.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DECLARED,
  DERIVED,
  LEGACY_DISPLAY_NAME,
  buildTypeModel,
  shimBrand,
  shimGroupKey,
  shimGroupSource,
  shimLabel,
  shimLabelSource,
} from '../../src/llm_rosetta/gateway/admin/js/core/shim-tree.js';

/** Preferred order for base types within a group, as the picker has always used. */
const BASE_TYPE_ORDER = [
  'chat_completions',
  'open_responses',
  'messages',
  'google_generate',
  'google_interactions',
  'decision',
];

/** The base-format roots the picker offers independently of any shim. */
const BASE_FORMATS = [{ name: 'chat_completions' }, { name: 'open_responses' }];

/**
 * The builder as it stood before `group` / `group_label` existed.
 *
 * Kept as the oracle for "renders exactly as it did": the top level is the name
 * segment before `--`, `display_name` titles that top level, and the shim's own
 * row label always comes from the name.
 */
function legacyTree(registeredShims, baseFormats) {
  const rootMap = new Map();
  const root = (name) => {
    if (!rootMap.has(name)) rootMap.set(name, { name, label: name, children: [] });
    return rootMap.get(name);
  };
  for (const s of registeredShims) {
    const sep = s.name.indexOf('--');
    const r = root(sep === -1 ? s.name : s.name.slice(0, sep));
    if (s.display_name) r.label = s.display_name;
    r.children.push({
      value: s.name,
      label: sep === -1 ? s.base : s.name.slice(sep + 2),
      base: s.base,
    });
  }
  for (const f of baseFormats) root(f.name);
  return { roots: [...rootMap.values()] };
}

/** Reduce a tree to what is rendered: root label + row labels, in a stable order. */
function rendered(tree) {
  const rank = (b) => {
    const i = BASE_TYPE_ORDER.indexOf(b);
    return i === -1 ? BASE_TYPE_ORDER.length : i;
  };
  return tree.roots
    .map((r) => ({
      name: r.name,
      label: r.label,
      children: [...r.children]
        .sort((a, b) => rank(a.base) - rank(b.base) || a.value.localeCompare(b.value))
        .map((c) => ({ value: c.value, label: c.label })),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** The children of a root, as `value -> label`. */
function childrenOf(tree, rootName) {
  const r = tree.roots.find((x) => x.name === rootName);
  assert.ok(r, `no root named ${rootName}`);
  return Object.fromEntries(r.children.map((c) => [c.value, c.label]));
}

/** Third-party shims as they could have been written against the older schema. */
const LEGACY_SCHEMA_SHIMS = [
  { name: 'extvendor--chat_completions', base: 'chat_completions', display_name: 'ExtVendor' },
  { name: 'moonshot', base: 'chat_completions', display_name: 'Moonshot AI' },
  { name: 'othervendor--open_responses', base: 'open_responses', display_name: 'OtherVendor' },
];

test('a shim declaring nothing keeps the name-derived tree, with derived provenance', () => {
  const shims = [
    { name: 'deepseek--chat_completions', base: 'chat_completions' },
    { name: 'xai', base: 'chat_completions' },
  ];
  const tree = buildTypeModel(shims, []);

  assert.deepEqual(rendered(tree), rendered(legacyTree(shims, [])));
  for (const s of shims) {
    assert.equal(shimGroupSource(s), DERIVED);
    assert.equal(shimLabelSource(s), DERIVED);
  }
  assert.equal(childrenOf(tree, 'deepseek')['deepseek--chat_completions'], 'chat_completions');
  assert.equal(childrenOf(tree, 'xai')['xai'], 'chat_completions');
});

test('a shim declaring only the old-schema display_name renders exactly as it did', () => {
  const tree = buildTypeModel(LEGACY_SCHEMA_SHIMS, []);
  const before = legacyTree(LEGACY_SCHEMA_SHIMS, []);

  // `display_name` titled the top level back then, and still does ...
  assert.deepEqual(rendered(tree), rendered(before));

  // ... and the row label is name-derived in both, so the vendor's name is not
  // printed twice (root "ExtVendor" + row "ExtVendor · chat_completions").
  assert.equal(childrenOf(tree, 'extvendor')['extvendor--chat_completions'], 'chat_completions');
  assert.equal(shimLabel(LEGACY_SCHEMA_SHIMS[0]), 'chat_completions');
  assert.equal(shimLabelSource(LEGACY_SCHEMA_SHIMS[0]), DERIVED);

  // The title branch is recorded as the legacy one, not as a declaration.
  const extvendor = tree.roots.find((r) => r.name === 'extvendor');
  assert.equal(extvendor.label, 'ExtVendor');
  assert.equal(extvendor.titleSource, LEGACY_DISPLAY_NAME);
  assert.equal(extvendor.groupSource, DERIVED);

  // A bare name takes its row label from the base, as before.
  assert.equal(childrenOf(tree, 'moonshot')['moonshot'], 'chat_completions');
  assert.equal(shimGroupKey(LEGACY_SCHEMA_SHIMS[1]), 'moonshot');
});

test('declaring a grouping field is what makes display_name the row label', () => {
  const shim = {
    name: 'alcf_metis--chat_completions',
    base: 'chat_completions',
    group: 'alcf',
    display_name: 'metis',
  };
  const tree = buildTypeModel([shim], []);

  assert.equal(shimGroupKey(shim), 'alcf');
  assert.equal(shimGroupSource(shim), DECLARED);
  assert.equal(shimLabel(shim), 'metis');
  assert.equal(shimLabelSource(shim), DECLARED);
  assert.equal(childrenOf(tree, 'alcf')[shim.name], 'metis');
});

test('the three ALCF clusters share one declared top level', () => {
  const shims = ['metis', 'minerva', 'sophia'].map((cluster) => ({
    name: `alcf_${cluster}--chat_completions`,
    base: 'chat_completions',
    group: 'alcf',
    display_name: cluster,
  }));
  const tree = buildTypeModel(shims, BASE_FORMATS);

  assert.equal(tree.roots.filter((r) => r.name === 'alcf').length, 1);
  assert.deepEqual(childrenOf(tree, 'alcf'), {
    'alcf_metis--chat_completions': 'metis',
    'alcf_minerva--chat_completions': 'minerva',
    'alcf_sophia--chat_completions': 'sophia',
  });
  for (const shim of shims) {
    assert.equal(shimGroupSource(shim), DECLARED);
    assert.equal(shimLabelSource(shim), DECLARED);
  }
  // A grouping field on any member settles the root's provenance.
  const alcf = tree.roots.find((r) => r.name === 'alcf');
  assert.equal(alcf.groupSource, DECLARED);
});

test('bare base formats and vendor groups are sectioned apart', () => {
  const tree = buildTypeModel(
    [{ name: 'acme--chat_completions', base: 'chat_completions', group: 'acme' }],
    BASE_FORMATS,
  );
  const byName = Object.fromEntries(tree.roots.map((r) => [r.name, r.kind]));
  assert.equal(byName.chat_completions, 'format');
  assert.equal(byName.open_responses, 'format');
  assert.equal(byName.acme, 'shim');

  // Every format root precedes every shim root, so the dropdown's two
  // sections never interleave.
  const kinds = tree.roots.map((r) => r.kind);
  const firstShim = kinds.indexOf('shim');
  assert.equal(kinds.lastIndexOf('format'), firstShim - 1);
});

test('Kilo keeps its look: group_label titles the root, the row stays name-derived', () => {
  const migrated = {
    name: 'kilo--chat_completions',
    base: 'chat_completions',
    group_label: 'Kilo',
    free_source: true,
  };
  const before = { name: 'kilo--chat_completions', base: 'chat_completions', display_name: 'Kilo' };

  // The migration this PR performs moves the field; the rendering does not move.
  assert.deepEqual(rendered(buildTypeModel([migrated], [])), rendered(legacyTree([before], [])));

  const tree = buildTypeModel([migrated], []);
  const kilo = tree.roots.find((r) => r.name === 'kilo');
  assert.equal(kilo.label, 'Kilo');
  assert.equal(kilo.titleSource, DECLARED);
  assert.equal(shimLabelSource(migrated), DERIVED);
  assert.equal(childrenOf(tree, 'kilo')['kilo--chat_completions'], 'chat_completions');
});

test('the free-source brand follows the group title', () => {
  // Older schema: the display_name was the brand.
  assert.equal(
    shimBrand({ name: 'extvendor--chat_completions', display_name: 'ExtVendor' }),
    'ExtVendor'
  );
  // Declared title wins.
  assert.equal(shimBrand({ name: 'kilo--chat_completions', group_label: 'Kilo' }), 'Kilo');
  assert.equal(
    shimBrand({ name: 'extvendor--chat_completions', group: 'extvendor', group_label: 'ExtVendor' }),
    'ExtVendor'
  );
  // A grouped vendor that declares no title gets its capitalized group key --
  // which is why `group_label` exists.
  assert.equal(shimBrand({ name: 'extvendor--chat_completions', group: 'extvendor' }), 'Extvendor');
  // No declaration at all: the name's first segment, capitalized.
  assert.equal(shimBrand({ name: 'extvendor--chat_completions' }), 'Extvendor');
  assert.equal(shimBrand({ name: 'moonshot' }), 'Moonshot');
});

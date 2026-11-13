/**
 * The Add Provider chooser's list and suggested id.
 *
 * A provider the chooser drops cannot be added from the screen at all, and a
 * suggestion that collides silently overwrites an existing entry — both are
 * invisible in a browser test that only checks that something rendered.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  pickDestination,
  providerPickRows,
  shimBadges,
  shimSubtitle,
  suggestedProviderName,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/provider-pick.js';

const SHIMS = [
  {
    name: 'openrouter--chat_completions',
    format: 'chat_completions',
    model_type: 'llm',
    display_name: 'OpenRouter',
    group: 'openrouter',
    default_base_url: 'https://openrouter.ai/api/v1',
  },
  {
    name: 'openai--open_responses',
    format: 'open_responses',
    model_type: 'llm',
    display_name: 'OpenAI Responses',
    group: 'openai',
  },
  {
    name: 'opencode_go--messages',
    format: 'messages',
    model_type: 'llm',
    display_name: 'OpenCode Go',
    group: 'opencode',
    free_source: true,
    keyless: true,
  },
  {
    name: 'argo--embeddings',
    format: 'openai',
    model_type: 'embeddings',
    display_name: 'Argo Embeddings',
    group: 'argo',
    mints_credential: true,
    credential_kind: 'oauth',
  },
];

test('every shim becomes a row, sorted by vendor', () => {
  const rows = providerPickRows(SHIMS);
  assert.deepEqual(
    rows.map((r) => r.title),
    ['Argo', 'Openai', 'Opencode', 'Openrouter'],
  );
});

test('the row names the format, and the model type above llm', () => {
  const rows = providerPickRows(SHIMS);
  const byName = Object.fromEntries(rows.map((r) => [r.name, r]));
  assert.match(byName['openrouter--chat_completions'].subtitle, /chat_completions/);
  assert.match(byName['argo--embeddings'].subtitle, /embeddings/);
  assert.doesNotMatch(byName['openrouter--chat_completions'].subtitle, /llm/);
});

test('the subtitle names the format, which tells same-vendor rows apart', () => {
  assert.equal(shimSubtitle(SHIMS[0]), 'chat_completions');
  assert.equal(shimSubtitle(SHIMS[1]), 'open_responses');
  assert.equal(shimSubtitle(SHIMS[3]), 'openai · embeddings');
});

test('a search matches title, id, format and vendor, case-insensitively', () => {
  const names = (q) => providerPickRows(SHIMS, q).map((r) => r.name);
  assert.deepEqual(names('openrouter'), ['openrouter--chat_completions']);
  assert.deepEqual(names('RESPONSES'), ['openai--open_responses']);
  assert.deepEqual(names('embeddings'), ['argo--embeddings']);
  assert.deepEqual(names('open'), [
    'argo--embeddings',
    'openai--open_responses',
    'opencode_go--messages',
    'openrouter--chat_completions',
  ]);
});

test('a badge is searchable, because it is on the row', () => {
  const names = (q) => providerPickRows(SHIMS, q).map((r) => r.name);
  // A row shows `free` / `keyless` / `oauth` on screen; a query for any of
  // them must find the rows that display it.
  assert.deepEqual(names('free'), ['opencode_go--messages']);
  assert.deepEqual(names('keyless'), ['opencode_go--messages']);
  assert.deepEqual(names('oauth'), ['argo--embeddings']);
});

test('an empty query lists everything', () => {
  assert.equal(providerPickRows(SHIMS, '').length, SHIMS.length);
  assert.equal(providerPickRows(SHIMS, '   ').length, SHIMS.length);
});

test('a query that matches nothing yields no rows', () => {
  assert.deepEqual(providerPickRows(SHIMS, 'zzzz-no-such-provider'), []);
});

test('an entry without a name is dropped, not rendered blank', () => {
  assert.deepEqual(providerPickRows([{ format: 'messages' }]), []);
});

test('badges say how the credential arrives and whether it costs', () => {
  const byName = Object.fromEntries(SHIMS.map((s) => [s.name, s]));
  assert.deepEqual(shimBadges(byName['opencode_go--messages']), ['free', 'keyless']);
  assert.deepEqual(shimBadges(byName['argo--embeddings']), ['oauth']);
  assert.deepEqual(shimBadges(byName['openai--open_responses']), []);
});

test('the suggested id is the vendor, lower-cased and addressable', () => {
  assert.equal(suggestedProviderName(SHIMS[0]), 'openrouter');
  assert.equal(suggestedProviderName(SHIMS[1]), 'openai');
  assert.equal(suggestedProviderName(SHIMS[2]), 'opencode');
});

test('a second copy of the same vendor gets a distinct id', () => {
  assert.equal(suggestedProviderName(SHIMS[0], ['openrouter']), 'openrouter-2');
  assert.equal(
    suggestedProviderName(SHIMS[0], ['openrouter', 'openrouter-2']),
    'openrouter-3',
  );
});

test('a name that cannot be a config key falls back to the shim id', () => {
  assert.equal(
    suggestedProviderName({ name: 'a--b', display_name: '***', group: '' }),
    'a--b',
  );
  assert.equal(suggestedProviderName({ name: '' }), '');
});

test('a de-duplicated id fits the same length limit as an original', () => {
  const long = { name: 'x--chat_completions', group: 'k'.repeat(60) };
  assert.equal(suggestedProviderName(long).length, 48);
  const second = suggestedProviderName(long, [suggestedProviderName(long)]);
  assert.ok(second.length <= 48, `id is ${second.length} characters`);
  assert.match(second, /-2$/);
  // Cutting to make room must not leave a trailing separator behind.
  const dashed = { name: 'x--chat_completions', group: `${'k'.repeat(45)}-abcdef` };
  const taken = [suggestedProviderName(dashed)];
  assert.doesNotMatch(suggestedProviderName(dashed, taken), /--/);
});

test('uniqueness compares the way the config loader does, lower-cased', () => {
  // A config holding `OpenRouter` already owns `openrouter`.
  assert.equal(
    suggestedProviderName(SHIMS[0], ['OpenRouter']),
    'openrouter-2',
  );
  // And a de-duplicated id that exists in another case is taken too.
  assert.equal(
    suggestedProviderName(SHIMS[0], ['openrouter', 'OPENROUTER-2']),
    'openrouter-3',
  );
});

test('a pick lands in the control that can hold it', () => {
  // The modal's type tree is built from the LLM shims, so a non-LLM row must
  // be routed to its endpoint section instead of an LLM-only picker.
  assert.deepEqual(pickDestination(SHIMS[0]), { type: 'llm', control: 'llm' });
  assert.deepEqual(pickDestination(SHIMS[3]), {
    type: 'embeddings',
    control: 'endpoint',
  });
  // A shim that declares no type is an LLM shim, as everywhere else.
  assert.deepEqual(pickDestination({ name: 'a--b' }), {
    type: 'llm',
    control: 'llm',
  });
});

// The wiring below runs only in a browser, so it is asserted against the
// source text: these are exactly the lines whose deletion every unit test
// above still passes.
const here = dirname(fileURLToPath(import.meta.url));
const providersSrc = readFileSync(
  join(
    here,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/providers.js',
  ),
  'utf8',
);
const pickFn = providersSrc.slice(
  providersSrc.indexOf('function pickProviderShim('),
  providersSrc.indexOf('function startBlankProvider('),
);

test('a picked shim opens the form as a create, not an edit', () => {
  // `_editingProviderName` is what decides create-or-overwrite; a prefill
  // passed without `isNew` would edit whatever already holds that id.
  assert.match(
    pickFn,
    /openProviderModal\([^;]*\{\s*isNew:\s*true\s*\}\s*\);/,
    'the pick must pass isNew: true',
  );
  assert.match(
    providersSrc,
    /_editingProviderName = isNew \? null : /,
    'openProviderModal must honour isNew',
  );
});

test('a pick reveals the control that holds it', () => {
  // The capability checkbox is what shows the section: unticked, an LLM pick
  // leaves the type tree invisible and Save answers "no capability".
  assert.match(pickFn, /_selectPickedCapability\(type, control === 'llm' \? '' : shim\.name\)/);
  const selectFn = providersSrc.slice(
    providersSrc.indexOf('function _selectPickedCapability('),
    providersSrc.indexOf('function startBlankProvider('),
  );
  // Each statement is asserted, not just the id it is built from: the
  // checkbox lookup alone is satisfied by `chk.checked = false`, and the
  // select alone by a call without the membership guard.
  assert.match(selectFn, /getElementById\(`provCap\$\{_capitalize\(type\)\}`\)/);
  assert.match(selectFn, /chk\.checked = true;/);
  assert.match(selectFn, /toggleProvCapSection\(\);/);
  assert.match(selectFn, /if \(picker && picker\.has\(shimName\)\) picker\.select\(shimName\);/);
});

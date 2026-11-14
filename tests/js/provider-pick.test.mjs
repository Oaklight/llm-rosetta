/**
 * Where a shim row on the Shims page lands, and what the Add form calls it.
 *
 * The chooser dialog these rules came from is gone; the rules are not.  A
 * non-LLM row routed to the LLM type tree opens a form that cannot hold it,
 * and a suggestion that collides silently overwrites an existing entry —
 * neither is visible in a browser test that only checks that something
 * rendered.
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
  assert.equal(suggestedProviderName(SHIMS[0], ['OpenRouter']), 'openrouter-2');
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
  providersSrc.indexOf('function startProviderFromShim('),
  providersSrc.indexOf('function _selectPickedCapability('),
);

test('a shim row opens the form as a create, not an edit', () => {
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
  // The shims page is the only entry point now, and both of its buttons have
  // to go through this function rather than open the form directly.
  const shimsPage = readFileSync(
    join(
      here,
      '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/shims-page.js',
    ),
    'utf8',
  );
  assert.doesNotMatch(shimsPage, /openProviderModal\(/);
  assert.equal(
    (shimsPage.match(/onclick="startProviderFromShim\(/g) || []).length,
    2,
    'both Add buttons (row and vendor header) call the same entry point',
  );
});

test('a shim row reveals the control that holds it', () => {
  // The capability checkbox is what shows the section: unticked, a pick
  // leaves the control invisible and Save answers "no capability".
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

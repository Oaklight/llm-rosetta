import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// The credential section's state machine runs only in a browser, so these two
// wiring points are asserted against the source text.  Both are exactly the
// kind of fix that passes every unit test after being deleted.

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(
    here,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/providers.js',
  ),
  'utf8',
);

test('the credential actions are reachable from an inline handler', () => {
  // The buttons are built with `fn: 'refreshProviderCredential()'`; an inline
  // handler resolves against window, and a module function is not there.
  const exportList = source.match(/Object\.assign\(window,\s*\{([\s\S]*?)\}\);/);
  assert.ok(exportList, 'providers.js must export its inline handlers');
  for (const name of ['refreshProviderCredential', 'testProviderCredential']) {
    assert.match(exportList[1], new RegExp(`\\b${name}\\b`), name);
  }
});

test('the save path asks whether the section read this provider', () => {
  // `credentialLoaded` decides whether an empty command box means "clear it".
  // It must come from the helper, not from a constant that happens to pass.
  assert.match(
    source,
    /credentialLoadedFor\(\s*_credentialFor,\s*_editingProviderName,\s*_credentialData\s*\)/,
  );
});

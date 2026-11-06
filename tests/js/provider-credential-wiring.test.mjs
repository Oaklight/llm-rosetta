import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// The credential section's state machine runs only in a browser, so these two
// wiring points are asserted against the source text.  Both are exactly the
// kind of fix that passes every unit test after being deleted.

const here = dirname(fileURLToPath(import.meta.url));
const ADMIN = join(
  here,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin',
);
const source = readFileSync(join(ADMIN, 'js/tabs/providers.js'), 'utf8');
const html = readFileSync(join(ADMIN, 'admin.html'), 'utf8');

test('the credential actions are reachable from an inline handler', () => {
  // The buttons are built with `fn: 'refreshProviderCredential()'`; an inline
  // handler resolves against window, and a module function is not there.
  const exportList = source.match(/Object\.assign\(window,\s*\{([\s\S]*?)\}\);/);
  assert.ok(exportList, 'providers.js must export its inline handlers');
  for (const name of [
    'refreshProviderCredential', 'testProviderCredential',
    'setProviderPane', 'selectCredentialSource', 'addProviderArg',
    'refreshProviderAuthPane',
  ]) {
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


test('the save tells the backend to delete a key it is replacing', () => {
  // `api_key: ""` has always meant "keep the stored key".  Moving to a
  // command or a sign-in has to say "remove it" in a different word.
  assert.match(
    source,
    /credentialSourceFields\(\{\s*to:\s*credSource,\s*needsKey\s*\}\)/,
    'the save sends clear_api_key through the helper',
  );
  // Leaving a command for another source clears an override the section may
  // not have read; without it the loader sees a key and a command at once.
  assert.match(source, /_credSourceOriginal === 'config'/);
  assert.match(source, /body\.token_command = \[\]/);
});

test('the drawer opens on Basic and the panes are toggled, not rebuilt', () => {
  const open = source.slice(source.indexOf('function openProviderModal('));
  assert.match(open, /setProviderPane\('basic'\)/, 'a new open resets to Basic');
  assert.match(open, /_clearProviderDrawerState\(/, 'the Auth state resets per provider');

  const pane = source.slice(
    source.indexOf('function setProviderPane('),
    source.indexOf('function selectCredentialSource('),
  );
  assert.match(pane, /\.hidden = /, 'panes are hidden, not destroyed');
  assert.doesNotMatch(pane, /innerHTML/, 'a pane is never rebuilt');
  assert.doesNotMatch(pane, /_applyType/, 'a tab switch must not re-run the type closure');

  // Both panes exist in the static markup, so no render path has to create
  // them — which is what keeps a multi-key row and a pasted record alive.
  assert.match(html, /id="provPaneBasic"/);
  assert.match(html, /id="provPaneAuth"[^>]*hidden/);
  assert.match(html, /id="provTabAuth"/);
});

test('the key box lives in the Auth pane, not above the form', () => {
  const paneAt = html.indexOf('id="provPaneAuth"');
  const basic = html.slice(html.indexOf('id="provPaneBasic"'), paneAt);
  const auth = html.slice(paneAt);
  const keyAt = html.indexOf('id="provApiKeyGroup"');
  assert.ok(keyAt > paneAt, 'the key group sits in Auth');
  assert.ok(auth.indexOf('id="provApiKeyGroup"') < auth.indexOf('id="provStatusCard"'),
    'and above the status card');
  assert.equal(basic.indexOf('id="provApiKeyGroup"'), -1);
});


test('a draft edit disables Refresh and Test at once', () => {
  // The buttons run the *saved* command.  Re-rendering the actions only when
  // the pane next opens left them enabled while the operator typed, which
  // was exactly the state the plan says must not exist.
  const edits = source.match(
    /_credFormDirty = true;[\s\S]{0,240}?_renderCredentialPanels\(\)/g,
  ) || [];
  assert.ok(
    edits.length >= 4,
    `every credential field re-renders the actions (found ${edits.length})`,
  );
  assert.match(source, /_credentialDraftDirty\(\)/);
  assert.match(source, /disabled: stale/);
});


test('the last argv row offers no delete button, and says so', () => {
  // Emptying the list could not clear the command anyway: `_seedArgv` fills
  // the rows from the entry, and the save refuses an empty argv.  The × is
  // hidden instead, so the operator is never shown a button that undoes
  // itself.
  assert.match(source, /_credArgv\.length > 1/, 'the × needs a second row');
  assert.match(html, /id="provArgvClearHint"/);
  assert.match(
    source,
    /clearHint\.hidden = readonly \|\| _credArgv\.length !== 1/,
  );
});

test('the key box repaints the preview like every other credential input', () => {
  const wire = source.slice(
    source.indexOf('function _wireCredentialFields('),
    source.indexOf('function _renderArgv('),
  );
  assert.match(wire, /getElementById\('provApiKey'\)[\s\S]{0,200}?_renderPreview\(\)/);
  // Multi-key rows are rebuilt on each edit; their container carries the
  // listener so every row repaints too.
  assert.match(wire, /getElementById\('provApiKeyMultiBox'\)[\s\S]{0,200}?_renderPreview\(\)/);
});


test('the Auth tab dot is named, and the name follows the state', () => {
  // An empty span with a colour is invisible to assistive tech; the dot's
  // meaning lives in its title and aria-label, which the summary rewrites as
  // the credential state changes.
  assert.match(html, /id="provAuthBadge"[^>]*role="img"/);
  assert.match(html, /id="provAuthBadge"[^>]*data-i18n-title="prov.authNeedsAttention"/);
  assert.match(source, /badge\.setAttribute\('aria-label', label\)/);
  assert.match(source, /_stateLabel\(state\)/);
});

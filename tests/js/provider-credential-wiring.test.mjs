import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
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
const utils = readFileSync(join(ADMIN, 'js/core/utils.js'), 'utf8');
const i18n = readFileSync(join(ADMIN, 'js/core/i18n.js'), 'utf8');
// These assertions pin wiring, so they must read code, not prose: a comment
// that names the pattern they look for would otherwise satisfy them.  A plain
// regex cannot do this — `//` also opens `https://` inside a string — so walk
// the source and drop comments while keeping string literals intact.
function stripJsComments(src) {
  let out = '';
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    const next = src[i + 1];
    if (ch === '/' && next === '/') {
      const end = src.indexOf('\n', i);
      i = end === -1 ? src.length : end;
      continue;
    }
    if (ch === '/' && next === '*') {
      const end = src.indexOf('*/', i + 2);
      i = end === -1 ? src.length : end + 2;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      const quote = ch;
      out += ch;
      i += 1;
      while (i < src.length) {
        out += src[i];
        if (src[i] === '\\') { out += src[i + 1] ?? ''; i += 2; continue; }
        if (src[i] === quote) { i += 1; break; }
        i += 1;
      }
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

const code = stripJsComments(source);

test('the comment stripper leaves string literals alone', () => {
  // `//` inside `https://` is not a comment; a text-level strip turned the
  // string into `https:` and would have made a future assertion red for the
  // wrong reason.
  const sample = stripJsComments("const url = 'https://api.openai.com/v1'; // note\nconst x = 1; /* block */");
  assert.match(sample, /'https:\/\/api\.openai\.com\/v1'/);
  assert.doesNotMatch(sample, /note|block/);
  assert.match(sample, /const x = 1;/);
});

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

test('the browser sign-in submits the callback URL and nothing else', () => {
  // Tokens stay server-side; the panel's payload is the operator's pasted URL.
  assert.match(source, /credential\/login`/);
  assert.match(source, /credential\/login\/callback`/);
  assert.match(source, /\{callback_url: callbackUrl\}/);
  assert.doesNotMatch(
    source.slice(source.indexOf('async function completeProviderBrowserLogin(')),
    /access_token|refresh_token|id_token/,
    'the completion request must not carry a token',
  );
});

test('the browser sign-in controls are statically present and exported', () => {
  for (const id of ['provBrowserLoginBox', 'provBrowserLoginLink', 'provBrowserLoginUrl', 'provBrowserCallback']) {
    assert.match(html, new RegExp(`id="${id}"`), id);
  }
  const exportList = source.match(/Object\.assign\(window,\s*\{([\s\S]*?)\}\);/);
  assert.ok(exportList);
  for (const name of [
    'beginProviderBrowserLogin', 'completeProviderBrowserLogin',
    'resetProviderBrowserLogin',
  ]) {
    assert.match(exportList[1], new RegExp(`\\b${name}\\b`), name);
  }
});

test('a CLI record stays installable beside the browser flow', () => {
  // The two are alternative ways to produce one record; hiding install would
  // strand a record the operator already generated on another machine.
  const panel = source.slice(
    source.indexOf('function _renderExchangePanel('),
    source.indexOf('function _credentialDraftDirty('),
  );
  assert.match(panel, /data\.can_begin_login/);
  assert.match(panel, /data\.can_install !== false/);
  assert.doesNotMatch(
    panel,
    /else if \(data\.can_install !== false\)/,
    'install must not be an exclusive fallback for the browser flow',
  );
});

test('the sign-in tab is opened inside the click, not after the await', () => {
  // A window.open() after an async round trip is what browsers block; the
  // blank tab is opened first and its location filled when the URL arrives.
  const fn = source.slice(
    source.indexOf('async function beginProviderBrowserLogin('),
    source.indexOf('async function completeProviderBrowserLogin('),
  );
  const openAt = fn.indexOf("window.open('about:blank'");
  const awaitAt = fn.indexOf('await api.post(');
  assert.ok(openAt >= 0, 'the blank tab is opened');
  assert.ok(openAt < awaitAt, 'the tab is opened before the request');
  assert.match(fn, /opened\.location\.href = result\.authorization_url/);
  assert.match(fn, /getElementById\('provBrowserLoginUrl'\)/);
  assert.match(fn, /opened\.close\(\)/, 'a failed start closes the blank tab');
});

test('an entry command is an override, not a peer credential source', () => {
  // Four equal cards made "sign in" and "run this argv instead" look like the
  // same kind of choice; the command is an override of the shim's own source.
  assert.match(html, /id="provOverrideBtn"/);
  assert.match(source, /const overriding = ui === 'config'/);
  assert.match(source, /overriding \? id !== 'config'/);
  assert.match(source, /id === 'config' \|\| !avail\[id\]/);
  const exportList = source.match(/Object\.assign\(window,\s*\{([\s\S]*?)\}\);/);
  assert.ok(exportList);
  assert.match(exportList[1], /\btoggleCommandOverride\b/);
});

test('the expiry row is readable and offers a countdown', () => {
  assert.match(html, /id="provExchangeRefreshHint"/);
  assert.match(code, /formatCredentialExpiry\(summary\.expires_at\)/);
  // The value is an action, so it is a button, and the row is only drawn when
  // the row builder is actually called.
  assert.match(code, /_renderExpiryRow\(rows, summary\.expires_at\)/);
  assert.match(code, /createElement\('button'\)/);
  assert.match(code, /prov\.autoRefreshOn/);
  // The countdown's word comes from the dictionary, not from the helper.
  assert.match(code, /formatCountdown\(expiresAt, Date\.now\(\),/);
  assert.match(code, /prov\.expiryExpired/);
});

test('a refresh keeps the expiry view the operator chose', () => {
  // The refresh path re-renders the panel; resetting to the wall clock here
  // would silently undo a countdown the operator had switched on.
  assert.match(code, /let _expiryCountdown = false;/);
  assert.match(code, /let showCountdown = _expiryCountdown;/);
  assert.match(code, /_expiryCountdown = showCountdown;/);
  assert.match(code, /if \(showCountdown\) _expiryTimer = setInterval\(paint, 1000\);/);
});

test('the refresher warning is said once, not once per card', () => {
  // The exchange hint sits beside the expiry; the status card lower down used
  // to repeat the same sentence when the refresher was off.
  assert.match(code, /const hintSaysIt = ui === 'exchange' && data\.state === 'connected';/);
  assert.match(code, /data\.refresher_enabled === false && !hintSaysIt/);
});

test('the countdown ticks only while it is on screen, and stops with the drawer', () => {
  // An always-running 1 s interval kept ticking after the drawer closed; the
  // timer now belongs to the countdown and the close path clears it.
  assert.match(code, /if \(showCountdown\) \{\s*_expiryTimer = setInterval\(paint, 1000\);/);
  assert.doesNotMatch(code, /_expiryTimer = setInterval\(\(\) => \{ if \(showCountdown\)/);
  assert.match(code, /else \{\s*_clearExpiryTimer\(\);/);
  assert.match(code, /if \(_provPane === 'basic'\) _clearExpiryTimer\(\);/);
  const exportList = code.match(/Object\.assign\(window,\s*\{([\s\S]*?)\}\);/);
  assert.ok(exportList, 'providers.js must export its inline handlers');
  assert.match(exportList[1], /\bclearProviderExpiryTimer\b/);
  assert.match(utils, /if \(id === 'providerModal'\) window\.clearProviderExpiryTimer\?\.\(\);/);
});

test('the override control describes the action it performs', () => {
  assert.match(code, /_renderOverrideButton\(avail, ui\);/);
  assert.match(code, /const removing = ui === 'config'/);
  assert.match(code, /prov\.overrideRemoveHint/);
  // Three close/open paths repaint the pane; the Override label is a built
  // string, so the language switch has to repaint it too.
  const refresh = code.slice(code.indexOf('function refreshProviderAuthPane('));
  assert.match(refresh.slice(0, refresh.indexOf('\n}')), /_renderSourceCards\(_availableSources\(\), _uiSource\(\)\)/);
});

test('the dictionary parses as a module, not only as text', () => {
  // The key checks below read the dictionary as text, so they pass over a
  // missing comma between two entries; the browser then fails to load the
  // module and the whole panel stays blank.  Parse it the way a browser does.
  assert.doesNotThrow(
    () => execFileSync(process.execPath, ['--input-type=module', '--check'], {
      input: i18n,
      stdio: ['pipe', 'pipe', 'pipe'],
    }),
    'i18n.js must parse as an ES module',
  );
});

test('a missing dictionary key falls back to the sentence, not to the key', () => {
  // `t(key, fallback)` passed the fallback as interpolation params, which made
  // it inert: a typo'd key rendered the key itself.
  assert.doesNotMatch(code, /\bt\(key, fallback\)/);
  assert.match(code, /return value && value !== key \? value : fallback;/);
  // The guard in tests/js/i18n-keys.mjs cannot see `_credentialText` call
  // sites, so check the new keys exist in both dictionaries here.  Counting
  // the whole file would accept two copies in one dictionary; read each
  // dictionary body instead.
  const enBody = i18n.slice(i18n.indexOf('  en: {'), i18n.indexOf('  zh: {'));
  const zhStart = i18n.indexOf('  zh: {');
  const zhBody = i18n.slice(zhStart, i18n.indexOf('\n};', zhStart));
  assert.ok(enBody.length && zhBody.length, 'both dictionaries are in i18n.js');
  for (const key of ['prov.expiryExpired', 'prov.overrideRemoveHint']) {
    assert.equal(enBody.split(`'${key}'`).length - 1, 1, `${key} in the en dictionary`);
    assert.equal(zhBody.split(`'${key}'`).length - 1, 1, `${key} in the zh dictionary`);
  }
});

test('the Basic pane states one source instead of offering a picker', () => {
  // Overrides live in Auth; the drawer home must not show two credential
  // forms as peer buttons.
  assert.match(code, /chip\.hidden = !current/);
  assert.doesNotMatch(code, /chip\.hidden = !avail\[id\]/);
  // The chip opens Auth; it does not toggle, so `aria-pressed` would promise
  // an action it does not perform.
  assert.match(code, /chip\.setAttribute\('aria-current', 'true'\)/);
  assert.doesNotMatch(code, /chip\.setAttribute\('aria-pressed'/);
});

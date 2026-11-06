import test from 'node:test';
import assert from 'node:assert/strict';

import {
  safeLinkUrl, parseRecord, credentialSource, credentialNeedsKey,
  credentialLoadedFor, credentialSaveFields, parseCommandArgv, shortCredential,
  credentialSourceId, credentialUiSource, availableCredentialSources,
  credentialFallbackSource, credentialSourceFields, credentialNeedsAttention,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/credential.js';

test('a link keeps https and normalises it', () => {
  assert.equal(safeLinkUrl('https://chatgpt.com/settings/usage'), 'https://chatgpt.com/settings/usage');
});

test('a link refuses every non-web scheme', () => {
  for (const bad of [
    'javascript:window.__pwned=1',
    'data:text/html,<script>alert(1)</script>',
    'file:///etc/passwd',
    'vbscript:msgbox(1)',
    // Vendor usage pages are https; plain http would only be mixed content.
    'http://example.test/x',
  ]) {
    assert.equal(safeLinkUrl(bad), null, bad);
  }
});

test('a link refuses what is not a URL', () => {
  assert.equal(safeLinkUrl(''), null);
  assert.equal(safeLinkUrl('not a url'), null);
  assert.equal(safeLinkUrl(undefined), null);
  assert.equal(safeLinkUrl(null), null);
  assert.equal(safeLinkUrl(42), null);
});

test('a record must be a JSON object', () => {
  assert.deepEqual(parseRecord('{"access_token":"a"}'), { access_token: 'a' });
  assert.deepEqual(parseRecord('{}'), {});
});

test('a record refuses everything else before a request leaves the page', () => {
  for (const bad of ['not json', '[1,2]', 'null', '"text"', '42', '']) {
    assert.equal(parseRecord(bad), null, bad);
  }
  assert.equal(parseRecord(undefined), null);
});

test('the credential source follows the builder precedence', () => {
  const shim = { mints_credential: true, credential_kind: 'oauth' };

  // A command written into the provider entry outranks the shim.
  assert.equal(credentialSource(shim, { token_command: ['t'] }), 'config_command');
  assert.equal(credentialSource({ mints_credential: true, credential_kind: 'command' }, { token_command: ['t'] }), 'config_command');
  // The shim's auth.py, then its command, then a plain key.
  assert.equal(credentialSource(shim, {}), 'auth_exchange');
  assert.equal(credentialSource({ mints_credential: true, credential_kind: 'command' }, {}), 'shim_command');
  assert.equal(credentialSource({ mints_credential: false, credential_kind: null }, {}), 'static_key');
  assert.equal(credentialSource(null, null), 'static_key');
  // A shim that no longer loads does not erase the live source.
  assert.equal(credentialSource(null, { credential_source: 'shim_command' }), 'shim_command');
  assert.equal(credentialSource(null, { mints_credential: true }), 'auth_exchange');
});

test('a command that consumes the key keeps the key box', () => {
  // AskSage: the account key is the command's input, not a second credential.
  assert.equal(credentialNeedsKey('shim_command', { default_token_command_env: 'ASK_KEY' }, {}), true);
  assert.equal(credentialNeedsKey('config_command', {}, { token_command_env: 'V' }), true);
  // A plain command stores no key.
  assert.equal(credentialNeedsKey('shim_command', {}, {}), false);
  assert.equal(credentialNeedsKey('config_command', {}, {}), false);
  // Neither does an exchange or a static key.
  assert.equal(credentialNeedsKey('auth_exchange', { default_token_command_env: 'X' }, {}), false);
  assert.equal(credentialNeedsKey('static_key', {}, {}), false);
});

test('an argv field is a non-empty array of non-empty strings', () => {
  assert.deepEqual(parseCommandArgv('["python3","token.py"]'), ['python3', 'token.py']);
  for (const bad of ['[]', '{}', '"python3"', '["python3", ""]', '[1, 2]', 'not json', '']) {
    assert.equal(parseCommandArgv(bad), null, bad);
  }
  assert.equal(parseCommandArgv(undefined), null);
});

test('a save trusts a read only when it is for the provider on screen', () => {
  const read = { state: 'connected' };
  assert.equal(credentialLoadedFor('p', 'p', read), true);
  // The section moved on before the read landed.
  assert.equal(credentialLoadedFor('q', 'p', read), false);
  // The drawer is new, or the read never ran.
  assert.equal(credentialLoadedFor(null, null, read), false);
  assert.equal(credentialLoadedFor('p', 'p', null), false);
  // The GET failed; the box shows a fallback, not the stored command.
  assert.equal(credentialLoadedFor('p', 'p', { state: 'error' }), false);
});

test('a save sends a command only after the section read the provider', () => {
  // The read failed, the box shows a fallback, and the operator saved
  // something else: an empty box is not "clear the command".
  assert.deepEqual(
    credentialSaveFields({ source: 'config_command', loaded: false, command: '', interval: '3600', env: '' }),
    { fields: { token_refresh_interval: 3600 } },
  );
  // Loaded: an empty box is the operator clearing it.
  assert.deepEqual(
    credentialSaveFields({ source: 'config_command', loaded: true, command: '', interval: '3600', env: 'KEY' }),
    { fields: { token_command: [], token_command_env: '', token_refresh_interval: 3600 } },
  );
  // An explicit argv is sent even when the live credential could not be read:
  // switching a static provider to a command has no credential to read.
  assert.deepEqual(
    credentialSaveFields({ source: 'config_command', loaded: false, command: '["v","x"]', interval: '60', env: '' }),
    { fields: { token_command: ['v', 'x'], token_command_env: '', token_refresh_interval: 60 } },
  );
  // A shim's argv is read-only, so there is nothing to send for it — but a
  // stored override is cleared, because choosing the shim means "the shim's
  // command", and an entry cannot carry both.  The env var is left alone:
  // the entry has none, and an empty one would override the shim's own.
  assert.deepEqual(
    credentialSaveFields({ source: 'shim_command', loaded: true, command: '["v","x"]', interval: '90', env: '' }),
    { fields: { token_command: [], token_refresh_interval: 90 } },
  );
  // A static key clears a stored command and has no interval of its own.
  assert.deepEqual(
    credentialSaveFields({ source: 'static_key', loaded: true, command: '', interval: '', env: '' }),
    { fields: { token_command: [] } },
  );
  // An entry that *does* carry an override env is the one case that removes
  // it; otherwise the field is not written at all.
  assert.deepEqual(
    credentialSaveFields({
      source: 'shim_command', loaded: true, command: '', interval: '60', env: '',
      clearCommandEnv: true,
    }),
    { fields: { token_command: [], token_command_env: '', token_refresh_interval: 60 } },
  );
  // An unread section leaves the stored command alone, whatever the source.
  assert.deepEqual(
    credentialSaveFields({ source: 'static_key', loaded: false, command: '', interval: '', env: '' }),
    { fields: {} },
  );
});

test('a save refuses an argv or an interval it cannot store', () => {
  assert.deepEqual(
    credentialSaveFields({ source: 'config_command', loaded: true, command: 'not json', interval: '3600' }),
    { error: 'token_command_argv' },
  );
  assert.deepEqual(
    credentialSaveFields({ source: 'config_command', loaded: true, command: '["v"]', interval: '30' }),
    { error: 'token_refresh_interval' },
  );
  assert.deepEqual(
    credentialSaveFields({ source: 'auth_exchange', loaded: true, interval: '' }),
    { error: 'token_refresh_interval' },
  );
});

test('a loaded command is stored as argv and its env var is trimmed', () => {
  assert.deepEqual(
    credentialSaveFields({
      source: 'config_command',
      loaded: true,
      command: ' ["python3", "token.py"] ',
      env: '  ACCOUNT_KEY  ',
      interval: '7200',
    }),
    {
      fields: {
        token_command: ['python3', 'token.py'],
        token_command_env: 'ACCOUNT_KEY',
        token_refresh_interval: 7200,
      },
    },
  );
});

test('a fingerprint names the kind and the first bytes of the digest', () => {
  assert.equal(shortCredential('cmd:0123456789abcdef'), 'cmd:0123456789ab…');
  assert.equal(shortCredential('key:fedcba9876543210'), 'key:fedcba987654…');
  for (const bad of ['', 'nocolon', ':', 'cmd:']) {
    assert.equal(shortCredential(bad), null, bad);
  }
  assert.equal(shortCredential(null), null);
});


test('the card ids and the loader source ids map both ways', () => {
  assert.equal(credentialSourceId('static'), 'static_key');
  assert.equal(credentialSourceId('config'), 'config_command');
  assert.equal(credentialSourceId('shim'), 'shim_command');
  assert.equal(credentialSourceId('exchange'), 'auth_exchange');
  // An unknown card never becomes a minting source by accident.
  assert.equal(credentialSourceId('nope'), 'static_key');

  assert.equal(credentialUiSource('static_key'), 'static');
  assert.equal(credentialUiSource('config_command'), 'config');
  assert.equal(credentialUiSource('shim_command'), 'shim');
  assert.equal(credentialUiSource('auth_exchange'), 'exchange');
  assert.equal(credentialUiSource(undefined), 'static');
});

test('a source card is offered only when the loader can honour it', () => {
  const keyed = { mints_credential: false, credential_kind: null };
  assert.deepEqual(availableCredentialSources(keyed, {}), {
    static: true, config: true, shim: false, exchange: false,
  });

  const command = { mints_credential: true, credential_kind: 'command' };
  assert.deepEqual(availableCredentialSources(command, {}), {
    static: false, config: true, shim: true, exchange: false,
  });

  const oauth = { mints_credential: true, credential_kind: 'oauth' };
  assert.deepEqual(availableCredentialSources(oauth, {}), {
    static: false, config: true, shim: false, exchange: true,
  });

  // A keyless shim mints nothing, so a paid key is still a legal choice.
  const keyless = { mints_credential: false, credential_kind: null, keyless: true };
  assert.equal(availableCredentialSources(keyless, {}).static, true);

  // The shim stopped loading: the entry's live source still gets a card.
  assert.equal(
    availableCredentialSources(null, { credential_source: 'auth_exchange', mints_credential: true }).exchange,
    true,
  );
  // ...but a provider whose command lives in the entry has no exchange to
  // install a record into, even though it mints and its shim is gone.
  assert.deepEqual(
    availableCredentialSources(null, { credential_source: 'config_command', mints_credential: true }),
    { static: false, config: true, shim: false, exchange: false },
  );
  // A registered shim that ships an exchange keeps the card available while
  // the entry is on a command override.
  assert.equal(
    availableCredentialSources(oauth, { credential_source: 'config_command', mints_credential: true }).exchange,
    true,
  );
  // No shim at all: an argv override and a key are the only real choices.
  assert.deepEqual(availableCredentialSources(null, null), {
    static: true, config: true, shim: false, exchange: false,
  });
});

test('a selection that becomes unavailable falls back where the save is honest', () => {
  const oauth = { static: false, config: true, shim: false, exchange: true };
  assert.equal(credentialFallbackSource(oauth, 'static'), 'exchange');
  assert.equal(credentialFallbackSource(oauth, 'exchange'), 'exchange');
  const plain = { static: true, config: true, shim: false, exchange: false };
  assert.equal(credentialFallbackSource(plain, 'static'), 'static');
  assert.equal(credentialFallbackSource(plain, 'shim'), 'config');
});

test('leaving the static source removes the stored key', () => {
  // An empty api_key has always meant "keep it"; deleting needs a flag.
  assert.deepEqual(
    credentialSourceFields({ to: 'config_command', needsKey: false }),
    { fields: { clear_api_key: true } },
  );
  assert.deepEqual(
    credentialSourceFields({ to: 'auth_exchange', needsKey: false }),
    { fields: { clear_api_key: true } },
  );
  // The exchange takes the operator's key as input: clearing it would break
  // the sign-in it is meant to enable.
  assert.deepEqual(
    credentialSourceFields({ to: 'auth_exchange', needsKey: true }),
    { fields: {} },
  );
  assert.deepEqual(
    credentialSourceFields({ to: 'static_key', needsKey: false }),
    { fields: {} },
  );
});

test('an exchange that consumes the key keeps the key box', () => {
  assert.equal(
    credentialNeedsKey('auth_exchange', { consumes_key: true }, {}),
    true,
  );
  assert.equal(credentialNeedsKey('auth_exchange', { consumes_key: false }, {}), false);
  assert.equal(credentialNeedsKey('auth_exchange', null, {}), false);
});

test('only a credential that needs an action badges the Auth tab', () => {
  assert.equal(credentialNeedsAttention('static', { state: 'error' }, false), false);
  assert.equal(credentialNeedsAttention('exchange', null, false), false);
  for (const state of ['needs_setup', 'unreadable', 'failing', 'error']) {
    assert.equal(credentialNeedsAttention('exchange', { state }, false), true, state);
  }
  for (const state of ['connected', 'initializing']) {
    assert.equal(credentialNeedsAttention('config', { state }, false), false, state);
  }
  // The legacy marker is worth a badge even on a static entry.
  assert.equal(credentialNeedsAttention('static', null, true), true);
});

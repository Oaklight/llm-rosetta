/**
 * What the panel says about its own configuration (#1502 §2).
 *
 * Two statements it used to get wrong or leave out: an unprotected panel said
 * nothing, and the API Keys page claimed the endpoints were open whatever
 * `server.open_on_no_keys` said — its default is `false`, so the page told the
 * operator their gateway was open while it was refusing every request.
 *
 * Pure functions, so this runs under node like `pagination.test.mjs`.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));

import {
  keysEmptyMessageKey,
  showsNoPasswordBanner,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/posture.js';

test('the empty keys page states the outcome that is in force', () => {
  assert.equal(keysEmptyMessageKey(true), 'keys.noKeysOpen');
  assert.equal(keysEmptyMessageKey(false), 'keys.noKeysBlocked');
});

test('a missing switch reads as closed, like the gateway does', () => {
  // `server.open_on_no_keys` is absent on most configs and the gateway then
  // refuses unauthenticated requests; the page must not claim otherwise.
  assert.equal(keysEmptyMessageKey(undefined), 'keys.noKeysBlocked');
});

test('the no-password notice waits for the server to answer', () => {
  assert.equal(showsNoPasswordBanner(false), true);
  assert.equal(showsNoPasswordBanner(true), false);
  // /auth-check has not answered: showing it here would flash on every load.
  assert.equal(showsNoPasswordBanner(undefined), false);
});

test('the password form gives way to an instruction, not to nothing', () => {
  // A form that can only 400 is worse than no form, but hiding it without a
  // path off the screen would strand the operator.  The guard pins both halves
  // and the key swap that keeps them translated.
  const text = readFileSync(
    join(
      HERE,
      '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/auth.js',
    ),
    'utf8',
  );
  assert.match(text, /function renderPasswordGroup\(unprotected\)/);
  assert.match(text, /'\.settings-group:has\(h4\[data-i18n="label\.changePassword"\]\)'/);
  assert.match(text, /part\.style\.display = unprotected \? 'none' : ''/);
  assert.match(text, /setAttribute\('data-i18n', unprotected \? 'hint\.noPasswordCli' : 'hint\.changePassword'\)/);
});

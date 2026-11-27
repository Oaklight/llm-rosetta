/**
 * The extension card's statuses.
 *
 * The listing says, for each external program an extension drives, whether it
 * is ready, missing or older than the minimum, and the command that installs
 * it.  These are the pure builders the panel paints from.  No DOM, so node runs
 * them directly.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  cardStatus, extensionCards, installOutcome, installRequest, programRow, programStatus,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/extension-cards.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const I18N = join(
  HERE,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/i18n.js',
);

const CMD = 'llm-rosetta ext install-cli claude';
const claude = (over = {}) => ({
  id: 'claude',
  version: '0.1.0',
  state: 'loaded',
  external: [{
    name: 'claude', status: 'missing', found: null, minimum: '2.1.299',
    install: { version: '2.1.299', source: 'npm' }, install_command: CMD, ...over,
  }],
});

test('a program is missing, too old or ready, and anything else is unknown', () => {
  assert.equal(programStatus({ status: 'ready' }), 'ready');
  assert.equal(programStatus({ status: 'missing' }), 'missing');
  assert.equal(programStatus({ status: 'too_old' }), 'too_old');
  assert.equal(programStatus({ status: 'broken' }), 'broken');
  assert.equal(programStatus({ status: 'something-new' }), 'unknown');
  assert.equal(programStatus({}), 'unknown');
  assert.equal(programStatus(null), 'unknown');
});

test('a program that is not ready carries the command that installs it', () => {
  const row = programRow(claude().external[0]);
  assert.equal(row.status, 'missing');
  assert.equal(row.command, CMD);
  assert.equal(row.pinned, '2.1.299');
  assert.equal(row.minimum, '2.1.299');
});

test('a ready program has no command, and a program with no install has none either', () => {
  assert.equal(programRow({ name: 'x', status: 'ready', install_command: null }).command, null);
  assert.equal(programRow({ name: 'x', status: 'missing', install: null }).command, null);
  assert.equal(programRow({ name: 'x', status: 'missing', install: null }).pinned, null);
});

test('a card takes the worst status of its programs', () => {
  const rows = [{ status: 'ready' }, { status: 'too_old' }, { status: 'missing' }];
  assert.equal(cardStatus(rows), 'missing');
  assert.equal(cardStatus([{ status: 'ready' }, { status: 'too_old' }]), 'too_old');
  assert.equal(cardStatus([{ status: 'too_old' }, { status: 'broken' }]), 'broken');
  assert.equal(cardStatus([{ status: 'broken' }, { status: 'missing' }]), 'missing');
  assert.equal(cardStatus([{ status: 'ready' }, { status: 'unknown' }]), 'unknown');
  assert.equal(cardStatus([{ status: 'ready' }]), 'ready');
});

test('only an extension that declares a program has a card', () => {
  const listing = [claude(), { id: 'plain', external: [] }, { id: 'bare' }];
  assert.deepEqual(extensionCards(listing).map((c) => c.id), ['claude']);
  assert.deepEqual(extensionCards(null), []);
});

test('cards that need attention come first, then by id', () => {
  const listing = [
    claude({ status: 'ready', install_command: null }),
    { ...claude(), id: 'zeta' },
    { ...claude({ status: 'too_old' }), id: 'alpha' },
  ];
  assert.deepEqual(extensionCards(listing).map((c) => [c.id, c.status]), [
    ['alpha', 'too_old'], ['zeta', 'missing'], ['claude', 'ready'],
  ]);
});

test('every status and the labels the card paints exist in both languages', () => {
  const source = readFileSync(I18N, 'utf8');
  const keys = [
    'ext.status.ready', 'ext.status.missing', 'ext.status.broken', 'ext.status.too_old', 'ext.status.unknown',
    'ext.found', 'ext.minimum', 'ext.pinned', 'ext.install', 'ext.empty', 'ext.copy',
    'ext.installBtn', 'ext.installing', 'ext.installDone',
    'shimPage.extensions',
  ];
  for (const key of keys) {
    const count = source.split(`'${key}'`).length - 1;
    assert.ok(count >= 2, `${key} is in ${count} of the 2 dictionaries`);
  }
});

test('the install request carries the program name and nothing else', () => {
  const { url, body } = installRequest('claude', 'claude');
  assert.equal(url, '/admin/api/extensions/claude/install-cli');
  assert.deepEqual(body, { program: 'claude' });
});

test('an extension id is encoded as one path segment', () => {
  assert.equal(
    installRequest('a/b c', 'x').url,
    '/admin/api/extensions/a%2Fb%20c/install-cli',
  );
});

test('the panel posts the request installRequest builds, and only for a pinned program', () => {
  const page = readFileSync(
    join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/shims-page.js'),
    'utf8',
  );
  assert.match(page, /const \{ url, body \} = installRequest\(extId, programName\);/);
  assert.match(page, /api\.post\(url, body\)/);
  // The button needs a pinned version: no install spec, no button.
  assert.match(page, /program\.pinned\s*\n?\s*\? `<button[^`]*installExtensionProgram\(this,/);
});

test('an install is a success only when the answer says ok', () => {
  assert.deepEqual(installOutcome({ ok: true, state: 'installed' }), { ok: true });
  assert.deepEqual(
    installOutcome({ ok: false, status: 502, error: 'no build of agy 1.5.15 is recorded for linux_x64_gnu' }),
    { ok: false, error: 'no build of agy 1.5.15 is recorded for linux_x64_gnu' },
  );
  // An answer with no `ok`, no answer, or an empty one is not a pass.
  assert.equal(installOutcome({ state: 'installed' }).ok, false);
  assert.equal(installOutcome(undefined).ok, false);
  assert.equal(installOutcome({ ok: false }).error, 'Install failed');
});

test('the panel reads the install answer through installOutcome', () => {
  const page = readFileSync(
    join(HERE, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/shims-page.js'),
    'utf8',
  );
  assert.match(page, /installOutcome\(await api\.post\(url, body\)\)/);
});

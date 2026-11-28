/**
 * The extension row: what it offers and what its requests carry (#1829).
 *
 * DOM-free builders, so node runs them directly.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  checkUpdatesRequest, declaredFor, extensionForVendor, humanSize, neededPrograms,
  planRequest, removalSummary, removeRequest, rowActions, skippedDeclaration,
  unreadyPrograms, updateLine, updateRequest, writeOutcome,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/extension-manage.js';

const program = (name, status, usedBy) => ({ name, status, used_by: usedBy });
const claude = (over = {}) => ({
  id: 'claude',
  version: '0.1.1',
  origin: 'directory',
  shims: ['claude--acp', 'claude--messages'],
  external: [
    program('claude', 'missing', ['claude--messages']),
    program('node', 'ready', ['claude--acp', 'claude--messages']),
  ],
  ...over,
});

test('the extension of a vendor is found by id, then by the shims it provides', () => {
  const list = [claude(), { id: 'kilo', shims: ['kilo--chat_completions'] }];
  assert.equal(extensionForVendor(list, 'claude').id, 'claude');
  assert.equal(extensionForVendor([{ id: 'x', shims: ['acme--chat'] }], 'acme').id, 'x');
  assert.equal(extensionForVendor(list, 'ghost'), null);
  assert.equal(extensionForVendor(null, 'claude'), null);
});

test('a shim lists only the programs it needs', () => {
  const ext = claude();
  assert.deepEqual(neededPrograms(ext, 'claude--acp').map((p) => p.name), ['node']);
  assert.deepEqual(
    neededPrograms(ext, 'claude--messages').map((p) => p.name), ['claude', 'node'],
  );
  assert.deepEqual(neededPrograms(claude({ external: [program('x', 'ready', [])] })), []);
});

test('only the needed programs that are not ready warn', () => {
  const ext = claude();
  assert.deepEqual(unreadyPrograms(ext, 'claude--acp'), []);
  assert.deepEqual(unreadyPrograms(ext, 'claude--messages').map((p) => p.name), ['claude']);
  assert.deepEqual(unreadyPrograms(null, 'claude--acp'), []);
});

test('a directory install offers remove, replace and update after a check', () => {
  const a = rowActions(claude(), { status: 'update', available: '0.2.0', versions: ['0.2.0', '0.1.1'] }, null);
  assert.equal(a.canRemove, true);
  assert.equal(a.canUpdate, true);
  assert.equal(a.updateTo, '0.2.0');
  assert.equal(a.replace, 'version');
  assert.deepEqual(a.versions, ['0.2.0', '0.1.1']);
});

test('update appears only once a check found a newer release', () => {
  assert.equal(rowActions(claude(), null, null).canUpdate, false);
  assert.equal(rowActions(claude(), { status: 'current' }, null).canUpdate, false);
  assert.equal(rowActions(claude(), { status: 'failed', error: 'x' }, null).canUpdate, false);
  assert.equal(rowActions(claude(), null, null).canCheck, true);
});

test('a declared source selecting the extension makes replace "declared"', () => {
  assert.equal(rowActions(claude(), null, { source: 'contrib' }).replace, 'declared');
});

test('an image extension offers neither a button that fails nor an update', () => {
  const ext = claude({ origin: 'image', remove_hint: 'pip uninstall llm-rosetta-ext-claude' });
  const a = rowActions(ext, { status: 'update', available: '9.9.9', versions: ['9.9.9'] }, { source: 'contrib' });
  assert.equal(a.origin, 'image');
  assert.equal(a.canRemove, false);
  assert.equal(a.canUpdate, false);
  assert.equal(a.replace, null);
  assert.equal(a.removeHint, 'pip uninstall llm-rosetta-ext-claude');
  assert.deepEqual(a.versions, []);
  assert.equal(a.canCheck, true);
});

test('an unknown origin is treated as removable only when it says directory', () => {
  assert.equal(rowActions(claude({ origin: undefined }), null, null).origin, 'directory');
});

test('an update line carries the reason of a failure and never says up to date for it', () => {
  const ext = claude();
  assert.equal(updateLine(ext, null), null);
  assert.deepEqual(updateLine(ext, { status: 'update', current: '0.1.1', available: '0.2.0' }).key, 'extMgr.updateAvail');
  assert.equal(updateLine(ext, { status: 'current', current: '0.1.1' }).key, 'extMgr.upToDate');
  assert.equal(updateLine(ext, { status: 'not_published' }).key, 'extMgr.notPublished');
  const failed = updateLine(ext, { status: 'failed', error: 'contrib: timed out' });
  assert.equal(failed.key, 'extMgr.checkFailed');
  assert.equal(failed.params.reason, 'contrib: timed out');
  assert.equal(failed.cls, 'is-bad');
  assert.equal(updateLine(ext, { status: 'no_compatible', error: 'requires surface 9' }).params.reason, 'requires surface 9');
});

const catalogue = {
  sources: [{
    id: 'contrib',
    vendors: [
      { vendor: 'kilo', skipped: 'already provided by installed plugin kilo 0.5.1', selected_by: 'contrib:kilo', installable: '0.6.0' },
      { vendor: 'claude', skipped: null, selected_by: null },
      { vendor: 'argo', skipped: null, selected_by: 'contrib:argo', installable: '0.3.1' },
    ],
  }],
};

test('a skipped declaration is found by the extension id', () => {
  const hit = skippedDeclaration(catalogue, 'kilo');
  assert.equal(hit.source, 'contrib');
  assert.match(hit.reason, /already provided/);
  assert.equal(hit.selectedBy, 'contrib:kilo');
  assert.equal(skippedDeclaration(catalogue, 'claude'), null);
  assert.equal(skippedDeclaration(null, 'kilo'), null);
});

test('a declared source is found only when a selector picks the extension', () => {
  assert.deepEqual(declaredFor(catalogue, 'argo'), { source: 'contrib', selectedBy: 'contrib:argo', version: '0.3.1' });
  assert.equal(declaredFor(catalogue, 'claude'), null);
});

test('requests carry only what the server reads', () => {
  assert.deepEqual(checkUpdatesRequest(null), { url: '/admin/api/extensions/check-updates', body: {} });
  assert.deepEqual(checkUpdatesRequest(['kilo']).body, { ids: ['kilo'] });
  assert.deepEqual(updateRequest('kilo', null), { url: '/admin/api/extensions/kilo/update', body: {} });
  assert.deepEqual(updateRequest('kilo', '0.6.0').body, { version: '0.6.0' });
  assert.deepEqual(removeRequest('kilo', false).body, {});
  assert.deepEqual(removeRequest('kilo', true).body, { replace: true });
  assert.equal(planRequest('kilo').url, '/admin/api/extensions/kilo/removal-plan');
  assert.equal(removeRequest('a/b', false).url, '/admin/api/extensions/a%2Fb/remove');
});

test('sizes are short and exact at the edges', () => {
  assert.equal(humanSize(0), '0 B');
  assert.equal(humanSize(1023), '1023 B');
  assert.equal(humanSize(1024), '1.0 KB');
  assert.equal(humanSize(214 * 1024 * 1024), '214 MB');
  assert.equal(humanSize(undefined), '0 B');
});

test('the removal summary lists what goes with sizes and what stays with the reason', () => {
  const s = removalSummary({
    delete: [{ path: '/config/tools/claude', size: 2048, note: '' }],
    keep: [{ path: '/config/tools/node', size: 0, note: 'still used by: codex' }],
  });
  assert.deepEqual(s.del, [{ path: '/config/tools/claude', size: '2.0 KB' }]);
  assert.deepEqual(s.keep, [{ path: '/config/tools/node', note: 'still used by: codex' }]);
  assert.equal(s.total, 1);
  assert.deepEqual(removalSummary({}), { del: [], keep: [], total: 0 });
});

test('a write is a success only when it says ok true', () => {
  assert.equal(writeOutcome({ ok: true, extension: 'kilo' }).ok, true);
  assert.deepEqual(writeOutcome({ ok: false, error: 'busy' }), { ok: false, error: 'busy' });
  assert.equal(writeOutcome({ extension: 'kilo' }).ok, false);
  assert.equal(writeOutcome(null).ok, false);
});

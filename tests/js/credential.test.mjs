import test from 'node:test';
import assert from 'node:assert/strict';

import { safeLinkUrl, parseRecord } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/credential.js';

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

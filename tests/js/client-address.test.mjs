/**
 * What the request log shows as a row's client (#1830 A12).
 *
 * DOM-free, so node runs it directly.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  clientAddress, firstForwarded,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/client-address.js';

test('a resolved client is shown as it is, verified', () => {
  const shown = clientAddress({
    client_ip: '198.51.100.7', peer_ip: '172.29.0.1', forwarded_for: '198.51.100.7, 127.0.0.1',
  });
  assert.equal(shown.ip, '198.51.100.7');
  assert.equal(shown.unverified, false);
});

test('when nothing trusted resolved, the first forwarded entry stands in, unverified', () => {
  const shown = clientAddress({
    client_ip: '172.29.0.1', peer_ip: '172.29.0.1', forwarded_for: '203.0.113.9, 127.0.0.1',
  });
  assert.equal(shown.ip, '203.0.113.9');
  assert.equal(shown.unverified, true);
  assert.equal(shown.peer, '172.29.0.1');
  assert.equal(shown.chain, '203.0.113.9, 127.0.0.1');
});

test('with no forwarded header the peer is shown, verified as what it is', () => {
  const shown = clientAddress({ client_ip: '172.29.0.1', peer_ip: '172.29.0.1' });
  assert.equal(shown.ip, '172.29.0.1');
  assert.equal(shown.unverified, false);
});

test('a row from before the columns existed keeps its stored client', () => {
  const shown = clientAddress({ client_ip: '172.29.0.1' });
  assert.equal(shown.ip, '172.29.0.1');
  assert.equal(shown.unverified, false);
});

test('a row with no address at all shows nothing', () => {
  assert.equal(clientAddress({}).ip, '');
});

test('the first entry is trimmed and an empty header has none', () => {
  assert.equal(firstForwarded('  1.2.3.4 , 5.6.7.8'), '1.2.3.4');
  assert.equal(firstForwarded(''), '');
  assert.equal(firstForwarded(undefined), '');
});

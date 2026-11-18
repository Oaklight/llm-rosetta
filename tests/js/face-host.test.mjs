/**
 * One endpoint's face: the host it forwards to, and the block that names it.
 *
 * The drawer used to inline all three rules — reading a setting block-first,
 * building the block on save, and deciding whether an inherited host belongs
 * to another vendor.  Six mutations of that code left the whole JS suite green
 * (#1595 review), including `if (baseUrl) block.base_url = baseUrl`, which is
 * the line the entire fix rests on.  The rules are in `core/face-host.js` now
 * so these cases can run them.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  faceBlock,
  faceCandidateHosts,
  faceHostMismatch,
  faceRow,
  faceSaveBlock,
  readFace,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/face-host.js';

// ── reading ─────────────────────────────────────────────────────────

test('the block wins over the flat key', () => {
  const cfg = { embeddings_shim: 'flat--embedding', embeddings: { shim: 'block--embedding' } };

  assert.equal(readFace(cfg, 'embeddings', 'shim', 'embeddings_shim'), 'block--embedding');
});

test('an empty block value falls through to the flat key', () => {
  const cfg = { embeddings_path: '/v1/embed', embeddings: { path: '' } };

  assert.equal(readFace(cfg, 'embeddings', 'path', 'embeddings_path'), '/v1/embed');
});

test('the flat key answers when there is no block', () => {
  assert.equal(readFace({ rerank_path: '/v1/rerank' }, 'rerank', 'path', 'rerank_path'), '/v1/rerank');
});

test('enabled false in the block is a value, not an absence', () => {
  const cfg = { embeddings_enabled: true, embeddings: { enabled: false } };

  assert.equal(readFace(cfg, 'embeddings', 'enabled'), false);
});

test('enabled reads the flat key when the block says nothing', () => {
  assert.equal(readFace({ embeddings_enabled: false, embeddings: { shim: 'x' } }, 'embeddings', 'enabled'), false);
});

test('the face host is read from the block', () => {
  const cfg = { base_url: 'https://chat.example', embeddings: { base_url: 'https://api.cohere.com' } };

  assert.equal(readFace(cfg, 'embeddings', 'base_url', 'embeddings_base_url'), 'https://api.cohere.com');
});

test('a block that is not a mapping is ignored', () => {
  const cfg = { embeddings: 'cohere--embedding', embeddings_shim: 'flat--embedding' };

  assert.equal(readFace(cfg, 'embeddings', 'shim', 'embeddings_shim'), 'flat--embedding');
});

test('no entry reads as undefined rather than throwing', () => {
  assert.equal(readFace(null, 'embeddings', 'shim', 'embeddings_shim'), undefined);
});

// ── writing ─────────────────────────────────────────────────────────

test('a typed host is written into the block', () => {
  const block = faceBlock({
    shim: 'cohere--embedding',
    format: 'cohere',
    path: '/v2/embed',
    baseUrl: 'https://api.cohere.com/v2',
  });

  assert.equal(block.base_url, 'https://api.cohere.com/v2');
});

test('a blank host is not written, so the endpoint inherits', () => {
  const block = faceBlock({ shim: 'cohere--embedding', baseUrl: '   ' });

  assert.equal('base_url' in block, false);
});

test('an enabled endpoint carries no enabled key', () => {
  assert.equal('enabled' in faceBlock({ shim: 'x', enabled: true }), false);
});

test('a switched-off endpoint carries enabled: false', () => {
  assert.equal(faceBlock({ shim: 'x', enabled: false }).enabled, false);
});

test('an empty shim is not written', () => {
  assert.equal('shim' in faceBlock({ shim: '', format: 'cohere' }), false);
});

// ── the mismatch ────────────────────────────────────────────────────

test('another origin is a mismatch', () => {
  assert.equal(
    faceHostMismatch('https://generativelanguage.googleapis.com', 'https://api.cohere.com'),
    true,
  );
});

test('the same vendor is not a mismatch', () => {
  assert.equal(
    faceHostMismatch('https://api.openai.com/v1', 'https://api.openai.com/v1/embeddings'),
    false,
  );
});

test('the scheme counts: plaintext against an https shim is a mismatch', () => {
  assert.equal(faceHostMismatch('http://api.cohere.com', 'https://api.cohere.com'), true);
});

test('the port counts: two local services are not one host', () => {
  assert.equal(faceHostMismatch('http://127.0.0.1:8001', 'http://127.0.0.1:8002'), true);
});

test('a missing side is not a mismatch', () => {
  assert.equal(faceHostMismatch('', 'https://api.cohere.com'), false);
  assert.equal(faceHostMismatch('https://x.example', ''), false);
});

// ── every shim of the format, not the first one ────────────────────

const SHIMS = [
  { name: 'openai--embedding', model_type: 'embeddings', format: 'openai', default_base_url: 'https://api.openai.com/v1' },
  { name: 'siliconflow--embedding', model_type: 'embeddings', format: 'openai', default_base_url: 'https://api.siliconflow.cn/v1' },
  { name: 'cohere--embedding', model_type: 'embeddings', format: 'cohere', default_base_url: 'https://api.cohere.com' },
];

test('a vendor serving its own dialect is not a mismatch', () => {
  // The measured false positive: SiliconFlow's own host, face spelled as the
  // bare format.  The gateway routed here correctly and warned anyway.
  assert.deepEqual(
    faceCandidateHosts(SHIMS, { modelType: 'embeddings', format: 'openai' }),
    ['https://api.openai.com/v1', 'https://api.siliconflow.cn/v1'],
  );
  assert.equal(faceHostMismatch('https://api.siliconflow.cn/v1', ['https://api.openai.com/v1', 'https://api.siliconflow.cn/v1']), false);
});

test('a host nobody serving the format declares is a mismatch', () => {
  assert.equal(
    faceHostMismatch('https://generativelanguage.googleapis.com', faceCandidateHosts(SHIMS, { modelType: 'embeddings', format: 'cohere' })),
    true,
  );
});

test('a named shim answers alone', () => {
  assert.deepEqual(
    faceCandidateHosts(SHIMS, { modelType: 'embeddings', format: 'openai', shim: 'siliconflow--embedding' }),
    ['https://api.siliconflow.cn/v1'],
  );
});

test('an unknown shim declares no host', () => {
  assert.deepEqual(faceCandidateHosts(SHIMS, { modelType: 'embeddings', format: 'openai', shim: 'gone--embedding' }), []);
  assert.equal(faceHostMismatch('https://api.openai.com/v1', []), false);
});

// ── the row the drawer shows, and the block it saves ───────────────

const EMB = { name: 'embeddings', config_shim_key: 'embeddings_shim', config_format_key: 'embeddings_format' };

test('the row shows the stored host and the one it would inherit', () => {
  const row = faceRow({
    entry: { base_url: 'https://chat.example/v1', embeddings: { format: 'cohere', base_url: 'https://api.cohere.com/v2' } },
    desc: EMB,
    shims: SHIMS,
  });

  assert.equal(row.host, 'https://api.cohere.com/v2');
  assert.equal(row.inherited, 'https://chat.example/v1');
  assert.equal(row.marked, false);
});

test('a format-only face on a foreign host is marked', () => {
  // This used to be silently unmarked: the panel looked the shim up by name,
  // and a format-only face has none, so it compared against nothing while the
  // gateway warned at startup (#1595 review).
  const row = faceRow({
    entry: { base_url: 'https://chat.example/v1', embeddings: { format: 'cohere' } },
    desc: EMB,
    shims: SHIMS,
  });

  assert.deepEqual(row.candidates, ['https://api.cohere.com']);
  assert.equal(row.marked, true);
});

test('a format-only face on the vendor that serves it is not marked', () => {
  const row = faceRow({
    entry: { base_url: 'https://api.siliconflow.cn/v1', embeddings: { format: 'openai' } },
    desc: EMB,
    shims: SHIMS,
  });

  assert.equal(row.marked, false);
});

test('the save writes the host the row holds', () => {
  const block = faceSaveBlock({ host: 'https://api.cohere.com/v2', shim: 'cohere--embedding', format: 'cohere', path: '/v2/embed' });

  assert.equal(block.base_url, 'https://api.cohere.com/v2');
});

test('the save drops a blank host, so the endpoint keeps inheriting', () => {
  const block = faceSaveBlock({ host: '   ', shim: 'cohere--embedding', format: 'cohere', path: '/v2/embed' });

  assert.equal('base_url' in block, false);
});

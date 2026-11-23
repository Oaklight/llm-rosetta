import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Who chose a capability's shim, the preset or the operator.  These cases state
// the rule itself, so a change that moves a face the badge calls Manual, or
// calls a face Preset that it will not follow, turns one of them red — which a
// test of the drawer's source text cannot do.

const here = dirname(fileURLToPath(import.meta.url));
const mod = await import(pathToFileURL(join(
  here,
  '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/face-origin.js',
)).href);
const { presetShimFor, initialManual, faceOrigin, afterPresetChange } = mod;

const SHIMS = [
  { name: 'openai--chat_completions', model_type: 'llm' },
  { name: 'openai--embedding', model_type: 'embeddings' },
  { name: 'cohere--chat_completions', model_type: 'llm' },
  { name: 'cohere--embedding', model_type: 'embeddings' },
  { name: 'cohere--rerank', model_type: 'rerank' },
  { name: 'jina--rerank', model_type: 'rerank' },
  // LLM-only vendors: they offer nothing for a typed face.
  { name: 'deepseek--chat_completions', model_type: 'llm' },
  { name: 'anthropic--messages', model_type: 'llm' },
];
const LLM = { name: 'llm', is_llm: true };
const EMB = { name: 'embeddings' };
const RERANK = { name: 'rerank' };

test('the LLM face is the preset; a typed face takes the preset vendor\'s own shim', () => {
  assert.equal(presetShimFor(SHIMS, LLM, 'cohere--chat_completions'), 'cohere--chat_completions');
  assert.equal(presetShimFor(SHIMS, EMB, 'cohere--chat_completions'), 'cohere--embedding');
  assert.equal(presetShimFor(SHIMS, RERANK, 'cohere--chat_completions'), 'cohere--rerank');
  assert.equal(presetShimFor(SHIMS, EMB, 'deepseek--chat_completions'), null);
  assert.equal(presetShimFor(SHIMS, EMB, ''), null);
  assert.equal(presetShimFor(SHIMS, LLM, ''), null);
});

test('a stored face opens as manual unless it is exactly what the preset offers', () => {
  assert.equal(initialManual('cohere--embedding', 'cohere--embedding'), false);
  assert.equal(initialManual('google--embedding', 'cohere--embedding'), true);
  assert.equal(initialManual('', 'cohere--embedding'), false);
  // The preset offers nothing: whatever the face holds is the operator's.
  assert.equal(initialManual('cohere--embedding', null), true);
  assert.equal(initialManual('', null), false);
});

test('the badge and the flag agree for every state', () => {
  for (const preset of ['', 'deepseek--chat_completions', 'cohere--chat_completions']) {
    for (const value of ['', 'cohere--embedding', 'openai--embedding']) {
      const target = presetShimFor(SHIMS, EMB, preset);
      const manual = initialManual(value, target);
      const badge = faceOrigin({ preset, value, target, manual });
      if (badge === 'manual') {
        // What is shown as the operator's must survive a preset change.
        const next = afterPresetChange({ manual, value, target: 'openai--embedding' });
        assert.deepEqual(next, { move: false, manual: true }, `${preset}|${value}`);
      }
    }
  }
});

test('an entry whose preset has no offer for a face keeps that face across a preset change', () => {
  // `type: deepseek--chat_completions` with a Cohere embeddings face.
  const value = 'cohere--embedding';
  const manual = initialManual(value, presetShimFor(SHIMS, EMB, 'deepseek--chat_completions'));
  const next = afterPresetChange({
    manual, value, target: presetShimFor(SHIMS, EMB, 'openai--chat_completions'),
  });
  assert.deepEqual(next, { move: false, manual: true });
});

test('a mixed entry saved before the preset existed keeps its other-vendor face', () => {
  // `type: anthropic--messages`, rerank `cohere--rerank`, no stored preset.
  const value = 'cohere--rerank';
  const manual = initialManual(value, presetShimFor(SHIMS, RERANK, 'anthropic--messages'));
  assert.equal(manual, true);
  assert.deepEqual(
    afterPresetChange({ manual, value, target: presetShimFor(SHIMS, RERANK, 'jina--chat') }),
    { move: false, manual: true },
  );
});

test('a face that follows moves with the preset, and a face with nothing yet is not claimed', () => {
  assert.deepEqual(
    afterPresetChange({ manual: false, value: 'cohere--embedding', target: 'openai--embedding' }),
    { move: true, manual: false },
  );
  // The new preset offers nothing and the face holds a value: it becomes the operator's.
  assert.deepEqual(
    afterPresetChange({ manual: false, value: 'cohere--embedding', target: null }),
    { move: false, manual: true },
  );
  assert.deepEqual(
    afterPresetChange({ manual: false, value: '', target: null }),
    { move: false, manual: false },
  );
});

test('after a reset the face reads Preset whether or not the picker moved', () => {
  // Reset clears the flag.  A face already holding the preset's offer fires no
  // picker change, so the origin has to be right from the flag alone.
  const preset = 'cohere--chat_completions';
  const target = presetShimFor(SHIMS, EMB, preset);
  assert.equal(faceOrigin({ preset, value: target, target, manual: true }), 'manual');
  assert.equal(faceOrigin({ preset, value: target, target, manual: false }), 'preset');
});

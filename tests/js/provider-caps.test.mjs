/**
 * What a bare provider `type` says about the provider's model type.
 *
 * A format can belong to more than one model type.  `google_generate` is a chat
 * format and an embedding dialect, so treating it as evidence of embeddings
 * gave `type: google_generate` an Embeddings badge and hid it from the LLM
 * facet.  A non-LLM type is only implied by a format that serves that type and
 * no LLM type; an embeddings provider must declare its endpoint.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { nonLlmFormatType } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/provider-endpoints.js';

// Shaped like the admin payload's model_types.
const TYPES = [
  {
    name: 'llm',
    is_llm: true,
    formats: [
      'chat_completions',
      'open_responses',
      'messages',
      'google_generate',
      'google_interactions',
    ],
  },
  {
    name: 'embeddings',
    is_llm: false,
    formats: ['openai', 'cohere', 'jina', 'voyage', 'google_generate', 'minimax'],
  },
  { name: 'rerank', is_llm: false, formats: ['jina', 'cohere', 'voyage'] },
  {
    name: 'decisions',
    is_llm: false,
    formats: ['systemone', 'openai_decisions'],
  },
];

test('a chat format that is also an embedding dialect implies no type', () => {
  // The reported bug: this was 'embeddings'.
  assert.equal(nonLlmFormatType(TYPES, 'google_generate'), null);
});

test('a chat-only format implies no type', () => {
  assert.equal(nonLlmFormatType(TYPES, 'chat_completions'), null);
  assert.equal(nonLlmFormatType(TYPES, 'messages'), null);
});

test('a non-LLM-only format still implies its type', () => {
  assert.equal(nonLlmFormatType(TYPES, 'voyage'), 'embeddings');
  assert.equal(nonLlmFormatType(TYPES, 'systemone'), 'decisions');
  assert.equal(nonLlmFormatType(TYPES, 'openai_decisions'), 'decisions');
});

test('a format shared by two non-LLM types takes the registry order', () => {
  assert.equal(nonLlmFormatType(TYPES, 'cohere'), 'embeddings');
  assert.equal(nonLlmFormatType(TYPES, 'jina'), 'embeddings');
});

test('an unknown or missing name implies no type', () => {
  assert.equal(nonLlmFormatType(TYPES, 'not-a-format'), null);
  assert.equal(nonLlmFormatType(TYPES, ''), null);
  assert.equal(nonLlmFormatType(TYPES, null), null);
  assert.equal(nonLlmFormatType(null, 'voyage'), null);
});

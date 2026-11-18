/**
 * The drawer's four wiring points for a face's host (#1595 review two rounds on).
 *
 * `core/face-host.js` holds the rules and `face-host.test.mjs` runs them.  What
 * that suite cannot see is the drawer *calling* them: four mutations of the
 * wiring in `tabs/providers.js` — the save posting no host, the field never
 * reading the stored one, the marker never firing, `_faceValue` answering
 * nothing — left the whole JS suite green at 498/498, and one of them is the
 * misroute #1595 exists to end.
 *
 * These cases read the source, so they run the same helper the credential
 * wiring test uses: strip comments first, or a comment naming the pattern
 * satisfies the assertion.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(
  join(here, '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/providers.js'),
  'utf8',
);

/** The source with comments dropped, string literals kept. */
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

/** The body of `function name(...) { ... }`, brace-matched. */
function bodyOf(name) {
  const start = code.search(new RegExp(`function ${name}\\s*\\(`));
  assert.ok(start !== -1, `${name} is still there`);
  const open = code.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < code.length; i += 1) {
    if (code[i] === '{') depth += 1;
    else if (code[i] === '}') {
      depth -= 1;
      if (depth === 0) return code.slice(open, i + 1);
    }
  }
  throw new Error(`${name} has no closing brace`);
}

test('the save posts the host the field holds', () => {
  const save = bodyOf('_saveProvider');

  assert.match(
    save,
    /faceSaveBlock\(\{[\s\S]{0,200}?host:\s*baseUrlEl\s*\?\s*baseUrlEl\.value\s*:\s*''/,
  );
  assert.ok(
    !/faceSaveBlock\(\{[\s\S]{0,200}?host:\s*''/.test(save),
    'a save that posts an empty host sends the lane back to the chat host',
  );
});

test('the drawer shows the host the entry stores', () => {
  assert.match(bodyOf('openProviderModal'), /faceBaseEl\.value\s*=\s*row\.host/);
});

test('the drawer resolves the candidates from the row, not from one shim', () => {
  const open = bodyOf('openProviderModal');

  assert.match(open, /faceRow\(\{[\s\S]{0,200}?shims:/);
  assert.match(open, /faceBaseEl\.dataset\.shimHosts\s*=\s*JSON\.stringify\(row\.candidates\)/);
});

test('the mismatch marker reads the candidates and toggles the class', () => {
  const mark = bodyOf('_markFaceHostMismatch');

  assert.match(mark, /JSON\.parse\(el\.dataset\.shimHosts/);
  // The rule itself: an empty field inheriting a host nobody serving the
  // endpoint declares.  Marking every row, or none, both red here.
  assert.match(
    mark,
    /const mismatch = !el\.value\.trim\(\)\s*\n?\s*&& faceHostMismatch\(el\.dataset\.inherited, candidates\)/,
  );
  assert.match(mark, /classList\.toggle\('is-host-mismatch',\s*mismatch\)/);
});

test('the marker runs when the row opens and while it is typed into', () => {
  const open = bodyOf('openProviderModal');

  assert.match(open, /_markFaceHostMismatch\(desc,\s*faceBaseEl\);/);
  assert.match(open, /faceBaseEl\.oninput\s*=\s*\(\)\s*=>\s*_markFaceHostMismatch\(desc,\s*faceBaseEl\)/);
});

test('the reader the drawer uses is the shared one', () => {
  const read = bodyOf('_faceValue');

  assert.match(read, /return readFace\(cfg,\s*desc\.name,\s*field,\s*flatKey\)/);
  assert.match(read, /base_url:\s*`\$\{desc\.name\}_base_url`/);
});

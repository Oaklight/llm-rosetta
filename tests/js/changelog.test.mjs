/**
 * Reading one release out of the bundled changelog.
 *
 * The panel shows a release's notes in a dialog after an update, so the
 * parser has to find exactly one `## vX.Y.Z — date` section in the docs
 * changelog — including a bullet that wraps onto a second line — and render
 * its inline markdown without letting the file inject markup.
 *
 * Run with `make test-js` (or `node --test tests/js/*.test.mjs`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  normaliseVersion,
  parseChangelog,
  renderEntryHtml,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/changelog.js';

const MARKDOWN = `# Changelog

## [Unreleased]

### Added
- Not released yet.

## v0.14.0 — 2026-10-31

### Added
- **One name per model type**: the badges now say \`Embeddings\`.
- A bullet that wraps
  onto a second line.

### Fixed
- Something.

## v0.13.0 — 2026-09-08

### Added
- Older.
`;

test('a version is the same with or without its v and local tag', () => {
  assert.equal(normaliseVersion('v0.14.0'), '0.14.0');
  assert.equal(normaliseVersion('0.14.0+g1a2b3c'), '0.14.0');
});

test('the section for the running version is found', () => {
  const section = parseChangelog(MARKDOWN, '0.14.0');
  assert.equal(section.date, '2026-10-31');
  assert.deepEqual(
    section.groups.map(group => group.name),
    ['Added', 'Fixed']
  );
  assert.equal(section.groups[0].items.length, 2);
});

test('the unreleased section is not mistaken for a release', () => {
  assert.equal(parseChangelog(MARKDOWN, 'unreleased'), null);
  assert.equal(parseChangelog(MARKDOWN, '0.15.0'), null);
});

test('a wrapped bullet stays one entry', () => {
  const section = parseChangelog(MARKDOWN, '0.14.0');
  assert.match(section.groups[0].items[1], /wraps onto a second line/);
});

test('the next release ends the section', () => {
  const section = parseChangelog(MARKDOWN, '0.13.0');
  assert.deepEqual(section.groups[0].items, ['Older.']);
});

test('an entry renders its inline markdown', () => {
  const html = renderEntryHtml('**Bold** and `code` and [link](https://x.test/a).');
  assert.equal(
    html,
    '<strong>Bold</strong> and <code>code</code> and <a href="https://x.test/a" target="_blank" rel="noopener">link</a>.'
  );
});

test('markup in the file is shown, not run', () => {
  assert.equal(
    renderEntryHtml('<img src=x onerror=alert(1)>'),
    '&lt;img src=x onerror=alert(1)&gt;'
  );
});

test('a non-http link stays text', () => {
  const html = renderEntryHtml('[x](javascript:alert(1))');
  assert.equal(html, '[x](javascript:alert(1))');
});

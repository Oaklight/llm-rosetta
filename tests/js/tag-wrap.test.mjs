/**
 * A tag is a one-line label.
 *
 * `.tag` is a fixed 18px box with 10.5px mono type, so any wrap inside it
 * spills over the pill: a CJK label such as 流式 stacks its two glyphs, and a
 * status code plus its expand caret splits across lines. Several seats size
 * to the content around them, so the component itself has to refuse the wrap.
 *
 * The request-log status cell holds two controls, the code pill and the
 * error-dump button; the phone card keeps them on one centred row instead of
 * letting the taller button fall to a second line.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const CSS = readFileSync(
  join(
    HERE,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/css/components.css',
  ),
  'utf8',
);

test('a tag never wraps inside its fixed-height seat', () => {
  const tag = CSS.match(/(?:^|\n)\.tag \{([^}]*)\}/);
  assert.ok(tag, '.tag is still defined');
  assert.match(tag[1], /white-space:\s*nowrap/, '.tag can wrap a CJK label or a status suffix');
});

test('the request-log status cell keeps its pill and dump button on one row', () => {
  assert.match(
    CSS,
    /#requestLogTable \.log-status\s*\{[^}]*white-space:\s*nowrap/,
    'the status cell can wrap between the code pill and the dump button',
  );
  assert.match(
    CSS,
    /#logTable \.log-status\s*\{[^}]*display:\s*flex[^}]*align-items:\s*center/,
    'the card status cell stacks or baseline-aligns its two controls',
  );
});

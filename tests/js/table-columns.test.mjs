/**
 * Every table's column count, and the header text at the columns the
 * stylesheet aligns by position.
 *
 * `components.css` right-aligns four tables' value columns.  Two of them by a
 * role class the renderer sets; two by `:nth-child(N)`, because those tables
 * are built from a fixed list of headers and nothing else identifies a column.
 * A positional rule fails silently: insert a column anywhere before the value
 * column and the numbers move left with no error, and the CSS still parses.
 *
 * This test pins what those indices land on.  Each entry is the table's tbody
 * id (the table element itself has no id for the four that a tab builds) and,
 * for the aligned columns, the header text — 1-based, the same numbers the
 * CSS comments carry.  Adding or removing a column reds the count assertion,
 * and renaming one reds the text assertion.
 *
 * The count check covers every table in the document, not only the aligned
 * ones: a new table has to be registered here, so the next positional rule
 * cannot be written against an index nobody measured.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const HTML = readFileSync(
  join(
    HERE,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/admin.html',
  ),
  'utf8',
);

/** Marker → column count and the 1-based columns the CSS aligns.
    The marker is the table's own id, or its tbody's id for the four tables
    a tab builds without one. */
const COLUMNS = {
  // 4 = "Models", a count — `.pc-col-num` on the header and the cell.
  providerTable: { count: 5, aligned: { 4: 'Models' } },
  modelTable: { count: 4, aligned: {} },
  // 5–8 = "Requests", "Input", "Output", "Total" — `.key-*-cell`.
  keysTableEl: {
    count: 9,
    aligned: { 5: 'Requests', 6: 'Input', 7: 'Output', 8: 'Total' },
  },
  // 8 = "Tokens", 9 = "Duration" — `.log-tokens`, `.log-duration`.
  requestLogTable: { count: 9, aligned: { 8: 'Tokens', 9: 'Duration' } },
  opsLogTableEl: { count: 5, aligned: {} },
  // 6 = "Duration" — `#logView-profiling thead th:nth-child(6)`.
  profilingResults: { count: 8, aligned: { 6: 'Duration' } },
  captureResults: { count: 7, aligned: {} },
  jobsTableEl: { count: 7, aligned: {} },
  dumpTableEl: { count: 8, aligned: {} },
  // 2–8 — `#tokenUsageModal … :nth-child(n+2)`; 1 = "Date" stays left.
  tokenUsageTable: {
    count: 8,
    aligned: {
      2: 'Requests',
      3: 'Input',
      4: 'Output',
      5: 'Cache↓',
      6: 'Cache↑',
      7: 'Reasoning',
      8: 'Total',
    },
  },
};

const ENTITIES = {
  '&rarr;': '→',
  '&larr;': '←',
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&nbsp;': ' ',
};

function headerText(cell) {
  return cell
    .replace(/<[^>]*>/g, '')
    .replace(/&[a-z]+;|&#\d+;/gi, (entity) => ENTITIES[entity] ?? entity)
    .replace(/\s+/g, ' ')
    .trim();
}

/** Every table in the document: marker, header texts, column count. */
function tables() {
  const found = [];
  const tablesRe = /<table\b[^>]*>([\s\S]*?)<\/table>/g;
  let match;
  while ((match = tablesRe.exec(HTML)) !== null) {
    const whole = match[0];
    const outer = whole.slice(0, whole.indexOf('>') + 1);
    const inner = match[1];
    const marker =
      /\bid="([^"]+)"/.exec(outer)?.[1] ??
      /<tbody\b[^>]*\bid="([^"]+)"/.exec(inner)?.[1];
    assert.ok(marker, `a table has neither an id nor an identifiable tbody`);
    const thead = /<thead\b[^>]*>([\s\S]*?)<\/thead>/.exec(inner)?.[1] ?? '';
    const headers = [...thead.matchAll(/<th\b[^>]*>([\s\S]*?)<\/th>/g)].map(
      (cell) => headerText(cell[1]),
    );
    found.push({ marker, headers });
  }
  return found;
}

test('every table in admin.html is registered here', () => {
  const markers = tables().map(({ marker }) => marker);
  assert.deepEqual(
    [...markers].sort(),
    Object.keys(COLUMNS).sort(),
    'a table was added or removed; register it and re-check the aligned indices',
  );
});

test('each table has the column count the stylesheet assumes', () => {
  for (const { marker, headers } of tables()) {
    assert.equal(
      headers.length,
      COLUMNS[marker].count,
      `${marker}: a column was added or removed, so every positional rule after it moved`,
    );
  }
});

test('each right-aligned column is still the one it was measured on', () => {
  for (const { marker, headers } of tables()) {
    for (const [column, text] of Object.entries(COLUMNS[marker].aligned)) {
      assert.equal(
        headers[Number(column) - 1],
        text,
        `${marker}, column ${column}: the alignment rule now lands on a different header`,
      );
    }
  }
});

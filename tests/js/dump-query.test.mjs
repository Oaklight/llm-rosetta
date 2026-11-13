/**
 * The Error Dumps toolbar → list query translation.
 *
 * The list used to be fetched once (`limit=10000`) and filtered in the
 * browser, so a filter only ever described the newest 500 rows the route
 * returned.  The filters now travel to the server, and a value that this
 * builder drops is invisible on screen: the panel looks like it filtered and
 * the server answers with everything.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { dumpQuery, dumpStatusValue } from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/dump-query.js';

test('an empty toolbar asks only for the page', () => {
  assert.equal(dumpQuery({}, { limit: 20, offset: 40 }).toString(), 'limit=20&offset=40');
});

test('an omitted page is omitted from the query', () => {
  assert.equal(dumpQuery({}).toString(), '');
  assert.equal(dumpQuery({}, { limit: 20 }).toString(), 'limit=20');
});

test('a keyset cursor travels as its two halves', () => {
  // The download walks the store while it is being written; an offset would
  // shift under it, so the page boundary is the last row it saw.
  const params = dumpQuery({}, {
    limit: 500,
    before: { timestamp: '2026-11-13T12:00:00+00:00', id: 'dump-9' },
  });
  assert.equal(params.get('before_timestamp'), '2026-11-13T12:00:00+00:00');
  assert.equal(params.get('before_id'), 'dump-9');
  assert.equal(params.get('offset'), null);
});

test('half a cursor is not sent: it does not name a position', () => {
  assert.equal(
    dumpQuery({}, { before: { timestamp: '2026-11-13T12:00:00+00:00' } }).toString(),
    '',
  );
  assert.equal(dumpQuery({}, { before: { id: 'dump-9' } }).toString(), '');
  assert.equal(dumpQuery({}, { before: {} }).toString(), '');
});

test('each filter reaches its own parameter', () => {
  const params = dumpQuery({
    phase: 'upstream',
    status: '4xx',
    provider: 'argo',
    idQuery: 'abc123',
  });
  assert.equal(params.get('error_phase'), 'upstream');
  assert.equal(params.get('status'), '4xx');
  assert.equal(params.get('provider'), 'argo');
  assert.equal(params.get('q'), 'abc123');
});

test('an exact model choice and a model substring are different filters', () => {
  const exact = dumpQuery({ model: 'gpt-5' });
  assert.equal(exact.get('model'), 'gpt-5');
  assert.equal(exact.get('model_search'), null);

  const search = dumpQuery({ modelSearch: 'haiku' });
  assert.equal(search.get('model_search'), 'haiku');
  assert.equal(search.get('model'), null);
});

test('empty values are not sent', () => {
  const params = dumpQuery({
    phase: '', status: '', provider: '', model: '', modelSearch: '',
    idQuery: '', timeRange: '',
  });
  assert.equal(params.toString(), '');
});

test('a preset range becomes an absolute start', () => {
  const now = Date.parse('2026-11-13T12:00:00Z');
  assert.equal(
    dumpQuery({ timeRange: '1h' }, {}, now).get('start'),
    '2026-11-13T11:00:00.000Z',
  );
  assert.equal(
    dumpQuery({ timeRange: '24h' }, {}, now).get('start'),
    '2026-11-12T12:00:00.000Z',
  );
  assert.equal(
    dumpQuery({ timeRange: '7d' }, {}, now).get('start'),
    '2026-11-06T12:00:00.000Z',
  );
  assert.equal(
    dumpQuery({ timeRange: '30d' }, {}, now).get('start'),
    '2026-10-14T12:00:00.000Z',
  );
});

test('a preset range sends only a start, never an end', () => {
  const params = dumpQuery({ timeRange: '24h' }, {}, Date.now());
  assert.equal(params.get('end'), null);
});

test('a custom range covers local midnight to local end of day', () => {
  const params = dumpQuery({ timeRange: 'custom', dateFrom: '2026-11-01', dateTo: '2026-11-02' });
  const start = new Date(params.get('start'));
  const end = new Date(params.get('end'));
  assert.deepEqual(
    [start.getFullYear(), start.getMonth(), start.getDate(), start.getHours(), start.getMinutes(), start.getSeconds()],
    [2026, 10, 1, 0, 0, 0],
  );
  assert.deepEqual(
    [end.getFullYear(), end.getMonth(), end.getDate(), end.getHours(), end.getMinutes(), end.getSeconds(), end.getMilliseconds()],
    [2026, 10, 2, 23, 59, 59, 999],
  );
});

test('a half-filled custom range sends only the bound it has', () => {
  const from = dumpQuery({ timeRange: 'custom', dateFrom: '2026-11-01' });
  assert.ok(from.get('start'));
  assert.equal(from.get('end'), null);

  const to = dumpQuery({ timeRange: 'custom', dateTo: '2026-11-02' });
  assert.equal(to.get('start'), null);
  assert.ok(to.get('end'));
});

test('an unreadable custom date is dropped, not sent as NaN', () => {
  const params = dumpQuery({ timeRange: 'custom', dateFrom: 'not-a-date' });
  assert.equal(params.get('start'), null);
});

test('a partial status code is not sent to a route that would refuse it', () => {
  assert.equal(dumpStatusValue('custom', '4'), '');
  assert.equal(dumpStatusValue('custom', '42'), '');
  assert.equal(dumpStatusValue('custom', '429'), '429');
  assert.equal(dumpStatusValue('custom', ''), '');
});

test('a status class from the select is passed through', () => {
  assert.equal(dumpStatusValue('4xx', ''), '4xx');
  assert.equal(dumpStatusValue('5xx', ''), '5xx');
  assert.equal(dumpStatusValue('', ''), '');
});

test('a search box with text wins over an empty select', () => {
  assert.equal(dumpStatusValue('', '500'), '500');
});

test('the query round-trips through the URL it builds', () => {
  const params = dumpQuery({ provider: 'a b&c', idQuery: 'x=y' }, { limit: 20, offset: 0 });
  const reparsed = new URLSearchParams(params.toString());
  assert.equal(reparsed.get('provider'), 'a b&c');
  assert.equal(reparsed.get('q'), 'x=y');
});

// The walk itself runs only in a browser, so the lines that decide how it
// pages are asserted against the source: reverting them to an offset leaves
// every test above green.  The assertions read code, not prose — the loop's
// own comment explains why the offset was dropped, so comments are removed
// first (string literals kept, or `//` in a URL would break the strip).
function stripComments(src) {
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

const here = dirname(fileURLToPath(import.meta.url));
const dashboard = stripComments(readFileSync(
  join(
    here,
    '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/tabs/dashboard.js',
  ),
  'utf8',
));
const walk = dashboard.slice(
  dashboard.indexOf('async function downloadAllDumps('),
  dashboard.indexOf('function changeDumpPage('),
);

test('the download walk pages by cursor, not by offset', () => {
  assert.ok(walk.length > 0, 'downloadAllDumps must exist');
  assert.match(walk, /page\.before = before/, 'the walk carries the cursor');
  assert.match(walk, /dumpQuery\(filters, page\)/, 'and sends it through the builder');
  assert.doesNotMatch(walk, /offset/, 'an offset would shift under the walk');
});

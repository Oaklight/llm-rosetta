/**
 * The job board's arithmetic: ids, badges, progress, results and actions.
 *
 * These are the pieces a rendering bug hides in — a bar that never fills, a
 * failed job that offers "cancel", a deleted count that divides by zero —
 * so they live in a DOM-free module and are pinned here.
 *
 * Run with `make test-js` (or `node --test tests/js/`).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  jobActions,
  jobDurationText,
  jobKindLabel,
  jobMatchesFilter,
  jobProgressPct,
  jobProgressText,
  jobResultText,
  jobShortId,
  jobStatusTag,
  jobTimeText,
  jobsHaveActive,
} from '../../packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin/js/core/jobs-view.js';

const job = overrides => ({
  id: '0123456789abcdef',
  kind: 'data.clear',
  status: 'queued',
  progress_done: 0,
  progress_total: null,
  progress_unit: null,
  result: null,
  error: null,
  started_at: null,
  finished_at: null,
  ...overrides,
});

test('the short id is eight characters', () => {
  assert.equal(jobShortId('0123456789abcdef'), '01234567');
  assert.equal(jobShortId(null), '');
});

test('each state has its badge, and an unknown state falls back', () => {
  assert.equal(jobStatusTag('succeeded'), 'tag is-pill tag-ok');
  assert.equal(jobStatusTag('failed'), 'tag is-pill tag-error');
  assert.equal(jobStatusTag('running'), 'tag is-pill tag-stream');
  assert.equal(jobStatusTag('queued'), 'tag is-pill tag-warning');
  assert.equal(jobStatusTag('nonsense'), 'tag is-pill tag-warning');
});

test('a known kind is translated, an unknown one shows its raw name', () => {
  const translate = key => `T:${key}`;
  assert.equal(jobKindLabel('data.trim', translate), 'T:jobs.kind.trim');
  assert.equal(jobKindLabel('batch.submit', translate), 'batch.submit');
  assert.equal(jobKindLabel('data.trim', null), 'data.trim');
});

test('progress is a percentage of the total, never out of range', () => {
  assert.equal(jobProgressPct(job({ progress_done: 50, progress_total: 200 })), 25);
  assert.equal(jobProgressPct(job({ progress_done: 300, progress_total: 200 })), 100);
  assert.equal(jobProgressPct(job({ progress_done: 5 })), 0);
  assert.equal(jobProgressPct(job({ status: 'succeeded' })), 100);
  assert.equal(jobProgressPct(null), 0);
});

test('a total is printed as done / total unit; a bare count prints alone', () => {
  assert.equal(
    jobProgressText(job({ progress_done: 5000, progress_total: 20000, progress_unit: 'rows' })),
    '5,000 / 20,000 rows',
  );
  assert.equal(jobProgressText(job({ progress_done: 5000, progress_unit: 'rows' })), '5,000 rows');
  assert.equal(jobProgressText(job({})), '');
});

test('the result names the deletion or the rebuild, and an error wins', () => {
  assert.equal(jobResultText(job({ result: { deleted: 1200 } })), '1,200 deleted');
  assert.equal(jobResultText(job({ result: { rebuilt_from: 40 } })), 'rebuilt from 40');
  assert.equal(jobResultText(job({ error: 'disk full', result: { deleted: 3 } })), 'disk full');
  assert.equal(jobResultText(job({})), '');
});

test('the unit and the result follow the dictionary when a t() is passed', () => {
  // An operator reading the board in Chinese should not get "5,001 rows" and
  // "0 deleted" under a Chinese page title.
  const dict = {
    'jobs.unit.rows': '\u884c',
    'jobs.result.deleted': '\u5df2\u5220\u9664 {count} \u6761',
    'jobs.result.rebuilt': '\u4ece {count} \u91cd\u5efa',
  };
  const zh = (key, params) => {
    let out = dict[key];
    if (out == null) return key;
    if (params) for (const [k, v] of Object.entries(params)) out = out.replace(`{${k}}`, v);
    return out;
  };

  assert.equal(
    jobProgressText(job({ progress_done: 5000, progress_total: 20000, progress_unit: 'rows' }), zh),
    '5,000 / 20,000 \u884c',
  );
  assert.equal(jobResultText(job({ result: { deleted: 1200 } }), zh), '\u5df2\u5220\u9664 1,200 \u6761');
  assert.equal(jobResultText(job({ result: { rebuilt_from: 40 } }), zh), '\u4ece 40 \u91cd\u5efa');
});

test('an unknown unit keeps the worker spelling instead of a raw key', () => {
  const t = key => key;
  assert.equal(
    jobProgressText(job({ progress_done: 2, progress_unit: 'shard' }), t),
    '2 shard',
  );
});

test('the created column reads like the log, not like an ISO string', () => {
  // The raw value has microseconds and an offset; the column shows the same
  // shape the request log uses and keeps the raw value in the title.
  const shown = jobTimeText('2026-11-14T01:56:20.781379+00:00');
  assert.match(shown, /^\d{2}\/\d{2}, \d{2}:\d{2}:\d{2}/);
  assert.ok(!shown.includes('T') && !shown.includes('+00:00'));
  assert.equal(jobTimeText(''), '');
  assert.equal(jobTimeText('not a date'), 'not a date');
});

test('a duration needs a start, and switches units', () => {
  const started = '2026-01-01T00:00:00+00:00';
  assert.equal(jobDurationText(job({}), Date.parse(started)), '');
  assert.equal(
    jobDurationText(
      job({ started_at: started, finished_at: '2026-01-01T00:00:00.250+00:00' }),
      Date.parse(started),
    ),
    '250 ms',
  );
  assert.equal(
    jobDurationText(
      job({ started_at: started, finished_at: '2026-01-01T00:00:02.000+00:00' }),
      Date.parse(started),
    ),
    '2.0 s',
  );
});

test('the board is active while anything is queued or running', () => {
  assert.equal(jobsHaveActive([job({ status: 'queued' })]), true);
  assert.equal(jobsHaveActive([job({ status: 'running' }), job({ status: 'succeeded' })]), true);
  assert.equal(jobsHaveActive([job({ status: 'succeeded' }), job({ status: 'failed' })]), false);
  assert.equal(jobsHaveActive([]), false);
});

test('the filter keeps the rows the segment names', () => {
  const rows = [
    job({ status: 'queued' }),
    job({ status: 'running' }),
    job({ status: 'failed' }),
    job({ status: 'succeeded' }),
    job({ status: 'cancelled' }),
  ];
  assert.equal(rows.filter(j => jobMatchesFilter(j, 'all')).length, 5);
  assert.equal(rows.filter(j => jobMatchesFilter(j, 'active')).length, 2);
  assert.equal(rows.filter(j => jobMatchesFilter(j, 'failed')).length, 1);
});

test('a row offers the actions its state allows', () => {
  assert.deepEqual(jobActions(job({ status: 'queued' })), ['cancel']);
  assert.deepEqual(jobActions(job({ status: 'running' })), ['cancel']);
  assert.deepEqual(jobActions(job({ status: 'failed' })), ['retry', 'unhide']);
  assert.deepEqual(jobActions(job({ status: 'cancelled' })), ['resume', 'unhide']);
  assert.deepEqual(jobActions(job({ status: 'succeeded' })), []);
});

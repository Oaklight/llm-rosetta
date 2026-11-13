/**
 * jobs-view.js — pure helpers for the background job board.
 *
 * DOM-free and dependency-free so the tests can run it under node: the board
 * itself (fetching, rendering, polling) lives in `tabs/jobs.js`.  What is here
 * is the arithmetic and the label choices a rendering bug would hide in.
 */

const STATUS_TAG = {
  queued: 'tag is-pill tag-warning',
  running: 'tag is-pill tag-stream',
  succeeded: 'tag is-pill tag-ok',
  failed: 'tag is-pill tag-error',
  cancelled: 'tag is-pill tag-warning',
};

const KIND_KEYS = {
  'data.clear': 'jobs.kind.clear',
  'data.cleanup': 'jobs.kind.cleanup',
  'data.trim': 'jobs.kind.trim',
  'data.vacuum': 'jobs.kind.vacuum',
  'data.rebuild': 'jobs.kind.rebuild',
};

const ACTIVE = new Set(['queued', 'running']);

/** The short id a table row shows; the full id is in the title attribute. */
export function jobShortId(id) {
  return String(id || '').slice(0, 8);
}

/** The tag classes a job state wears (`tag is-pill tag-ok` and the like). */
export function jobStatusTag(status) {
  return STATUS_TAG[status] || STATUS_TAG.queued;
}

/** The operator-facing name of an operation, given a `t()` function. */
export function jobKindLabel(kind, translate) {
  const key = KIND_KEYS[kind];
  return key && translate ? translate(key) : String(kind || '');
}

/** How full the progress bar is, 0-100. */
export function jobProgressPct(job) {
  if (!job) return 0;
  const total = Number(job.progress_total);
  if (Number.isFinite(total) && total > 0) {
    const done = Number(job.progress_done) || 0;
    return Math.max(0, Math.min(100, Math.round((done * 100) / total)));
  }
  return job.status === 'succeeded' ? 100 : 0;
}

/** Translate when the caller has a `t()`, else keep the server's spelling. */
function _unitLabel(unit, translate) {
  if (!translate) return unit;
  const key = 'jobs.unit.' + unit;
  const text = translate(key);
  return text === key ? unit : text;
}

/** Translate when the caller has a `t()`, else fall back to English. */
function _text(translate, key, params, fallback) {
  if (!translate) return fallback;
  const text = translate(key, params);
  return text === key ? fallback : text;
}

/** "12,000 / 20,000 rows" and the like; empty when there is nothing to say.
 *
 * The unit reaches the row from the worker (`rows`, `pass`), so it is
 * translated here rather than baked into the server's progress report.
 */
export function jobProgressText(job, translate) {
  if (!job) return '';
  const done = Number(job.progress_done) || 0;
  const total = Number(job.progress_total);
  const unit = job.progress_unit ? ` ${_unitLabel(job.progress_unit, translate)}` : '';
  if (Number.isFinite(total) && total > 0) {
    return `${done.toLocaleString()} / ${total.toLocaleString()}${unit}`;
  }
  return done ? `${done.toLocaleString()}${unit}` : '';
}

/** What the job did: the deletion count, the rebuild count, or its error. */
export function jobResultText(job, translate) {
  if (!job) return '';
  if (job.error) return job.error;
  const result = job.result || {};
  if (typeof result.deleted === 'number') {
    const count = result.deleted.toLocaleString();
    return _text(translate, 'jobs.result.deleted', {count}, `${count} deleted`);
  }
  if (typeof result.rebuilt_from === 'number') {
    const count = result.rebuilt_from.toLocaleString();
    return _text(translate, 'jobs.result.rebuilt', {count}, `rebuilt from ${count}`);
  }
  return '';
}

/** The row's timestamp in the log's own format, never the raw ISO string.
 *
 * `created_at` carries microseconds and a UTC offset, which reads as noise in
 * a column and pushes it half a screen wide; the request log already decided
 * what a timestamp looks like in this panel.
 */
export function jobTimeText(iso) {
  if (!iso) return '';
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return String(iso);
  return when.toLocaleString(undefined, {
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** How long a job ran, or has been running. */
export function jobDurationText(job, now = Date.now()) {
  if (!job || !job.started_at) return '';
  const started = Date.parse(job.started_at);
  const finished = job.finished_at ? Date.parse(job.finished_at) : now;
  const ms = finished - started;
  if (!Number.isFinite(ms) || ms < 0) return '';
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

/** Whether a job belongs in the board's current filter. */
export function jobMatchesFilter(job, filter) {
  if (!job) return false;
  if (filter === 'active') return ACTIVE.has(job.status);
  if (filter === 'failed') return job.status === 'failed';
  return true;
}

/** True while anything on the board is queued or running. */
export function jobsHaveActive(jobs) {
  return (jobs || []).some(job => ACTIVE.has(job && job.status));
}

/** The actions a row offers, given its state. */
export function jobActions(job) {
  if (!job) return [];
  if (ACTIVE.has(job.status)) return ['cancel'];
  if (job.status === 'failed') return ['retry', 'unhide'];
  if (job.status === 'cancelled') return ['resume', 'unhide'];
  return [];
}

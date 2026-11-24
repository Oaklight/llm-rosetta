/**
 * jobs.js — the background job board, its own page in the rail.
 *
 * The destructive operations return 202 and do their work in the worker, so
 * this is where an operator watches them: a row per job, live progress, and
 * the actions the job's state allows.  The polling is self-scheduled — fast
 * while something is active, slow when the board is idle — because a board
 * that polls every two seconds forever is a request every two seconds.
 */

import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, esc } from '../core/utils.js';
import { cssValue, jsArg } from '../core/html.js';
import {
  jobActions,
  jobActiveCount,
  jobMatchesCategory,
  jobMatchesFilter,
  jobDurationText,
  jobKindLabel,
  jobProgressPct,
  jobProgressText,
  jobResultText,
  jobShortId,
  jobTimeText,
  jobStatusTag,
  jobsHaveActive,
} from '../core/jobs-view.js';

const ACTIVE_POLL_MS = 2000;
const IDLE_POLL_MS = 10000;

/**
 * Enter the page: paint once, then keep the self-scheduled poll running.
 *
 * The poll is started here rather than from the segment that used to host the
 * board, because the page fetch and the timer have to be one decision —
 * `stopTimers()` stops the timer when the rail moves away.
 */
export function startJobsBoard() {
  loadJobs();
  scheduleJobsPoll(true);
}

/** Stop the board's own poll (called when the view is left or the app idles). */
export function stopJobsTimer() {
  if (S.jobsTimer) {
    clearInterval(S.jobsTimer);
    S.jobsTimer = null;
  }
  S._jobsPollMs = 0;
}

function scheduleJobsPoll(active) {
  const ms = active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
  if (S.jobsTimer && S._jobsPollMs === ms) return;
  stopJobsTimer();
  S._jobsPollMs = ms;
  S.jobsTimer = setInterval(() => { loadJobs(); }, ms);
}

/** Fetch the board and render it.
 *
 * The pause switch belongs to the log streams: its button lives in the
 * Observability panel, so honouring it here would freeze this page with no
 * visible way to resume it.  The board carries its own Refresh button.
 */
export async function loadJobs() {
  try {
    const category = S._jobsCategory || 'ops';
    const data = await api.get(
      `/admin/api/jobs?limit=50&category=${encodeURIComponent(category)}`,
    );
    renderJobs(data.jobs || [], data.total, data.active);
    return data;
  } catch (e) {
    return null;
  }
}

/** Pick the rows the segment shows; `all` keeps the board as it came. */
export function setJobsFilter(view) {
  S._jobsFilter = view || 'all';
  const seg = document.getElementById('jobsSeg');
  if (seg) {
    seg.querySelectorAll('[data-view]').forEach(el => {
      const active = el.dataset.view === S._jobsFilter;
      el.classList.toggle('active', active);
      el.setAttribute('aria-checked', String(active));
      el.tabIndex = active ? 0 : -1;
    });
  }
  loadJobs();
}

/** Pick the half of the board (request work or gateway maintenance). */
export function setJobsCategory(view) {
  S._jobsCategory = view === 'requests' ? 'requests' : 'ops';
  const seg = document.getElementById('jobsKindSeg');
  if (seg) {
    seg.querySelectorAll('[data-view]').forEach(el => {
      const active = el.dataset.view === S._jobsCategory;
      el.classList.toggle('active', active);
      el.setAttribute('aria-checked', String(active));
      el.tabIndex = active ? 0 : -1;
    });
  }
  loadJobs();
}

/** Paint the rows, and choose the next poll interval from what they show. */
export function renderJobs(allJobs, total, activeCount) {
  const body = document.getElementById('jobsBody');
  if (!body) return;
  const all = allJobs || [];
  const category = S._jobsCategory || 'ops';
  const inCategory = all.filter(job => jobMatchesCategory(job, category));
  const jobs = inCategory.filter(job => jobMatchesFilter(job, S._jobsFilter || 'all'));
  // The server counts active work across the whole category; a page-local
  // count would read "0 active" whenever the active job sat on page two.
  const active = jobActiveCount(activeCount, inCategory);
  const count = document.getElementById('jobsCount');
  const shown = Number.isFinite(total) ? total : inCategory.length;
  if (count) count.textContent = t('jobs.count', { count: shown, active });
  if (!jobs.length) {
    const empty = S._jobsFilter && S._jobsFilter !== 'all'
      ? t('jobs.emptyFiltered')
      : category === 'requests' ? t('jobs.emptyRequests') : t('jobs.emptyOps');
    body.innerHTML = `<tr><td colspan="7" style="color:var(--text-dim)">${esc(empty)}</td></tr>`;
    scheduleJobsPoll(false);
    return;
  }
  body.innerHTML = jobs.map(job => {
    const pct = jobProgressPct(job);
    const actions = jobActions(job).map(action => {
      const label = esc(t('jobs.' + action));
      return `<button class="btn btn-sm" onclick="jobAction(${jsArg(action)}, ${jsArg(job.id)})">${label}</button>`;
    }).join(' ');
    const duration = jobDurationText(job);
    return `<tr>
      <td class="mono" title="${esc(job.id)}">${esc(jobShortId(job.id))}</td>
      <td>${esc(jobKindLabel(job.kind, t))}</td>
      <td><span class="${jobStatusTag(job.status)}">${esc(t('jobs.status.' + job.status))}</span></td>
      <td><div class="job-progress"><div class="job-progress-bar" style="width:${cssValue(pct + '%')}"></div></div>
          <span class="job-progress-text">${esc(jobProgressText(job, t))}</span></td>
      <td title="${esc(job.created_at || '')}">${esc(jobTimeText(job.created_at))}${duration ? `<div class="job-progress-text">${esc(duration)}</div>` : ''}</td>
      <td>${esc(jobResultText(job, t))}</td>
      <td>${actions}</td>
    </tr>`;
  }).join('');
  scheduleJobsPoll(jobsHaveActive(jobs));
}

/** Run one of a row's actions, then refresh the board. */
export async function jobAction(action, id) {
  const paths = {
    cancel: 'cancel',
    retry: 'retry',
  };
  const path = paths[action];
  if (!path) return;
  const result = await api.post(`/admin/api/jobs/${encodeURIComponent(id)}/${path}`);
  if (!result.ok) {
    showToast(result.error || t('toast.jobFailed'), 'error');
    return;
  }
  showToast(t(action === 'cancel' ? 'toast.jobCancelRequested' : 'toast.jobActionDone'));
  await loadJobs();
  window.loadMetrics?.();
}

Object.assign(window, { loadJobs, jobAction, setJobsFilter, setJobsCategory, startJobsBoard, stopJobsTimer });

export { jobShortId };

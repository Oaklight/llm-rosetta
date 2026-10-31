/**
 * ops-log.js — Server Ops Log: loading, rendering, filtering,
 * pagination, and view switching within the Logs tab.
 */

import { S, OPS_LOG_LIMIT } from '../core/state.js';
import { t } from '../core/i18n.js';
import { pageCount, clampPage, offsetForPage, shouldResync } from '../core/pagination.js';
import { api, esc } from '../core/utils.js';

const EVENT_KEYS = {
  startup: 'event.startup',
  shutdown: 'event.shutdown',
  config_reload: 'event.configReload',
  key_create: 'event.keyCreate',
  key_update: 'event.keyUpdate',
  key_delete: 'event.keyDelete',
  key_rotate: 'event.keyRotate',
  health_status_change: 'event.healthChange',
  admin_setup: 'event.adminSetup',
  ops_log_cleared: 'event.logCleared',
};

const SEVERITY_BADGE = {
  info: 'badge-ok',
  warning: 'badge-warning',
  error: 'badge-error',
};

// ===================== Ops Log =====================

async function loadOpsLog() {
  const eventType = document.getElementById('filterOpsEventType').value;
  const severity = document.getElementById('filterOpsSeverity').value;
  const source = document.getElementById('filterOpsSource').value;
  let url = `/admin/api/ops-log?limit=${OPS_LOG_LIMIT}&offset=${S.opsLogOffset}`;
  if (eventType) url += `&event_type=${encodeURIComponent(eventType)}`;
  if (severity) url += `&severity=${encodeURIComponent(severity)}`;
  if (source) url += `&source=${encodeURIComponent(source)}`;

  try {
    const data = await api.get(url);
    renderOpsLog(data.entries, data.total);
  } catch (_) { /* auth redirect handled by api.get */ }
}

// Page the jump box was last synced to, and whether the user has edited it
// since.  See shouldResync() for why both signals are needed.
let _opsJumpShown = 1;
let _opsJumpDirty = false;

const _opsJumpInput = document.getElementById('opsLogPageJump');
_opsJumpInput?.addEventListener?.('input', () => { _opsJumpDirty = true; });
_opsJumpInput?.addEventListener?.('blur', () => { _opsJumpDirty = false; });

function renderOpsLog(entries, total) {
  const tbody = document.getElementById('opsLogTable');
  if (entries.length === 0) {
    tbody.innerHTML = `<tr class="ops-empty-row" role="row"><td colspan="5" role="cell" style="color:var(--text-dim)">${t('empty.opsLog')}</td></tr>`;
  } else {
    tbody.innerHTML = entries.map((e, idx) => {
      const time = new Date(e.timestamp).toLocaleString(undefined, {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit'});
      const label = EVENT_KEYS[e.event_type] ? t(EVENT_KEYS[e.event_type]) : e.event_type;
      const sevCls = SEVERITY_BADGE[e.severity] || '';
      const hasDetails = !!e.details;
      const rowId = e.id;
      const detailId = 'opsErr-' + idx;
      const isExpanded = S.expandedOpsLogRows.has(rowId);
      const rowCursor = hasDetails ? 'cursor:pointer;' : '';
      const rowClick = hasDetails ? ' onclick="toggleOpsLogRow(this.dataset.opsLogId,this)"' : '';
      // Same keyboard contract as the Request Log's error rows: a tab stop,
      // Enter/Space runs the click path, and the nested pre does not re-toggle.
      const rowKeyboard = hasDetails
        ? ` tabindex="0" aria-expanded="${isExpanded}" aria-controls="${detailId}" onkeydown="rowToggleKey(event,this)"`
        : '';
      const expandHint = hasDetails ? ' ▸' : '';
      let rows = `<tr data-ops-log-id="${esc(rowId)}" role="row"${rowCursor ? ` style="${rowCursor}"` : ''}${rowClick}${rowKeyboard}>
        <td class="ops-time" role="cell" aria-labelledby="thOpsTime">${time}</td>
        <td class="ops-event" role="cell" aria-labelledby="thOpsEvent"><span class="badge badge-stream">${esc(label)}</span></td>
        <td class="ops-severity" role="cell" aria-labelledby="thOpsSeverity"><span class="badge ${sevCls}">${esc(e.severity)}${expandHint}</span></td>
        <td class="ops-message" role="cell" aria-labelledby="thOpsMessage">${esc(e.message)}</td>
        <td class="ops-source" role="cell" aria-labelledby="thOpsSource" style="font-size:12px;color:var(--text-dim)">${esc(e.source || '—')}</td>
      </tr>`;
      if (hasDetails) {
        rows += `<tr class="ops-error-row" id="${detailId}" role="row"${isExpanded ? '' : ' hidden'}><td colspan="5" role="cell"><pre style="margin:0;padding:8px;background:var(--bg);border-radius:6px;font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap;word-break:break-all">${esc(JSON.stringify(e.details, null, 2))}</pre></td></tr>`;
      }
      return rows;
    }).join('');
  }

  const totalPages = pageCount(total, OPS_LOG_LIMIT);
  const currentPage = Math.floor(S.opsLogOffset / OPS_LOG_LIMIT) + 1;
  S.opsLogTotal = total;
  document.getElementById('opsLogPageInfo').textContent = t('page.info', {current: currentPage, total: totalPages, count: total});
  document.getElementById('opsLogPrevPage').disabled = S.opsLogOffset === 0;
  document.getElementById('opsLogNextPage').disabled = S.opsLogOffset + OPS_LOG_LIMIT >= total;
  const jump = document.getElementById('opsLogPageJump');
  if (jump) {
    jump.max = String(totalPages);
    jump.disabled = totalPages <= 1;
    const focused = document.activeElement === jump;
    if (shouldResync(jump.value, _opsJumpShown, { focused, dirty: _opsJumpDirty })) {
      jump.value = String(currentPage);
      _opsJumpShown = currentPage;
    }
  }
}

function toggleOpsLogRow(rowId, tr) {
  const detail = tr.nextElementSibling;
  if (!detail) return;
  detail.hidden = !detail.hidden;
  tr.setAttribute('aria-expanded', String(!detail.hidden));
  if (detail.hidden) S.expandedOpsLogRows.delete(rowId);
  else S.expandedOpsLogRows.add(rowId);
}

function changeOpsLogPage(dir) {
  S.opsLogOffset = Math.max(0, S.opsLogOffset + dir * OPS_LOG_LIMIT);
  S.expandedOpsLogRows.clear();
  _opsJumpDirty = false;
  loadOpsLog();
}

// Same contract as the Request Log's jump box: clamp out-of-range numbers,
// ignore an empty box, and re-ask the server for the chosen offset.
function jumpToOpsLogPage() {
  const input = document.getElementById('opsLogPageJump');
  if (!input) return;
  const totalPages = pageCount(S.opsLogTotal || 0, OPS_LOG_LIMIT);
  const page = clampPage(input.value, totalPages);
  if (page === null) {
    const current = Math.floor(S.opsLogOffset / OPS_LOG_LIMIT) + 1;
    input.value = String(current);
    _opsJumpShown = current;
    _opsJumpDirty = false;
    return;
  }
  input.value = String(page);
  _opsJumpShown = page;
  _opsJumpDirty = false;
  S.opsLogOffset = offsetForPage(page, OPS_LOG_LIMIT);
  S.expandedOpsLogRows.clear();
  loadOpsLog();
}

function resetOpsLogFilters() {
  document.getElementById('filterOpsEventType').value = '';
  document.getElementById('filterOpsSeverity').value = '';
  document.getElementById('filterOpsSource').value = '';
  S.opsLogOffset = 0;
  S.expandedOpsLogRows.clear();
  loadOpsLog();
}

// ===================== Filter population =====================

async function populateOpsLogFilters() {
  try {
    const [etData, srcData] = await Promise.all([
      api.get('/admin/api/ops-log/event-types'),
      api.get('/admin/api/ops-log/sources'),
    ]);
    const etSel = document.getElementById('filterOpsEventType');
    const etVal = etSel.value;
    etSel.innerHTML = `<option value="">${t('filter.allEventTypes')}</option>` +
      (etData.event_types || []).map(et => `<option value="${esc(et)}">${esc(EVENT_KEYS[et] ? t(EVENT_KEYS[et]) : et)}</option>`).join('');
    etSel.value = etVal;

    const srcSel = document.getElementById('filterOpsSource');
    const srcVal = srcSel.value;
    srcSel.innerHTML = `<option value="">${t('filter.allSources')}</option>` +
      (srcData.sources || []).map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join('');
    srcSel.value = srcVal;
  } catch (_) { /* auth redirect handled by api.get */ }
}

// ===================== View switching =====================

/** The views this panel serves, in rail order.  Requests and Ops are the
 *  paged logs; the other three are the record-level diagnostic tools that
 *  used to sit in the dashboard, where a single bad request was hard to find. */
const LOG_VIEWS = ['requests', 'ops', 'dumps', 'profiling', 'capture'];

function switchLogView(view) {
  if (S._logView === view) return;
  S._logView = view;

  for (const name of LOG_VIEWS) {
    const el = document.getElementById('logView-' + name);
    if (el) el.style.display = name === view ? '' : 'none';
  }

  const seg = document.getElementById('logViewSeg');
  if (seg) {
    seg.querySelectorAll('label').forEach(l => {
      const active = l.dataset.view === view;
      l.classList.toggle('active', active);
      l.setAttribute('aria-checked', String(active));
      l.tabIndex = active ? 0 : -1;
    });
  }

  // Each view owns its loader: the paged ones reset their offset, the
  // diagnostic ones read the side that changed while they were hidden.
  if (view === 'requests') {
    S.logOffset = 0;
    window.loadLogs();
  } else if (view === 'ops') {
    S.opsLogOffset = 0;
    loadOpsLog();
  } else if (view === 'dumps') {
    window.invalidateDumpCache?.();
    window.renderDumps?.();
  } else if (view === 'profiling') {
    window.loadProfilingStatus?.();
    window.loadProfilingResults?.();
  } else if (view === 'capture') {
    window.loadCaptureStatus?.();
    window.loadCaptureResults?.();
  }
  // The shared starter respects the pause state and the active view.
  window.startLogTimers?.();
}

// ===================== Exports =====================

Object.assign(window, {
  loadOpsLog, renderOpsLog, toggleOpsLogRow, changeOpsLogPage, jumpToOpsLogPage,
  resetOpsLogFilters, switchLogView, populateOpsLogFilters,
});

export { loadOpsLog, switchLogView, populateOpsLogFilters };

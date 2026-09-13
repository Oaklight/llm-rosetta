/**
 * ops-log.js — Server Ops Log: loading, rendering, filtering,
 * pagination, and view switching within the Logs tab.
 */

import { S, OPS_LOG_LIMIT } from './state.js';
import { t } from './i18n.js';
import { api, esc } from './core.js';

const EVENT_LABELS = {
  startup: 'Startup',
  shutdown: 'Shutdown',
  config_reload: 'Config Reload',
  key_create: 'Key Create',
  key_update: 'Key Update',
  key_delete: 'Key Delete',
  key_rotate: 'Key Rotate',
  health_status_change: 'Health Change',
  admin_setup: 'Admin Setup',
  ops_log_cleared: 'Log Cleared',
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

function renderOpsLog(entries, total) {
  const tbody = document.getElementById('opsLogTable');
  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim)">${t('empty.opsLog')}</td></tr>`;
  } else {
    tbody.innerHTML = entries.map(e => {
      const time = new Date(e.timestamp).toLocaleString(undefined, {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit'});
      const label = EVENT_LABELS[e.event_type] || e.event_type;
      const sevCls = SEVERITY_BADGE[e.severity] || '';
      const hasDetails = !!e.details;
      const rowId = e.id;
      const isExpanded = S.expandedOpsLogRows.has(rowId);
      const rowCursor = hasDetails ? 'cursor:pointer;' : '';
      const rowClick = hasDetails ? ` onclick="toggleOpsLogRow('${esc(rowId)}',this)"` : '';
      const expandHint = hasDetails ? ' ▸' : '';
      let rows = `<tr${rowCursor ? ` style="${rowCursor}"` : ''}${rowClick}>
        <td>${time}</td>
        <td><span class="badge badge-stream">${esc(label)}</span></td>
        <td><span class="badge ${sevCls}">${esc(e.severity)}${expandHint}</span></td>
        <td>${esc(e.message)}</td>
        <td style="font-size:12px;color:var(--text-dim)">${esc(e.source || '—')}</td>
      </tr>`;
      if (hasDetails) {
        rows += `<tr${isExpanded ? '' : ' hidden'}><td colspan="5"><pre style="margin:0;padding:8px;background:var(--bg);border-radius:6px;font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap;word-break:break-all">${esc(JSON.stringify(e.details, null, 2))}</pre></td></tr>`;
      }
      return rows;
    }).join('');
  }

  const totalPages = Math.ceil(total / OPS_LOG_LIMIT) || 1;
  const currentPage = Math.floor(S.opsLogOffset / OPS_LOG_LIMIT) + 1;
  document.getElementById('opsLogPageInfo').textContent = t('page.info', {current: currentPage, total: totalPages, count: total});
  document.getElementById('opsLogPrevPage').disabled = S.opsLogOffset === 0;
  document.getElementById('opsLogNextPage').disabled = S.opsLogOffset + OPS_LOG_LIMIT >= total;
}

function toggleOpsLogRow(rowId, tr) {
  const detail = tr.nextElementSibling;
  if (!detail) return;
  detail.hidden = !detail.hidden;
  if (detail.hidden) S.expandedOpsLogRows.delete(rowId);
  else S.expandedOpsLogRows.add(rowId);
}

function changeOpsLogPage(dir) {
  S.opsLogOffset = Math.max(0, S.opsLogOffset + dir * OPS_LOG_LIMIT);
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
      (etData.event_types || []).map(et => `<option value="${esc(et)}">${esc(EVENT_LABELS[et] || et)}</option>`).join('');
    etSel.value = etVal;

    const srcSel = document.getElementById('filterOpsSource');
    const srcVal = srcSel.value;
    srcSel.innerHTML = `<option value="">${t('filter.allSources')}</option>` +
      (srcData.sources || []).map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join('');
    srcSel.value = srcVal;
  } catch (_) { /* auth redirect handled by api.get */ }
}

// ===================== View switching =====================

function switchLogView(view) {
  if (S._logView === view) return;
  S._logView = view;

  const reqView = document.getElementById('logView-requests');
  const opsView = document.getElementById('logView-ops');
  reqView.style.display = view === 'requests' ? '' : 'none';
  opsView.style.display = view === 'ops' ? '' : 'none';

  // Update seg-control active state
  const seg = document.getElementById('logViewSeg');
  if (seg) {
    const labels = seg.querySelectorAll('label');
    labels.forEach(l => {
      const isReq = l.getAttribute('data-i18n') === 'logView.requests';
      const active = (view === 'requests') === isReq;
      l.classList.toggle('active', active);
      l.setAttribute('aria-checked', String(active));
      l.tabIndex = active ? 0 : -1;
    });
  }

  // Stop both timers, start the active one
  if (S.logTimer) { clearInterval(S.logTimer); S.logTimer = null; }
  if (S.opsLogTimer) { clearInterval(S.opsLogTimer); S.opsLogTimer = null; }

  const interval = S._dashboardRefreshMs > 0 ? S._dashboardRefreshMs : 5000;
  if (view === 'requests') {
    S.logOffset = 0;
    window.loadLogs();
    S.logTimer = setInterval(window.loadLogs, interval);
  } else {
    S.opsLogOffset = 0;
    loadOpsLog();
    S.opsLogTimer = setInterval(loadOpsLog, interval);
  }
}

// ===================== Exports =====================

Object.assign(window, {
  loadOpsLog, renderOpsLog, toggleOpsLogRow, changeOpsLogPage,
  resetOpsLogFilters, switchLogView, populateOpsLogFilters,
});

export { loadOpsLog, switchLogView, populateOpsLogFilters };

/**
 * logs.js — Request Log tab: log loading, rendering, filtering,
 * pagination, and log-context model deletion.
 */

import { S, LOG_LIMIT } from './state.js';
import { t } from './i18n.js';
import { api, esc, formatDuration, inlineConfirm, showToast } from './core.js';

// ===================== Request Log =====================

async function loadLogs() {
  const model = document.getElementById('filterModel').value;
  const provider = document.getElementById('filterProvider').value;
  const status = document.getElementById('filterStatus').value;
  const apiKey = document.getElementById('filterApiKey').value;
  let url = `/admin/api/requests?limit=${LOG_LIMIT}&offset=${S.logOffset}`;
  if (model) url += `&model=${encodeURIComponent(model)}`;
  if (provider) url += `&provider=${encodeURIComponent(provider)}`;
  if (status) url += `&status=${encodeURIComponent(status)}`;
  if (apiKey) url += `&api_key_label=${encodeURIComponent(apiKey)}`;

  const data = await api.get(url);
  renderLogs(data.entries, data.total);
}

// Reverse-lookup: provider API type → display name from config
function resolveProviderName(apiType) {
  if (!S.configData) return apiType;
  for (const [name, cfg] of Object.entries(S.configData.providers || {})) {
    if (cfg.type === apiType) return name;
  }
  return apiType;
}

function fmtTokens(n) {
  if (n >= 1e9) return (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return String(n);
}

function renderLogs(entries, total) {
  const tbody = document.getElementById('logTable');
  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="color:var(--text-dim)">${t('empty.logs')}</td></tr>`;
  } else {
    tbody.innerHTML = entries.map(e => {
      const time = new Date(e.timestamp).toLocaleString(undefined, {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit'});
      const statusCls = e.status_code < 400 ? 'badge-ok' : 'badge-error';
      const modeBadge = e.is_stream ? '<span class="badge badge-stream">stream</span>' : '';
      const keyLabel = e.api_key_label || '—';
      const hasError = !!e.error_detail;
      const rowId = e.timestamp + '_' + e.model;
      const isExpanded = S.expandedLogRows.has(rowId);
      const expandHint = hasError ? ' title="Click to expand error"' : '';
      const clientIp = e.client_ip || '—';
      const hlBg = S._highlightLogId === e.id ? 'background:color-mix(in srgb, var(--accent) 20%, transparent);' : '';
      const rowCursor = hasError ? 'cursor:pointer;' : '';
      const styleAttr = (hlBg || rowCursor) ? ` style="${hlBg}${rowCursor}"` : '';
      const rowClick = hasError ? ` onclick="toggleLogRow('${rowId}',this)"` : '';
      let rows = `<tr data-log-id="${esc(e.id)}"${styleAttr}${rowClick}${expandHint}>
        <td>${time}</td>
        <td><code>${esc(e.model)}</code></td>
        <td>${esc(e.source_provider)} &rarr; ${esc(e.target_provider_name || resolveProviderName(e.target_provider))}</td>
        <td>${modeBadge}</td>
        <td style="font-size:12px;color:var(--text-dim)">${esc(keyLabel)}</td>
        <td style="font-size:12px;color:var(--text-dim)">${esc(clientIp)}</td>
        <td><span class="badge ${statusCls}">${e.status_code}${hasError ? ' ▸' : ''}</span>${e.status_code >= 400 ? ` <button class="btn btn-sm" onclick="event.stopPropagation();jumpToErrorDump('${esc(e.id)}')" title="View error dump" style="padding:2px 4px;margin-left:2px"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></button>` : ''}</td>
        <td style="font-size:12px">${e.total_tokens != null ? fmtTokens(e.total_tokens) : '—'}</td>
        <td>${e.duration_ms.toFixed(0)} ms</td>
      </tr>`;
      if (hasError) {
        rows += `<tr${isExpanded ? '' : ' hidden'}><td colspan="9"><pre style="margin:0;padding:8px;background:var(--bg);border-radius:6px;font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap;word-break:break-all">${esc(e.error_detail)}</pre></td></tr>`;
      }
      return rows;
    }).join('');
  }

  // Auto-highlight and scroll
  if (S._highlightLogId) {
    const hlRow = document.querySelector('#logTable tr[data-log-id="' + S._highlightLogId + '"]');
    if (hlRow) hlRow.scrollIntoView({behavior:'smooth', block:'center'});
    if (S._highlightLogIdTimer) clearTimeout(S._highlightLogIdTimer);
    S._highlightLogIdTimer = setTimeout(function() {
      S._highlightLogId = null;
      S._highlightLogIdTimer = null;
    }, 10000);
  }

  // Pagination
  const totalPages = Math.ceil(total / LOG_LIMIT) || 1;
  const currentPage = Math.floor(S.logOffset / LOG_LIMIT) + 1;
  document.getElementById('pageInfo').textContent = t('page.info', {current: currentPage, total: totalPages, count: total});
  document.getElementById('prevPage').disabled = S.logOffset === 0;
  document.getElementById('nextPage').disabled = S.logOffset + LOG_LIMIT >= total;
}

function toggleLogRow(rowId, tr) {
  const detail = tr.nextElementSibling;
  if (!detail) return;
  detail.hidden = !detail.hidden;
  if (detail.hidden) S.expandedLogRows.delete(rowId);
  else S.expandedLogRows.add(rowId);
}

function changePage(dir) {
  S.logOffset = Math.max(0, S.logOffset + dir * LOG_LIMIT);
  S.expandedLogRows.clear();
  loadLogs();
}

function deleteModel(name, btn) {
  if (!btn) { _doDeleteModel(name); return; }
  inlineConfirm(btn, () => _doDeleteModel(name));
}

async function _doDeleteModel(name) {
  const res = await api.del(`/admin/api/config/models/${encodeURIComponent(name)}`);
  if (res.ok) { showToast(t('toast.modelDeleted', {name})); window.loadConfig(); }
  else { showToast(res.error || 'Failed', 'error'); }
}

function resetLogFilters() {
  document.getElementById('filterModel').value = '';
  document.getElementById('filterProvider').value = '';
  document.getElementById('filterStatus').value = '';
  document.getElementById('filterApiKey').value = '';
  S.logOffset = 0;
  S.expandedLogRows.clear();
  loadLogs();
}

function updateFilterOptions() {
  if (!S.configData) return;
  const models = Object.keys(S.configData.models || {});
  const providers = Object.keys(S.configData.providers || {});
  const mSel = document.getElementById('filterModel');
  const pSel = document.getElementById('filterProvider');
  // preserve selection
  const mVal = mSel.value, pVal = pSel.value;
  mSel.innerHTML = `<option value="">${t('filter.allModels')}</option>` + models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
  pSel.innerHTML = `<option value="">${t('filter.allProviders')}</option>` + providers.map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('');
  mSel.value = mVal; pSel.value = pVal;
  updateKeyFilterOptions();
}

function updateKeyFilterOptions() {
  const kSel = document.getElementById('filterApiKey');
  const kVal = kSel.value;
  const keys = (S.keysData && S.keysData.keys) || [];
  const labels = [...new Set([...keys.map(k => k.label), ...S.logKeyLabels].filter(Boolean))].sort();
  kSel.innerHTML = `<option value="">${t('filter.allKeys')}</option>` + labels.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('');
  kSel.value = kVal;
}

// ===================== Window globals =====================

function jumpToErrorDump(requestLogId) {
  goToTab('dashboard', function() {
    function tryFind(attempts) {
      const entries = S._dumpAllEntries || [];
      const match = entries.find(e => e.request_log_id === requestLogId);
      if (match) {
        const dumpId = match.dump_id || match.id;
        S._highlightDumpId = dumpId;
        if (S._highlightDumpIdTimer) clearTimeout(S._highlightDumpIdTimer);
        S._highlightDumpIdTimer = setTimeout(function() {
          S._highlightDumpId = null;
          S._highlightDumpIdTimer = null;
        }, 10000);
        const idx = entries.indexOf(match);
        S._dumpPage = Math.floor(idx / 20);
        renderDumps();
        setTimeout(function() {
          const section = document.getElementById('errorDumpSection');
          if (section) section.scrollIntoView({behavior:'smooth', block:'start'});
          setTimeout(function() {
            const btn = document.querySelector('#dumpTable tr [onclick*="' + dumpId + '"]');
            if (btn) btn.closest('tr').scrollIntoView({behavior:'smooth', block:'center'});
          }, 300);
        }, 200);
      } else if (attempts < 5) {
        setTimeout(function() { tryFind(attempts + 1); }, 500);
      } else {
        const section = document.getElementById('errorDumpSection');
        if (section) section.scrollIntoView({behavior:'smooth', block:'start'});
      }
    }
    setTimeout(function() { tryFind(0); }, 600);
  });
}

Object.assign(window, {
  loadLogs, renderLogs, toggleLogRow, changePage, jumpToErrorDump,
  resetLogFilters, updateFilterOptions, updateKeyFilterOptions,
  deleteModel,
});

export { loadLogs, renderLogs, updateFilterOptions, updateKeyFilterOptions };

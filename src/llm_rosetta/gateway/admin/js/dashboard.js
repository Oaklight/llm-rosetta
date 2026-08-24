/**
 * dashboard.js — Dashboard tab: metrics, profiling, capture, error dumps,
 * and canvas charts.
 */

import { S, DUMP_PAGE_SIZE } from './state.js';
import { t } from './i18n.js';
import { api, _adminHeaders, showToast, esc, formatDuration, closeModal, fmtBytesShort, fmtBytesLong } from './core.js';

// ===================== Metrics =====================

async function loadMetrics() {
  try {
    const data = await api.get('/admin/api/metrics?seconds=60');
    renderStats(data);
    drawChart('chartThroughput', data.series, 'count', 'req/s');
    drawChart('chartLatency', data.series, 'avg_ms', 'ms');
    renderProviderBreakdown(data);
    renderPersistence(data.persistence, data.total_requests);
    loadProfilingStatus();
    loadProfilingResults();
    loadCaptureStatus();
    loadCaptureResults();
  } catch { /* metrics endpoint unavailable — keep stale data */ }
}

// ===================== Profiling =====================

async function loadProfilingStatus() {
  try {
    const s = await api.get('/admin/api/profiling/status');
    S._profilingEnabled = s.enabled;
    const badge = document.getElementById('profilingBadge');
    const btn = document.getElementById('profilingToggleBtn');
    if (badge) {
      if (s.enabled) {
        badge.textContent = t('profiling.remaining').replace('{n}', s.remaining);
        badge.style.background = 'var(--green, #22c55e)';
        badge.style.color = '#fff';
      } else {
        badge.textContent = 'OFF';
        badge.style.background = 'var(--text-dim, #888)';
        badge.style.color = '#fff';
      }
    }
    if (btn) btn.textContent = s.enabled ? t('profiling.disable') : t('profiling.enable');
  } catch (e) { /* ignore */ }
}

async function loadProfilingResults() {
  try {
    const data = await api.get('/admin/api/profiling/results');
    const tbody = document.getElementById('profilingResults');
    const empty = document.getElementById('profilingEmpty');
    if (!tbody) return;
    const results = data.results || [];
    const dlAll = document.getElementById('profilingDownloadAll');
    if (results.length === 0) {
      tbody.innerHTML = '';
      if (empty) empty.style.display = '';
      if (dlAll) dlAll.style.display = 'none';
      return;
    }
    if (empty) empty.style.display = 'none';
    if (dlAll) dlAll.style.display = '';
    tbody.innerHTML = results.map((r, i) => {
      const ts = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '-';
      const mode = r.is_stream ? 'stream' : 'sync';
      const dur = typeof r.duration_ms === 'number' ? r.duration_ms.toFixed(0) + ' ms' : '-';
      return `<tr>
        <td>${esc(ts)}</td>
        <td>${esc(r.model || '-')}</td>
        <td>${esc(r.source || '-')} → ${esc(r.target || '-')}</td>
        <td>${mode}</td>
        <td>${dur}</td>
        <td style="white-space:nowrap"><button class="btn btn-sm" onclick="viewFlamegraph(${i})" title="View"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 14c-2.5 0-5-1.5-5-5 0-2 1-3.5 2.5-5.5C7 1.5 8 1 8 1s1 .5 2.5 2.5C12 5.5 13 7 13 9c0 3.5-2.5 5-5 5z"/><path d="M8 14c-1.5 0-2.5-1-2.5-3 0-1 .5-2 1.5-3 .5-.5 1-1 1-1s.5.5 1 1c1 1 1.5 2 1.5 3 0 2-1 3-2.5 3z"/></svg></button> <button class="btn btn-sm" onclick="downloadFlamegraph(${i}, '${esc(r.model||"profile")}')" title="Download"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button></td>
      </tr>`;
    }).join('');
  } catch (e) { /* ignore */ }
}

async function toggleProfiling() {
  if (S._profilingEnabled) {
    await api.post('/admin/api/profiling/disable');
  } else {
    const n = parseInt(document.getElementById('profilingCount')?.value || '5', 10);
    const res = await api.post('/admin/api/profiling/enable', { requests: Math.max(1, Math.min(100, n)) });
    if (res && res.error) { showToast(res.error, 'error'); return; }
  }
  await loadProfilingStatus();
}

async function clearProfilingResults() {
  await api.del('/admin/api/profiling/results');
  await loadProfilingResults();
}

async function _fetchFlamegraphHtml(index) {
  const url = '/admin/api/profiling/results/' + index + '?format=html';
  const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store'});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.text();
}

async function viewFlamegraph(index) {
  try {
    const html = await _fetchFlamegraphHtml(index);
    const w = window.open('', '_blank');
    if (w) { w.document.write(html); w.document.close(); }
  } catch (e) { showToast('Failed to load flamegraph', 'error'); }
}

async function downloadFlamegraph(index, model) {
  try {
    const html = await _fetchFlamegraphHtml(index);
    const blob = new Blob([html], {type: 'text/html'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `profile-${model}-${index}.html`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download flamegraph', 'error'); }
}

async function downloadAllFlamegraphs() {
  try {
    const url = '/admin/api/profiling/results/download';
    const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store'});
    if (!r.ok) { showToast('Failed to download', 'error'); return; }
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'profiling-results.zip';
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download', 'error'); }
}

// ===================== Content Capture =====================

async function loadCaptureStatus() {
  try {
    const s = await api.get('/admin/api/capture/status');
    S._captureEnabled = s.enabled;
    const badge = document.getElementById('captureBadge');
    const btn = document.getElementById('captureToggleBtn');
    if (badge) {
      if (s.enabled) {
        badge.textContent = t('capture.remaining').replace('{n}', s.remaining);
        badge.style.background = 'var(--green, #22c55e)';
        badge.style.color = '#fff';
      } else {
        badge.textContent = 'OFF';
        badge.style.background = 'var(--text-dim, #888)';
        badge.style.color = '#fff';
      }
    }
    if (btn) btn.textContent = s.enabled ? t('capture.disable') : t('capture.enable');
  } catch (e) { /* ignore */ }
}

async function loadCaptureResults() {
  try {
    const data = await api.get('/admin/api/capture/results');
    const tbody = document.getElementById('captureResults');
    const empty = document.getElementById('captureEmpty');
    if (!tbody) return;
    const results = data.results || [];
    const dlBtn = document.getElementById('captureDownloadAll');
    if (results.length === 0) {
      tbody.innerHTML = '';
      if (empty) empty.style.display = '';
      if (dlBtn) dlBtn.style.display = 'none';
      return;
    }
    if (empty) empty.style.display = 'none';
    if (dlBtn) dlBtn.style.display = '';
    tbody.innerHTML = results.map((r, i) => {
      const ts = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '-';
      const mode = r.is_stream ? 'stream' : 'sync';
      const st = r.status_code != null ? r.status_code : '-';
      return `<tr>
        <td>${esc(ts)}</td>
        <td>${esc(r.model || '-')}</td>
        <td>${esc(r.source_provider || '-')} → ${esc(r.target_provider || '-')}</td>
        <td>${mode}</td>
        <td>${st}</td>
        <td><button class="btn btn-sm" onclick="viewCapture(${i})" title="View"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button> <button class="btn btn-sm" onclick="downloadCapture(${i})" title="Download"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button></td>
      </tr>`;
    }).join('');
  } catch (e) { /* ignore */ }
}

async function toggleCapture() {
  if (S._captureEnabled) {
    await api.post('/admin/api/capture/disable');
  } else {
    const n = parseInt(document.getElementById('captureCount')?.value || '5', 10);
    await api.post('/admin/api/capture/enable', { requests: Math.max(1, Math.min(100, n)) });
  }
  await loadCaptureStatus();
}

async function clearCaptureResults() {
  await api.del('/admin/api/capture/results');
  await loadCaptureResults();
}

async function viewCapture(index) {
  try {
    const data = await api.get('/admin/api/capture/results/' + index);
    const w = window.open('', '_blank');
    if (!w) return;
    const sections = [
      { title: 'Original Request', body: data.original_request },
      { title: 'Converted Body', body: data.converted_body },
      { title: 'Upstream Response', body: data.upstream_response },
    ];
    const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Capture #${index}</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}
h1{font-size:18px;color:#a0c4ff}h2{font-size:15px;margin-top:24px;color:#bbb;cursor:pointer;user-select:none}
.meta{font-size:13px;color:#888;margin-bottom:16px}
pre{background:#0d1117;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.5;border:1px solid #333;white-space:pre-wrap;word-break:break-word}
.collapsed pre{display:none}
</style></head><body>
<h1>Content Capture Detail</h1>
<div class="meta">Model: ${esc(data.model||'-')} | ${esc(data.source_provider||'-')} → ${esc(data.target_provider||'-')} | ${data.is_stream?'stream':'sync'} | Status: ${data.status_code??'-'} | ${esc(data.timestamp||'-')}</div>
${sections.map(s => `<div><h2 onclick="this.parentElement.classList.toggle('collapsed')">${s.title} ▼</h2><pre>${esc(JSON.stringify(s.body, null, 2))}</pre></div>`).join('')}
</body></html>`;
    w.document.write(html);
    w.document.close();
  } catch (e) { showToast('Failed to load capture detail', 'error'); }
}

async function downloadCapture(index) {
  try {
    const data = await api.get('/admin/api/capture/results/' + index);
    const model = data.model || 'capture';
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], {type: 'application/json'});
    const ts = data.timestamp ? new Date(data.timestamp).toISOString().replace(/[:.]/g, '-').slice(0, 19) : index;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `capture-${model}-${ts}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download capture', 'error'); }
}

async function downloadAllCaptures() {
  try {
    const data = await api.get('/admin/api/capture/results');
    const results = data.results || [];
    if (results.length === 0) { showToast('No captures to download', 'error'); return; }
    const json = JSON.stringify(results, null, 2);
    const blob = new Blob([json], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `captures-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download captures', 'error'); }
}

// ===================== Error Dumps =====================

async function loadDumps() {
  S._dumpAllEntries = [];
  renderDumps();
}

async function renderDumps() {
  const phase = document.getElementById('dumpPhaseFilter').value;
  const status = document.getElementById('dumpStatusFilter').value;
  const provider = document.getElementById('dumpProviderFilter').value;
  const modelDrop = document.getElementById('dumpModelFilter').value;
  const search = document.getElementById('dumpModelSearch').value.toLowerCase();
  const timeRange = document.getElementById('dumpTimeRange').value;
  const dateFrom = document.getElementById('dumpDateFrom').value;
  const dateTo = document.getElementById('dumpDateTo').value;

  try {
    // Fetch all entries on first load or refresh
    if (S._dumpAllEntries.length === 0) {
      const data = await api.get('/admin/api/error-dumps?limit=10000&offset=0');
      S._dumpAllEntries = data.entries || [];
    }

    // Populate dropdowns from full dataset (before filtering)
    const provSelect = document.getElementById('dumpProviderFilter');
    const prevProv = provSelect.value;
    const allProvSet = new Set(S._dumpAllEntries.map(e => e.provider_name).filter(Boolean));
    provSelect.innerHTML = `<option value="">${t('filter.allProviders')}</option>` +
      [...allProvSet].sort().map(p => `<option value="${p}"${p===prevProv?' selected':''}>${esc(p)}</option>`).join('');

    const modelSelect = document.getElementById('dumpModelFilter');
    const prevModel = modelSelect.value;
    const allModelSet = new Set(S._dumpAllEntries.map(e => e.model).filter(Boolean));
    modelSelect.innerHTML = `<option value="">${t('filter.allModels')}</option>` +
      [...allModelSet].sort().map(m => `<option value="${m}"${m===prevModel?' selected':''}>${esc(m)}</option>`).join('') +
      '<option value="custom">Custom...</option>';

    // Apply all filters client-side
    let entries = S._dumpAllEntries;
    if (phase) entries = entries.filter(e => e.error_phase === phase);
    if (provider) entries = entries.filter(e => e.provider_name === provider);
    const filterModel = (modelDrop && modelDrop !== 'custom') ? modelDrop : '';
    if (filterModel) entries = entries.filter(e => e.model === filterModel);
    if (search) entries = entries.filter(e => (e.model||'').toLowerCase().includes(search));
    if (status) {
      if (status === '4xx') entries = entries.filter(e => e.status_code >= 400 && e.status_code < 500);
      else if (status === '5xx') entries = entries.filter(e => e.status_code >= 500 && e.status_code < 600);
      else entries = entries.filter(e => e.status_code === parseInt(status));
    }
    if (timeRange && timeRange !== 'custom') {
      const now = Date.now();
      const ms = {'1h':3600e3,'24h':86400e3,'7d':604800e3,'30d':2592000e3}[timeRange];
      if (ms) entries = entries.filter(e => now - new Date(e.timestamp).getTime() < ms);
    } else if (timeRange === 'custom') {
      if (dateFrom) entries = entries.filter(e => new Date(e.timestamp) >= new Date(dateFrom));
      if (dateTo) entries = entries.filter(e => new Date(e.timestamp) <= new Date(dateTo + 'T23:59:59'));
    }

    const total = entries.length;

    const tbody = document.getElementById('dumpTable');
    const countEl = document.getElementById('dumpCount');
    const emptyEl = document.getElementById('dumpEmpty');
    const dlBtn = document.getElementById('dumpDownloadAllBtn');

    countEl.textContent = total > 0 ? `${total} error${total !== 1 ? 's' : ''}` : '';
    emptyEl.style.display = total === 0 ? '' : 'none';
    if (dlBtn) dlBtn.style.display = S._dumpAllEntries.length > 0 ? '' : 'none';

    // Show/hide reset
    const hasFilters = phase || status || provider || search || filterModel || (timeRange && timeRange !== '');
    const resetBtn = document.getElementById('dumpResetBtn');
    if (resetBtn) resetBtn.style.display = hasFilters ? 'inline-flex' : 'none';

    // Client-side pagination on filtered results
    const totalPages = Math.ceil(total / DUMP_PAGE_SIZE) || 1;
    if (S._dumpPage >= totalPages) S._dumpPage = Math.max(0, totalPages - 1);
    const pageStart = S._dumpPage * DUMP_PAGE_SIZE;
    const pageEntries = entries.slice(pageStart, pageStart + DUMP_PAGE_SIZE);
    document.getElementById('dumpPageInfo').textContent = `Page ${S._dumpPage + 1} / ${totalPages}`;
    document.getElementById('dumpPrevPage').disabled = S._dumpPage === 0;
    document.getElementById('dumpNextPage').disabled = (S._dumpPage + 1) * DUMP_PAGE_SIZE >= total;

    if (total === 0) { tbody.innerHTML = ''; return; }

    tbody.innerHTML = pageEntries.map(e => {
      const time = new Date(e.timestamp).toLocaleString(undefined, {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
      let errPreview = '—';
      if (e.response_text) {
        try { errPreview = JSON.parse(e.response_text).error?.message || JSON.parse(e.response_text).detail || e.response_text; } catch { errPreview = e.response_text; }
        if (errPreview.length > 60) errPreview = errPreview.slice(0, 60) + '…';
      }
      return `<tr>
        <td>${time}</td>
        <td><code>${esc(e.model||'-')}</code></td>
        <td>${esc(e.source_provider||'-')} → ${esc(e.target_provider||'-')}</td>
        <td><span class="badge badge-stream">${esc(e.error_phase||'-')}</span></td>
        <td><span class="badge badge-error">${e.status_code||'-'}</span></td>
        <td><span style="font-size:11px;color:var(--red);font-family:var(--mono);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block" title="${esc(e.response_text||'')}">${esc(errPreview)}</span></td>
        <td style="white-space:nowrap">
          <button class="btn btn-sm" onclick="viewDump(${JSON.stringify(e.dump_id||e.id)})" title="View"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button>
          <button class="btn btn-sm" onclick="downloadDump(${JSON.stringify(e.dump_id||e.id)})" title="Download"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button>
        </td>
      </tr>`;
    }).join('');
  } catch (err) { /* ignore load errors */ }
}

async function viewDump(dumpId) {
  try {
    const e = await api.get(`/admin/api/error-dumps/${encodeURIComponent(dumpId)}`);
    const w = window.open('', '_blank');
    if (!w) return;
    const sections = [];
    if (e.response_text) {
      let formatted;
      try { formatted = JSON.stringify(JSON.parse(e.response_text), null, 2); } catch { formatted = e.response_text; }
      sections.push({title:'Response / Error', body:formatted, cls:'error'});
    }
    if (e.request_body) {
      const size = JSON.stringify(e.request_body).length;
      sections.push({title:`Request Body (${fmtBytesLong(size)})`, body:JSON.stringify(e.request_body, null, 2)});
    }
    if (e.converted_body) {
      const size = JSON.stringify(e.converted_body).length;
      sections.push({title:`Converted Body (${fmtBytesLong(size)})`, body:JSON.stringify(e.converted_body, null, 2)});
    }
    const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Error Dump — ${esc(e.model||'')}</title>
<style>body{font-family:system-ui,sans-serif;margin:0;background:#0f1117;color:#e4e7ef}
.header{padding:16px 24px;border-bottom:1px solid #2d3148;display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:16px;font-weight:600} .header button{padding:6px 14px;border-radius:6px;border:1px solid #2d3148;background:#1a1d27;color:#e4e7ef;cursor:pointer;font-size:13px}
.header button:hover{background:#242838} .content{padding:24px;max-width:1000px;margin:0 auto}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px 24px;margin-bottom:20px;padding:12px;background:#1a1d27;border:1px solid #2d3148;border-radius:8px;font-size:13px}
.meta .label{color:#8b90a5;font-size:11px;text-transform:uppercase;letter-spacing:.5px} .meta .value{color:#e4e7ef;font-family:'SF Mono',Consolas,monospace;font-size:12px;word-break:break-all} .meta .value.error{color:#ef4444}
.section{margin-bottom:16px} .section-header{font-size:14px;font-weight:600;color:#8b90a5;margin-bottom:8px;cursor:pointer;user-select:none}
.section-header:hover{color:#e4e7ef} .section-header::before{content:'▾ ';font-size:10px} .section.collapsed .section-header::before{content:'▸ '} .section.collapsed pre{display:none}
pre{background:#0d1117;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px;line-height:1.6;border:1px solid #333;white-space:pre-wrap;word-break:break-word;font-family:'SF Mono',Consolas,monospace;max-height:400px;overflow-y:auto}
pre.error{color:#ef4444}</style></head><body>
<div class="header"><h1>Error Dump Detail</h1><div><button onclick="downloadThis()">↓ Download JSON</button></div></div>
<div class="content"><div class="meta">
<div><div class="label">Model</div><div class="value">${esc(e.model||'-')}</div></div>
<div><div class="label">Routing</div><div class="value">${esc(e.source_provider||'-')} → ${esc(e.target_provider||'-')}</div></div>
<div><div class="label">Provider</div><div class="value">${esc(e.provider_name||'-')}</div></div>
<div><div class="label">Phase</div><div class="value">${esc(e.error_phase||'-')}</div></div>
<div><div class="label">Status</div><div class="value error">${e.status_code||'-'}</div></div>
<div><div class="label">Time</div><div class="value">${esc(e.timestamp||'-')}</div></div>
<div><div class="label">Upstream URL</div><div class="value">${esc(e.upstream_url||'-')}</div></div>
<div><div class="label">Dump ID</div><div class="value">${esc(e.dump_id||e.id||'-')}</div></div>
</div>${sections.map(s => `<div class="section"><div class="section-header" onclick="this.parentElement.classList.toggle('collapsed')">${s.title}</div><pre${s.cls?` class="${s.cls}"`:''}>` + (function(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;})(s.body) + `</pre></div>`).join('')}
</div><`+`script>function downloadThis(){const b=new Blob([${JSON.stringify(JSON.stringify(e,null,2)).replace(/</g,'\\u003c')}],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='error-dump-${esc(e.model||'error')}-${esc(e.dump_id||e.id||'')}.json';a.click();URL.revokeObjectURL(a.href);}<`+`/script></body></html>`;
    w.document.write(html);
    w.document.close();
  } catch (err) { showToast(t('error.loadFailed'), 'error'); }
}

async function downloadDump(dumpId) {
  try {
    const data = await api.get(`/admin/api/error-dumps/${encodeURIComponent(dumpId)}`);
    const model = data.model || 'error';
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], {type:'application/json'});
    const ts = data.timestamp ? new Date(data.timestamp).toISOString().replace(/[:.]/g,'-').slice(0,19) : dumpId;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `error-dump-${model}-${ts}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download', 'error'); }
}

async function downloadAllDumps() {
  try {
    const data = await api.get('/admin/api/error-dumps?limit=10000&offset=0');
    const entries = data.entries || [];
    if (entries.length === 0) { showToast('No dumps to download', 'error'); return; }
    const json = JSON.stringify(entries, null, 2);
    const blob = new Blob([json], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `error-dumps-${new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast('Failed to download', 'error'); }
}

function changeDumpPage(dir) {
  S._dumpPage = Math.max(0, S._dumpPage + dir);
  renderDumps();
}

function toggleDumpMoreMenu(btn) {
  const menu = document.getElementById('dumpMoreMenu');
  const show = menu.style.display === 'none';
  menu.style.display = show ? '' : 'none';
  if (show) {
    const close = (e) => { if (!btn.parentElement.contains(e.target)) { menu.style.display = 'none'; document.removeEventListener('click', close); } };
    setTimeout(() => document.addEventListener('click', close), 0);
  }
}

function openClearDumpsConfirm() {
  document.getElementById('clearDumpsInput').value = '';
  const btn = document.getElementById('clearDumpsBtn');
  btn.disabled = true; btn.style.opacity = '0.4'; btn.style.cursor = 'not-allowed';
  document.getElementById('clearDumpsModal').classList.add('open');
  document.getElementById('clearDumpsInput').focus();
}

function onClearDumpsInput() {
  const matched = document.getElementById('clearDumpsInput').value === 'CLEAR';
  const btn = document.getElementById('clearDumpsBtn');
  btn.disabled = !matched;
  btn.style.opacity = matched ? '1' : '0.4';
  btn.style.cursor = matched ? 'pointer' : 'not-allowed';
}

async function confirmClearDumps() {
  closeModal('clearDumpsModal');
  await api.del('/admin/api/error-dumps');
  showToast('Cleared all error dumps');
  S._dumpPage = 0;
  S._dumpAllEntries = [];
  renderDumps();
}

function onDumpModelFilterChange() {
  const val = document.getElementById('dumpModelFilter').value;
  if (val === 'custom') {
    document.getElementById('dumpModelFilter').style.display = 'none';
    document.getElementById('dumpModelSearchWrap').style.display = 'inline-flex';
    document.getElementById('dumpModelSearch').focus();
  }
  renderDumps();
}

function closeDumpModelSearch() {
  document.getElementById('dumpModelSearch').value = '';
  document.getElementById('dumpModelFilter').value = '';
  document.getElementById('dumpModelFilter').style.display = '';
  document.getElementById('dumpModelSearchWrap').style.display = 'none';
  renderDumps();
}

function onDumpTimeRangeChange() {
  const val = document.getElementById('dumpTimeRange').value;
  if (val === 'custom') {
    document.getElementById('dumpTimeRange').style.display = 'none';
    document.getElementById('dumpCustomDateRange').style.display = 'inline-flex';
  }
  renderDumps();
}

function closeDumpTimeCustom() {
  document.getElementById('dumpDateFrom').value = '';
  document.getElementById('dumpDateTo').value = '';
  document.getElementById('dumpTimeRange').value = '';
  document.getElementById('dumpTimeRange').style.display = '';
  document.getElementById('dumpCustomDateRange').style.display = 'none';
  renderDumps();
}

function resetDumpFilters() {
  document.getElementById('dumpPhaseFilter').value = '';
  document.getElementById('dumpStatusFilter').value = '';
  document.getElementById('dumpProviderFilter').value = '';
  closeDumpModelSearch();
  closeDumpTimeCustom();
}

// ===================== Charts / Canvas =====================

function renderPersistence(p, totalReq) {
  S._lastPersistence = p;
  S._lastTotalReq = totalReq;
  const el = document.getElementById('dbFooter');
  if (!el) return;
  if (!p) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');

  const successCap = p.log_max_success || 0;
  const errorCap = p.log_max_error || 0;
  const successN = p.log_success_entries || 0;
  const errorN = p.log_error_entries || 0;
  const successPct = successCap > 0 ? Math.round((successN / successCap) * 100) : 0;
  const errorPct = errorCap > 0 ? Math.round((errorN / errorCap) * 100) : 0;
  const pctClass = (pct) => pct >= 95 ? 'crit' : (pct >= 75 ? 'warn' : '');
  const fmtN = (n) => Number.isFinite(n) ? n.toLocaleString() : '–';

  el.innerHTML = `
    <span class="seg" title="${t('footer.tip.req')}"><span class="k">${t('footer.req')}</span><span class="v">${fmtN(totalReq)}</span></span>
    <span class="seg" title="${t('footer.tip.ok')}"><span class="k">${t('footer.ok')}</span><span class="v ${pctClass(successPct)}">${successN}/${successCap} (${successPct}%)</span></span>
    <span class="seg" title="${t('footer.tip.err')}"><span class="k">${t('footer.err')}</span><span class="v ${pctClass(errorPct)}">${errorN}/${errorCap} (${errorPct}%)</span></span>
    <span class="seg" title="${t('footer.tip.db')}"><span class="k">${t('footer.db')}</span><span class="v">${fmtBytesShort(p.db_bytes)}</span></span>
    <span class="seg" title="${t('footer.tip.wal')}"><span class="k">WAL</span><span class="v">${fmtBytesShort(p.wal_bytes)}</span></span>
  `;
}

function renderStats(d) {
  const uptime = formatDuration(d.uptime_seconds);
  const errRate = (d.error_rate * 100).toFixed(1) + '%';
  document.getElementById('statsGrid').innerHTML = `
    <div class="stat-card"><div class="label">${t('stat.totalRequests')}</div><div class="value">${d.total_requests}</div></div>
    <div class="stat-card"><div class="label">${t('stat.errorRate')}</div><div class="value ${d.error_rate > 0.05 ? 'red' : 'green'}">${errRate}</div></div>
    <div class="stat-card"><div class="label">${t('stat.activeStreams')}</div><div class="value blue">${d.active_streams}</div></div>
    <div class="stat-card"><div class="label">${t('stat.uptime')}</div><div class="value">${uptime}</div></div>
  `;
}

function renderProviderBreakdown(d) {
  const tbody = document.getElementById('providerBreakdown');
  const entries = Object.entries(d.by_target_provider || {}).sort((a,b) => b[1]-a[1]);
  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="2" style="color:var(--text-dim)">${t('empty.data')}</td></tr>`;
    return;
  }
  tbody.innerHTML = entries.map(([p,c]) => `<tr><td>${esc(p)}</td><td>${c}</td></tr>`).join('');
}

async function rebuildMetrics() {
  try {
    const res = await api.post('/admin/api/metrics/rebuild');
    if (res && res.error) { showToast(res.error, 'error'); return; }
    showToast(t('toast.metricsRebuilt').replace('{n}', res.rebuilt_from || 0));
    loadMetrics();
  } catch (e) { showToast('Rebuild failed', 'error'); }
}

function drawChart(canvasId, series, key, unit) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;

  ctx.clearRect(0, 0, w, h);

  // Read theme colors from CSS variables
  const cs = getComputedStyle(document.documentElement);
  const gridColor = cs.getPropertyValue('--border').trim();
  const dimColor = cs.getPropertyValue('--text-dim').trim();
  const accentColor = cs.getPropertyValue('--accent').trim();

  const values = series.map(s => s[key]);
  const maxVal = Math.max(...values, 1);

  const padL = 40, padR = 8, padT = 8, padB = 24;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  // Grid lines
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = dimColor;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH / 4) * i;
    const val = maxVal * (1 - i/4);
    ctx.fillText(val.toFixed(val >= 10 ? 0 : 1), padL - 6, y + 3);
  }

  // X-axis labels
  ctx.textAlign = 'center';
  ctx.fillText('-60s', padL, h - 4);
  ctx.fillText('-30s', padL + chartW/2, h - 4);
  ctx.fillText('now', padL + chartW, h - 4);

  if (values.length === 0 || Math.max(...values) === 0) {
    ctx.fillStyle = dimColor;
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(t('empty.data'), padL + chartW/2, padT + chartH/2);
    return;
  }

  // Line
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = padL + (i / (values.length - 1)) * chartW;
    const y = padT + chartH - (values[i] / maxVal) * chartH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.lineTo(padL, padT + chartH);
  ctx.closePath();
  ctx.fillStyle = accentColor.startsWith('#') ? accentColor + '14' : 'rgba(99,102,241,0.08)';
  ctx.fill();
}

// ===================== Window globals =====================

Object.assign(window, {
  loadMetrics, loadProfilingStatus, loadProfilingResults,
  toggleProfiling, clearProfilingResults, viewFlamegraph,
  downloadFlamegraph, downloadAllFlamegraphs,
  loadCaptureStatus, loadCaptureResults, toggleCapture,
  clearCaptureResults, viewCapture, downloadCapture, downloadAllCaptures,
  loadDumps, renderDumps, viewDump, downloadDump, downloadAllDumps,
  changeDumpPage, toggleDumpMoreMenu, openClearDumpsConfirm,
  onClearDumpsInput, confirmClearDumps,
  onDumpModelFilterChange, closeDumpModelSearch,
  onDumpTimeRangeChange, closeDumpTimeCustom, resetDumpFilters,
  renderPersistence, renderStats, renderProviderBreakdown,
  rebuildMetrics, drawChart,
});

export { loadMetrics, loadDumps, renderPersistence, renderStats, renderProviderBreakdown, rebuildMetrics };

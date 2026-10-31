/**
 * dashboard.js — Dashboard tab: metrics, profiling, capture, error dumps,
 * and canvas charts.
 */

import { S, DUMP_PAGE_SIZE, LOG_LIMIT } from '../core/state.js';
import { t } from '../core/i18n.js';
import { pageCount, clampPage, shouldResync } from '../core/pagination.js';
import { api, _adminHeaders, showToast, esc, formatDuration, openModal, closeModal, fmtBytesShort, fmtBytesLong, fmtTokens, copyText } from '../core/utils.js';

// ===================== Metrics =====================

async function loadMetrics() {
  try {
    const data = await api.get('/admin/api/metrics?seconds=60');
    renderStats(data);
    drawThroughputChart('chartThroughput', data.series);
    drawLatencyChart('chartLatency', data.series);
    renderProviderBreakdown(data);
    renderPersistence(data.persistence, data.total_requests);
    loadProfilingStatus();
    loadProfilingResults();
    loadCaptureStatus();
    loadCaptureResults();
    loadRequestHistogram();
  } catch { /* metrics endpoint unavailable — keep stale data */ }
}

// ===================== Request Histogram =====================

// The in-memory series stops at five minutes, so the five-hour view reads the
// persisted log.  That is a table scan, not a counter: refresh it a minute at
// most, however often the dashboard itself refreshes.
let _histogramFetchedAt = 0;
let _histogramData = null;

//: The two windows the card offers.  The 24-hour view uses wider buckets so
//: the bars stay wide enough to hover.
const HISTOGRAM_RANGES = {
  '5h': {minutes: 300, buckets: 60},
  '24h': {minutes: 1440, buckets: 48},
};
const HISTOGRAM_RANGE_KEY = 'llm-rosetta-hist-range';
let _histogramRange = HISTOGRAM_RANGES[localStorage.getItem(HISTOGRAM_RANGE_KEY)]
  ? localStorage.getItem(HISTOGRAM_RANGE_KEY) : '5h';

function _syncHistogramRangeButtons() {
  document.querySelectorAll('.chart-range-btn').forEach(b => {
    const on = b.dataset.range === _histogramRange;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  const title = document.getElementById('chartHistogramTitle');
  if (title) title.textContent = t('chart.histogram', {window: _histogramRange});
}

/** Switch the chart between the five-hour and the day-long window. */
function setHistogramRange(key) {
  if (!HISTOGRAM_RANGES[key]) return;
  _histogramRange = key;
  localStorage.setItem(HISTOGRAM_RANGE_KEY, key);
  _histogramData = null;
  _histogramFetchedAt = 0;
  _syncHistogramRangeButtons();
  loadRequestHistogram();
}
window.setHistogramRange = setHistogramRange;

/** Paint from the last answer; used when the data is still fresh, and when a
 *  language switch has to relabel a chart it already has. */
function _renderHistogram(data) {
  _syncHistogramRangeButtons();
  drawRequestHistogram('chartHistogram', data);
  _renderHistogramLegend();
  const sub = document.getElementById('chartHistogramSub');
  if (sub) {
    const peak = Math.max(0, ...(data.buckets || []).map(b => b.count));
    sub.textContent = t('chart.histogramSub', {total: String(data.total || 0), peak: String(peak)});
  }
}

async function loadRequestHistogram() {
  if (_histogramData && Date.now() - _histogramFetchedAt < 60000) {
    _renderHistogram(_histogramData);
    return;
  }
  const range = HISTOGRAM_RANGES[_histogramRange] || HISTOGRAM_RANGES['5h'];
  try {
    const data = await api.get(
      `/admin/api/metrics/histogram?minutes=${range.minutes}&buckets=${range.buckets}`);
    _histogramFetchedAt = Date.now();
    _histogramData = data;
    _renderHistogram(data);
  } catch { /* persistence off, or the log is unreadable — leave the chart empty */ }
}

/** Bars per time bucket: total height, the error share painted red on top. */
/** Bars per time bucket: total height, the error share painted red on top.
 *
 *  @param {string} canvasId Canvas to paint.
 *  @param {object} data     The histogram payload.
 *  @param {number} hoverIdx Bucket under the cursor, drawn at full strength.
 */
function drawRequestHistogram(canvasId, data, hoverIdx = -1) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h, cs } = _chartSetup(canvasId);
  const padL = 40, padR = 8, padT = 8, padB = 20;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;
  const buckets = data.buckets || [];
  const maxVal = Math.max(1, ...buckets.map(b => b.count));

  _drawAxes(ctx, w, h, cs, maxVal, padL, padR, padT, padB, {labels: false});

  const accent = cs.getPropertyValue('--accent').trim() || '#2f7d5c';
  const red = cs.getPropertyValue('--red').trim() || '#ef4444';
  const dim = cs.getPropertyValue('--text-dim').trim();
  const border = cs.getPropertyValue('--border').trim();
  const n = buckets.length;
  if (!n) return;
  const lastB = buckets[n - 1];
  const bw = chartW / n;
  const barW = Math.max(1, bw - 1);

  if (hoverIdx >= 0 && hoverIdx < n) {
    ctx.fillStyle = border;
    ctx.globalAlpha = 0.35;
    ctx.fillRect(padL + hoverIdx * bw, padT, bw, chartH);
    ctx.globalAlpha = 1;
  }

  for (let i = 0; i < n; i++) {
    const b = buckets[i];
    if (!b.count) continue;
    const x = padL + i * bw;
    const totalH = (b.count / maxVal) * chartH;
    const bottom = padT + chartH;
    const errH = Math.min(totalH, (b.errors / maxVal) * chartH);
    const okH = totalH - errH;
    // The segments stack: green is what went through, red is the error cap on
    // top of it.  Overlaying red on the whole bar made a failing bucket read
    // as a red bar rather than as a tall bar with a red top.
    ctx.fillStyle = accent;
    ctx.globalAlpha = i === hoverIdx ? 0.85 : 0.5;
    // Success sits on the baseline; the error cap sits on top of it.  Both
    // anchored at the top of the bar left the baseline empty, so a bucket
    // with any success in it floated.
    ctx.fillRect(x, bottom - totalH + errH, barW, okH);
    ctx.globalAlpha = 1;
    if (errH > 0) {
      ctx.fillStyle = red;
      ctx.globalAlpha = i === hoverIdx ? 1 : 0.85;
      ctx.fillRect(x, bottom - totalH, barW, errH);
      ctx.globalAlpha = 1;
    }
  }

  // Time marks under the bars, spaced wide enough to stay apart.
  const labelCount = Math.max(2, Math.min(6, Math.floor(chartW / 120)));
  const marks = new Set();
  for (let k = 0; k < labelCount; k++) {
    marks.add(Math.round((k * (n - 1)) / (labelCount - 1)));
  }
  ctx.fillStyle = dim;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  marks.forEach(i => {
    const b = buckets[i];
    if (!b) return;
    const dd = new Date(b.t);
    // A mark from another day has to name it; today's marks stay short.
    ctx.fillText(_markLabel(dd, _dayKey(new Date(lastB.t))), padL + (i + 0.5) * bw, h - 4);
  });
  _histogramGeom = {
    padL, bw, buckets,
    width: data.bucket_seconds || 0,
    minutes: data.minutes || 0,
    lastDay: lastB ? _dayKey(new Date(lastB.t)) : '',
  };
  _wireHistogramHover(canvas);
}

/** Layout of the last paint, so the hover handler can map x to a bucket. */
let _histogramGeom = null;
let _histogramHover = -1;
let _histogramTip = null;

function _histogramTipEl() {
  if (_histogramTip && document.contains(_histogramTip)) return _histogramTip;
  const box = document.getElementById('chartHistogram')?.closest('.chart-box');
  if (!box) return null;
  _histogramTip = document.createElement('div');
  _histogramTip.className = 'chart-tip';
  box.appendChild(_histogramTip);
  return _histogramTip;
}

/** The colour key for the bars; the words follow the language. */
function _renderHistogramLegend() {
  const box = document.getElementById('chartHistogramLegend');
  if (!box) return;
  box.replaceChildren();
  for (const [cls, key] of [['ok', 'chart.legendOk'], ['err', 'chart.legendErr']]) {
    const item = document.createElement('span');
    item.className = 'chart-legend-item';
    const swatch = document.createElement('span');
    swatch.className = `chart-swatch ${cls}`;
    const label = document.createElement('span');
    label.textContent = t(key);
    item.append(swatch, label);
    box.appendChild(item);
  }
}

function _dayKey(dd) {
  return `${dd.getFullYear()}-${dd.getMonth() + 1}-${dd.getDate()}`;
}

/** ``12:35`` on the reference day, ``10-29 23:50`` on any other. */
function _markLabel(dd, refDay) {
  const clock = `${String(dd.getHours()).padStart(2, '0')}:${String(dd.getMinutes()).padStart(2, '0')}`;
  return _dayKey(dd) === refDay ? clock : `${dd.getMonth() + 1}-${dd.getDate()} ${clock}`;
}

function _hideHistogramTip() {
  if (_histogramTip) _histogramTip.classList.remove('show');
  if (_histogramHover !== -1) {
    _histogramHover = -1;
    if (_histogramData) drawRequestHistogram('chartHistogram', _histogramData, -1);
  }
}

function _onHistogramMove(e) {
  const g = _histogramGeom;
  const canvas = e.currentTarget;
  if (!g) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const idx = Math.floor((x - g.padL) / g.bw);
  if (idx < 0 || idx >= g.buckets.length) { _hideHistogramTip(); return; }
  const b = g.buckets[idx];
  if (idx !== _histogramHover) {
    _histogramHover = idx;
    if (_histogramData) drawRequestHistogram('chartHistogram', _histogramData, idx);
  }
  const tip = _histogramTipEl();
  if (!tip) return;
  const from = new Date(b.t);
  const to = new Date(from.getTime() + (g.width || 0) * 1000);
  // Across a day, a bare clock time reads as if it were today.
  const hm = (dd) => _markLabel(dd, g.lastDay || '');
  tip.replaceChildren();
  const head = document.createElement('div');
  head.className = 'chart-tip-head';
  head.textContent = `${hm(from)} \u2013 ${hm(to)}`;
  const body = document.createElement('div');
  body.textContent = t('chart.tipBody', {
    count: String(b.count),
    errors: String(b.errors),
    tokens: fmtTokens(b.output_tokens || 0),
  });
  tip.append(head, body);
  const box = canvas.closest('.chart-box');
  const boxRect = box.getBoundingClientRect();
  const left = Math.min(x + 14, boxRect.width - tip.offsetWidth - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(6, e.clientY - boxRect.top - 12)}px`;
  tip.classList.add('show');
}

/** Wire hover once; the canvas element itself survives every repaint. */
function _wireHistogramHover(canvas) {
  if (canvas._histHoverWired) return;
  canvas._histHoverWired = true;
  canvas.addEventListener('mousemove', _onHistogramMove);
  canvas.addEventListener('mouseleave', _hideHistogramTip);
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
        badge.textContent = t('label.off');
        badge.style.background = 'var(--text-dim, #888)';
        badge.style.color = '#fff';
      }
      // The stored text is a placeholder; reveal the badge only once the
      // server has answered, so a collapsed summary never claims OFF while
      // profiling is actually on.
      badge.style.visibility = 'visible';
    }
    if (btn) btn.textContent = s.enabled ? t('profiling.disable') : t('profiling.enable');
    const cb = document.getElementById('profilingTracing');
    if (cb && s.enabled) cb.checked = !!s.tracing;
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
        <td><input type="checkbox" class="row-check" data-index="${i}" onchange="updateProfilingBulk()"></td>
        <td>${esc(ts)}</td>
        <td>${esc(r.model || '-')}</td>
        <td>${esc(r.source || '-')} → ${esc(r.target || '-')}</td>
        <td>${mode}</td>
        <td>${dur}</td>
        <td>${r.tracing ? '<span style="color:var(--orange,#f59e0b);font-weight:600">trace</span>' : 'cProfile'}</td>
        <td style="white-space:nowrap"><button class="btn btn-sm" onclick="viewFlamegraph(${i})" title="${esc(t('btn.view'))}"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 14c-2.5 0-5-1.5-5-5 0-2 1-3.5 2.5-5.5C7 1.5 8 1 8 1s1 .5 2.5 2.5C12 5.5 13 7 13 9c0 3.5-2.5 5-5 5z"/><path d="M8 14c-1.5 0-2.5-1-2.5-3 0-1 .5-2 1.5-3 .5-.5 1-1 1-1s.5.5 1 1c1 1 1.5 2 1.5 3 0 2-1 3-2.5 3z"/></svg></button> <button class="btn btn-sm" onclick="downloadFlamegraph(${i}, '${esc(r.model||"profile")}')" title="${esc(t('btn.downloadFile'))}"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button></td>
      </tr>`;
    }).join('');
  } catch (e) { /* ignore */ }
}

async function toggleProfiling() {
  if (S._profilingEnabled) {
    await api.post('/admin/api/profiling/disable');
  } else {
    const n = parseInt(document.getElementById('profilingCount')?.value || '5', 10);
    const tracing = !!document.getElementById('profilingTracing')?.checked;
    const res = await api.post('/admin/api/profiling/enable', { requests: Math.max(1, Math.min(100, n)), tracing });
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
  const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store', credentials: 'include'});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.text();
}

async function viewFlamegraph(index) {
  try {
    const html = await _fetchFlamegraphHtml(index);
    const w = window.open('', '_blank');
    if (w) { w.document.write(html); w.document.close(); }
  } catch (e) { showToast(t('toast.loadFailed'), 'error'); }
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
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
}

async function downloadAllFlamegraphs() {
  try {
    const url = '/admin/api/profiling/results/download';
    const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store', credentials: 'include'});
    if (!r.ok) { showToast(t('toast.downloadFailed'), 'error'); return; }
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'profiling-results.zip';
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
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
        badge.textContent = t('label.off');
        badge.style.background = 'var(--text-dim, #888)';
        badge.style.color = '#fff';
      }
      // Same rule as the profiling badge: no state claim before the fetch.
      badge.style.visibility = 'visible';
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
        <td><input type="checkbox" class="row-check" data-index="${i}" onchange="updateCaptureBulk()"></td>
        <td>${esc(ts)}</td>
        <td>${esc(r.model || '-')}</td>
        <td>${esc(r.source_provider || '-')} → ${esc(r.target_provider || '-')}</td>
        <td>${mode}</td>
        <td>${st}</td>
        <td><button class="btn btn-sm" onclick="viewCapture(${i})" title="${esc(t('btn.view'))}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button> <button class="btn btn-sm" onclick="downloadCapture(${i})" title="${esc(t('btn.downloadFile'))}"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button></td>
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
<h1>${t('capture.detailTitle')}</h1>
<div class="meta">${t('col.model')}: ${esc(data.model||'-')} | ${esc(data.source_provider||'-')} → ${esc(data.target_provider||'-')} | ${data.is_stream?t('phase.stream'):t('sync.label')} | ${t('col.status')}: ${data.status_code??'-'} | ${esc(data.timestamp||'-')}</div>
${sections.map(s => `<div><h2 onclick="this.parentElement.classList.toggle('collapsed')">${s.title} ▼</h2><pre>${esc(JSON.stringify(s.body, null, 2))}</pre></div>`).join('')}
</body></html>`;
    w.document.write(html);
    w.document.close();
  } catch (e) { showToast(t('toast.loadFailed'), 'error'); }
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
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
}

async function downloadAllCaptures() {
  try {
    const data = await api.get('/admin/api/capture/results');
    const results = data.results || [];
    if (results.length === 0) { showToast(t('toast.nothingToDownload'), 'error'); return; }
    const json = JSON.stringify(results, null, 2);
    const blob = new Blob([json], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `captures-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
}

// ===================== Error Dumps =====================

function invalidateDumpCache() {
  S._dumpAllEntries = [];
}

async function loadDumps() {
  invalidateDumpCache();
  renderDumps();
}

// Page the dump jump box was last synced to, and whether the user has edited
// it since.  See shouldResync() for why both signals are needed.
let _dumpJumpShown = 1;
let _dumpJumpDirty = false;

const _dumpJumpInput = document.getElementById('dumpPageJump');
_dumpJumpInput?.addEventListener?.('input', () => { _dumpJumpDirty = true; });
_dumpJumpInput?.addEventListener?.('blur', () => { _dumpJumpDirty = false; });

async function renderDumps() {
  const phase = document.getElementById('dumpPhaseFilter').value;
  const statusDrop = document.getElementById('dumpStatusFilter').value;
  const statusSearch = document.getElementById('dumpStatusSearch').value.trim();
  const status = (statusDrop === 'custom' || (!statusDrop && statusSearch)) ? statusSearch : statusDrop;
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
      `<option value="custom">${t('filter.custom')}</option>`;

    // Apply all filters client-side
    let entries = S._dumpAllEntries;
    // An id search is how a linked dump is found without paging to it.
    const idQuery = (document.getElementById('dumpIdSearch')?.value || '').trim().toLowerCase();
    if (idQuery) {
      entries = entries.filter(e =>
        String(e.dump_id || e.id || '').toLowerCase().includes(idQuery)
        || String(e.request_log_id || '').toLowerCase().includes(idQuery));
    }
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

    countEl.textContent = total > 0 ? t('dump.count', {n: total}) : '';
    emptyEl.style.display = total === 0 ? '' : 'none';
    if (dlBtn) dlBtn.style.display = S._dumpAllEntries.length > 0 ? '' : 'none';

    // Show/hide reset
    const hasFilters = phase || status || provider || search || filterModel
      || idQuery || (timeRange && timeRange !== '');
    const resetBtn = document.getElementById('dumpResetBtn');
    if (resetBtn) resetBtn.style.display = hasFilters ? 'inline-flex' : 'none';

    // Client-side pagination on filtered results
    const totalPages = pageCount(total, DUMP_PAGE_SIZE);
    if (S._dumpPage >= totalPages) S._dumpPage = Math.max(0, totalPages - 1);
    const pageStart = S._dumpPage * DUMP_PAGE_SIZE;
    const pageEntries = entries.slice(pageStart, pageStart + DUMP_PAGE_SIZE);
    S._dumpTotalPages = totalPages;
    document.getElementById('dumpPageInfo').textContent = t('page.info', {current: S._dumpPage + 1, total: totalPages, count: ''}).replace(/\s*\(\s*\)\s*$/, '');
    document.getElementById('dumpPrevPage').disabled = S._dumpPage === 0;
    document.getElementById('dumpNextPage').disabled = (S._dumpPage + 1) * DUMP_PAGE_SIZE >= total;
    const jump = document.getElementById('dumpPageJump');
    if (jump) {
      jump.max = String(totalPages);
      jump.disabled = totalPages <= 1;
      const focused = document.activeElement === jump;
      if (shouldResync(jump.value, _dumpJumpShown, { focused, dirty: _dumpJumpDirty })) {
        jump.value = String(S._dumpPage + 1);
        _dumpJumpShown = S._dumpPage + 1;
      }
    }

    // A selection outlives the rows it was made on.  Drop ids the dump store
    // no longer has, so the bar never counts a dump that is already gone.
    const liveIds = new Set(S._dumpAllEntries.map(e => esc(e.dump_id || e.id || '')));
    for (const id of [...S._selectedDumpIds]) if (!liveIds.has(id)) S._selectedDumpIds.delete(id);

    // The header box covers one page; these two cover the whole filtered set,
    // which is what an operator filtering to a provider actually means.
    S._dumpFilteredIds = pageEntries.length ? entries.map(e => esc(e.dump_id || e.id || '')) : [];
    _syncDumpScopeButtons(total);

    if (total === 0) { tbody.innerHTML = ''; _syncDumpHeader(); return; }

    tbody.innerHTML = pageEntries.map(e => {
      const time = new Date(e.timestamp).toLocaleString(undefined, {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
      let errPreview = '—';
      if (e.response_text) {
        try { errPreview = JSON.parse(e.response_text).error?.message || JSON.parse(e.response_text).detail || e.response_text; } catch { errPreview = e.response_text; }
        if (errPreview.length > 60) errPreview = errPreview.slice(0, 60) + '…';
      }
      const did = esc(e.dump_id||e.id||'');
      const chk = S._selectedDumpIds && S._selectedDumpIds.has(did) ? ' checked' : '';
      const dumpHlStyle = S._highlightDumpId === did ? ' style="background:color-mix(in srgb, var(--accent) 20%, transparent)"' : '';
      return `<tr role="row"${dumpHlStyle}>
        <td class="dump-check" role="cell" aria-labelledby="dumpThSelect"><input type="checkbox" class="row-check" data-id="${did}" onchange="updateDumpBulk()"${chk}></td>
        <td class="dump-time" role="cell" aria-labelledby="dumpThTime">${time}</td>
        <td class="dump-model" role="cell" aria-labelledby="dumpThModel"><code>${esc(e.model||'-')}</code></td>
        <td class="dump-route" role="cell" aria-labelledby="dumpThSourceTarget">${esc(e.source_provider||'-')} → ${esc(e.target_provider||'-')}</td>
        <td class="dump-phase" role="cell" aria-labelledby="dumpThPhase"><span class="badge badge-stream">${esc(e.error_phase||'-')}</span></td>
        <td class="dump-status" role="cell" aria-labelledby="dumpThStatus"><span class="badge badge-error">${e.status_code||'-'}</span></td>
        <td class="dump-error" role="cell" aria-labelledby="dumpThError"><span class="dump-error-text" title="${esc(e.response_text||'')}">${esc(errPreview)}</span></td>
        <td class="dump-actions" role="cell" aria-labelledby="dumpThActions" style="white-space:nowrap">
          <button class="btn btn-sm" onclick="viewDump('${esc(e.dump_id||e.id)}')" title="${esc(t('btn.view'))}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></button>
          <button class="btn btn-sm" onclick="downloadDump('${esc(e.dump_id||e.id)}')" title="${esc(t('btn.downloadFile'))}"><svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" style="vertical-align:middle"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10"/></svg></button>${e.request_log_id ? ` <button class="btn btn-sm" onclick="jumpToRequestLog('${esc(e.request_log_id)}')" title="${esc(t('btn.viewInRequestLog'))}"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg></button>` : ''}
        </td>
      </tr>`;
    }).join('');
    _syncDumpHeader();
  } catch (err) { /* ignore load errors */ }
}

/** Open the dump detail in its own tab, at a URL that names the dump. */
function viewDump(dumpId) {
  window.open(`${location.pathname}#dump/${encodeURIComponent(dumpId)}`, '_blank');
}

// ===================== Error Dump Page =====================
//
// #dump/<id> is a real, shareable URL.  The page lives inside the admin app,
// so it answers to the same session check as every other screen: no cookie,
// no dump.

let _dumpPageEntry = null;

//: A body larger than this is not painted on open; it is a download.
const _DP_INLINE_LIMIT = 64 * 1024;

/** Save one body on its own, without the rest of the dump around it. */
function _dpDownload(text, filename, mime) {
  const blob = new Blob([text], {type: mime || 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** A small action in a section header.  It must not toggle the section. */
function _dpToolBtn(label, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'dp-tool';
  b.textContent = label;
  b.addEventListener('click', e => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  });
  return b;
}

/**
 * One body of the dump: a title, its size, copy and download.
 *
 * @param {string} title Section name.
 * @param {string} text  The body, already formatted for reading or saving.
 * @param {object} opts  ``cls`` (``is-error``), ``filename`` for the download,
 *   ``mime`` when it is not JSON.
 */
function _dumpPageSection(title, text, opts = {}) {
  const wrap = document.createElement('details');
  wrap.className = 'dp-section';
  wrap.open = true;

  const sum = document.createElement('summary');
  const label = document.createElement('span');
  label.className = 'dp-title';
  label.textContent = title;
  const size = document.createElement('span');
  size.className = 'dp-size';
  size.textContent = fmtBytesLong(new Blob([text]).size);
  const tools = document.createElement('span');
  tools.className = 'dp-tools';
  tools.append(_dpToolBtn(t('btn.copy'), () => copyText(text)));
  if (opts.filename) {
    tools.append(_dpToolBtn(t('btn.download'), () =>
      _dpDownload(text, opts.filename, opts.mime)));
  }
  sum.append(label, size, tools);
  wrap.append(sum);

  const pre = document.createElement('pre');
  if (opts.cls) pre.className = opts.cls;
  pre.textContent = text;

  if (text.length > _DP_INLINE_LIMIT) {
    // Painting a multi-megabyte body freezes the tab, and nobody reads it
    // in place: the download is the way out, with an opt-in to show it.
    const note = document.createElement('div');
    note.className = 'dp-big';
    const msg = document.createElement('span');
    msg.textContent = t('dump.tooLarge', {size: fmtBytesLong(new Blob([text]).size)});
    note.append(msg, _dpToolBtn(t('dump.showAnyway'), () => { note.replaceWith(pre); }));
    wrap.append(note);
  } else {
    wrap.append(pre);
  }
  return wrap;
}

function _dumpPageMetaRow(label, value, cls) {
  const cell = document.createElement('div');
  const l = document.createElement('div');
  l.className = 'dp-label';
  l.textContent = label;
  const v = document.createElement('div');
  v.className = 'dp-value' + (cls ? ' ' + cls : '');
  v.textContent = value == null || value === '' ? '-' : String(value);
  cell.append(l, v);
  return cell;
}

async function openDumpPage(dumpId) {
  const page = document.getElementById('dumpPage');
  if (!page) return;
  page.hidden = false;
  const title = document.getElementById('dumpPageTitle');
  const meta = document.getElementById('dumpPageMeta');
  const body = document.getElementById('dumpPageBody');
  title.textContent = dumpId;
  meta.replaceChildren();
  body.replaceChildren();
  document.title = t('dump.pageTitle', {id: dumpId});
  try {
    const e = await api.get(`/admin/api/error-dumps/${encodeURIComponent(dumpId)}`);
    _dumpPageEntry = e;
    meta.replaceChildren(
      _dumpPageMetaRow(t('dump.metaModel'), e.model),
      _dumpPageMetaRow(t('dump.metaRouting'), `${e.source_provider || '-'} \u2192 ${e.target_provider || '-'}`),
      _dumpPageMetaRow(t('dump.metaProvider'), e.provider_name),
      _dumpPageMetaRow(t('dump.metaPhase'), e.error_phase),
      _dumpPageMetaRow(t('dump.metaStatus'), e.status_code, 'is-error'),
      _dumpPageMetaRow(t('dump.metaTime'), e.timestamp),
      _dumpPageMetaRow(t('dump.metaUpstream'), e.upstream_url),
      _dumpPageMetaRow(t('dump.metaCollisions'), e.tool_name_collisions),
      _dumpPageMetaRow(t('dump.metaDumpId'), e.dump_id || e.id),
      _dumpPageMetaRow(t('dump.metaRequestLog'), e.request_log_id),
    );
    const stem = `error-dump-${e.model || 'error'}-${e.dump_id || e.id}`;
    const asText = (raw) => {
      // Keep the body readable: pretty JSON when it is JSON, the text as-is
      // when it is not.
      if (typeof raw !== 'string') return {text: JSON.stringify(raw, null, 2), json: true};
      try { return {text: JSON.stringify(JSON.parse(raw), null, 2), json: true}; }
      catch { return {text: raw, json: false}; }
    };
    if (e.response_text) {
      const {text, json} = asText(e.response_text);
      body.appendChild(_dumpPageSection(t('dump.sectionError'), text, {
        cls: 'is-error', filename: `${stem}-response.${json ? 'json' : 'txt'}`,
        mime: json ? 'application/json' : 'text/plain',
      }));
    }
    for (const [key, label, suffix] of [
      ['request_body', t('dump.sectionRequest'), 'request'],
      ['converted_body', t('dump.sectionConverted'), 'converted-request'],
      ['upstream_request_body', t('dump.sectionUpstreamRequest'), 'upstream-request'],
      ['upstream_response_text', t('dump.sectionUpstreamResponse'), 'upstream-response'],
    ]) {
      const raw = e[key];
      if (raw == null || raw === '') continue;
      const {text, json} = asText(raw);
      body.appendChild(_dumpPageSection(label, text, {
        filename: `${stem}-${suffix}.${json ? 'json' : 'txt'}`,
        mime: json ? 'application/json' : 'text/plain',
      }));
    }
    const logBtn = document.getElementById('dumpPageLogBtn');
    if (logBtn) logBtn.disabled = !e.request_log_id;
  } catch (err) {
    body.appendChild(_dumpPageSection(t('dump.notFound'), String(err && err.message || err), {cls: 'is-error'}));
  }
}

/** Hide the dump page.  Navigation is the caller's business: the tab strip
 *  closes it as a side effect of leaving, and the Back button also rewrites
 *  the URL. */
function closeDumpPage() {
  const page = document.getElementById('dumpPage');
  if (page) page.hidden = true;
  document.title = t('app.title');
  _dumpPageEntry = null;
}

/** The Back button: leave the dump and land on the dump list. */
function exitDumpPage() {
  closeDumpPage();
  if (location.hash.startsWith('#dump/')) {
    history.pushState({tab: 'logs'}, '', '#logs');
    window.switchLogView?.('dumps');
    window.goToTab?.('logs');
  }
}

/** The dump as one JSON document, to paste somewhere else. */
function copyDumpPageJson() {
  if (!_dumpPageEntry) return;
  copyText(JSON.stringify(_dumpPageEntry, null, 2));
}

function downloadDumpPage() {
  if (!_dumpPageEntry) return;
  const id = _dumpPageEntry.dump_id || _dumpPageEntry.id;
  const blob = new Blob([JSON.stringify(_dumpPageEntry, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `error-dump-${_dumpPageEntry.model || 'error'}-${id}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function openDumpPageLog() {
  const id = _dumpPageEntry?.request_log_id;
  if (!id) return;
  closeDumpPage();
  window.jumpToRequestLog?.(id);
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
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
}

async function downloadAllDumps() {
  try {
    const data = await api.get('/admin/api/error-dumps?limit=10000&offset=0');
    const entries = data.entries || [];
    if (entries.length === 0) { showToast(t('toast.nothingToDownload'), 'error'); return; }
    const json = JSON.stringify(entries, null, 2);
    const blob = new Blob([json], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `error-dumps-${new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { showToast(t('toast.downloadFailed'), 'error'); }
}

function changeDumpPage(dir) {
  S._dumpPage = Math.max(0, S._dumpPage + dir);
  _dumpJumpDirty = false;
  renderDumps();
}

// The dump list paginates in the browser, so a jump is just a re-render at the
// clamped page.  Out-of-range numbers clamp; an empty box is ignored.
function jumpToDumpPage() {
  const input = document.getElementById('dumpPageJump');
  if (!input) return;
  const page = clampPage(input.value, S._dumpTotalPages || 1);
  if (page === null) {
    input.value = String(S._dumpPage + 1);
    _dumpJumpShown = S._dumpPage + 1;
    _dumpJumpDirty = false;
    return;
  }
  input.value = String(page);
  _dumpJumpShown = page;
  _dumpJumpDirty = false;
  S._dumpPage = page - 1;
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
  openModal('clearDumpsModal');
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
  showToast(t('toast.dumpsCleared'));
  S._dumpPage = 0;
  invalidateDumpCache();
  renderDumps();
  loadMetrics();
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

function closeDumpModelSearch(silent) {
  document.getElementById('dumpModelSearch').value = '';
  document.getElementById('dumpModelFilter').value = '';
  document.getElementById('dumpModelFilter').style.display = '';
  document.getElementById('dumpModelSearchWrap').style.display = 'none';
  if (!silent) renderDumps();
}

function onDumpStatusFilterChange() {
  const val = document.getElementById('dumpStatusFilter').value;
  if (val === 'custom') {
    document.getElementById('dumpStatusFilter').style.display = 'none';
    document.getElementById('dumpStatusSearchWrap').style.display = 'inline-flex';
    document.getElementById('dumpStatusSearch').focus();
  }
  renderDumps();
}

function onDumpStatusSearchInput() {
  const el = document.getElementById('dumpStatusSearch');
  el.value = el.value.replace(/\D/g, '');
  renderDumps();
}

function closeDumpStatusSearch(silent) {
  document.getElementById('dumpStatusSearch').value = '';
  document.getElementById('dumpStatusFilter').value = '';
  document.getElementById('dumpStatusFilter').style.display = '';
  document.getElementById('dumpStatusSearchWrap').style.display = 'none';
  if (!silent) renderDumps();
}

function onDumpTimeRangeChange() {
  const val = document.getElementById('dumpTimeRange').value;
  if (val === 'custom') {
    document.getElementById('dumpTimeRange').style.display = 'none';
    document.getElementById('dumpCustomDateRange').style.display = 'inline-flex';
  }
  renderDumps();
}

function closeDumpTimeCustom(silent) {
  document.getElementById('dumpDateFrom').value = '';
  document.getElementById('dumpDateTo').value = '';
  document.getElementById('dumpTimeRange').value = '';
  document.getElementById('dumpTimeRange').style.display = '';
  document.getElementById('dumpCustomDateRange').style.display = 'none';
  if (!silent) renderDumps();
}

/** A new id query starts from the first page, or the match stays off-screen. */
function onDumpIdSearchInput() {
  S._dumpPage = 0;
  renderDumps();
}

function resetDumpFilters() {
  const ids = document.getElementById('dumpIdSearch');
  if (ids) ids.value = '';
  document.getElementById('dumpPhaseFilter').value = '';
  document.getElementById('dumpProviderFilter').value = '';
  closeDumpStatusSearch(true);
  closeDumpModelSearch(true);
  closeDumpTimeCustom(true);
  renderDumps();
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
  const successN = p.log_success_entries || 0;
  const errorN = p.log_error_entries || 0;
  const dumpN = p.dump_entries || 0;
  const dumpCap = p.dump_max || 0;
  const successPct = successCap > 0 ? Math.round((successN / successCap) * 100) : 0;
  const dumpPct = dumpCap > 0 ? Math.round((dumpN / dumpCap) * 100) : 0;
  const pctClass = (pct) => pct >= 95 ? 'crit' : (pct >= 75 ? 'warn' : '');
  const fmtN = (n) => Number.isFinite(n) ? n.toLocaleString() : '–';

  el.innerHTML = `
    <span class="seg seg-req" title="${t('footer.tip.req')}"><span class="k">${t('footer.req')}</span><span class="v">${fmtN(totalReq)}</span></span>
    <span class="seg seg-ok" title="${t('footer.tip.ok')}"><span class="k">${t('footer.ok')}</span><span class="v ${pctClass(successPct)}">${successN}/${successCap} (${successPct}%)</span></span>
    <span class="seg seg-err" title="${t('footer.tip.err')}"><span class="k">${t('footer.err')}</span><span class="v">${fmtN(errorN)}</span></span>
    <span class="seg seg-dumps" title="${t('footer.tip.dumps')}"><span class="k">${t('footer.dumps')}</span><span class="v ${pctClass(dumpPct)}">${dumpN}/${dumpCap} (${dumpPct}%)</span></span>
    <span class="seg seg-db" title="${t('footer.tip.db')}"><span class="k">${t('footer.db')}</span><span class="v">${fmtBytesShort(p.db_bytes)}</span></span>
    <span class="seg seg-wal" title="${t('footer.tip.wal')}"><span class="k">WAL</span><span class="v">${fmtBytesShort(p.wal_bytes)}</span></span>
  `;
}

function renderStats(d) {
  const uptime = formatDuration(d.uptime_seconds);
  const errRate = (d.error_rate * 100).toFixed(1) + '%';
  // The headline is the canonical total — the same number the API Keys panel
  // and the usage modal show, so the three can be reconciled.  The breakdown
  // below it is exactly that: reasoning rides inside the output, and cache
  // tokens are additive for some providers and a subset of the input for
  // others, so neither is added to the headline.
  const has24h = d.total_tokens_24h != null;
  const tokTotal = has24h ? (d.total_tokens_24h||0) : (d.total_tokens||0);
  const tokIn = has24h ? (d.input_tokens_24h||0) : (d.total_input_tokens||0);
  const tokOut = has24h ? (d.output_tokens_24h||0) : (d.total_output_tokens||0);
  const tokCacheR = has24h ? (d.cache_read_tokens_24h||0) : (d.total_cache_read_tokens||0);
  const tokCacheW = has24h ? (d.cache_creation_tokens_24h||0) : (d.total_cache_creation_tokens||0);
  const tokReason = has24h ? (d.reasoning_tokens_24h||0) : (d.total_reasoning_tokens||0);
  const tokLabel = has24h ? t('stat.tokens24h') : t('stat.totalTokens');
  document.getElementById('statsGrid').innerHTML = `
    <div class="stat-card"><div class="label">${t('stat.totalRequests')}</div><div class="value">${(d.total_requests || 0).toLocaleString()}</div></div>
    <div class="stat-card"><div class="label">${t('stat.errorRate')}</div><div class="value ${d.error_rate > 0.05 ? 'red' : 'green'}">${errRate}</div></div>
    <div class="stat-card"><div class="label">${t('stat.activeStreams')}</div><div class="value blue">${(d.active_streams || 0).toLocaleString()}</div></div>
    <div class="stat-card"><div class="label">${t('stat.uptime')}</div><div class="value">${uptime}</div></div>
    <div class="stat-card"><div class="label">${tokLabel}</div><div class="value">${fmtTokens(tokTotal)}</div><div class="sub">↑${fmtTokens(tokIn)} ↓${fmtTokens(tokOut)}${tokCacheR>0?` · ${t('stat.cacheRead')} ${fmtTokens(tokCacheR)}`:''}${tokCacheW>0?` · ${t('stat.cacheWrite')} ${fmtTokens(tokCacheW)}`:''}${tokReason>0?` · ${t('stat.reasoning')} ${fmtTokens(tokReason)}`:''}</div><div class="sub" style="margin-top:4px"><a href="#" onclick="event.preventDefault();openTokenUsageModal()" style="color:var(--accent);font-size:11px;text-decoration:none">${t('tokenUsage.details')}</a></div></div>
  `;
}

function renderProviderBreakdown(d) {
  const grid = document.getElementById('providerBreakdownGrid');
  const entries = Object.entries(d.by_target_provider || {}).sort((a,b) => b[1]-a[1]);
  if (entries.length === 0) {
    grid.innerHTML = `<div style="color:var(--text-dim);padding:10px">${t('empty.data')}</div>`;
    return;
  }
  grid.style.columnCount = 2;
  grid.innerHTML = entries.map(([p,c], i) => `<div class="pb-item"><span class="pb-rank">${i+1}</span><span class="pb-name">${esc(p)}</span><span class="pb-count">${c}</span></div>`).join('');
}

async function rebuildMetrics() {
  try {
    const res = await api.post('/admin/api/metrics/rebuild');
    if (res && res.error) { showToast(res.error, 'error'); return; }
    const n = res.rebuilt_from || 0;
    const before = res.before || {};
    const after = res.counters || {};
    const diffs = [];
    const check = (label, b, a) => { if (b !== a) diffs.push(`${label}: ${b} → ${a}`); };
    const fields = ['total_requests', 'total_errors', 'total_streams',
      'total_input_tokens', 'total_output_tokens', 'total_cache_read_tokens',
      'total_cache_creation_tokens', 'total_reasoning_tokens'];
    for (const f of fields) check(f, before[f] || 0, after[f] || 0);
    if (diffs.length === 0) {
      showToast(t('toast.metricsRebuilt').replace('{n}', n) + ' — counters consistent ✓');
    } else {
      showToast(t('toast.metricsRebuilt').replace('{n}', n) + ' — fixed: ' + diffs.join(', '));
    }
    loadMetrics();
  } catch (e) { showToast(t('toast.metricsRebuildFailed'), 'error'); }
}

function _chartSetup(canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  return { canvas, ctx, w, h, cs };
}

function _drawAxes(ctx, w, h, cs, maxVal, padL, padR, padT, padB, opts = {}) {
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;
  const gridColor = cs.getPropertyValue('--border').trim();
  const dimColor = cs.getPropertyValue('--text-dim').trim();

  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
  }

  ctx.fillStyle = dimColor;
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH / 4) * i;
    const val = maxVal * (1 - i / 4);
    ctx.fillText(val.toFixed(val >= 10 ? 0 : 1), padL - 6, y + 3);
  }

  // A chart with its own time span draws its own marks.
  if (opts.labels === false) return;
  ctx.textAlign = 'center';
  ctx.fillText('-60s', padL, h - 4);
  ctx.fillText('-30s', padL + chartW / 2, h - 4);
  ctx.fillText('now', padL + chartW, h - 4);
}

function drawThroughputChart(canvasId, series) {
  const { ctx, w, h, cs } = _chartSetup(canvasId);
  const padL = 40, padR = 8, padT = 8, padB = 24;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const values = series.map(s => s.count);
  const maxVal = Math.max(...values, 1);

  _drawAxes(ctx, w, h, cs, maxVal, padL, padR, padT, padB);

  if (values.length === 0 || Math.max(...values) === 0) {
    ctx.fillStyle = cs.getPropertyValue('--text-dim').trim();
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(t('empty.data'), padL + chartW / 2, padT + chartH / 2);
    return;
  }

  const accentColor = cs.getPropertyValue('--accent').trim();
  const fillColor = cs.getPropertyValue('--accent-subtle').trim() || 'rgba(0,0,0,0.08)';
  const n = values.length;
  const stepW = chartW / n;

  // Step-after fill
  ctx.beginPath();
  ctx.moveTo(padL, padT + chartH);
  for (let i = 0; i < n; i++) {
    const x = padL + i * stepW;
    const y = padT + chartH - (values[i] / maxVal) * chartH;
    ctx.lineTo(x, y);
    ctx.lineTo(x + stepW, y);
  }
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // Step-after stroke
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = padL + i * stepW;
    const y = padT + chartH - (values[i] / maxVal) * chartH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    ctx.lineTo(x + stepW, y);
  }
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawLatencyChart(canvasId, series) {
  const { ctx, w, h, cs } = _chartSetup(canvasId);
  const padL = 40, padR = 8, padT = 8, padB = 24;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  // Collect all individual latency points with their time offset
  const points = [];
  for (const entry of series) {
    if (entry.latencies) {
      for (const ms of entry.latencies) {
        points.push({ t: entry.t, ms });
      }
    }
  }

  const maxMs = points.reduce((m, p) => Math.max(m, p.ms), 1);

  _drawAxes(ctx, w, h, cs, maxMs, padL, padR, padT, padB);

  if (points.length === 0) {
    ctx.fillStyle = cs.getPropertyValue('--text-dim').trim();
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(t('empty.data'), padL + chartW / 2, padT + chartH / 2);
    return;
  }

  const blueColor = cs.getPropertyValue('--blue').trim();
  const tMin = series[0].t;
  const tMax = series[series.length - 1].t;
  const tRange = tMax - tMin || 1;

  ctx.strokeStyle = blueColor;
  ctx.lineWidth = 1;
  for (const p of points) {
    const x = padL + ((p.t - tMin) / tRange) * chartW;
    const y = padT + chartH - (p.ms / maxMs) * chartH;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = blueColor;
    ctx.fill();
    ctx.globalAlpha = 1.0;
    ctx.stroke();
  }
}


// ===================== Token Usage Modal =====================

let _tokenUsageDays = 1;

function openTokenUsageModal() {
  _tokenUsageDays = 1;
  const rangeCtrl = document.getElementById('tokenUsageRange');
  if (rangeCtrl) {
    rangeCtrl.querySelectorAll('label').forEach((l, i) => {
      l.classList.toggle('active', i === 0);
    });
  }
  const kSel = document.getElementById('tokenUsageKeyFilter');
  if (kSel) {
    const labels = (S.keysData && S.keysData.keys) || [];
    const allLabels = [...new Set([...labels.map(k => k.label), ...S.logKeyLabels].filter(Boolean))].sort();
    kSel.innerHTML = `<option value="">${t('filter.allKeys')}</option>` + allLabels.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('');
    kSel.value = '';
  }
  openModal('tokenUsageModal');
  loadTokenUsage();
}

function setTokenUsageRange(days) {
  _tokenUsageDays = days;
  const rangeCtrl = document.getElementById('tokenUsageRange');
  if (rangeCtrl) {
    const vals = [1, 7, 30];
    rangeCtrl.querySelectorAll('label').forEach((l, i) => {
      l.classList.toggle('active', vals[i] === days);
    });
  }
  loadTokenUsage();
}

async function loadTokenUsage() {
  const keyFilter = document.getElementById('tokenUsageKeyFilter')?.value || '';
  let url = `/admin/api/token-usage?days=${_tokenUsageDays}`;
  if (keyFilter) url += `&api_key_label=${encodeURIComponent(keyFilter)}`;
  try {
    const data = await api.get(url);
    renderTokenUsageTable(data);
  } catch { /* ignore */ }
}

function renderTokenUsageTable(data) {
  const tbody = document.getElementById('tokenUsageTable');
  const summaryEl = document.getElementById('tokenUsageSummary');
  const days = data.days || [];
  const totals = data.totals || {};

  if (days.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-dim)">${t('tokenUsage.noData')}</td></tr>`;
    summaryEl.innerHTML = '';
    return;
  }

  summaryEl.innerHTML = `<div style="display:flex;gap:16px;flex-wrap:wrap;padding:10px 12px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);font-size:12px">
    <span><strong>${t('tokenUsage.summary')}</strong></span>
    <span>${t('tokenUsage.col.requests')}: <strong>${(totals.request_count||0).toLocaleString()}</strong></span>
    <span>↑ <strong>${fmtTokens(totals.input_tokens||0)}</strong></span>
    <span>↓ <strong>${fmtTokens(totals.output_tokens||0)}</strong></span>
    <span>${t('stat.cacheRead')} <strong>${fmtTokens(totals.cache_read_tokens||0)}</strong></span>
    <span>${t('stat.cacheWrite')} <strong>${fmtTokens(totals.cache_creation_tokens||0)}</strong></span>
    <span>${t('stat.reasoning')} <strong>${fmtTokens(totals.reasoning_tokens||0)}</strong></span>
    <span>${t('tokenUsage.col.total')}: <strong>${fmtTokens(totals.total_tokens||0)}</strong></span>
  </div>`;

  tbody.innerHTML = days.map(d => `<tr>
    <td>${esc(d.day)}</td>
    <td>${d.request_count.toLocaleString()}</td>
    <td>${fmtTokens(d.input_tokens)}</td>
    <td>${fmtTokens(d.output_tokens)}</td>
    <td>${fmtTokens(d.cache_read_tokens)}</td>
    <td>${fmtTokens(d.cache_creation_tokens)}</td>
    <td>${fmtTokens(d.reasoning_tokens)}</td>
    <td><strong>${fmtTokens(d.total_tokens)}</strong></td>
  </tr>`).join('');
}

// ===================== Cross-linking =====================

async function backfillDumpLogIds() {
  try {
    const res = await api.post('/admin/api/error-dumps/backfill-log-ids');
    if (res && res.error) { showToast(res.error, 'error'); return; }
    showToast(t('toast.dumpsMatched', {n: res.updated || 0}));
    if (res.updated > 0) { invalidateDumpCache(); renderDumps(); }
  } catch(e) { showToast(t('toast.backfillFailed'), 'error'); }
}



function jumpToRequestLog(requestLogId) {
  api.get('/admin/api/requests/' + requestLogId).then(function(entry) {
    if (!entry || entry.error) { showToast(t('toast.logEntryNotFound'), 'error'); return; }
    S._highlightLogId = requestLogId;
    if (S._highlightLogIdTimer) clearTimeout(S._highlightLogIdTimer);
    S._highlightLogIdTimer = setTimeout(function() {
      S._highlightLogId = null;
      S._highlightLogIdTimer = null;
    }, 10000);
    const offset = entry._offset || 0;
    const pageSize = LOG_LIMIT;
    S.logOffset = Math.floor(offset / pageSize) * pageSize;
    S._keepLogOffset = true;
    // The log panel may be showing dumps or ops; the jump is to the request
    // list, so put that view up first.
    if (S._logView !== 'requests') window.switchLogView?.('requests');
    goToTab('logs', function() {
      ['filterModel','filterProvider','filterStatus','filterApiKey'].forEach(function(id) {
        const el = document.getElementById(id);
        if (el) el.value = '';
      });
      setTimeout(function() { loadLogs(); }, 300);
    });
  }, function() { showToast(t('toast.logEntryLoadFailed'), 'error'); });
}

// ===================== Bulk Select =====================

if (!S._selectedDumpIds) S._selectedDumpIds = new Set();

function _updateBulk(tbodyId, barId, countId) {
  const checked = document.querySelectorAll('#' + tbodyId + ' .row-check:checked');
  document.getElementById(countId).textContent = checked.length;
  document.getElementById(barId).style.display = checked.length > 0 ? 'flex' : 'none';
  if (checked.length === 0) {
    const headerCb = document.getElementById(tbodyId)?.closest('table')?.querySelector('thead .row-check');
    if (headerCb) headerCb.checked = false;
  }
}
function _selectAll(headerCb, tbodyId, barId, countId) {
  document.querySelectorAll('#' + tbodyId + ' .row-check').forEach(cb => cb.checked = headerCb.checked);
  _updateBulk(tbodyId, barId, countId);
}

function selectAllProfiling(cb) { _selectAll(cb, 'profilingResults', 'profilingBulkBar', 'profilingBulkCount'); }
function updateProfilingBulk() { _updateBulk('profilingResults', 'profilingBulkBar', 'profilingBulkCount'); }
async function bulkDownloadProfiling() {
  const indices = [...document.querySelectorAll('#profilingResults .row-check:checked')].map(cb => parseInt(cb.dataset.index));
  if (!indices.length) return;
  for (const i of indices) { await downloadFlamegraph(i, 'profile'); }
  showToast(t('toast.downloaded', {count: String(indices.length)}));
}

function selectAllCapture(cb) { _selectAll(cb, 'captureResults', 'captureBulkBar', 'captureBulkCount'); }
function updateCaptureBulk() { _updateBulk('captureResults', 'captureBulkBar', 'captureBulkCount'); }
async function bulkDownloadCapture() {
  const indices = [...document.querySelectorAll('#captureResults .row-check:checked')].map(cb => parseInt(cb.dataset.index));
  if (!indices.length) return;
  const results = [];
  for (const i of indices) {
    try { results.push(await api.get('/admin/api/capture/results/' + i)); } catch {}
  }
  if (!results.length) return;
  const blob = new Blob([JSON.stringify(results, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `captures-selected-${new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')}.json`;
  a.click(); URL.revokeObjectURL(a.href);
  showToast(t('toast.downloaded', {count: String(results.length)}));
}

function selectAllDumps(cb) {
  document.querySelectorAll('#dumpTable .row-check').forEach(c => {
    c.checked = cb.checked;
    if (cb.checked) S._selectedDumpIds.add(c.dataset.id);
    else S._selectedDumpIds.delete(c.dataset.id);
  });
  _updateDumpBulkBar();
}
function updateDumpBulk() {
  document.querySelectorAll('#dumpTable .row-check').forEach(c => {
    if (c.checked) S._selectedDumpIds.add(c.dataset.id);
    else S._selectedDumpIds.delete(c.dataset.id);
  });
  _updateDumpBulkBar();
}
/** Label the whole-set buttons with what they would act on. */
function _syncDumpScopeButtons(total) {
  const btn = document.getElementById('dumpSelectFilteredBtn');
  if (btn) {
    btn.textContent = t('btn.selectFiltered', {count: String(total)});
    btn.disabled = total === 0;
  }
  const inv = document.getElementById('dumpInvertBtn');
  if (inv) inv.disabled = total === 0;
}

/** Add every row the filters keep, on every page, to the selection. */
function selectAllFilteredDumps() {
  (S._dumpFilteredIds || []).forEach(id => S._selectedDumpIds.add(id));
  renderDumps();
  _updateDumpBulkBar();
}

/** Flip every filtered row: selected becomes not, and the other way round. */
function invertDumpSelection() {
  (S._dumpFilteredIds || []).forEach(id => {
    if (S._selectedDumpIds.has(id)) S._selectedDumpIds.delete(id);
    else S._selectedDumpIds.add(id);
  });
  renderDumps();
  _updateDumpBulkBar();
}

/** Make the header box describe the page that is on screen.
 *
 *  The box sits in static markup while the rows are re-rendered on every load
 *  and page change, so it used to keep the state of the page that just went
 *  away: still checked while the new rows were not.  The next click then
 *  "cleared" rows that were already clear and did nothing visible, and only
 *  the click after that selected anything.  Deriving it from the rows fixes
 *  the two-click report and adds the partial state for free.
 */
function _syncDumpHeader() {
  const head = document.querySelector('#dumpTable')?.closest('table')?.querySelector('thead .row-check');
  if (!head) return;
  const rows = [...document.querySelectorAll('#dumpTable .row-check')];
  const checked = rows.filter(c => c.checked).length;
  head.checked = rows.length > 0 && checked === rows.length;
  head.indeterminate = checked > 0 && checked < rows.length;
}

function _updateDumpBulkBar() {
  document.getElementById('dumpBulkCount').textContent = S._selectedDumpIds.size;
  document.getElementById('dumpBulkBar').style.display = S._selectedDumpIds.size > 0 ? 'flex' : 'none';
  _syncDumpHeader();
}
/** Drop every selected dump, on this page or another. */
function clearDumpSelection() {
  S._selectedDumpIds.clear();
  document.querySelectorAll('#dumpTable .row-check').forEach(c => { c.checked = false; });
  _updateDumpBulkBar();
}

function bulkDownloadDumps() {
  if (!S._selectedDumpIds.size) return;
  const selected = S._dumpAllEntries.filter(e => S._selectedDumpIds.has(e.dump_id || e.id));
  const blob = new Blob([JSON.stringify(selected, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `error-dumps-selected-${new Date().toISOString().slice(0,19).replace(/[:.]/g,'-')}.json`;
  a.click(); URL.revokeObjectURL(a.href);
  showToast(t('toast.downloaded', {count: String(selected.length)}));
}
async function bulkDeleteDumps() {
  if (!S._selectedDumpIds.size) return;
  if (!confirm(t('confirm.deleteDumps', {count: String(S._selectedDumpIds.size)}))) return;
  let ok = 0;
  for (const id of S._selectedDumpIds) {
    try { await api.del('/admin/api/error-dumps/' + encodeURIComponent(id)); ok++; } catch {}
  }
  S._selectedDumpIds.clear();
  _updateDumpBulkBar();
  showToast(t('toast.deleted', {count: String(ok)}));
  invalidateDumpCache();
  renderDumps();
  loadMetrics();
}

// ===================== Window globals =====================

// Profiling and Content Capture are opt-in debugging tools, so their panels
// stay closed until asked for; the choice survives a reload.  The badge in the
// summary still reports OFF / remaining while collapsed.
(function initDashboardSections() {
  for (const id of ['profilingSection', 'captureSection']) {
    const el = document.getElementById(id);
    if (!el) continue;
    const key = 'llm-rosetta-' + id + '-open';
    el.open = localStorage.getItem(key) === '1';
    el.addEventListener('toggle', () => localStorage.setItem(key, el.open ? '1' : '0'));
  }
})();

Object.assign(window, {
  loadMetrics, loadProfilingStatus, loadProfilingResults,
  toggleProfiling, clearProfilingResults, viewFlamegraph,
  downloadFlamegraph, downloadAllFlamegraphs,
  loadCaptureStatus, loadCaptureResults, toggleCapture,
  clearCaptureResults, viewCapture, downloadCapture, downloadAllCaptures,
  loadDumps, renderDumps, invalidateDumpCache, viewDump, downloadDump, downloadAllDumps,
  selectAllFilteredDumps, invertDumpSelection,
  openDumpPage, closeDumpPage, exitDumpPage, copyDumpPageJson, downloadDumpPage, openDumpPageLog, onDumpIdSearchInput,
  changeDumpPage, jumpToDumpPage, toggleDumpMoreMenu, openClearDumpsConfirm,
  onClearDumpsInput, confirmClearDumps,
  onDumpModelFilterChange, closeDumpModelSearch,
  onDumpStatusFilterChange, onDumpStatusSearchInput, closeDumpStatusSearch,
  onDumpTimeRangeChange, closeDumpTimeCustom, resetDumpFilters,
  renderPersistence, renderStats, renderProviderBreakdown,
  jumpToRequestLog, backfillDumpLogIds,
  selectAllProfiling, updateProfilingBulk, bulkDownloadProfiling,
  selectAllCapture, updateCaptureBulk, bulkDownloadCapture,
  selectAllDumps, updateDumpBulk, clearDumpSelection, bulkDownloadDumps, bulkDeleteDumps,
  rebuildMetrics, drawThroughputChart, drawLatencyChart,
  loadRequestHistogram, drawRequestHistogram, setHistogramRange,
  openTokenUsageModal, setTokenUsageRange, loadTokenUsage,
});

export { loadMetrics, loadDumps, invalidateDumpCache, renderPersistence, renderStats, renderProviderBreakdown,
  jumpToRequestLog, backfillDumpLogIds,
  selectAllProfiling, updateProfilingBulk, bulkDownloadProfiling,
  selectAllCapture, updateCaptureBulk, bulkDownloadCapture,
  selectAllDumps, updateDumpBulk, clearDumpSelection, bulkDownloadDumps, bulkDeleteDumps, rebuildMetrics };

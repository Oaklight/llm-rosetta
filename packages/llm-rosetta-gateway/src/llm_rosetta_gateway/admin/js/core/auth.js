// auth.js — Authentication, login, settings, token management
import { S } from './state.js';
import { t, applyI18n } from './i18n.js';
import { api, _adminHeaders, showToast, closeModal, fmtBytesLong, _startInactivityTracking, _stopInactivityTracking, paintThemeButton } from './utils.js';
import { renderShimSources } from '../tabs/shims.js';

// --- Admin Token ---
let _tokenRotateTimer = null;
let _tokenRotateEnd = 0;
let _tokenCountdownTimer = null;

function openSettings() {
  // Sync theme & language
  const ss = document.getElementById('settingsSchemeSelect');
  if (ss) ss.value = S.currentScheme;
  const ms = document.getElementById('settingsModeSelect');
  if (ms) ms.value = S.currentMode;
  const ls = document.getElementById('settingsLangSelect');
  if (ls) ls.value = S.currentLang;
  // Version — the gateway's own, plus the core when they differ (they are
  // released together but may drift by a patch).
  const ver = document.getElementById('settingsVersion');
  if (ver) {
    const gw = S.configData?.version;
    const core = S.configData?.core_version;
    ver.textContent = gw
      ? ('v' + gw + (core && core !== gw ? t('settings.coreVersion', {v: core}) : ''))
      : '';
  }
  // Sync log level
  const ll = document.getElementById('settingsLogLevel');
  if (ll && S.configData?.server) {
    const dbg = S.configData.debug || {};
    if (dbg.log_bodies) ll.value = 'debug';
    else if (dbg.verbose) ll.value = 'verbose';
    else ll.value = 'normal';
  }
  // Sync auto-refresh
  const ar = document.getElementById('settingsAutoRefresh');
  if (ar) ar.value = localStorage.getItem('dashboardRefreshMs') || '3000';
  const ed = document.getElementById('settingsErrorDumps');
  if (ed) { ed.checked = S.configData?.debug?.error_dumps !== false; ed.setAttribute('aria-checked', ed.checked); }
  // Sync log retention
  if (S.configData?.server?.request_log) {
    const sm = document.getElementById('settingsSuccessMax');
    const em = document.getElementById('settingsErrorDumpMax');
    const mad = document.getElementById('settingsMaxAgeDays');
    if (sm) sm.value = S.configData.server.request_log.success_max || 50000;
    if (em) em.value = S.configData.server.request_log.error_dump_max || 10000;
    if (mad) mad.value = S.configData.server.request_log.max_age_days || 90;
  }
  const olCfg = S.configData?.server?.ops_log || {};
  const oim = document.getElementById('settingsOpsInfoMax');
  const owm = document.getElementById('settingsOpsWarnMax');
  const omad = document.getElementById('settingsOpsMaxAgeDays');
  if (oim) oim.value = olCfg.info_max || 10000;
  if (owm) owm.value = olCfg.warn_max || 5000;
  if (omad) omad.value = olCfg.max_age_days || 90;
  // Sync rate limiting
  const rl = S.configData?.server?.rate_limit || {};
  const rlEn = document.getElementById('rlEnabled');
  if (rlEn) { rlEn.checked = !!rl.enabled; rlEn.setAttribute('aria-checked', rlEn.checked); onRlToggle(); }
  const rlAlgo = document.getElementById('rlAlgorithm');
  if (rlAlgo && rl.algorithm) rlAlgo.value = rl.algorithm;
  const rlG = document.getElementById('rlGlobal');
  if (rlG) rlG.value = _quotaToDisplay(rl.global);
  const rlI = document.getElementById('rlPerIp');
  if (rlI) rlI.value = _quotaToDisplay(rl.per_ip);
  const rlK = document.getElementById('rlPerKey');
  if (rlK) rlK.value = _quotaToDisplay(rl.per_key);
  const rlM = document.getElementById('rlPerModel');
  if (rlM) rlM.value = _quotaToDisplay(rl.per_model);
  _validateRlQuotas();
  // Sync credential visibility
  const cv = document.getElementById('settingsCredentialVisible');
  if (cv) {
    const noAuth = !S.configData?.has_admin_password;
    cv.checked = S.configData?.credential_visible === true;
    cv.setAttribute('aria-checked', cv.checked);
    cv.disabled = noAuth;
    if (noAuth) cv.closest('.settings-popup-item')?.classList.add('settings-item-disabled');
    else cv.closest('.settings-popup-item')?.classList.remove('settings-item-disabled');
    updateCvLabel(cv.checked);
  }
  // Sync token
  _refreshTokenDisplay();
  // Sync rotate interval
  const ri = document.getElementById('settingsRotateInterval');
  _localizeDurationOptions(ri, 'minutes');
  if (ri) ri.value = localStorage.getItem('tokenRotateMinutes') || '15';
  // Clear password fields
  ['settingsCurrentPw','settingsNewPw','settingsConfirmPw'].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = '';
  });
  document.getElementById('settingsPwError').textContent = '';
  // Sync the global proxy
  const gp = document.getElementById('globalProxy');
  if (gp) gp.value = (S.configData?.server && S.configData.server.proxy) || '';
  // Sync the server theme override
  renderAdminTheme();
  // Sync session lifetimes
  renderSessionSettings();
  // Sync external shim sources
  renderShimSources();
}

/** Rewrite a duration <select>'s option labels in the current language. */
function _localizeDurationOptions(sel, kind) {
  if (!sel) return;
  [...sel.options].forEach((opt) => {
    const v = Number(opt.value);
    if (!v) return;
    if (kind === 'days') opt.textContent = t(v === 1 ? 'duration.day' : 'duration.days', {n: v});
    else if (v < 60) opt.textContent = t('duration.minutes', {n: v});
    else opt.textContent = t('duration.hours', {n: v / 60});
  });
}

/** Paint the session group from the effective (defaults-included) values. */
function renderSessionSettings() {
  const sess = S.configData?.admin_session || {};
  const idle = document.getElementById('settingsSessionIdle');
  _localizeDurationOptions(idle, 'minutes');
  if (idle) idle.value = String(sess.idle_minutes ?? 30);
  const remember = document.getElementById('settingsSessionRemember');
  _localizeDurationOptions(remember, 'days');
  if (remember) remember.value = String(sess.remember_days ?? 30);
  const err = document.getElementById('settingsSessionError');
  if (err) err.textContent = '';
}

/** Persist both session lifetimes; the gateway applies them immediately. */
async function saveSessionSettings() {
  const err = document.getElementById('settingsSessionError');
  if (err) err.textContent = '';
  const body = {
    admin_session: {
      idle_minutes: Number(document.getElementById('settingsSessionIdle').value),
      remember_days: Number(document.getElementById('settingsSessionRemember').value),
    },
  };
  try {
    await api.put('/admin/api/config/server', body);
    await window.loadConfig();
    // The panel's own idle timer follows the value it just saved.
    S.idleMinutes = body.admin_session.idle_minutes;
    _startInactivityTracking();
    renderSessionSettings();
    showToast(t('toast.saved'));
  } catch (e) {
    const msg = String(e.message || e);
    if (err) err.textContent = msg; else showToast(msg, 'error');
  }
}

async function saveSettingsField(field, value) {
  const body = {};
  if (field === 'logLevel') {
    body.verbose = (value === 'verbose' || value === 'debug');
    body.log_bodies = (value === 'debug');
  } else if (field === 'errorDumps') {
    body.error_dumps = value;
  } else if (field === 'credentialVisible') {
    body.credential_visible = value;
  }
  try {
    await api.put('/admin/api/config/server', body);
    await window.loadConfig(); showToast(t('toast.saved'));
  } catch { showToast(t('toast.error'), 'error'); }
}

function updateCvLabel(on) {
  const lbl = document.getElementById('cvLabel');
  if (lbl) lbl.textContent = on ? t('label.enabled') : t('label.disabled');
}

function saveAutoRefresh(val) {
  S._dashboardRefreshMs = parseInt(val, 10);
  localStorage.setItem('dashboardRefreshMs', val);
  // Restart dashboard timer if active
  if (S.dashboardTimer) {
    clearInterval(S.dashboardTimer);
    S.dashboardTimer = S._dashboardRefreshMs > 0
      ? setInterval(window.loadMetrics, S._dashboardRefreshMs)
      : null;
  }
}

/** Save one group of the logging pane.
 *
 *  The endpoint merges the fields it is given, so each group saves what it
 *  owns: editing the ops cap no longer rewrites the request-log caps with
 *  whatever happened to be in those boxes.
 *
 *  @param {string} part ``request``, ``errors`` or ``ops``.
 */
async function saveRetentionPart(part) {
  const num = (id) => parseInt(document.getElementById(id)?.value || '0', 10);
  const body = {};
  if (part === 'request') {
    const sm = num('settingsSuccessMax');
    const mad = num('settingsMaxAgeDays');
    if (!(sm >= 50000) || !(mad >= 1)) { showToast(t('toast.retentionMin'), 'error'); return; }
    body.request_log = { success_max: sm, max_age_days: mad };
  } else if (part === 'errors') {
    const em = num('settingsErrorDumpMax');
    if (!(em >= 1000)) { showToast(t('toast.retentionMin'), 'error'); return; }
    body.request_log = { error_dump_max: em };
  } else if (part === 'ops') {
    const oim = num('settingsOpsInfoMax');
    const owm = num('settingsOpsWarnMax');
    const omad = num('settingsOpsMaxAgeDays');
    if (!(oim >= 100) || !(owm >= 100) || !(omad >= 1)) {
      showToast(t('toast.error'), 'error');
      return;
    }
    body.ops_log = { info_max: oim, warn_max: owm, max_age_days: omad };
  } else {
    return;
  }
  try {
    await api.put('/admin/api/config/server', body);
    await window.loadConfig();
    showToast(t('toast.saved'));
  } catch {
    showToast(t('toast.error'), 'error');
  }
}

// --- Rate Limiting ---
const _RL_UNIT_MAP = {s:1,sec:1,second:1,m:60,min:60,minute:60,h:3600,hr:3600,hour:3600,d:86400,day:86400};
function _parseQuota(q) {
  if (!q || !q.trim()) return null;
  const m = q.trim().match(/^(\d+)\s*(?:\/|per)\s*([a-z]+)(?:\s+burst\s+(\d+))?$/i);
  if (!m) return {error: 'Invalid format. Use: N/unit (e.g. 100/m, 60/s, 10000/d)'};
  let unit = m[2].toLowerCase();
  if (unit.length > 1 && unit.endsWith('s')) unit = unit.slice(0, -1);
  if (!(unit in _RL_UNIT_MAP)) return {error: 'Unknown unit: ' + m[2]};
  return {limit: parseInt(m[1]), period: _RL_UNIT_MAP[unit]};
}
function _quotaToDisplay(v) {
  if (!v) return '';
  return Array.isArray(v) ? v.join(', ') : String(v);
}
function _splitQuotas(raw) {
  if (!raw || !raw.trim()) return [];
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}
function _quotaToPayload(raw) {
  const parts = _splitQuotas(raw);
  if (parts.length === 0) return null;
  return parts.length === 1 ? parts[0] : parts;
}
function _toRpm(p) { return p ? (p.limit / p.period) * 60 : null; }
function _fmtRpm(v) { return v >= 1 ? Math.round(v) + ' ' + t('unit.reqPerMin')
                 : (v * 60).toFixed(1) + ' ' + t('unit.reqPerHour'); }

function onRlToggle() {
  const on = document.getElementById('rlEnabled').checked;
  const lbl = document.getElementById('rlEnabledLabel');
  lbl.textContent = on ? t('label.enabled') : t('label.disabled');
  lbl.setAttribute('data-i18n', on ? 'label.enabled' : 'label.disabled');
  document.getElementById('rlFields').classList.toggle('disabled', !on);
  document.getElementById('rlAlgorithm').disabled = !on;
}

function _validateRlQuotas() {
  const fields = [{id:'rlGlobal',name:'Global'},{id:'rlPerIp',name:'Per IP'},{id:'rlPerKey',name:'Per API Key'},{id:'rlPerModel',name:'Per Model'}];
  const warnings = [], errors = [], parsed = {};
  for (const f of fields) {
    const raw = document.getElementById(f.id)?.value?.trim();
    if (!raw) { parsed[f.id] = null; continue; }
    const parts = _splitQuotas(raw);
    let hasError = false;
    const allParsed = [];
    for (const part of parts) {
      const p = _parseQuota(part);
      if (p && p.error) { errors.push(f.name + ' ("' + part + '"): ' + p.error); hasError = true; }
      else if (p) { allParsed.push(p); }
    }
    if (hasError) { document.getElementById(f.id).style.borderColor = 'var(--red)'; }
    else { document.getElementById(f.id).style.borderColor = ''; parsed[f.id] = allParsed.length === 1 ? allParsed[0] : allParsed[0] || null; }
  }
  const rpm = {};
  for (const f of fields) if (parsed[f.id] && !Array.isArray(parsed[f.id])) rpm[f.id] = _toRpm(parsed[f.id]);
  if (rpm.rlGlobal && rpm.rlPerIp && rpm.rlPerIp > rpm.rlGlobal) warnings.push(t('rl.warnOrder', {a: t('rl.perIp'), av: _fmtRpm(rpm.rlPerIp), b: t('rl.perGlobal'), bv: _fmtRpm(rpm.rlGlobal)}));
  if (rpm.rlPerIp && rpm.rlPerKey && rpm.rlPerKey > rpm.rlPerIp) warnings.push('Per API Key (' + _fmtRpm(rpm.rlPerKey) + ') > Per IP (' + _fmtRpm(rpm.rlPerIp) + ')');
  if (rpm.rlGlobal && rpm.rlPerKey && rpm.rlPerKey > rpm.rlGlobal) warnings.push(t('rl.warnOrder', {a: t('rl.perKey'), av: _fmtRpm(rpm.rlPerKey), b: t('rl.perGlobal'), bv: _fmtRpm(rpm.rlGlobal)}));
  if (rpm.rlPerModel && rpm.rlPerKey && rpm.rlPerModel > rpm.rlPerKey) warnings.push('Per Model (' + _fmtRpm(rpm.rlPerModel) + ') > Per API Key (' + _fmtRpm(rpm.rlPerKey) + ')');
  const el = document.getElementById('rlWarning');
  if (errors.length) { el.innerHTML = '⛔ ' + errors.join('<br>⛔ '); el.classList.add('visible'); el.style.color = 'var(--red)'; el.style.background = 'rgba(207,34,46,0.06)'; el.style.borderColor = 'rgba(207,34,46,0.2)'; return false; }
  else if (warnings.length) { el.innerHTML = '⚠ ' + warnings.join('<br>⚠ '); el.classList.add('visible'); el.style.color = ''; el.style.background = ''; el.style.borderColor = ''; return true; }
  else { el.classList.remove('visible'); return true; }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('#rlGlobal,#rlPerIp,#rlPerKey,#rlPerModel').forEach(el => el.addEventListener('input', _validateRlQuotas));
});

async function saveRateLimitSettings() {
  if (!_validateRlQuotas()) { showToast(t('toast.error'), 'error'); return; }
  const payload = { rate_limit: { enabled: document.getElementById('rlEnabled').checked, algorithm: document.getElementById('rlAlgorithm').value } };
  for (const [id, key] of [['rlGlobal','global'],['rlPerIp','per_ip'],['rlPerKey','per_key'],['rlPerModel','per_model']]) {
    const v = document.getElementById(id)?.value?.trim();
    payload.rate_limit[key] = _quotaToPayload(v);
  }
  try { await api.put('/admin/api/config/server', payload); await window.loadConfig(); showToast(t('toast.saved')); } catch { showToast(t('toast.error'), 'error'); }
}

// --- Database Cleanup ---
let _cleanupTarget = 'all';
let _cleanupMode = 'olderThan';

function _showWorkingToast(key) {
  const el = document.getElementById('toast');
  el.textContent = t(key);
  el.className = 'toast show info';
}

function setCleanupMode(mode) {
  _cleanupMode = mode;
  document.querySelectorAll('.cleanupModeBtn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  document.querySelectorAll('.cleanupModePanel').forEach(p => { p.style.display = 'none'; });
  const panelId = mode === 'olderThan' ? 'cleanupModeOlderThan'
    : mode === 'beforeDate' ? 'cleanupModeBeforeDate'
    : mode === 'dateRange' ? 'cleanupModeDateRange'
    : 'cleanupModeTrimToCap';
  document.getElementById(panelId).style.display = '';
  if (mode === 'trimToCap') {
    const sm = S.configData?.server?.request_log?.success_max || 50000;
    const em = S.configData?.server?.request_log?.error_dump_max || 10000;
    document.getElementById('cleanupTrimInfo').textContent =
      t('cleanup.trimInfo', {success: sm.toLocaleString(), error: em.toLocaleString()});
  }
  document.getElementById('cleanupConfirmInput').value = '';
  onCleanupConfirmInput();
}

function openCleanupConfirm(target) {
  _cleanupTarget = target || 'all';
  _cleanupMode = 'olderThan';
  const daysEl = target === 'ops' ? 'settingsOpsMaxAgeDays' : 'settingsMaxAgeDays';
  const days = parseInt(document.getElementById(daysEl)?.value || '90', 10);
  if (days < 1) { showToast(t('toast.error'), 'error'); return; }
  const msgKey = target === 'logs' ? 'confirm.cleanupLogsMsg'
    : target === 'errors' ? 'confirm.cleanupErrorsMsg'
    : target === 'ops' ? 'confirm.cleanupOpsMsg'
    : 'confirm.cleanupMsg';
  document.getElementById('cleanupConfirmMsg').innerHTML = t(msgKey, {days});
  document.getElementById('cleanupConfirmInput').value = '';
  document.getElementById('cleanupBeforeDate').value = '';
  document.getElementById('cleanupRangeStart').value = '';
  document.getElementById('cleanupRangeEnd').value = '';
  setCleanupMode('olderThan');
  const btn = document.getElementById('cleanupConfirmBtn');
  btn.disabled = true;
  btn.style.background = '#ccc'; btn.style.color = '#888'; btn.style.cursor = 'not-allowed';
  openModal('cleanupConfirmModal');
  document.getElementById('cleanupConfirmInput').focus();
}

function onCleanupConfirmInput() {
  const matched = document.getElementById('cleanupConfirmInput').value === 'CLEANUP';
  const btn = document.getElementById('cleanupConfirmBtn');
  btn.disabled = !matched;
  btn.style.background = matched ? 'var(--red)' : '#ccc';
  btn.style.color = matched ? '#fff' : '#888';
  btn.style.cursor = matched ? 'pointer' : 'not-allowed';
}

const _TARGET_LABELS = {logs: 'log', errors: 'error dump', ops: 'ops log', all: ''};

async function onCleanupConfirmClick() {
  const target = _cleanupTarget;
  const mode = _cleanupMode;
  closeModal('cleanupConfirmModal');
  _showWorkingToast('toast.cleanupWorking');

  try {
    let d;
    if (mode === 'olderThan') {
      const daysEl = target === 'ops' ? 'settingsOpsMaxAgeDays' : 'settingsMaxAgeDays';
      const days = parseInt(document.getElementById(daysEl)?.value || '90', 10);
      const url = target === 'logs' ? '/admin/api/requests/cleanup'
        : target === 'errors' ? '/admin/api/error-dumps/cleanup'
        : target === 'ops' ? '/admin/api/ops-log/cleanup'
        : '/admin/api/db/cleanup';
      d = await api.post(url, { max_age_days: days });
      _showOlderThanResult(target, d, days);
    } else if (mode === 'beforeDate') {
      const before = document.getElementById('cleanupBeforeDate').value;
      if (!before) { showToast(t('toast.error'), 'error'); return; }
      d = await api.post('/admin/api/db/cleanup-before', { target, before: before + 'T00:00:00Z' });
      const total = Object.keys(d).filter(k => k.endsWith('_deleted')).reduce((s, k) => s + (d[k] || 0), 0);
      if (total === 0) showToast(t('toast.cleanupNone', {days: before}));
      else showToast(t('toast.cleanupBeforeDone', {total, date: before, freed: fmtBytesLong(d.freed_bytes)}));
    } else if (mode === 'dateRange') {
      const start = document.getElementById('cleanupRangeStart').value;
      const end = document.getElementById('cleanupRangeEnd').value;
      if (!start || !end) { showToast(t('toast.error'), 'error'); return; }
      d = await api.post('/admin/api/db/cleanup-range', { target, start: start + 'T00:00:00Z', end: end + 'T23:59:59Z' });
      const total = Object.keys(d).filter(k => k.endsWith('_deleted')).reduce((s, k) => s + (d[k] || 0), 0);
      if (total === 0) showToast(t('toast.cleanupNone', {days: `${start} ~ ${end}`}));
      else showToast(t('toast.cleanupRangeDone', {total, freed: fmtBytesLong(d.freed_bytes)}));
    } else if (mode === 'trimToCap') {
      d = await api.post('/admin/api/db/trim');
      const total = (d.request_log_trimmed || 0) + (d.error_dumps_trimmed || 0) + (d.ops_log_trimmed || 0);
      if (total === 0) showToast(t('toast.trimNone'));
      else showToast(t('toast.trimDone', {rl: d.request_log_trimmed, ed: d.error_dumps_trimmed, ops: d.ops_log_trimmed, freed: fmtBytesLong(d.freed_bytes)}));
    }
  } catch { showToast(t('toast.error'), 'error'); }
  window.loadMetrics?.();
  if (target === 'errors' || target === 'all') window.invalidateDumpCache?.();
}

function _showOlderThanResult(target, d, days) {
  if (target === 'logs') {
    if (d.deleted === 0) showToast(t('toast.cleanupNone', {days}));
    else showToast(t('toast.cleanupLogsDone', {count: d.deleted, freed: fmtBytesLong(d.freed_bytes)}));
  } else if (target === 'errors') {
    const total = d.error_dumps_deleted + d.dump_bodies_deleted;
    if (total === 0) showToast(t('toast.cleanupNone', {days}));
    else showToast(t('toast.cleanupErrorsDone', {ed: d.error_dumps_deleted, db: d.dump_bodies_deleted, freed: fmtBytesLong(d.freed_bytes)}));
  } else if (target === 'ops') {
    if (d.deleted === 0) showToast(t('toast.cleanupNone', {days}));
    else showToast(t('toast.cleanupOpsDone', {count: d.deleted, freed: fmtBytesLong(d.freed_bytes)}));
  } else {
    const total = d.request_log_deleted + d.error_dumps_deleted + d.dump_bodies_deleted;
    if (total === 0) showToast(t('toast.cleanupNone', {days}));
    else showToast(t('toast.cleanupDone', {rl: d.request_log_deleted, ed: d.error_dumps_deleted, db: d.dump_bodies_deleted, freed: fmtBytesLong(d.freed_bytes)}));
  }
}

// --- Trim to Cap (quick action) ---
async function doTrimToCap() {
  _showWorkingToast('toast.trimWorking');
  try {
    const d = await api.post('/admin/api/db/trim');
    const total = (d.request_log_trimmed || 0) + (d.error_dumps_trimmed || 0) + (d.ops_log_trimmed || 0);
    if (total === 0) showToast(t('toast.trimNone'));
    else showToast(t('toast.trimDone', {rl: d.request_log_trimmed, ed: d.error_dumps_trimmed, ops: d.ops_log_trimmed, freed: fmtBytesLong(d.freed_bytes)}));
  } catch { showToast(t('toast.error'), 'error'); }
  window.loadMetrics?.();
}

// --- Vacuum ---
async function doVacuum() {
  _showWorkingToast('toast.vacuumWorking');
  try {
    const d = await api.post('/admin/api/db/vacuum');
    if (d.freed_bytes > 0) showToast(t('toast.vacuumDone', {freed: fmtBytesLong(d.freed_bytes)}));
    else showToast(t('toast.vacuumNone'));
  } catch { showToast(t('toast.error'), 'error'); }
  window.loadMetrics?.();
}

// --- Error Dump Export ---
function openExportDumpsModal() {
  document.getElementById('exportStartDate').value = '';
  document.getElementById('exportEndDate').value = '';
  openModal('exportDumpsModal');
}

async function doExportDumps() {
  closeModal('exportDumpsModal');
  const start = document.getElementById('exportStartDate').value;
  const end = document.getElementById('exportEndDate').value;
  let url = '/admin/api/error-dumps/export';
  const params = [];
  if (start) params.push('start=' + encodeURIComponent(start + 'T00:00:00Z'));
  if (end) params.push('end=' + encodeURIComponent(end + 'T23:59:59Z'));
  if (params.length) url += '?' + params.join('&');
  try {
    const r = await fetch(url, {headers: _adminHeaders(), credentials: 'include'});
    if (!r.ok) { showToast(t('toast.error'), 'error'); return; }
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const parts = ['error-dumps'];
    if (start) parts.push(start);
    if (end) parts.push('to', end);
    a.download = parts.join('-') + '.tar.gz';
    a.click();
    URL.revokeObjectURL(a.href);
    showToast(t('toast.exported'));
  } catch { showToast(t('toast.error'), 'error'); }
}

function _refreshTokenDisplay() {
  const el = document.getElementById('settingsTokenDisplay');
  if (!el || !S.internalToken) { if (el) el.textContent = '—'; return; }
  // Mask: show first 14 chars + ... + last 4
  const masked = S.internalToken.length > 20
    ? S.internalToken.slice(0, 14) + '...' + S.internalToken.slice(-4)
    : S.internalToken;
  el.textContent = masked;
}

async function copyAdminToken() {
  if (!S.internalToken) return;
  try {
    await navigator.clipboard.writeText(S.internalToken);
    showToast(t('toast.copied'));
  } catch { showToast(t('toast.copyFailed'), 'error'); return; }
  // Start auto-rotate countdown
  const minutes = parseInt(localStorage.getItem('tokenRotateMinutes') || '15', 10);
  _tokenRotateEnd = Date.now() + minutes * 60 * 1000;
  if (_tokenRotateTimer) clearTimeout(_tokenRotateTimer);
  if (_tokenCountdownTimer) clearInterval(_tokenCountdownTimer);
  _tokenRotateTimer = setTimeout(_doTokenRotate, minutes * 60 * 1000);
  _tokenCountdownTimer = setInterval(_updateTokenCountdown, 1000);
  _updateTokenCountdown();
}

function _updateTokenCountdown() {
  const el = document.getElementById('tokenCountdown');
  if (!el) return;
  const remaining = Math.max(0, _tokenRotateEnd - Date.now());
  if (remaining <= 0) {
    el.textContent = '';
    if (_tokenCountdownTimer) { clearInterval(_tokenCountdownTimer); _tokenCountdownTimer = null; }
    return;
  }
  const mins = Math.floor(remaining / 60000);
  const secs = Math.floor((remaining % 60000) / 1000);
  el.innerHTML = t('label.rotatingIn', { time: `${mins}:${String(secs).padStart(2,'0')}` });
}

async function _doTokenRotate() {
  _tokenRotateTimer = null;
  if (_tokenCountdownTimer) { clearInterval(_tokenCountdownTimer); _tokenCountdownTimer = null; }
  document.getElementById('tokenCountdown').textContent = '';
  try {
    await api.post('/admin/api/token/rotate');
    const td = await api.get('/admin/api/internal-token');
    S.internalToken = td.token;
    _refreshTokenDisplay();
    showToast(t('toast.tokenRotated'));
  } catch (e) {
    showToast(t('toast.error'), 'error');
  }
}

async function logoutAllSessions() {
  try {
    const data = await api.post('/admin/api/sessions/logout-all');
    showToast(t('toast.logoutAll', { count: data.sessions_cleared || 0 }));
    showLoginOverlay();
  } catch (e) {
    showToast(t('toast.error'), 'error');
  }
}

// --- Theme override ---

/** Show the server's CSS override on this page without a reload.
 *
 *  The server also injects the same text into `<head>` when it serves the
 *  page, so a fresh load already has it; this keeps Save instant and drops
 *  the block the moment the override is cleared.
 */
function applyAdminCustomCss(css) {
  const text = (css || '').trim();
  let el = document.getElementById('admin-custom-css');
  if (!text) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('style');
    el.id = 'admin-custom-css';
    document.head.appendChild(el);
  }
  el.textContent = text;
}

/** Copy the stored override into the textarea and apply it. */
function renderAdminTheme() {
  const ta = document.getElementById('settingsCustomCss');
  const css = S.configData?.server?.admin_custom_css || '';
  if (ta) ta.value = css;
  applyAdminCustomCss(css);
}

/** Save the textarea as `server.admin_custom_css`. */
async function saveAdminTheme() {
  const css = document.getElementById('settingsCustomCss')?.value || '';
  try {
    await api.put('/admin/api/config/server', { admin_custom_css: css });
    await window.loadConfig();
    renderAdminTheme();
    showToast(t('toast.saved'));
  } catch {
    showToast(t('toast.error'), 'error');
  }
}

/** Clear the override (an empty value removes the key from the config). */
async function resetAdminTheme() {
  const ta = document.getElementById('settingsCustomCss');
  if (ta) ta.value = '';
  await saveAdminTheme();
}

// --- Change Password ---
async function changeAdminPassword() {
  const current = document.getElementById('settingsCurrentPw')?.value || '';
  const newPw = document.getElementById('settingsNewPw')?.value || '';
  const confirm = document.getElementById('settingsConfirmPw')?.value || '';
  const errEl = document.getElementById('settingsPwError');
  errEl.textContent = '';
  if (!current || !newPw) { errEl.textContent = t('pw.required'); return; }
  if (newPw !== confirm) { errEl.textContent = t('pw.mismatch'); return; }
  if (newPw.length < 4) { errEl.textContent = t('pw.tooShort'); return; }
  try {
    const data = await api.put('/admin/api/config/password', { current_password: current, new_password: newPw });
    // Session cookie updated automatically by server response
    ['settingsCurrentPw','settingsNewPw','settingsConfirmPw'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = '';
    });
    showToast(t('pw.changed'));
  } catch (e) {
    errEl.textContent = e.message || t('toast.error');
  }
}

function showLoginOverlay() {
  document.body.classList.add('auth-pending');
  // Stop background polling timers to prevent repeated 401 → showLoginOverlay loops
  if (S.dashboardTimer) { clearInterval(S.dashboardTimer); S.dashboardTimer = null; }
  if (S.logTimer) { clearInterval(S.logTimer); S.logTimer = null; }
  const overlay = document.getElementById('loginOverlay');
  // Repaint the settings-dependent parts before the early return: an API call
  // can hit a 401 before /auth-check has answered, and that first paint would
  // otherwise keep the fallback lifetimes until the page is reloaded.
  _renderLoginForm();
  // Don't re-clear fields if already showing (would dismiss password manager popup)
  if (overlay.style.display === 'flex') return;
  overlay.style.display = 'flex';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginError').textContent = '';
  document.getElementById('loginPassword').focus();
}

// ── Login page ──────────────────────────────────────────────────────
// Three things the page has to answer before a password is typed: whether
// "stay signed in" exists at all, how long the cookie lasts, and how long an
// idle session survives.  All three come from /auth-check.

const _EYE_OPEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const _EYE_OFF = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.9 17.9A10.9 10.9 0 0 1 12 19c-7 0-11-7-11-7a20 20 0 0 1 5.1-5.9"/><path d="M9.9 4.2A10.9 10.9 0 0 1 12 4c7 0 11 7 11 7a20 20 0 0 1-2.2 3.2"/><path d="M1 1l22 22"/></svg>';

/** Repaint the parts of the login page that depend on server settings. */
function _renderLoginForm() {
  const days = Number(S.rememberDays);
  const row = document.getElementById('loginRememberRow');
  if (row) {
    row.hidden = !(days > 0);
    const box = document.getElementById('loginRemember');
    // Default on: a private admin panel is not a kiosk.  The last choice wins
    // so an operator who unchecked it once is not asked again every day.
    if (box) box.checked = localStorage.getItem('login-remember') !== '0';
  }
  const box = document.getElementById('loginRemember');
  if (box && !box.dataset.wired) {
    box.dataset.wired = '1';
    box.addEventListener('change', _updateLoginHint);
  }
  _updateLoginHint();
  const reveal = document.getElementById('loginReveal');
  if (reveal) {
    reveal.innerHTML = _EYE_OPEN;
    reveal.setAttribute('aria-label', t('login.showPassword'));
    reveal.setAttribute('title', t('login.showPassword'));
    reveal.setAttribute('aria-pressed', 'false');
  }
  paintThemeButton(document.getElementById('loginThemeBtn'));
  // The language button shows what you are in, not what you would switch to.
  const langBtn = document.getElementById('loginLangBtn');
  if (langBtn) langBtn.textContent = S.currentLang === 'zh' ? '\u4e2d' : 'EN';
  _watchCapsLock();
}

/**
 * One line, because the two answers exclude each other: a remembered session
 * ends when you sign out or its own lifetime ends, everything else ends after
 * a period of not being used.
 */
function _updateLoginHint() {
  const foot = document.getElementById('loginFoot');
  if (!foot) return;
  const days = Number(S.rememberDays);
  const idle = Number(S.idleMinutes);
  const box = document.getElementById('loginRemember');
  const remembered = days > 0 && !!box && box.checked;
  foot.textContent = remembered
    ? t('login.rememberHint', { days })
    : idle > 0
      ? t('login.idleHint', { minutes: _formatIdle(idle) })
      : t('login.idleNever');
}

/** "30" → "30 min", "90" → "1.5 h", with the unit translated. */
function _formatIdle(minutes) {
  if (minutes < 60) return t('session.durationMin', { n: minutes });
  const hours = minutes / 60;
  return t('session.durationHour', {
    n: Number.isInteger(hours) ? hours : hours.toFixed(1),
  });
}

function _watchCapsLock() {
  const pw = document.getElementById('loginPassword');
  const warn = document.getElementById('loginCapsLock');
  if (!pw || !warn || pw.dataset.capsWired === '1') return;
  pw.dataset.capsWired = '1';
  const sync = (e) => {
    const on = typeof e.getModifierState === 'function' && e.getModifierState('CapsLock');
    warn.hidden = !on;
  };
  pw.addEventListener('keydown', sync);
  pw.addEventListener('keyup', sync);
  pw.addEventListener('blur', () => { warn.hidden = true; });
}

/** Show or hide the typed password. */
function toggleLoginReveal() {
  const input = document.getElementById('loginPassword');
  const btn = document.getElementById('loginReveal');
  const show = input.type === 'password';
  input.type = show ? 'text' : 'password';
  if (btn) {
    btn.innerHTML = show ? _EYE_OFF : _EYE_OPEN;
    btn.setAttribute('aria-pressed', String(show));
    const label = t(show ? 'login.hidePassword' : 'login.showPassword');
    btn.setAttribute('aria-label', label);
    btn.setAttribute('title', label);
  }
  input.focus();
}

async function doLogin() {
  const pwEl = document.getElementById('loginPassword');
  const pw = pwEl.value;
  if (!pw) return;
  const errEl = document.getElementById('loginError');
  const btn = document.getElementById('loginSubmit');
  const rememberBox = document.getElementById('loginRemember');
  const remember = !!(rememberBox && !rememberBox.hidden && rememberBox.checked);
  localStorage.setItem('login-remember', remember ? '1' : '0');
  errEl.textContent = '';
  if (btn) btn.disabled = true;
  try {
    const r = await fetch('/admin/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pw, remember}),
      credentials: 'include',
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok) {
      document.getElementById('loginOverlay').style.display = 'none';
      document.body.classList.remove('auth-pending');
      const out = document.getElementById('logoutBtn');
      if (out) out.style.display = '';
      _startInactivityTracking();
      window.initApp();
    } else {
      // A lockout carries the remaining seconds, which the generic line cannot
      // say; anything else is indistinguishable from a wrong password.
      errEl.textContent = (data && data.error && data.error !== 'Invalid password')
        ? data.error
        : t('login.error');
      pwEl.select();
    }
  } catch(e) {
    errEl.textContent = String(e);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function checkAuthAndInit() {
  try {
    const [authRes, configRes] = await Promise.all([
      fetch('/admin/api/auth-check'),
      fetch('/admin/api/config', {credentials: 'include'}),
    ]);
    const authData = await authRes.json();
    if (typeof authData.idle_minutes === 'number') S.idleMinutes = authData.idle_minutes;
    if (typeof authData.remember_days === 'number') S.rememberDays = authData.remember_days;
    if (authData.requires_auth && configRes.status === 401) {
      showLoginOverlay();
      return;
    }
    if (authData.requires_auth) {
      const btn = document.getElementById('logoutBtn');
      if (btn) btn.style.display = '';
      _startInactivityTracking();
    }
    document.body.classList.remove('auth-pending');
    const configData = await configRes.json();
    window.initApp(configData);
  } catch(e) {
    console.error('Auth check failed:', e);
    document.body.classList.remove('auth-pending');
    window.initApp();
  }
}

// --- Named exports for cross-module imports ---
export {
  openSettings,
  checkAuthAndInit,
  showLoginOverlay,
  _refreshTokenDisplay,
  _doTokenRotate,
};

// --- Expose to global scope for inline handlers ---
Object.assign(window, {
  openSettings, saveSettingsField, saveAutoRefresh, saveRetentionPart, updateCvLabel,
  applyAdminCustomCss, saveAdminTheme, resetAdminTheme, renderAdminTheme,
  copyAdminToken, changeAdminPassword, showLoginOverlay, doLogin, toggleLoginReveal,
  renderSessionSettings, saveSessionSettings, renderLoginForm: _renderLoginForm,
  checkAuthAndInit, _doTokenRotate, logoutAllSessions,
  onRlToggle, saveRateLimitSettings,
  openCleanupConfirm, onCleanupConfirmInput, onCleanupConfirmClick, setCleanupMode, doTrimToCap, doVacuum,
  openExportDumpsModal, doExportDumps,
});

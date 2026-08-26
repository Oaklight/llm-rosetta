// auth.js — Authentication, login, settings, token management
import { S } from './state.js';
import { t, applyI18n } from './i18n.js';
import { api, _adminHeaders, showToast, closeModal, fmtBytesLong, _startInactivityTracking, _stopInactivityTracking } from './core.js';

// --- Admin Token ---
let _tokenRotateTimer = null;
let _tokenRotateEnd = 0;
let _tokenCountdownTimer = null;

function openSettings() {
  const popup = document.getElementById('settingsPopup');
  // Sync theme & language
  const ts = document.getElementById('settingsThemeSelect');
  if (ts) ts.value = S.currentTheme;
  const ls = document.getElementById('settingsLangSelect');
  if (ls) ls.value = S.currentLang;
  // Version
  const ver = document.getElementById('settingsVersion');
  if (ver) ver.textContent = S.configData?.version ? 'v' + S.configData.version : '';
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
  if (ed) ed.checked = S.configData?.debug?.error_dumps !== false;
  // Sync log retention
  if (S.configData?.server?.request_log) {
    const sm = document.getElementById('settingsSuccessMax');
    const em = document.getElementById('settingsErrorMax');
    const mad = document.getElementById('settingsMaxAgeDays');
    if (sm) sm.value = S.configData.server.request_log.success_max || 50000;
    if (em) em.value = S.configData.server.request_log.error_max || 10000;
    if (mad) mad.value = S.configData.server.request_log.max_age_days || 90;
  }
  // Sync rate limiting
  const rl = S.configData?.server?.rate_limit || {};
  const rlEn = document.getElementById('rlEnabled');
  if (rlEn) { rlEn.checked = !!rl.enabled; onRlToggle(); }
  const rlAlgo = document.getElementById('rlAlgorithm');
  if (rlAlgo && rl.algorithm) rlAlgo.value = rl.algorithm;
  const rlG = document.getElementById('rlGlobal');
  if (rlG) rlG.value = rl.global || '';
  const rlI = document.getElementById('rlPerIp');
  if (rlI) rlI.value = rl.per_ip || '';
  const rlK = document.getElementById('rlPerKey');
  if (rlK) rlK.value = rl.per_key || '';
  const rlM = document.getElementById('rlPerModel');
  if (rlM) rlM.value = rl.per_model || '';
  _validateRlQuotas();
  // Sync credential visibility
  const cv = document.getElementById('settingsCredentialVisible');
  if (cv) cv.checked = S.configData?.server?.credential_visible !== false;
  // Sync token
  _refreshTokenDisplay();
  // Sync rotate interval
  const ri = document.getElementById('settingsRotateInterval');
  if (ri) ri.value = localStorage.getItem('tokenRotateMinutes') || '15';
  // Clear password fields
  ['settingsCurrentPw','settingsNewPw','settingsConfirmPw'].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = '';
  });
  document.getElementById('settingsPwError').textContent = '';
  popup.classList.toggle('open');
}

async function saveSettingsField(field, value) {
  const body = {};
  if (field === 'logLevel') {
    body.verbose = (value === 'verbose' || value === 'debug');
    body.log_bodies = (value === 'debug');
  } else if (field === 'errorDumps') {
    body.error_dumps = value;
  }
  try {
    await api.put('/admin/api/config/server', body);
    await window.loadConfig(); showToast(t('toast.saved'));
  } catch { showToast(t('toast.error'), 'error'); }
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

async function saveLogRetention() {
  const sm = parseInt(document.getElementById('settingsSuccessMax')?.value || '50000', 10);
  const em = parseInt(document.getElementById('settingsErrorMax')?.value || '10000', 10);
  const mad = parseInt(document.getElementById('settingsMaxAgeDays')?.value || '90', 10);
  if (sm < 50000 || em < 5000) { showToast(t('toast.retentionMin'), 'error'); return; }
  if (mad < 1) { showToast(t('toast.error'), 'error'); return; }
  try {
    await api.put('/admin/api/config/server', { request_log: { success_max: sm, error_max: em, max_age_days: mad } });
    await window.loadConfig(); showToast(t('toast.saved'));
  } catch { showToast(t('toast.error'), 'error'); }
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
function _toRpm(p) { return p ? (p.limit / p.period) * 60 : null; }
function _fmtRpm(v) { return v >= 1 ? Math.round(v) + ' req/min' : (v * 60).toFixed(1) + ' req/hr'; }

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
    const val = document.getElementById(f.id)?.value?.trim();
    if (!val) { parsed[f.id] = null; continue; }
    const p = _parseQuota(val);
    if (p && p.error) { errors.push(f.name + ': ' + p.error); document.getElementById(f.id).style.borderColor = 'var(--red)'; }
    else { document.getElementById(f.id).style.borderColor = ''; parsed[f.id] = p; }
  }
  const rpm = {};
  for (const f of fields) if (parsed[f.id]) rpm[f.id] = _toRpm(parsed[f.id]);
  if (rpm.rlGlobal && rpm.rlPerIp && rpm.rlPerIp > rpm.rlGlobal) warnings.push('Per IP (' + _fmtRpm(rpm.rlPerIp) + ') > Global (' + _fmtRpm(rpm.rlGlobal) + ')');
  if (rpm.rlPerIp && rpm.rlPerKey && rpm.rlPerKey > rpm.rlPerIp) warnings.push('Per API Key (' + _fmtRpm(rpm.rlPerKey) + ') > Per IP (' + _fmtRpm(rpm.rlPerIp) + ')');
  if (rpm.rlGlobal && rpm.rlPerKey && rpm.rlPerKey > rpm.rlGlobal) warnings.push('Per API Key (' + _fmtRpm(rpm.rlPerKey) + ') > Global (' + _fmtRpm(rpm.rlGlobal) + ')');
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
    payload.rate_limit[key] = v || null;
  }
  try { await api.put('/admin/api/config/server', payload); await window.loadConfig(); showToast(t('toast.saved')); } catch { showToast(t('toast.error'), 'error'); }
}

// --- Database Cleanup ---
let _cleanupTarget = 'all';

function openCleanupConfirm(target) {
  _cleanupTarget = target || 'all';
  const days = parseInt(document.getElementById('settingsMaxAgeDays')?.value || '90', 10);
  if (days < 1) { showToast(t('toast.error'), 'error'); return; }
  const msgKey = target === 'logs' ? 'confirm.cleanupLogsMsg'
    : target === 'errors' ? 'confirm.cleanupErrorsMsg'
    : 'confirm.cleanupMsg';
  document.getElementById('cleanupConfirmMsg').innerHTML = t(msgKey, {days});
  document.getElementById('cleanupConfirmInput').value = '';
  document.getElementById('cleanupConfirmBtn').disabled = true;
  const btn = document.getElementById('cleanupConfirmBtn');
  btn.style.background = '#ccc'; btn.style.color = '#888'; btn.style.cursor = 'not-allowed';
  document.getElementById('cleanupConfirmModal').classList.add('open');
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

async function onCleanupConfirmClick() {
  const days = parseInt(document.getElementById('settingsMaxAgeDays')?.value || '90', 10);
  closeModal('cleanupConfirmModal');
  const target = _cleanupTarget;
  try {
    const url = target === 'logs' ? '/admin/api/requests/cleanup'
      : target === 'errors' ? '/admin/api/error-dumps/cleanup'
      : '/admin/api/db/cleanup';
    const d = await api.post(url, { max_age_days: days });
    if (target === 'logs') {
      if (d.deleted === 0) showToast(t('toast.cleanupNone', {days}));
      else showToast(t('toast.cleanupLogsDone', {count: d.deleted, freed: fmtBytesLong(d.freed_bytes)}));
    } else if (target === 'errors') {
      const total = d.error_dumps_deleted + d.dump_bodies_deleted;
      if (total === 0) showToast(t('toast.cleanupNone', {days}));
      else showToast(t('toast.cleanupErrorsDone', {ed: d.error_dumps_deleted, db: d.dump_bodies_deleted, freed: fmtBytesLong(d.freed_bytes)}));
    } else {
      const total = d.request_log_deleted + d.error_dumps_deleted + d.dump_bodies_deleted;
      if (total === 0) showToast(t('toast.cleanupNone', {days}));
      else showToast(t('toast.cleanupDone', {rl: d.request_log_deleted, ed: d.error_dumps_deleted, db: d.dump_bodies_deleted, freed: fmtBytesLong(d.freed_bytes)}));
    }
  } catch { showToast(t('toast.error'), 'error'); }
}

// --- Error Dump Export ---
function openExportDumpsModal() {
  document.getElementById('exportStartDate').value = '';
  document.getElementById('exportEndDate').value = '';
  document.getElementById('exportDumpsModal').classList.add('open');
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
    const r = await fetch(url, {headers: _adminHeaders()});
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
    const data = await api.post('/admin/api/token/rotate');
    // api.post returns parsed JSON directly, not a Response
    if (data.token) {
      localStorage.setItem('admin_token', data.token);
    }
    // Refresh internal token display
    const td = await api.get('/admin/api/internal-token');
    S.internalToken = td.token;
    _refreshTokenDisplay();
    showToast(t('toast.tokenRotated'));
  } catch (e) {
    showToast(t('toast.error'), 'error');
  }
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
    if (data.token) localStorage.setItem('admin_token', data.token);
    ['settingsCurrentPw','settingsNewPw','settingsConfirmPw'].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = '';
    });
    showToast(t('pw.changed'));
  } catch (e) {
    errEl.textContent = e.message || t('toast.error');
  }
}

function showLoginOverlay() {
  localStorage.removeItem('admin_token');
  document.body.classList.add('auth-pending');
  // Stop background polling timers to prevent repeated 401 → showLoginOverlay loops
  if (S.dashboardTimer) { clearInterval(S.dashboardTimer); S.dashboardTimer = null; }
  if (S.logTimer) { clearInterval(S.logTimer); S.logTimer = null; }
  const overlay = document.getElementById('loginOverlay');
  // Don't re-clear fields if already showing (would dismiss password manager popup)
  if (overlay.style.display === 'flex') return;
  overlay.style.display = 'flex';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginError').textContent = '';
  document.getElementById('loginPassword').focus();
}

async function doLogin() {
  const pw = document.getElementById('loginPassword').value;
  if (!pw) return;
  try {
    const r = await fetch('/admin/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pw}),
    });
    const data = await r.json();
    if (r.ok && data.token) {
      localStorage.setItem('admin_token', data.token);
      document.getElementById('loginOverlay').style.display = 'none';
      document.body.classList.remove('auth-pending');
      const btn = document.getElementById('logoutBtn');
      if (btn) btn.style.display = '';
      _startInactivityTracking();
      window.initApp();
    } else {
      document.getElementById('loginError').textContent = t('login.error');
    }
  } catch(e) {
    document.getElementById('loginError').textContent = String(e);
  }
}

async function checkAuthAndInit() {
  try {
    const r = await fetch('/admin/api/auth-check');
    const data = await r.json();
    if (data.requires_auth) {
      // Check if we have a valid stored token
      const token = localStorage.getItem('admin_token');
      if (token) {
        // Verify token by trying to load config
        const test = await fetch('/admin/api/config', {headers: {'X-Admin-Token': token}});
        if (test.status === 401) {
          showLoginOverlay();
          return;
        }
        const btn = document.getElementById('logoutBtn');
        if (btn) btn.style.display = '';
        _startInactivityTracking();
      } else {
        showLoginOverlay();
        return;
      }
    }
    document.body.classList.remove('auth-pending');
    window.initApp();
  } catch(e) {
    console.error('Auth check failed:', e);
    document.body.classList.remove('auth-pending');
    window.initApp(); // fallback: try to load anyway
  }
}

// --- Named exports for cross-module imports ---
export {
  checkAuthAndInit,
  showLoginOverlay,
  _refreshTokenDisplay,
  _doTokenRotate,
};

// --- Expose to global scope for inline handlers ---
Object.assign(window, {
  openSettings, saveSettingsField, saveAutoRefresh, saveLogRetention,
  copyAdminToken, changeAdminPassword, showLoginOverlay, doLogin,
  checkAuthAndInit, _doTokenRotate,
  onRlToggle, saveRateLimitSettings,
  openCleanupConfirm, onCleanupConfirmInput, onCleanupConfirmClick,
  openExportDumpsModal, doExportDumps,
});

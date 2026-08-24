// auth.js — Authentication, login, settings, token management
import { S } from './state.js';
import { t, applyI18n } from './i18n.js';
import { api, showToast, _startInactivityTracking, _stopInactivityTracking } from './core.js';

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
    if (sm) sm.value = S.configData.server.request_log.success_max || 50000;
    if (em) em.value = S.configData.server.request_log.error_max || 10000;
  }
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
  if (sm < 50000 || em < 5000) { showToast(t('toast.retentionMin'), 'error'); return; }
  try {
    await api.put('/admin/api/config/server', { request_log: { success_max: sm, error_max: em } });
    await window.loadConfig(); showToast(t('toast.saved'));
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
});

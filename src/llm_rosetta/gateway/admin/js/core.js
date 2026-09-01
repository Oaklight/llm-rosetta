/**
 * core.js — shared utilities for the admin panel.
 *
 * Every tab-specific module depends on this module.  It provides theming,
 * the authenticated API wrapper, inactivity auto-logout, toast / modal /
 * clipboard helpers, inline confirmation, and small formatting utilities.
 */

import { S, INACTIVITY_TIMEOUT_MS } from './state.js';
import { t } from './i18n.js';

// ===================== Scheme + Mode =====================
const VALID_SCHEMES = ['minimal', 'emerald'];
const VALID_MODES = ['light', 'dark'];

function setScheme(scheme) {
  if (!VALID_SCHEMES.includes(scheme)) scheme = 'minimal';
  document.documentElement.setAttribute('data-scheme', scheme);
  S.currentScheme = scheme;
  localStorage.setItem('llm-rosetta-scheme', scheme);
  const el = document.getElementById('settingsSchemeSelect');
  if (el) el.value = scheme;
  if (S.currentTab === 'dashboard') window.loadMetrics?.();
}

function setMode(mode) {
  if (!VALID_MODES.includes(mode)) mode = 'light';
  document.documentElement.setAttribute('data-mode', mode);
  S.currentMode = mode;
  localStorage.setItem('llm-rosetta-mode', mode);
  const el = document.getElementById('settingsModeSelect');
  if (el) el.value = mode;
  if (S.currentTab === 'dashboard') window.loadMetrics?.();
}

function setTheme(name) {
  if (name === 'light') { setScheme('minimal'); setMode('light'); }
  else if (name === 'dark') { setScheme('minimal'); setMode('dark'); }
  else { setScheme('minimal'); setMode('light'); }
}

// ===================== API =====================
function _adminHeaders(extra) {
  const h = extra ? {...extra} : {};
  const token = localStorage.getItem('admin_token');
  if (token) h['X-Admin-Token'] = token;
  return h;
}

const api = {
  async get(url) {
    const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store'});
    if (r.status === 401) { window.showLoginOverlay?.(); throw new Error('Unauthorized'); }
    const data = await r.json().catch(() => null);
    if (!r.ok) throw new Error(data?.error || (r.status === 504 ? 'Request timed out — the upstream service may be unreachable' : `HTTP ${r.status}`));
    if (data === null) throw new Error('Invalid JSON response');
    return data;
  },
  async put(url, body) {
    const r = await fetch(url, {method:'PUT', headers:_adminHeaders({'Content-Type':'application/json'}), body:JSON.stringify(body)});
    if (r.status === 401) { window.showLoginOverlay?.(); throw new Error('Unauthorized'); }
    return r.json();
  },
  async del(url) {
    const r = await fetch(url, {method:'DELETE', headers: _adminHeaders()});
    if (r.status === 401) { window.showLoginOverlay?.(); throw new Error('Unauthorized'); }
    return r.json();
  },
  async post(url, body) {
    const h = body !== undefined ? _adminHeaders({'Content-Type':'application/json'}) : _adminHeaders();
    const opts = {method:'POST', headers: h};
    if (body !== undefined) opts.body = JSON.stringify(body);
    const r = await fetch(url, opts);
    if (r.status === 401) { window.showLoginOverlay?.(); throw new Error('Unauthorized'); }
    return r.json();
  }
};

// ===================== Inactivity auto-logout =====================
let _inactivityTimer = null;

function _resetInactivityTimer() {
  if (_inactivityTimer) clearTimeout(_inactivityTimer);
  _inactivityTimer = setTimeout(() => doLogout(), INACTIVITY_TIMEOUT_MS);
}

function _startInactivityTracking() {
  ['mousemove', 'keydown', 'mousedown', 'touchstart', 'scroll', 'click'].forEach(evt => {
    document.addEventListener(evt, _resetInactivityTimer, { passive: true });
  });
  _resetInactivityTimer();
}

function _stopInactivityTracking() {
  if (_inactivityTimer) { clearTimeout(_inactivityTimer); _inactivityTimer = null; }
}

function doLogout() {
  localStorage.removeItem('admin_token');
  _stopInactivityTracking();
  const btn = document.getElementById('logoutBtn');
  if (btn) btn.style.display = 'none';
  window.showLoginOverlay?.();
}

// ===================== Copy Helper =====================
async function copyText(text, toastMsg) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(toastMsg || t('toast.copied'));
  } catch(e) {
    showToast('Copy failed', 'error');
  }
}

function copyProviderEntry(name) {
  const cfg = S.configData.providers[name] || {};
  // Open Add Provider modal pre-filled with non-sensitive fields; API key is intentionally omitted
  window.openProviderModal?.('', cfg.base_url, '', cfg.proxy, cfg.type || name);
}

// ===================== Toast =====================
function showToast(msg, type='success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + type;
  setTimeout(() => el.className = 'toast', 3000);
}

// ===================== Modal =====================
const _modalTriggers = new Map();

function openModal(id) {
  _modalTriggers.set(id, document.activeElement);
  document.getElementById(id).classList.add('open');
}

function closeModal(id) {
  document.getElementById(id).classList.remove('open');
  if (id === 'testModal') window._abortPendingTest?.();
  const trigger = _modalTriggers.get(id);
  _modalTriggers.delete(id);
  if (trigger && typeof trigger.focus === 'function') trigger.focus();
}

// ===================== Inline Confirm =====================
// Inline two-step delete confirmation — replaces native confirm()
function inlineConfirm(btn, action) {
  if (btn.dataset.confirming) return;
  btn.dataset.confirming = '1';
  const orig = btn.innerHTML;
  const origClass = btn.className;
  const origOnclick = btn.onclick;
  btn.innerHTML = `${t('confirm.sure')} <span style="text-decoration:underline;cursor:pointer">${t('confirm.yes')}</span>`;
  btn.className = btn.className.replace('btn-danger', 'btn-warning');
  btn.style.minWidth = btn.offsetWidth + 'px';
  const revert = () => { btn.innerHTML = orig; btn.className = origClass; btn.style.minWidth = ''; btn.onclick = origOnclick; delete btn.dataset.confirming; };
  const timer = setTimeout(revert, 3000);
  btn.querySelector('span').onclick = (e) => { e.stopPropagation(); clearTimeout(timer); revert(); action(); };
  btn.onclick = (e) => { e.stopPropagation(); clearTimeout(timer); revert(); };
}

// ===================== Helpers =====================
// esc() escapes HTML special chars AND single quotes so the result is
// safe in both HTML text nodes and single-quoted JS-string attributes
// such as onclick="fn('${esc(name)}')".
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  // innerHTML escapes &, <, > and " but NOT '.  Replace ' last to avoid
  // double-encoding the & in the &#39; entity.
  return d.innerHTML.replace(/'/g, '&#39;');
}

function formatDuration(seconds) {
  if (seconds < 60) return Math.floor(seconds) + 's';
  if (seconds < 3600) return Math.floor(seconds/60) + 'm ' + Math.floor(seconds%60) + 's';
  const h = Math.floor(seconds/3600);
  const m = Math.floor((seconds%3600)/60);
  return h + 'h ' + m + 'm';
}

// Verbose byte formatter for request/response body sizes (e.g. "1.5 KB")
function fmtBytesLong(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

// Compact byte formatter for dashboard footer (e.g. "832 K", "1.4 M")
function fmtBytesShort(n) {
  if (!Number.isFinite(n) || n < 0) return '–';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' K';
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' M';
  return (n / (1024 * 1024 * 1024)).toFixed(2) + ' G';
}

// ===================== Window globals =====================
Object.assign(window, {
  setScheme, setMode, setTheme, api, doLogout, copyText, copyProviderEntry,
  showToast, openModal, closeModal, inlineConfirm, esc, formatDuration,
  fmtBytesShort, fmtBytesLong,
  _startInactivityTracking, _stopInactivityTracking,
});

export {
  setScheme, setMode, setTheme, _adminHeaders, api, showToast, openModal, closeModal,
  copyText, copyProviderEntry, esc, formatDuration,
  fmtBytesShort, fmtBytesLong,
  inlineConfirm, doLogout,
  _startInactivityTracking, _stopInactivityTracking,
};

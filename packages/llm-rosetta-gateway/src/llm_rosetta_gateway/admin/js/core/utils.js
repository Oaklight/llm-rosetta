/**
 * core.js — shared utilities for the admin panel.
 *
 * Every tab-specific module depends on this module.  It provides theming,
 * the authenticated API wrapper, inactivity auto-logout, toast / modal /
 * clipboard helpers, inline confirmation, and small formatting utilities.
 */

import { S, INACTIVITY_TIMEOUT_MS } from './state.js';
import { t } from './i18n.js';
import { shimBrand } from './shim-tree.js';
import { esc } from './html.js';
import { defaultThemeName, resolveScheme, setThemeSheet } from './themes.js';
import { openDialog, closeDialog } from './dialog.js';
import { writeResult, networkResult } from './write-result.js';

// ===================== Scheme + Mode =====================
// The schemes themselves live in the gateway's theme registry (themes.js);
// this module only applies the choice.
const VALID_MODES = ['light', 'dark', 'system'];
const SYSTEM_DARK = () => window.matchMedia?.('(prefers-color-scheme: dark)').matches;

function setScheme(scheme) {
  // A stored name this build no longer ships — a theme removed in an upgrade,
  // or a hand-edited value — resolves to the registry's default.
  scheme = resolveScheme(scheme);
  document.documentElement.setAttribute('data-scheme', scheme);
  setThemeSheet(scheme);
  S.currentScheme = scheme;
  localStorage.setItem('llm-rosetta-scheme', scheme);
  const el = document.getElementById('settingsSchemeSelect');
  if (el) el.value = scheme;
  window._syncRailPrefs?.();
  if (S.currentTab === 'dashboard') window.loadMetrics?.();
}

function setMode(mode) {
  if (!VALID_MODES.includes(mode)) mode = 'light';
  // "System" stores the choice, not the resolved colour, so the panel follows
  // the OS when it changes later.
  const resolved = mode === 'system' ? (SYSTEM_DARK() ? 'dark' : 'light') : mode;
  document.documentElement.setAttribute('data-mode', resolved);
  S.currentMode = mode;
  localStorage.setItem('llm-rosetta-mode', mode);
  const el = document.getElementById('settingsModeSelect');
  if (el) el.value = mode;
  window._syncRailPrefs?.();
  if (S.currentTab === 'dashboard') window.loadMetrics?.();
}

// Follow the OS while "system" is the stored choice.
window.matchMedia?.('(prefers-color-scheme: dark)')
  .addEventListener?.('change', () => {
    if (S.currentMode === 'system') setMode('system');
  });

function setTheme(name) {
  // The login card and the rail call this with a legacy pseudo-theme name;
  // both land on the registry's default scheme.
  const scheme = defaultThemeName();
  setScheme(scheme);
  setMode(name === 'dark' ? 'dark' : 'light');
}

// ===================== Theme icons =====================
// One set, painted by whoever owns a mode button: the rail's and the login
// card's must be the same three glyphs, or the two controls read as different
// features.
const ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.5 14.6A8.6 8.6 0 0 1 9.4 3.5a8.5 8.5 0 1 0 11.1 11.1Z"/></svg>';
const ICON_SYSTEM = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="4" width="19" height="13" rx="2"/><path d="M8.5 21h7M12 17v4"/></svg>';

/** Paint a mode button: the icon is the state, the title names the next. */
export function paintThemeButton(btn) {
  if (!btn) return;
  const mode = S.currentMode || 'light';
  btn.innerHTML = mode === 'dark' ? ICON_MOON
    : mode === 'system' ? ICON_SYSTEM : ICON_SUN;
  const next = mode === 'light' ? t('mode.dark')
    : mode === 'dark' ? t('mode.system') : t('mode.light');
  btn.title = next;
  btn.setAttribute('aria-label', next);
}

// ===================== API =====================
function _adminHeaders(extra) {
  return extra ? {...extra} : {};
}

/**
 * One mutating request, in the shape every call site already branches on.
 *
 * Never rejects.  A refusal, an answer that is not JSON and a request that
 * never reached the server all resolve to `{ok: false, error}` — see
 * `write-result.js` for why (#1502).  A success hands back the server's body.
 * `api.get` keeps its own contract (it throws): every read path wraps it.
 */
const PASSWORD_CHANGE_URL = /^\/admin\/api\/config\/password(?:[?#]|$)/;

async function _send(method, url, body) {
  const hasBody = body !== undefined;
  const opts = {
    method,
    headers: hasBody ? _adminHeaders({'Content-Type':'application/json'}) : _adminHeaders(),
    credentials: 'include',
  };
  if (hasBody) opts.body = JSON.stringify(body);
  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) {
    // Transport failure: say so in the panel, and keep the cause for the console.
    console.warn('admin request failed:', method, url, e);
    return networkResult(t('toast.error'));
  }
  // A 401 means the session ended — except on the password-change endpoint,
  // where it means the *current password typed into the form* is wrong.
  // Sending the operator to the login card there hides the real reason.
  //
  // The login card is the whole signal: it covers the panel at a far higher
  // stacking order than the toast, so a call site's error toast on the same
  // 401 stays behind it and clears itself.
  if (r.status === 401 && !PASSWORD_CHANGE_URL.test(url)) window.showLoginOverlay?.();
  return writeResult(r);
}

const api = {
  async get(url) {
    const r = await fetch(url, {headers: _adminHeaders(), cache: 'no-store', credentials: 'include'});
    if (r.status === 401) { window.showLoginOverlay?.(); throw new Error('Unauthorized'); }
    const data = await r.json().catch(() => null);
    if (!r.ok) throw new Error(data?.error || (r.status === 504 ? 'Request timed out — the upstream service may be unreachable' : `HTTP ${r.status}`));
    if (data === null) throw new Error('Invalid JSON response');
    return data;
  },
  async put(url, body) { return _send('PUT', url, body); },
  async del(url) { return _send('DELETE', url); },
  async post(url, body) { return _send('POST', url, body); },
};

/**
 * Run one action with its submit button held disabled until it settles.
 *
 * A second click used to send the request twice — two POSTs for one
 * double-clicked *Generate*, measured in #1502 §1.2.  `doLogin` already
 * toggles its own button; this is the same rule for the dialogs, in one place
 * so no handler re-implements a flag.  A click that arrives while the first is
 * still in flight is dropped, not queued.
 *
 * @param {HTMLElement|null} btn
 * @param {() => Promise<unknown>} action
 * @returns {Promise<unknown>}
 */
async function withBusyButton(btn, action) {
  if (!btn) return action();
  if (btn.dataset.busy === '1') return undefined;
  // A caller may hand over a button that was already disabled for its own
  // reason; put it back the way it was found, not unconditionally enabled.
  const wasDisabled = btn.disabled;
  btn.dataset.busy = '1';
  btn.disabled = true;
  try {
    return await action();
  } finally {
    btn.dataset.busy = '';
    btn.disabled = wasDisabled;
  }
}

// ===================== Inactivity auto-logout =====================
// The window comes from the server (server.admin_session.idle_minutes), so the
// panel and the session store agree on when an idle session ends.  Zero means
// the operator turned the idle sign-out off: only an explicit sign-out ends
// the session, and no timer is armed.
let _inactivityTimer = null;
const _ACTIVITY_EVENTS = ['mousemove', 'keydown', 'mousedown', 'touchstart', 'scroll', 'click'];

function _idleTimeoutMs() {
  // A remembered session is exempt from the idle window on the server, and
  // this timer ends the session on the server (`doLogout` posts the logout).
  // Arming it here is what signed "stay signed in" users out after half an
  // hour of quiet (issue #1381): their cookie was still good, the panel had
  // already destroyed the session.
  if (S.rememberedSession) return 0;
  // undefined until /auth-check answers; the constant is the fallback so a
  // page that never reaches it still signs out.
  return typeof S.idleMinutes === 'number' ? S.idleMinutes * 60 * 1000 : INACTIVITY_TIMEOUT_MS;
}

function _resetInactivityTimer() {
  if (_inactivityTimer) clearTimeout(_inactivityTimer);
  const ms = _idleTimeoutMs();
  _inactivityTimer = ms > 0 ? setTimeout(() => doLogout(), ms) : null;
}

function _startInactivityTracking() {
  _stopInactivityTracking();
  _ACTIVITY_EVENTS.forEach(evt => {
    document.addEventListener(evt, _resetInactivityTimer, { passive: true });
  });
  _resetInactivityTimer();
}

function _stopInactivityTracking() {
  if (_inactivityTimer) { clearTimeout(_inactivityTimer); _inactivityTimer = null; }
  _ACTIVITY_EVENTS.forEach(evt => {
    document.removeEventListener(evt, _resetInactivityTimer);
  });
}

async function doLogout() {
  try {
    await fetch('/admin/api/logout', {method: 'POST', credentials: 'include'});
  } catch (_) { /* network error — cookie may persist until next login */ }
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
    showToast(t('toast.copyFailed'), 'error');
  }
}

function copyProviderEntry(name) {
  const cfg = S.configData.providers[name] || {};
  // Open Add Provider modal pre-filled with non-sensitive fields; API key is intentionally omitted
  window.openProviderModal?.('', cfg.base_url, '', cfg.proxy, providerTypeName(cfg, name));
}

// ===================== Toast =====================
let _toastTimer = null;

function _renderToast(type, delay, apply) {
  if (_toastTimer) clearTimeout(_toastTimer);
  const el = document.getElementById('toast');
  apply(el);
  el.className = 'toast show ' + type;
  _toastTimer = setTimeout(() => { el.className = 'toast'; el.textContent = ''; _toastTimer = null; }, delay);
}

/** Show a plain-text toast. Text is never parsed as HTML. */
function showToast(msg, type = 'success') {
  _renderToast(type, 3000, el => { el.textContent = msg; });
}

/**
 * Show a toast built from markup. The caller owns escaping — every
 * interpolated value must already have gone through esc().
 * Prefer showToast() unless you need structured rows.
 */
function showToastHtml(trustedHtml, type = 'success') {
  _renderToast(type, 5000, el => { el.innerHTML = trustedHtml; });
}

// ===================== Modal =====================
// The dialog rules (scroll lock, Tab trap, Escape) live in `dialog.js`,
// shared with the model side panel.  This module keeps only what is specific
// to an overlay-modal: the `open` class, the element lookup and the trigger
// to give focus back to.
const _modalTriggers = new Map();

/** The dialog panel inside an overlay: the thing that carries
 *  `role="dialog"` and `tabindex="-1"`. */
function _modalPanel(id) {
  const overlay = document.getElementById(id);
  return overlay ? overlay.querySelector(':scope > .modal') : null;
}

function openModal(id) {
  const overlay = document.getElementById(id);
  const panel = _modalPanel(id);
  if (!overlay || !panel || overlay.classList.contains('open')) return;
  _modalTriggers.set(id, document.activeElement);
  overlay.classList.add('open');
  openDialog(panel, { onEscape: () => closeModal(id) });
}

function closeModal(id) {
  const overlay = document.getElementById(id);
  if (!overlay || !overlay.classList.contains('open')) return;
  overlay.classList.remove('open');
  closeDialog(_modalPanel(id));
  if (id === 'testModal') window._abortPendingTest?.();
  // The credential drawer's countdown owns an interval; a closed drawer must
  // not keep ticking (X, Cancel and Escape all arrive here).
  if (id === 'providerModal') window.clearProviderExpiryTimer?.();
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

// ===================== Dropdown menus =====================
// The row actions "⋯" menu and the Test "▾" menu.  Both are `position:fixed`
// so they escape scrolling ancestors that would clip them — the models table
// sits in a `.table-scroll` box whose `overflow-x:auto` clips vertically too —
// and therefore need positioning by hand (positionMenu) and closing on
// scroll, not only on an outside click.
const MENU_SELECTOR = '.more-menu.open, .test-menu.open';

// How many menus are open, so the window-level scroll handler can return
// without touching the DOM — the usual answer is none.
let _menusOpen = 0;

/**
 * Close every open dropdown menu, clearing the geometry positionMenu() wrote
 * and the `aria-expanded` openMenu() set.
 *
 * Clearing matters because the menus stay in the DOM: a stale `top`/`right`
 * would place the next open before positionMenu() recomputes it.
 */
function closeMenus() {
  if (!_menusOpen) return;
  _menusOpen = 0;
  document.querySelectorAll(MENU_SELECTOR).forEach(m => {
    m.classList.remove('open');
    m.style.top = '';
    m.style.right = '';
    // Both menus are the trigger's next sibling, so this is that trigger.
    m.previousElementSibling?.setAttribute('aria-expanded', 'false');
  });
}

/** Open *menu* against the trigger that owns it, and flag the trigger as expanded. */
function openMenu(menu, trigger) {
  if (!menu.classList.contains('open')) _menusOpen++;
  menu.classList.add('open');
  trigger.setAttribute('aria-expanded', 'true');
  positionMenu(menu, trigger);
}

/**
 * Place a dropdown menu against the element that opened it: right-aligned with
 * its trigger, flipped above the trigger when the layout viewport has no room
 * left below it, and clamped to an EDGE-px margin on both axes.
 *
 * The fixed containing block is the layout viewport, which excludes the
 * scrollbars — window.innerWidth/Height include them, so this reads
 * documentElement instead.
 */
function positionMenu(menu, trigger) {
  const GAP = 4, EDGE = 8;
  const r = trigger.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const mh = menu.offsetHeight, mw = menu.offsetWidth;
  const below = vh - r.bottom - GAP;
  const above = r.top - GAP;
  const top = (below < mh && above > below) ? r.top - mh - GAP : r.bottom + GAP;
  menu.style.top = Math.max(EDGE, Math.min(top, vh - mh - EDGE)) + 'px';
  menu.style.right = Math.max(EDGE, Math.min(vw - r.right, vw - mw - EDGE)) + 'px';
}

// ===================== Helpers =====================
// esc() lives in core/html.js so markup can be tested without a DOM; it is
// re-exported here for the modules that already import it from utils.

function formatDuration(seconds) {
  const sec = t('unit.secondsShort'), min = t('unit.minutesShort'), hr = t('unit.hoursShort');
  if (seconds < 60) return Math.floor(seconds) + sec;
  if (seconds < 3600) return Math.floor(seconds/60) + min + ' ' + Math.floor(seconds%60) + sec;
  const h = Math.floor(seconds/3600);
  const m = Math.floor((seconds%3600)/60);
  return h + hr + ' ' + m + min;
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

// Compact token formatter (e.g. "1.2K", "3.5M")
function fmtTokens(n) {
  if (n >= 1e9) return (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return String(n);
}

// ===================== Window globals =====================
// ── Free-source entries (shared by the Providers and Models tabs) ──

/** The shim / base-format name a provider entry resolves to: an explicit
 *  `shim` wins, then a real `type`, then the non-LLM endpoint shim, then the
 *  provider name.  A config that declares `shim:` instead of `type:` arrives
 *  with `type` filled in from the provider name (see admin/routes/config.py),
 *  so a `type` equal to the name is synthetic and must not win.  A provider
 *  defined only by `embeddings_shim` / `rerank_shim` / `decision_shim` has no
 *  chat type at all. */
export function providerTypeName(cfg, name) {
  if (!cfg) return name;
  if (cfg.shim) return cfg.shim;
  if (cfg.type && cfg.type !== name) return cfg.type;
  for (const desc of S.configData?.model_types || []) {
    if (!desc.is_llm && desc.config_shim_key && cfg[desc.config_shim_key]) {
      return cfg[desc.config_shim_key];
    }
  }
  return name;
}

/** The registered shim backing provider *name*, or null. */
export function shimOf(providers, shims, name) {
  const cfg = (providers || {})[name];
  const typeName = (cfg && cfg.type) || name;
  return (shims || []).find(s => s.name === typeName) || null;
}

/** True when *name* is a keyless entry of a free-source shim. */
export function isFreeEntry(providers, shims, name) {
  const cfg = (providers || {})[name];
  const shim = shimOf(providers, shims, name);
  return !!(shim && shim.free_source && cfg && cfg.keyless === true);
}

/** Display label for a free entry — "<Brand> <suffix>" (e.g. "Kilo (Free)"). */
export function freeEntryLabel(providers, shims, name, suffix) {
  const shim = shimOf(providers, shims, name);
  const brand = shim ? shimBrand(shim) : name;
  return `${brand} ${suffix}`;
}

/**
 * Keyboard path for an expandable table row: Enter or Space runs the row's
 * own click handler, so the keyboard and pointer paths cannot drift.  Events
 * from controls inside the row are ignored, so activating the error-dump
 * button does not also fold the row.
 *
 * @param {KeyboardEvent} event
 * @param {HTMLElement} row
 */
function rowToggleKey(event, row) {
  if (event.target !== row) return;
  if (event.key !== 'Enter' && event.key !== ' ' && event.key !== 'Spacebar') return;
  event.preventDefault();  // Space would otherwise scroll the page
  row.click();
}

Object.assign(window, {
  setScheme, setMode, setTheme, api, doLogout, copyText, copyProviderEntry,
  showToast, showToastHtml, openModal, closeModal, inlineConfirm, esc, formatDuration, rowToggleKey,
  fmtBytesShort, fmtBytesLong, fmtTokens, closeMenus,
  _startInactivityTracking, _stopInactivityTracking,
});

export {
  setScheme, setMode, setTheme, _adminHeaders, api, showToast, showToastHtml, openModal, closeModal,
  copyText, copyProviderEntry, esc, formatDuration,
  fmtBytesShort, fmtBytesLong, fmtTokens,
  inlineConfirm, closeMenus, openMenu, positionMenu, doLogout, withBusyButton,
  _startInactivityTracking, _stopInactivityTracking,
};

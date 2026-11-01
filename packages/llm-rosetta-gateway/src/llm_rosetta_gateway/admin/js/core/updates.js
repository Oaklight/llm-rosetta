/**
 * updates.js — the "a newer release exists" bubble and the changelog dialog.
 *
 * The server does the asking (PyPI, cached for six hours) and the deciding
 * (may this process replace itself); this module only paints the answer.  Two
 * things appear in the corner: a bubble when there is a release to move to,
 * and — once per version — a dialog with what changed, shown the first time
 * the panel loads after an update.
 */

import { S } from './state.js';
import { t } from './i18n.js';
import { api, showToast, openModal, closeModal, esc } from './utils.js';
import { normaliseVersion, parseChangelog, renderEntryHtml } from './changelog.js';

const SEEN_KEY = 'llm-rosetta-changelog-seen';
const RESTART_POLL_MS = 1500;
const RESTART_TIMEOUT_MS = 120000;
const DOCS_CHANGELOG = 'https://llm-rosetta.readthedocs.io/{lang}/latest/changelog/';

let _payload = null;
let _changelog = {lang: null, markdown: ''};
let _shownVersion = '';

function _seen() {
  try { return localStorage.getItem(SEEN_KEY) || ''; } catch { return ''; }
}

function _markSeen(version) {
  try { localStorage.setItem(SEEN_KEY, version); } catch { /* private window */ }
}

function _target() {
  return (_payload && _payload.update) || null;
}

function _docsLang() {
  return S.currentLang === 'zh' ? 'zh-cn' : 'en';
}

// ===================== The bubble =====================

export async function initUpdates() {
  let payload;
  try { payload = await api.get('/admin/api/updates'); } catch { return; }
  _payload = payload;
  _paintBadge();
  await _offerChangelog();
}

export async function refreshUpdates() {
  try { _payload = await api.get('/admin/api/updates?force=1'); } catch { return; }
  _paintBadge();
  const pop = document.getElementById('updatePopover');
  if (pop && !pop.hidden) _renderPopover();
}

function _paintBadge() {
  const badge = document.getElementById('updateBadge');
  if (!badge) return;
  const target = _target();
  badge.hidden = !target;
  if (!target) return;
  const text = document.getElementById('updateBadgeText');
  if (text) text.textContent = t('update.badge', {version: target.version});
  badge.title = t('update.hint');
  badge.setAttribute('aria-label', badge.title);
}

export function toggleUpdatePopover() {
  const pop = document.getElementById('updatePopover');
  if (!pop) return;
  if (!pop.hidden) { pop.hidden = true; return; }
  _renderPopover();
  pop.hidden = false;
}

export function closeUpdatePopover() {
  const pop = document.getElementById('updatePopover');
  if (pop) pop.hidden = true;
}

/** Repaint the labels after a language change; no request is made. */
export function repaintUpdateLabels() {
  _paintBadge();
  const pop = document.getElementById('updatePopover');
  if (pop && !pop.hidden) _renderPopover();
}

function _renderPopover() {
  const body = document.getElementById('updatePopoverBody');
  if (!body) return;
  const current = (_payload && _payload.current) || {};
  const target = _target();
  if (!target) {
    body.innerHTML = `<p class="update-empty">${esc(t('update.none', {version: current.gateway || ''}))}</p>`;
    return;
  }
  const gate = (_payload && _payload.self_update) || {allowed: false, reason: 'refused'};
  const reason = gate.allowed
    ? ''
    : `<p class="update-note">${esc(t('update.reason.' + (gate.reason || 'refused')))}</p>`;
  const docs = DOCS_CHANGELOG.replace('{lang}', _docsLang());
  body.innerHTML = `
    <div class="update-row"><span>${esc(t('update.current'))}</span><code>${esc(current.gateway || '')}</code></div>
    <div class="update-row"><span>${esc(t('update.latest'))}</span><code>${esc(target.version)}</code></div>
    ${reason}
    <div class="update-actions">
      <a class="btn btn-sm" href="${esc(docs)}" target="_blank" rel="noopener">${esc(t('update.whatsNew'))}</a>
      <button type="button" class="btn btn-sm btn-primary" id="updateApplyBtn" onclick="applyUpdate()" ${gate.allowed ? '' : 'disabled'}>${esc(t('update.apply'))}</button>
    </div>`;
}

// ===================== Installing =====================

export async function applyUpdate() {
  const target = _target();
  if (!target) return;
  if (!window.confirm(t('confirm.selfUpdate', {version: target.version}))) return;
  const button = document.getElementById('updateApplyBtn');
  if (button) { button.disabled = true; button.textContent = t('update.applying'); }
  let result = null;
  try {
    result = await api.post('/admin/api/updates/apply', {version: target.version});
  } catch { result = null; }
  if (!result || result.status !== 'restarting') {
    if (button) { button.disabled = false; button.textContent = t('update.apply'); }
    // pip's own output is the only thing that says which step failed; the
    // toast carries the reason, the console carries the detail.
    if (result && result.output) console.error(result.output);
    const reason = result && result.reason ? t('update.reason.' + result.reason) : '';
    showToast(reason || (result && result.error) || t('update.failed'), 'error');
    return;
  }
  showToast(t('update.restarting'));
  await _waitForRestart(target.version);
}

async function _waitForRestart(version) {
  const started = Date.now();
  let sawDown = false;
  let mismatches = 0;
  while (Date.now() - started < RESTART_TIMEOUT_MS) {
    await new Promise(resolve => setTimeout(resolve, RESTART_POLL_MS));
    let running = null;
    try {
      const response = await fetch('/health', {cache: 'no-store'});
      if (!response.ok) continue;
      const data = await response.json();
      running = (data.versions && data.versions.gateway) || '';
    } catch {
      // The old process is gone and the new one has not bound the port yet.
      sawDown = true;
      continue;
    }
    if (running === version) {
      // The new build answers; a reload picks up its assets and its version.
      location.reload();
      return;
    }
    if (sawDown) {
      // The gateway came back running something else: the restart did not
      // take, and waiting longer will not change the answer.
      mismatches += 1;
      if (mismatches >= 2) {
        showToast(t('update.restartFailed'), 'error');
        return;
      }
    }
  }
  showToast(t('update.restartTimeout'), 'error');
}

// ===================== The changelog dialog =====================

async function _markdown() {
  const lang = S.currentLang === 'zh' ? 'zh' : 'en';
  if (_changelog.lang === lang) return _changelog.markdown;
  let markdown = '';
  try {
    const data = await api.get('/admin/api/changelog?lang=' + lang);
    markdown = (data && data.markdown) || '';
  } catch { markdown = ''; }
  _changelog = {lang, markdown};
  return markdown;
}

async function _offerChangelog() {
  const running = ((_payload || {}).current || {}).gateway || '';
  if (!running || _seen() === running) return;
  await showChangelog(running);
}

/**
 * Open the "what changed" dialog for one version.
 *
 * Returns false when the build carries no entry for it — a development build
 * released from a branch has no changelog section, and an empty dialog would
 * be worse than no dialog.
 */
export async function showChangelog(version) {
  const wanted = normaliseVersion(version);
  const section = parseChangelog(await _markdown(), wanted);
  if (!section) return false;
  _shownVersion = wanted;
  const title = document.getElementById('changelogTitle');
  if (title) title.textContent = t('update.whatsNewVersion', {version: wanted});
  const link = document.getElementById('changelogFullLink');
  if (link) link.href = DOCS_CHANGELOG.replace('{lang}', _docsLang());
  const body = document.getElementById('changelogBody');
  if (body) {
    body.innerHTML = section.groups.map(group => `
      <div class="changelog-group">
        ${group.name ? `<h4>${esc(group.name)}</h4>` : ''}
        <ul>${group.items.map(item => `<li>${renderEntryHtml(item)}</li>`).join('')}</ul>
      </div>`).join('') || `<p class="update-empty">${esc(t('update.noChangelog'))}</p>`;
  }
  openModal('changelogModal');
  return true;
}

/** Close without remembering: the dialog comes back on the next load. */
export function closeChangelog() {
  closeModal('changelogModal');
}

/** Remember this version, so the dialog stops coming back for it. */
export function dismissChangelog() {
  if (_shownVersion) _markSeen(_shownVersion);
  closeModal('changelogModal');
}

// ---- Window bindings for inline handlers ----
Object.assign(window, {
  initUpdates,
  refreshUpdates,
  repaintUpdateLabels,
  toggleUpdatePopover,
  closeUpdatePopover,
  applyUpdate,
  showChangelog,
  closeChangelog,
  dismissChangelog,
});

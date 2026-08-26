import { S } from './state.js';
import { t, setLang, applyI18n } from './i18n.js';
import {
  setTheme, api, showToast, closeModal, esc,
  _startInactivityTracking, _stopInactivityTracking,
} from './core.js';
import { checkAuthAndInit, showLoginOverlay } from './auth.js';
import { loadConfig, renderProviders } from './providers.js';
import { renderModels } from './models.js';
import './fetch-models.js';
import { loadKeys, loadLogKeyLabels, renderKeys } from './keys.js';
import { loadMetrics, loadDumps, renderPersistence } from './dashboard.js';
import { loadLogs, renderLogs, updateFilterOptions, updateKeyFilterOptions } from './logs.js';
import './test.js';

// Disabled tabs from branding — skip data fetching for these
const _dt = (window.__branding && window.__branding.disabled_tabs) || [];
function _tabEnabled(id) { return _dt.indexOf(id) === -1; }

// ===================== Init =====================
function initApp() {
  loadConfig();
  if (_tabEnabled('keys')) { loadKeys(); loadLogKeyLabels(); }
  api.get('/admin/api/internal-token').then(r => { S.internalToken = r.token; }).catch(() => {});
  api.get('/admin/api/metrics?seconds=1').then(data => {
    renderPersistence(data.persistence, data.total_requests);
  }).catch(() => {});
  stopTimers();
  if (S.currentTab === 'dashboard' && _tabEnabled('dashboard')) { loadMetrics(); S.dashboardTimer = (S._dashboardRefreshMs > 0 ? setInterval(loadMetrics, S._dashboardRefreshMs) : null); }
  if (S.currentTab === 'logs' && _tabEnabled('logs')) { S.logOffset = 0; loadLogs(); S.logTimer = setInterval(loadLogs, 5000); }
}

// ===================== Tabs =====================
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const id = tab.dataset.tab;
    document.getElementById('tab-' + id).classList.add('active');
    S.currentTab = id;
    localStorage.setItem('llm-rosetta-tab', id);
    stopTimers();
    if (id === 'dashboard' && _tabEnabled('dashboard')) { loadMetrics(); loadDumps(); S.dashboardTimer = (S._dashboardRefreshMs > 0 ? setInterval(loadMetrics, S._dashboardRefreshMs) : null); }
    if (id === 'logs' && _tabEnabled('logs')) { S.logOffset = 0; loadLogs(); S.logTimer = setInterval(loadLogs, 5000); }
    if (id === 'providers' || id === 'models') { loadConfig(); }
    if (id === 'keys' && _tabEnabled('keys')) { loadKeys(); }
  });
});

if (S.currentTab !== 'providers') {
  const savedTab = document.querySelector(`.tab[data-tab="${S.currentTab}"]`);
  if (savedTab) savedTab.click();
}

function stopTimers() {
  if (S.dashboardTimer) { clearInterval(S.dashboardTimer); S.dashboardTimer = null; }
  if (S.logTimer) { clearInterval(S.logTimer); S.logTimer = null; }
}

// Filter change handlers
document.getElementById('filterModel').addEventListener('change', () => { S.logOffset = 0; S.expandedLogRows.clear(); loadLogs(); });
document.getElementById('filterProvider').addEventListener('change', () => { S.logOffset = 0; S.expandedLogRows.clear(); loadLogs(); });
document.getElementById('filterStatus').addEventListener('change', () => { S.logOffset = 0; S.expandedLogRows.clear(); loadLogs(); });
document.getElementById('filterApiKey').addEventListener('change', () => { S.logOffset = 0; S.expandedLogRows.clear(); loadLogs(); });

// Close modal on overlay click
document.querySelectorAll('.modal-overlay').forEach(m => {
  m.addEventListener('click', e => { if (e.target === m) closeModal(m.id); });
});

// Close popups/modals on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.getElementById('settingsPopup').classList.remove('open');
    const testModal = document.getElementById('testModal');
    if (testModal && testModal.classList.contains('open')) closeModal('testModal');
  }
});

// ===================== System Clock =====================
(function initClock() {
  const el = document.getElementById('systemClock');
  function tick() { el.textContent = new Date().toLocaleTimeString(); }
  tick();
  setInterval(tick, 1000);
})();

// ===================== Branding =====================
(function applyBranding() {
  const b = window.__branding; if (!b) return;
  const _icons = {
    github:'<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M12 .3a12 12 0 0 0-3.8 23.38c.6.12.83-.26.83-.57L9 20.86c-3.37.73-4.08-1.63-4.08-1.63-.55-1.4-1.34-1.77-1.34-1.77-1.1-.75.08-.73.08-.73 1.21.08 1.85 1.24 1.85 1.24 1.08 1.85 2.83 1.32 3.52 1 .1-.78.42-1.32.76-1.62-2.69-.3-5.52-1.34-5.52-5.98 0-1.32.47-2.4 1.24-3.24-.12-.3-.54-1.54.12-3.2 0 0 1.01-.33 3.3 1.24a11.5 11.5 0 0 1 6.02 0c2.3-1.57 3.3-1.24 3.3-1.24.66 1.66.24 2.9.12 3.2a4.65 4.65 0 0 1 1.24 3.24c0 4.65-2.83 5.67-5.53 5.97.43.37.82 1.1.82 2.22l-.01 3.29c0 .31.22.69.83.57A12 12 0 0 0 12 .3"/></svg>',
    pypi:'<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z"/></svg>',
    docker:'<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M13.983 11.078h2.119a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.119a.185.185 0 00-.185.185v1.888c0 .102.083.185.185.185m-2.954-5.43h2.118a.186.186 0 00.186-.186V3.574a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m0 2.716h2.118a.187.187 0 00.186-.186V6.29a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.887c0 .102.082.185.185.186m-2.93 0h2.12a.186.186 0 00.184-.186V6.29a.185.185 0 00-.185-.185H8.1a.185.185 0 00-.185.185v1.887c0 .102.083.185.185.186m-2.964 0h2.119a.186.186 0 00.185-.186V6.29a.185.185 0 00-.185-.185H5.136a.186.186 0 00-.186.185v1.887c0 .102.084.185.186.186m5.893 2.715h2.118a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m-2.93 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.083.185.185.185m-2.964 0h2.119a.185.185 0 00.185-.185V9.006a.185.185 0 00-.184-.186h-2.12a.186.186 0 00-.186.186v1.887c0 .102.084.185.186.185m-2.92 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.082.185.185.185M23.763 9.89c-.065-.051-.672-.51-1.954-.51-.338.001-.676.03-1.01.087-.248-1.7-1.653-2.53-1.716-2.566l-.344-.199-.226.327c-.284.438-.49.922-.612 1.43-.23.97-.09 1.882.403 2.661-.595.332-1.55.413-1.744.42H.751a.751.751 0 00-.75.748 11.376 11.376 0 00.692 4.062c.545 1.428 1.355 2.48 2.41 3.124 1.18.723 3.1 1.137 5.275 1.137.983.003 1.963-.086 2.93-.266a12.248 12.248 0 003.823-1.389c.98-.567 1.86-1.288 2.61-2.136 1.252-1.418 1.998-2.997 2.553-4.4h.221c1.372 0 2.215-.549 2.68-1.009.309-.293.55-.65.707-1.046l.098-.288Z"/></svg>',
    docs:'<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
  };
  function _esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
  // Header
  const h1 = document.getElementById('brandTitle');
  if (h1 && (b.title || b.subtitle)) {
    h1.innerHTML = _esc(b.title || 'llm-rosetta') + ' <span>' + _esc(b.subtitle || 'gateway admin') + '</span>';
  }
  // Login
  const h2 = document.getElementById('brandLoginTitle');
  if (h2 && b.title) {
    h2.innerHTML = '<strong>' + _esc(b.title) + '</strong> <span style="font-weight:400;color:var(--text-dim)">' + _esc(b.subtitle || 'gateway') + '</span>';
  }
  // Settings footer — project name
  const fn = document.getElementById('brandFooterName');
  if (fn && b.title) fn.textContent = b.title;
  // Settings footer — version
  if (b.version) {
    const sv = document.getElementById('settingsVersion');
    if (sv) sv.textContent = 'v' + b.version;
  }
  // Settings footer — links
  if (b.links) {
    const fl = document.getElementById('brandFooterLinks');
    if (fl) {
      fl.innerHTML = b.links.map(function(lk) {
        const ico = lk.icon && _icons[lk.icon] ? _icons[lk.icon] + ' ' : '';
        return '<a href="' + _esc(lk.url) + '" target="_blank" rel="noopener" class="about-link">' + ico + _esc(lk.label) + '</a>';
      }).join('');
    }
  }
  // Attribution
  if (b.attribution) {
    const fn2 = document.getElementById('brandFooterName');
    if (fn2) {
      const attr = document.createElement('div');
      attr.style.cssText = 'font-size:11px;color:var(--text-dim);margin-top:4px';
      attr.textContent = b.attribution;
      fn2.parentNode.insertBefore(attr, fn2.nextSibling);
    }
  }
  // Redirect away from disabled tabs
  if (b.disabled_tabs && b.disabled_tabs.indexOf(S.currentTab) !== -1) {
    const _allTabs = b.all_tabs || ['providers', 'models', 'keys', 'dashboard', 'logs'];
    for (let i = 0; i < _allTabs.length; i++) {
      if (b.disabled_tabs.indexOf(_allTabs[i]) === -1) { S.currentTab = _allTabs[i]; break; }
    }
    localStorage.setItem('llm-rosetta-tab', S.currentTab);
  }
})();

// Global click handler for model more-menus
document.addEventListener('click', function(e) {
  if (!e.target.closest('.model-more-menu')) {
    document.querySelectorAll('.more-menu').forEach(m => m.style.display = 'none');
  }
});

// ===================== Start =====================
setLang(S.currentLang);
checkAuthAndInit();

Object.assign(window, { initApp, stopTimers });

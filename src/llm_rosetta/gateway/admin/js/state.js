/**
 * state.js — Single mutable state object and constants for admin UI.
 *
 * Every other module imports { S } (or individual constants) from here
 * instead of declaring its own globals.
 */

// ---- Constants --------------------------------------------------------

export const LOG_LIMIT = 30;
export const DUMP_PAGE_SIZE = 20;
export const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
export const _TEST_TIMEOUT_MS = 120_000;

export const _CAP_ICONS = {
  llm: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 4c0-1.1.9-2 2-2h8a2 2 0 012 2v5a2 2 0 01-2 2H5l-3 3V4z"/></svg>',
  embedding: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="4" cy="5" r="1.3"/><circle cx="10" cy="3" r="1.3"/><circle cx="12" cy="9" r="1.3"/><circle cx="6" cy="11" r="1.3"/><circle cx="3" cy="9" r="1"/><circle cx="9" cy="7" r="1"/><circle cx="13" cy="13" r="1"/></svg>',
  rerank: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M4 3h8M4 7h6M4 11h10"/><path d="M13 1.5l1.5 1.5-1.5 1.5" stroke-linejoin="round"/><path d="M3 9.5L1.5 11 3 12.5" stroke-linejoin="round"/></svg>',
};

// ---- Mutable state ----------------------------------------------------

export const S = {
  currentTab: localStorage.getItem('llm-rosetta-tab') || 'providers',
  currentTheme: localStorage.getItem('llm-rosetta-theme') || 'light',
  currentLang: localStorage.getItem('llm-rosetta-lang') || 'en',
  configData: null,
  _credentialVisible: true,
  _providerFilter: 'all',
  _modelDomain: 'all',
  keysData: null,
  logKeyLabels: [],
  internalToken: null,
  logOffset: 0,
  dashboardTimer: null,
  logTimer: null,
  expandedLogRows: new Set(),
  _dashboardRefreshMs: parseInt(localStorage.getItem('dashboardRefreshMs') || '3000', 10),
  _editingProviderName: null,
  _providerViewMode: localStorage.getItem('provider-view') || 'grid',
  _modelSortKey: 'name',
  _modelSortDir: 'asc',
  _pendingDeleteProvider: '',
  _keyFieldIsMulti: false,
  _keyFieldVisible: false,
  _fetchedModels: [],
  _fetchProvider: '',
  _profilingEnabled: false,
  _captureEnabled: false,
  _dumpPage: 0,
  _dumpAllEntries: [],
  _lastPersistence: null,
  _lastTotalReq: 0,
  _testAbortCtrl: null,
  _testTaskId: null,
  _testPollTimer: null,
  _testElapsedTimer: null,
  _matryoshkaModel: '',
};

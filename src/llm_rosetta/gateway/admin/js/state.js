/**
 * state.js — Single mutable state object and constants for admin UI.
 *
 * Every other module imports { S } (or individual constants) from here
 * instead of declaring its own globals.
 */

// ---- Constants --------------------------------------------------------

export const LOG_LIMIT = 30;
export const OPS_LOG_LIMIT = 30;
export const DUMP_PAGE_SIZE = 20;
export const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
export const _TEST_TIMEOUT_MS = 120_000;

export const _CAP_ICONS = {};

/**
 * Populate _CAP_ICONS from model_types metadata returned by the backend.
 * Called once after config is first loaded.
 */
export function populateCapIcons(modelTypes) {
  for (const desc of modelTypes) {
    if (desc.icon_svg) _CAP_ICONS[desc.name] = desc.icon_svg;
  }
}

// ---- Mutable state ----------------------------------------------------

export const S = {
  currentTab: localStorage.getItem('llm-rosetta-tab') || 'providers',
  currentScheme: localStorage.getItem('llm-rosetta-scheme') || 'minimal',
  currentMode: localStorage.getItem('llm-rosetta-mode') || 'light',
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
  healthTimer: null,
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
  opsLogOffset: 0,
  opsLogTimer: null,
  expandedOpsLogRows: new Set(),
  _logView: 'requests',
};

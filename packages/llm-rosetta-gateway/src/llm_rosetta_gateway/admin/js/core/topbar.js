/**
 * topbar.js — the shell's sticky header: the section index and the page title.
 *
 * The index comes from the rail's own order, and the title from the active
 * panel's `.section-header h2` — which the page already translates — so no tab
 * changes its markup and no new `data-i18n` key is introduced.  When the
 * language switches, `applyI18n()` re-runs the fill, so the heading that is
 * hidden stays the single source of the text.
 *
 * The pure helpers take their inputs as arguments so the node tests can cover
 * them without a DOM.
 */

/**
 * The bracketed section index, e.g. `[03]`.
 *
 * @param {number} position 1-based position in the rail.
 * @returns {string} The label, or `''` for a position outside the rail.
 */
export function indexLabel(position) {
  if (!Number.isInteger(position) || position < 1) return '';
  return '[' + String(position).padStart(2, '0') + ']';
}

/**
 * The 1-based position of *tabId* among the rail's tab ids.
 *
 * @param {string} tabId
 * @param {string[]} railTabIds
 * @returns {number} 0 when the tab is not in the rail.
 */
export function sectionIndexOf(tabId, railTabIds) {
  const i = (railTabIds || []).indexOf(tabId);
  return i === -1 ? 0 : i + 1;
}

/**
 * The title to show: the panel's own heading when it has one, else the rail
 * label, else the fallback.
 *
 * @param {string|null|undefined} headingText
 * @param {string|null|undefined} tabLabel
 * @param {string} [fallback]
 * @returns {string}
 */
export function topbarTitle(headingText, tabLabel, fallback = '') {
  for (const candidate of [headingText, tabLabel, fallback]) {
    const text = (candidate || '').trim();
    if (text) return text;
  }
  return '';
}

/** Fill the topbar from the active tab and panel. */
export function syncTopbarTitle() {
  const title = document.getElementById('topbarTitle');
  const index = document.getElementById('topbarIndex');
  if (!title && !index) return;

  const railIds = [...document.querySelectorAll('.sidebar .tab[data-tab]')]
    .map(el => el.dataset.tab);
  const activeTab = document.querySelector('.sidebar .tab.active');
  const tabId = activeTab?.dataset.tab || '';
  const panel = tabId ? document.getElementById('tab-' + tabId) : null;
  // The same element the stylesheet hides: the panel's own heading at
  // `.tab-panel > .section > .section-header`.  Deeper headers (the
  // observability board's two) are sub-sections and are not the page title.
  const heading = panel?.querySelector(':scope > .section > .section-header > h2');

  if (title) {
    const fallback = tabId === 'settings'
      ? (window.t?.('modal.settings') || 'Settings')
      : '';
    title.textContent = topbarTitle(
      heading?.textContent, activeTab?.querySelector('.tab-label')?.textContent, fallback,
    );
  }
  if (index) {
    // The settings tab is in the rail but hidden; it keeps its position so the
    // numbers do not renumber when the operator opens it.
    index.textContent = indexLabel(sectionIndexOf(tabId, railIds));
  }
}

/** Expose the fill for the modules that cannot import it (i18n, init).
 *  Guarded so the pure helpers stay importable from node, where the page's
 *  globals do not exist. */
if (typeof window !== 'undefined') window.syncTopbarTitle = syncTopbarTitle;

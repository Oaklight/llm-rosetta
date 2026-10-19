/**
 * sort.js — keyboard and assistive-tech support for sortable table headers.
 *
 * Kept free of imports so it can be unit-tested under node, which has no DOM.
 */

/**
 * Expose a table's sort state to assistive tech: the active header carries
 * `aria-sort`, the others carry none (the implicit `none`).  The ▲/▼ drawn by
 * `th.sortable::after` is a pseudo-element, so it is never announced — this
 * attribute is the only state a screen reader gets.
 *
 * @param {Array<[HTMLElement|null, string]>} headers - `[element, sortKey]` pairs.
 * @param {string} activeKey - The key currently sorted by, or '' for none.
 * @param {'asc'|'desc'} dir - The current direction.
 */
export function syncAriaSort(headers, activeKey, dir) {
  const value = dir === 'asc' ? 'ascending' : 'descending';
  for (const [el, key] of headers) {
    if (!el) continue;
    if (key === activeKey) el.setAttribute('aria-sort', value);
    else el.removeAttribute('aria-sort');
  }
}

/**
 * Make every `th.sortable` under *root* a keyboard-operable tab stop.  Enter
 * or Space runs the header's own click handler, so the keyboard path cannot
 * drift from the mouse path.  Idempotent, and exported for any renderer that
 * builds a header row after startup.
 *
 * @param {ParentNode} [root] - Subtree to wire; defaults to the document.
 */
export function wireSortableHeaders(root = document) {
  for (const th of root.querySelectorAll('th.sortable')) {
    if (th.dataset.kbdSort) continue;
    th.dataset.kbdSort = '1';
    th.tabIndex = 0;
    th.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
      e.preventDefault();  // Space would otherwise scroll the page
      th.click();
    });
  }
}

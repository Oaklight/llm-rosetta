/**
 * The one sentence for "the filter matched nothing", and its way out.
 *
 * Four pages used to answer the same situation four ways — two of them
 * claiming the config held nothing while it held dozens (#1519).  The choice
 * between "nothing exists yet" and "nothing matches" is a rule, not copy: a
 * page that has rows but shows none is filtered, and the message has to say
 * so and offer the way back.  This module keeps that rule and the markup in
 * one place so a page cannot drift again.
 *
 * DOM-free builders: the caller decides the container (a table row versus a
 * block), the builder decides the words.
 */

import { t } from './i18n.js';

/** The shared wording for a filter that matched nothing. */
export const FILTERED_EMPTY_KEY = 'empty.filtered';

/**
 * The message, with a reset control, for a page that has rows but shows none.
 *
 * @param {string} clearCall The reset function to call, e.g. `resetModelFilters()`.
 *   Always a literal from the caller's own module — never user input.
 * @returns {string} HTML.
 */
export function filteredEmptyHtml(clearCall) {
  // The call is interpolated into an attribute: only one plain function call
  // passes, so a caller cannot hand this builder a statement.
  const raw = String(clearCall || '');
  const call = /^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\(\)$/.test(raw) ? raw : '';
  return `<span class="empty-filtered-text">${t(FILTERED_EMPTY_KEY)}</span>`
    + ` <button type="button" class="empty-clear" onclick="${call}">${t('empty.clearFilter')}</button>`;
}

/**
 * The same message as a table row, for the pages whose list is a `<tbody>`.
 *
 * @param {string} clearCall The reset function to call.
 * @param {number} colSpan How many columns the row spans.
 * @returns {string} HTML.
 */
export function filteredEmptyRowHtml(clearCall, colSpan) {
  return `<tr role="row"><td colspan="${Number(colSpan) || 1}" role="cell" class="empty-filtered">${filteredEmptyHtml(clearCall)}</td></tr>`;
}

/**
 * Which sentence is true: the store has rows, or it does not.
 *
 * @param {number} total Rows the config holds before filtering.
 * @param {number} shown Rows the filter lets through.
 * @returns {'none'|'filtered'|'shown'}
 */
export function emptyChoice(total, shown) {
  if (shown > 0) return 'shown';
  return total > 0 ? 'filtered' : 'none';
}

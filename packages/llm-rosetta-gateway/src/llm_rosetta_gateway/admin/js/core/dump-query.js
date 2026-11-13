/**
 * The Error Dumps list request, built from the toolbar's values.
 *
 * The panel used to ask for `limit=10000` and filter in the browser, while
 * the route clamps a limit to 500 — so every count and filter described the
 * newest 500 rows of a store holding thousands.  The list now pages and
 * filters on the server, and this module owns the translation from the
 * toolbar to its query string.
 *
 * It is DOM-free so `node --test` can pin it: a filter that is silently
 * dropped here looks exactly like a server that returned the wrong rows.
 */

/** The time ranges the toolbar offers, in milliseconds. */
const RANGE_MS = {
  '1h': 3_600_000,
  '24h': 86_400_000,
  '7d': 604_800_000,
  '30d': 2_592_000_000,
};

/** Local midnight at the start of *date* (`YYYY-MM-DD`), as UTC ISO. */
function dayStart(date) {
  const d = new Date(`${date}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

/** The last millisecond of *date*, in the operator's own timezone. */
function dayEnd(date) {
  const d = new Date(`${date}T23:59:59.999`);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

/**
 * The status filter to send, or `''` for none.
 *
 * The route accepts a class (`4xx`) or a three-digit code; anything else is
 * a 400.  The search box is typed a digit at a time, so a partial code is
 * normal input — it selects nothing rather than asking the server to refuse
 * the request and leaving the old rows on screen.
 *
 * @param {string} drop    The select's value (`''`, `4xx`, `5xx`, `custom`).
 * @param {string} search  The custom box's text.
 * @returns {string}
 */
export function dumpStatusValue(drop, search) {
  const raw = (drop === 'custom' || (!drop && search)) ? search : drop;
  if (!raw) return '';
  if (/^[1-5]xx$/.test(raw)) return raw;
  if (/^\d{3}$/.test(raw)) return raw;
  return '';
}

/**
 * Build the query for one page of the Error Dumps list.
 *
 * An empty filter is omitted rather than sent empty: the route treats
 * `provider=` and no provider alike, but a URL that states what it does not
 * filter is harder to read in a log.
 *
 * @param {object} filters   Toolbar values: `phase`, `status`, `provider`,
 *   `model` (an exact choice), `modelSearch` (a substring), `idQuery`,
 *   `timeRange`, `dateFrom`, `dateTo`.
 * @param {object} page      `limit` and `offset`; both omitted when undefined.
 *   `before` is a `{timestamp, id}` keyset cursor — the last row of the
 *   previous page — and replaces `offset` for a walk that must not shift
 *   while rows are being written.
 * @param {number} [now]     Milliseconds since epoch, injectable for tests.
 * @returns {URLSearchParams}
 */
export function dumpQuery(filters, page = {}, now = Date.now()) {
  const p = new URLSearchParams();
  if (page.limit != null) p.set('limit', String(page.limit));
  if (page.offset != null) p.set('offset', String(page.offset));
  if (page.before && page.before.timestamp && page.before.id) {
    p.set('before_timestamp', page.before.timestamp);
    p.set('before_id', page.before.id);
  }
  if (filters.phase) p.set('error_phase', filters.phase);
  if (filters.status) p.set('status', filters.status);
  if (filters.provider) p.set('provider', filters.provider);
  if (filters.model) p.set('model', filters.model);
  if (filters.modelSearch) p.set('model_search', filters.modelSearch);
  if (filters.idQuery) p.set('q', filters.idQuery);

  const range = filters.timeRange;
  if (range && range !== 'custom') {
    const ms = RANGE_MS[range];
    if (ms) p.set('start', new Date(now - ms).toISOString());
  } else if (range === 'custom') {
    if (filters.dateFrom) {
      const start = dayStart(filters.dateFrom);
      if (start) p.set('start', start);
    }
    if (filters.dateTo) {
      const end = dayEnd(filters.dateTo);
      if (end) p.set('end', end);
    }
  }
  return p;
}

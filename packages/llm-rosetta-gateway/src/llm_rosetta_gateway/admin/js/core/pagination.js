/**
 * pagination.js — Page-jump math shared by the paginated admin views.
 *
 * Kept free of DOM and module imports so the arithmetic and the sync rule can
 * be pinned by `node --test tests/js/pagination.test.mjs`; the callers own the
 * elements.
 */

/** Number of pages a result set of `total` rows spans, never below 1. */
export function pageCount(total, limit) {
  const n = Math.trunc(Number(total));
  const per = Math.trunc(Number(limit));
  if (!Number.isFinite(n) || !Number.isFinite(per) || per <= 0 || n <= 0) return 1;
  return Math.ceil(n / per);
}

/**
 * Clamp a page typed into a jump box.
 *
 * Returns `null` for input that carries no page number (empty, non-numeric),
 * so the caller can leave the current page alone instead of jumping to 1.
 */
export function clampPage(value, totalPages) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const page = Math.trunc(Number(value));
  if (!Number.isFinite(page)) return null;
  const last = Math.max(1, Math.trunc(Number(totalPages)) || 1);
  return Math.min(Math.max(page, 1), last);
}

/**
 * Zero-based offset of a 1-based page, for the offset-based endpoints.
 *
 * An unusable `limit` yields 0 rather than `NaN`: every caller passes a
 * constant today, but a `NaN` offset would reach the query string as
 * `offset=NaN` and only surface as a 400 from the server.
 */
export function offsetForPage(page, limit) {
  const per = Math.trunc(Number(limit));
  const size = Number.isFinite(per) && per > 0 ? per : 0;
  return (Math.max(1, Math.trunc(Number(page)) || 1) - 1) * size;
}

/**
 * Decide whether a re-render may write its page number into the jump box.
 *
 * The box holds what the user is typing, so a re-render must not eat it.  Two
 * signals say "hands off", and they cover different windows:
 *
 * - `dirty` — the user has edited the box since it was last synced.  This is
 *   the one that matters on the first keystroke: on page 1, typing `1` of `12`
 *   leaves a value that still *equals* the shown page, so a value comparison
 *   alone would let a refresh replace it with the (possibly moved) page and the
 *   next keystroke would append to that instead — `3` + `2` = `32`.
 * - `focused` with a diverged value — the box is in use and holds something
 *   other than the last synced page, even though nothing marked it dirty.
 *
 * Anything else is safe to overwrite, which is what keeps a box the user has
 * not touched following Prev/Next and the rows-per-page control.
 */
export function shouldResync(value, lastShown, { focused = false, dirty = false } = {}) {
  if (dirty) return false;
  if (focused && String(value) !== String(lastShown)) return false;
  return true;
}

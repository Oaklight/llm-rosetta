/**
 * The ink for a retention counter in the status bar.
 *
 * Both stores the footer counts — the success log and the error dumps — are
 * trimmed to a rolling cap, so `n === cap` is what a healthy gateway spends
 * its life at.  Colouring that red trains the operator to ignore the bar, so
 * red is kept for a store *over* its cap, where the trimmer is behind, and
 * amber for one on its way there (§13 of #1502).
 *
 * DOM-free, so `node --test` can pin the thresholds.
 */

/**
 * @param {number} n Rows held.
 * @param {number} cap The configured cap; 0 or absent means no cap.
 * @returns {string} `''`, `'warn'` or `'crit'`.
 */
export function capClass(n, cap) {
  if (!(cap > 0)) return '';
  if (n > cap) return 'crit';
  return n / cap >= 0.75 ? 'warn' : '';
}

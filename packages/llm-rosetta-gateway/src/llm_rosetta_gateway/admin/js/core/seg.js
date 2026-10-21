// seg.js — keyboard model for .seg-control groups.
//
// Kept import-free so the node suite can exercise it without a DOM.  The
// admin uses two flavours of segmented control and they need different
// handling:
//
//   * pseudo-radio items (role="radio" on a div/label) have no native
//     activation, so Enter and Space must be translated into a click;
//   * real <input type="radio"> items activate natively, so these helpers are
//     only consulted for the items that carry role="radio" — a group whose
//     items are inputs returns index -1 and the browser keeps its behaviour.
//
// Movement follows the WAI-ARIA radio-group pattern: focus and selection move
// together, Arrow keys wrap, Home/End jump to the ends.

/**
 * Index the arrow/Home/End keys move to, or -1 when the key is not one of
 * them (or the current index is not inside the group).
 *
 * @param {number} length - number of items in the group
 * @param {number} index - index of the currently focused item
 * @param {string} key - KeyboardEvent.key
 * @returns {number} target index, or -1 to leave the event alone
 */
export function segKeyTargetIndex(length, index, key) {
  if (length <= 0 || index < 0 || index >= length) return -1;
  if (key === 'ArrowRight') return (index + 1) % length;
  if (key === 'ArrowLeft') return (index - 1 + length) % length;
  if (key === 'Home') return 0;
  if (key === 'End') return length - 1;
  return -1;
}

/**
 * Whether the key should activate the focused pseudo-radio item.  Both the
 * modern and legacy names for Space are accepted, as elsewhere in the panel.
 *
 * @param {string} key - KeyboardEvent.key
 * @returns {boolean}
 */
export function isSegActivationKey(key) {
  return key === 'Enter' || key === ' ' || key === 'Spacebar';
}

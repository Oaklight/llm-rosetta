/**
 * shim-selection.js — the `shims.include` / `shims.exclude` rule, in one place.
 *
 * A package is installed whole by default; a selection narrows it to the shims
 * the operator ticked.  Both halves have to agree with the gateway, which reads
 * the same section: the panel shows what is kept, and writes what is ticked.
 */

/**
 * Whether a selection holds an `fnmatch` pattern rather than plain names.
 *
 * A pattern is the operator's own edit of the config file; the checkboxes can
 * render the package's declared shims but not expand a pattern, so the card
 * says so instead of pretending the selection is a list of names.
 *
 * @param {string[]} list - One `include` or `exclude` entry.
 * @returns {boolean} True when the entry cannot be shown as checkboxes.
 */
export function hasPattern(list) {
  return (list || []).some((x) => /[*?[]/.test(String(x)));
}

/**
 * Which of a package's declared shims a saved selection keeps.
 *
 * @param {string[]} names - The shims the package declares.
 * @param {{include?: string[], exclude?: string[]}} selection - Its saved entry.
 * @returns {string[]} The shims that load, in declaration order.
 */
export function keptShims(names, selection) {
  const include = Array.isArray(selection && selection.include) ? selection.include : [];
  const exclude = Array.isArray(selection && selection.exclude) ? selection.exclude : [];
  let kept = Array.isArray(names) ? names : [];
  if (include.length) kept = kept.filter((n) => include.includes(n));
  return kept.filter((n) => !exclude.includes(n));
}

/**
 * The `include` / `exclude` maps that add *key* with *checked* of *total*.
 *
 * Everything ticked clears the package's entry, so the default stays "all of
 * it" — a shim the package adds later then switches itself on.  Anything less
 * is a closed `include` list, which is what the operator picked.  Every other
 * package's entry is carried over untouched, and a stale `exclude` for this
 * package is dropped: the two maps must not both speak for one key.
 *
 * @param {object} section - The saved `shims` config section.
 * @param {string} key - The catalogue key being saved.
 * @param {string[]} checked - The ticked shim names.
 * @param {number} total - How many the card offered.
 * @returns {{include: object, exclude: object}} The maps to write.
 */
export function selectionMaps(section, key, checked, total) {
  const include = { ...((section && section.include) || {}) };
  const exclude = { ...((section && section.exclude) || {}) };
  if (checked.length === total) {
    delete include[key];
  } else {
    include[key] = [...checked];
  }
  delete exclude[key];
  return { include, exclude };
}

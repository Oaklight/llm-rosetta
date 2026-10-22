/**
 * reasoning-override.js — Merge the model card's reasoning controls onto what was loaded.
 *
 * The form renders four controls (thinking type, budget ratio, disabled
 * strategy, replay missing), but a stored ``reasoning_override`` may carry
 * fields it cannot express — ``effort_field``, ``effort_range``,
 * ``visibility_modes``, ``unsigned_blocks`` — and the card *shows* those as
 * inherited values.  Rebuilding the override from the controls alone therefore
 * deletes whatever the operator set by hand the next time they save for an
 * unrelated reason, and it can never clear a field: "inherit" would simply omit
 * the key, which the backend reads as "leave it".
 *
 * Kept free of DOM and module imports so the rule can be pinned by
 * `node --test tests/js/reasoning-override.test.mjs`; the caller owns the
 * elements.
 */

/**
 * Apply the form's edits to a loaded override.
 *
 * @param {object|null|undefined} loaded - the stored ``reasoning_override``.
 * @param {object} edits - one entry per rendered control; a falsy value
 *   (`''`, `null`, `undefined`) means "inherit", which removes the key.
 * @returns {object} the override to save — `{}` when nothing is set, which is
 *   also how the backend is told to drop the key entirely.
 */
export function mergeReasoningOverride(loaded, edits) {
  const out = { ...(loaded || {}) };
  for (const [key, value] of Object.entries(edits || {})) {
    if (value === null || value === undefined || value === '') delete out[key];
    else out[key] = value;
  }
  return out;
}

/**
 * Where a shim picked from the shims page lands, and what to call it.
 *
 * The page lists the same catalogue the Add Provider modal used to list in a
 * dialog, so the chooser went away with the dialog; what it decided did not.
 * A shim's `model_type` decides which control can hold it, and its vendor name
 * decides the provider id — both are rules, not layout, so they live here and
 * stay DOM-free for `node --test`.
 */

import { shimBrand } from './shim-tree.js';

/**
 * Where a picked shim lands in the form.
 *
 * The modal's type tree is the LLM one — `buildTypeModel` is built from the
 * LLM shims alone — so a non-LLM shim has no row in it.  Such a pick belongs
 * to its endpoint section instead: the caller ticks that section's capability
 * and chooses the shim in the section's own picker.  A form that cannot hold
 * the shim it was opened for is a promise it does not keep.
 *
 * @param {object} shim A `registered_shims` entry.
 * @returns {{type: string, control: 'llm'|'endpoint'}}
 */
export function pickDestination(shim) {
  const type = (shim && shim.model_type) || 'llm';
  return { type, control: type === 'llm' ? 'llm' : 'endpoint' };
}

/** The cap this rule puts on a suggested id: long enough for a vendor's name,
 *  short enough to read as a config key.  The form itself accepts a longer
 *  one, so this only decides what is prefilled. */
const MAX_ID_LENGTH = 48;

/**
 * A provider id suggested for *shim*, unique against *taken*.
 *
 * The id is the vendor's own name, lower-cased and reduced to the characters
 * a config key can hold.  A second copy becomes `-2`, `-3`, … rather than
 * silently overwriting the first.  The length cap covers the whole id, the
 * suffix included, so a de-duplicated name is never longer than a first one.
 *
 * @param {object} shim
 * @param {string[]} [taken] Provider ids already in the config.
 * @returns {string}
 */
export function suggestedProviderName(shim, taken = []) {
  const clean = (text) =>
    String(text || '')
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, '-')
      .replace(/^-+|-+$/g, '');
  // A shim whose declared name is only punctuation still has its own id to
  // fall back to, so the Add form never opens with no name at all.
  const base = (clean(shimBrand(shim)) || clean(shim.display_name) || clean(shim.name))
    .slice(0, MAX_ID_LENGTH)
    .replace(/-+$/, '');
  if (!base) return '';
  // Config keys are compared as the loader compares them, lower-cased: a
  // config holding `OpenRouter` already owns `openrouter`.
  const used = new Set((taken || []).map((id) => String(id).toLowerCase()));
  if (!used.has(base)) return base;
  // The suffix has to fit inside the same limit, and trimming it back can
  // leave a trailing separator — so trim again after the cut.
  const withSuffix = (n) => {
    const suffix = `-${n}`;
    return base.slice(0, MAX_ID_LENGTH - suffix.length).replace(/-+$/, '') + suffix;
  };
  let n = 2;
  while (used.has(withSuffix(n))) n += 1;
  return withSuffix(n);
}

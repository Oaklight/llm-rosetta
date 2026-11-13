/**
 * The Add Provider chooser: which shims it offers, and the id it suggests.
 *
 * The modal used to open blank, with every known vendor buried in the
 * "top level" dropdown inside it.  An operator who wanted to add, say,
 * OpenRouter had to know that the vendor existed before they could look for
 * it.  The chooser lists every shim the gateway can serve, one row each, so
 * the catalogue is visible before anything is typed.
 *
 * This module is DOM-free so `node --test` can pin the list and the id.  A
 * row that goes missing here is invisible on screen — the provider simply
 * cannot be chosen.
 */

import { shimBrand } from './shim-tree.js';

/** The name a row shows: the vendor, which is what an operator looks for. */
export function shimTitle(shim) {
  return shimBrand(shim) || shim.display_name || shim.name || '';
}

/**
 * The row's second line: the wire format, and the model type when it is not
 * `llm`.  A vendor appears once per format it serves, so the format is what
 * tells two rows of the same vendor apart.
 *
 * @param {object} shim A `registered_shims` entry.
 * @returns {string}
 */
export function shimSubtitle(shim) {
  const parts = [];
  const format = shim.format || shim.base;
  if (format) parts.push(format);
  const type = shim.model_type || 'llm';
  if (type !== 'llm') parts.push(type);
  return parts.join(' · ');
}

/** The short tags a row carries: how its credential arrives, and its cost. */
export function shimBadges(shim) {
  const badges = [];
  if (shim.free_source) badges.push('free');
  if (shim.keyless) badges.push('keyless');
  else if (shim.mints_credential) badges.push(shim.credential_kind || 'sign-in');
  return badges;
}

/** One rendered row. */
function pickRow(shim) {
  return {
    name: shim.name || '',
    title: shimTitle(shim) || shim.name || '',
    subtitle: shimSubtitle(shim),
    logo: shim.logo || '',
    badges: shimBadges(shim),
  };
}

/**
 * Where a picked shim lands in the form.
 *
 * The modal's type tree is the LLM one — `buildTypeModel` is built from the
 * LLM shims alone — so a non-LLM shim has no row in it.  Such a pick belongs
 * to its endpoint section instead: the caller ticks that section's capability
 * and chooses the shim in the section's own picker.  A chooser that lists
 * rows the chosen control cannot hold is a promise the form does not keep.
 *
 * @param {object} shim A `registered_shims` entry.
 * @returns {{type: string, control: 'llm'|'endpoint'}}
 */
export function pickDestination(shim) {
  const type = (shim && shim.model_type) || 'llm';
  return { type, control: type === 'llm' ? 'llm' : 'endpoint' };
}

// The badges are on the row, so they are searchable: typing `free` or
// `keyless` must find what the row displays, or the tag reads as decoration.
const haystack = (row) =>
  `${row.title} ${row.name} ${row.subtitle} ${row.badges.join(' ')}`.toLowerCase();

/**
 * Rows for *query*, sorted by title.
 *
 * An empty query lists everything: the point of the screen is that the
 * catalogue is visible without knowing what to search for.
 *
 * @param {object[]} shims `registered_shims` from the config payload.
 * @param {string} [query]
 * @returns {object[]}
 */
export function providerPickRows(shims, query = '') {
  const rows = (shims || [])
    .filter((s) => s && s.name)
    .map(pickRow)
    .sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' }));
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) return rows;
  return rows.filter((row) => haystack(row).includes(needle));
}

/** The cap this chooser puts on a suggested id: long enough for a vendor's
 *  name, short enough to read as a config key.  The form itself accepts a
 *  longer one, so this only decides what is prefilled. */
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
  // fall back to, so the chooser never opens an Add form with no name at all.
  const base = (clean(shimBrand(shim)) || clean(shim.display_name) || clean(shim.name))
    .slice(0, MAX_ID_LENGTH)
    .replace(/-+$/, '');
  if (!base) return '';
  // Config keys are compared as the loader compares them, lower-cased: a
  // config holding `OpenRouter` already owns `openrouter`.
  const used = new Set((taken || []).map((id) => String(id).toLowerCase()));
  if (!used.has(base)) return base;
  // The suffix has to fit inside the same limit, and trimming it back to 46
  // can leave a trailing separator — so trim again after the cut.
  const withSuffix = (n) => {
    const suffix = `-${n}`;
    return base.slice(0, MAX_ID_LENGTH - suffix.length).replace(/-+$/, '') + suffix;
  };
  let n = 2;
  while (used.has(withSuffix(n))) n += 1;
  return withSuffix(n);
}

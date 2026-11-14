/**
 * One model-type switch, used wherever the panel filters by type.
 *
 * The Providers page and the Models page each grew their own builder for the
 * same control — same options, same order, one with a mark and one without —
 * so the two switches looked different on the same screen (#1502 §13).  This
 * module owns the option markup and the type's mark; the pages keep only what
 * is theirs (which filter moves, and how).
 *
 * The marks come from the model-type registry the server sends with the
 * config, so a new type needs no code here.
 */

import { typeLabel } from './i18n.js';
import { S, _CAP_ICONS } from './state.js';

/**
 * The registry descriptor for a model type.
 *
 * @param {string} name
 * @returns {{name: string, icon_svg?: string}|null}
 */
export function typeDescriptor(name) {
  const types = (S.configData && S.configData.model_types) || [];
  return types.find((d) => d.name === name) || null;
}

/**
 * The type's own mark, sized by `.seg-icon`.
 *
 * @param {string} name
 * @returns {string} HTML, or `''` when the registry knows no mark.
 */
export function typeMark(name) {
  const desc = typeDescriptor(name);
  // The registry's mark, or the one the config load already collected for the
  // capability tags — the same glyph either way.
  const svg = (desc && desc.icon_svg) || _CAP_ICONS[name];
  return svg ? `<span class="seg-icon" aria-hidden="true">${svg}</span>` : '';
}

/**
 * One option of the switch.
 *
 * Two shapes because two pages already behave differently and the behaviour
 * is theirs: `radio` wraps a real radio input (the Models filter), `button`
 * is a `role="radio"` box the caller wires itself (the Providers filter,
 * which cannot use an attribute string — issue #1394).
 *
 * @param {string} name The type name, or `''` for the "all" option.
 * @param {{checked?: boolean, label?: string, mark?: boolean}} [opts]
 * @returns {string} HTML for one direct child of `.seg-control`.
 */
export function typeOptionHtml(name, opts = {}) {
  const checked = !!opts.checked;
  const label = opts.label ?? (name ? typeLabel(name) : '');
  const mark = opts.mark === false || !name ? '' : typeMark(name);
  const attrs = checked ? ' class="active"' : '';
  const dataName = name ? ` data-type-name="${name}"` : '';
  if (name && opts.mode === 'radio') {
    const divAttrs = opts.divAttrs ? ` ${opts.divAttrs}` : '';
    const inputAttrs = opts.inputAttrs ? ` ${opts.inputAttrs}` : '';
    return `<div${attrs}${divAttrs}><input type="radio" name="modelType" value="${name}"`
      + `${checked ? ' checked' : ''}${inputAttrs}>`
      + `${mark}<span${dataName}>${label}</span></div>`;
  }
  return `<div${attrs} role="radio" aria-checked="${checked}" tabindex="${checked ? 0 : -1}">`
    + `${mark}<span${dataName}>${label}</span></div>`;
}

/** provider-picker.js — logo rows for a provider menu, shared by both pickers.
 *
 *  A native <select> cannot render an image, so every place that picks a
 *  provider from a roster renders the same menu: a logo, then the name.
 */
import { S } from '../core/state.js';
import { esc } from '../core/utils.js';
import { _providerLogo } from '../tabs/models.js';

/** The logo for *name*, or an empty placeholder when it has none. */
export function providerLogoHtml(name, cls = 'pp-logo') {
  const logo = _providerLogo(name);
  return logo
    ? `<img class="${cls}" src="${esc(logo)}" alt="">`
    : `<span class="${cls} pp-logo-empty"></span>`;
}

/**
 * The roster's rows.
 *
 * @param {string[]} names - provider names, in display order.
 * @param {(name: string) => string} onclickJs - inline handler body per row.
 * @param {Set<string>} [muted] - names to render dimmed (disabled providers).
 */
export function providerRowsHtml(names, onclickJs, muted = new Set()) {
  return names.map(name => {
    const off = muted.has(name) ? ' pm-off' : '';
    return `<div class="more-menu-item pp-row${off}" role="option" data-provider="${esc(name)}"
        onclick="${onclickJs(name)}">${providerLogoHtml(name)}<span class="pp-name">${esc(name)}</span></div>`;
  }).join('');
}

/** Show *name* (and its logo) on a picker's trigger, or *placeholder*. */
export function syncPickerBtn(btn, name, placeholder) {
  if (!btn) return;
  btn.querySelector('.pp-value').innerHTML = name
    ? `${providerLogoHtml(name)}<span>${esc(name)}</span>`
    : `<span>${esc(placeholder)}</span>`;
}

/** Open a picker menu at least as wide as the control it belongs to. */
export function openPickerMenu(menu, btn, openMenu, closeMenus) {
  closeMenus();
  menu.style.minWidth = `${Math.round(btn.getBoundingClientRect().width)}px`;
  openMenu(menu, btn);
}

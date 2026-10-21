/**
 * logo-select.js — drop-in replacement for a `<select>` whose rows carry a
 * logo.  A native `<select>` cannot render images, and the Models provider
 * filter wants the same mark the provider card shows.
 *
 * Markup contract (one root per instance):
 *
 *   <div class="logo-select">
 *     <input type="hidden" id="…">              ← the value, as a select's
 *     <button class="logo-select-trigger" …>    ← what the user sees
 *       <img class="logo-select-img">
 *       <span class="logo-select-label"></span>
 *     </button>
 *     <div class="logo-select-panel" role="listbox" tabindex="-1"></div>
 *   </div>
 *
 * The hidden input keeps `document.getElementById(id).value` working for the
 * call sites that already write it, so only the render path changes.
 *
 * Keyboard: Enter/Space/ArrowDown opens, Up/Down/Home/End move the active
 * row, Enter/Space commits it, Escape closes and returns focus to the
 * trigger, Tab closes.
 */

import { esc } from '../core/utils.js';

const _instances = new Map();

function _el(id) { return document.getElementById(id); }

function _syncTrigger(inst) {
  const item = inst.items.find(i => i.value === inst.input.value) || inst.items[0];
  const img = inst.trigger.querySelector('.logo-select-img');
  const label = inst.trigger.querySelector('.logo-select-label');
  if (item && item.logo) {
    img.src = item.logo;
    img.hidden = false;
  } else {
    img.removeAttribute('src');
    img.hidden = true;
  }
  if (label) label.textContent = item ? item.label : '';
  inst.trigger.title = item ? item.label : '';
}

function _renderPanel(inst) {
  inst.panel.innerHTML = inst.items.map((item, i) => {
    const selected = item.value === inst.input.value;
    const logo = item.logo
      ? `<img class="logo-select-img" src="${esc(item.logo)}" alt="">`
      : '<span class="logo-select-img logo-select-img-empty"></span>';
    return `<div class="logo-select-option${selected ? ' selected' : ''}" role="option"
      id="${inst.id}-opt-${i}" data-index="${i}" aria-selected="${selected}">
      ${logo}<span class="logo-select-option-label">${esc(item.label)}</span></div>`;
  }).join('');
}

function _setActive(inst, index) {
  const options = [...inst.panel.querySelectorAll('.logo-select-option')];
  if (!options.length) return;
  inst.active = Math.max(0, Math.min(index, options.length - 1));
  options.forEach((o, i) => o.classList.toggle('active', i === inst.active));
  const active = options[inst.active];
  inst.panel.setAttribute('aria-activedescendant', active.id);
  active.scrollIntoView({ block: 'nearest' });
}

function _open(inst) {
  if (inst.open) return;
  inst.open = true;
  inst.root.classList.add('open');
  inst.trigger.setAttribute('aria-expanded', 'true');
  _renderPanel(inst);
  const selectedIndex = Math.max(0, inst.items.findIndex(i => i.value === inst.input.value));
  _setActive(inst, selectedIndex);
  inst.panel.focus();
}

function _close(inst, { focusTrigger = false } = {}) {
  if (!inst.open) return;
  inst.open = false;
  inst.root.classList.remove('open');
  inst.trigger.setAttribute('aria-expanded', 'false');
  inst.panel.removeAttribute('aria-activedescendant');
  if (focusTrigger) inst.trigger.focus();
}

function _select(inst, index) {
  const item = inst.items[index];
  if (!item) return;
  const changed = item.value !== inst.input.value;
  inst.input.value = item.value;
  _syncTrigger(inst);
  _close(inst, { focusTrigger: true });
  if (changed) inst.onChange(item.value);
}

/** First-letter type-ahead, so a long list keeps the native select's
 *  affordance.  A fresh character starts after the active row, so repeated
 *  presses cycle through the rows that share it. */
function _typeAhead(inst, ch) {
  const now = Date.now();
  if (now - (inst.typedAt || 0) > 600) inst.typeBuffer = '';
  inst.typeBuffer = (inst.typeBuffer || '') + ch.toLowerCase();
  inst.typedAt = now;
  const query = inst.typeBuffer;
  const count = inst.items.length;
  const start = query.length === 1 ? inst.active + 1 : inst.active;
  for (let k = 0; k < count; k++) {
    const i = (start + k + count) % count;
    if (inst.items[i].label.toLowerCase().startsWith(query)) { _setActive(inst, i); return; }
  }
}

function _onPanelKeydown(inst, e) {
  if (e.key === ' ') { e.preventDefault(); _select(inst, inst.active); return; }
  if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    _typeAhead(inst, e.key);
    return;
  }
  switch (e.key) {
    case 'ArrowDown': e.preventDefault(); _setActive(inst, inst.active + 1); break;
    case 'ArrowUp': e.preventDefault(); _setActive(inst, inst.active - 1); break;
    case 'PageDown': e.preventDefault(); _setActive(inst, inst.active + 5); break;
    case 'PageUp': e.preventDefault(); _setActive(inst, inst.active - 5); break;
    case 'Home': e.preventDefault(); _setActive(inst, 0); break;
    case 'End': e.preventDefault(); _setActive(inst, inst.items.length - 1); break;
    case 'Enter': e.preventDefault(); _select(inst, inst.active); break;
    case 'Escape': e.preventDefault(); _close(inst, { focusTrigger: true }); break;
    case 'Tab': _close(inst); break;
    default: break;
  }
}

/** Bind the trigger and panel once.  Safe to call repeatedly. */
export function initLogoSelect(id, onChange) {
  if (_instances.has(id)) {
    if (onChange) _instances.get(id).onChange = onChange;
    return _instances.get(id);
  }
  const input = _el(id);
  if (!input) return null;
  const root = input.closest('.logo-select');
  const trigger = root?.querySelector('.logo-select-trigger');
  const panel = root?.querySelector('.logo-select-panel');
  if (!root || !trigger || !panel) return null;

  const inst = { id, input, root, trigger, panel, items: [], active: -1, open: false, onChange: onChange || (() => {}) };
  _instances.set(id, inst);

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (inst.open) _close(inst, { focusTrigger: true }); else _open(inst);
  });
  trigger.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); _open(inst); }
  });
  panel.addEventListener('keydown', (e) => _onPanelKeydown(inst, e));
  panel.addEventListener('click', (e) => {
    const option = e.target.closest('.logo-select-option');
    if (option) _select(inst, Number(option.dataset.index));
  });
  document.addEventListener('click', (e) => {
    if (!root.contains(e.target)) _close(inst);
  });

  _syncTrigger(inst);
  return inst;
}

/**
 * Replace the option list.  `items` is [{value, label, logo}]; `value` may be
 * passed to force the current selection (otherwise the hidden input keeps it).
 */
export function setLogoSelectOptions(id, items, value) {
  const inst = _instances.get(id);
  if (!inst) return;
  inst.items = items || [];
  if (typeof value === 'string') inst.input.value = value;
  if (!inst.items.some(i => i.value === inst.input.value)) inst.input.value = '';
  _syncTrigger(inst);
  // Rebuilding the rows drops their `.active` class, so restore the active
  // row when the list is replaced under an open panel -- `applyI18n()` calls
  // renderModels(), which lands here.
  if (inst.open) { _renderPanel(inst); _setActive(inst, inst.active); }
}

/** Re-read the hidden input after a programmatic write. */
export function syncLogoSelect(id) {
  const inst = _instances.get(id);
  if (inst) _syncTrigger(inst);
}

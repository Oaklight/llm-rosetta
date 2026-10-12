/**
 * type-picker.js — Rich provider-type dropdown for the provider modal.
 *
 * The modal splits "Provider Type" into two controls: a rich **Top level**
 * dropdown (this module) and a plain **Base type** `<select>` (owned by
 * providers.js).  The top-level dropdown shows each vendor's logo and, in
 * tree view, expands the base/variant children under each root the way the
 * `tree` command does — logos appear on the root row only.
 *
 * The chosen value is the shim (or bare base-format) name, written to the
 * hidden `#provType` input so the save path in providers.js is unchanged.
 *
 * Keyboard: the trigger opens on click / Enter / Space / ArrowDown; once the
 * panel is open the rows take roving focus (Up/Down/Home/End) and Enter or
 * Space commits the focused row.
 */

import { t } from '../core/i18n.js';

const _VIEW_KEY = 'provider-type-view';

let _roots = [];          // [{ name, label, logo, defaultValue, children: [...] }]
let _byValue = new Map(); // value -> { root, child }
let _view = localStorage.getItem(_VIEW_KEY) === 'tree' ? 'tree' : 'top';
let _selected = '';
let _open = false;
let _onSelect = null;

/** Register the selection callback (fired on user and programmatic changes). */
export function onTypeSelect(cb) { _onSelect = cb; }

/** Attach trigger / toggle / outside-click handlers.  Idempotent. */
export function initTypePicker() {
  const trigger = document.getElementById('typePickerTrigger');
  if (!trigger || trigger.dataset.bound) return;
  trigger.dataset.bound = '1';

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_open) _closePanel(); else _openPanel();
  });
  // ArrowDown opens the panel without committing (Enter/Space already click).
  trigger.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && !_open) { e.preventDefault(); _openPanel(); }
  });

  const toggle = document.getElementById('typePickerDropdown')?.querySelector('.tp-view-toggle');
  if (toggle) {
    toggle.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-view]');
      if (!btn) return;
      e.stopPropagation();
      _setView(btn.dataset.view);
    });
  }

  document.getElementById('typePickerList')?.addEventListener('keydown', _onListKeydown);

  // Close when focus leaves the picker entirely (e.g. Tab out of the rows).
  // Closing on the Tab key itself can drop focus to <body> once the focused
  // row is hidden.
  const wrapper = document.getElementById('typePickerWrapper');
  wrapper?.addEventListener('focusout', (e) => {
    if (!_open) return;
    if (e.relatedTarget && wrapper.contains(e.relatedTarget)) return;
    _closePanel();
  });

  // The panel is anchored to the trigger; if the modal scrolls or the window
  // resizes while it is open, close it rather than let it drift.
  document.querySelector('#providerModal .modal')?.addEventListener(
    'scroll', () => { if (_open) _closePanel(); }, { passive: true });
  window.addEventListener('resize', () => { if (_open) _closePanel(); });

  document.addEventListener('click', (e) => {
    const w = document.getElementById('typePickerWrapper');
    if (w && !w.contains(e.target)) _closePanel();
  }, true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _open) {
      e.preventDefault();
      _closePanel();
      document.getElementById('typePickerTrigger')?.focus();
    }
  });
}

/** Populate the tree and select *value* ('' leaves the picker unselected). */
export function setTypeTree(roots, value) {
  _roots = roots || [];
  _byValue = new Map();
  for (const root of _roots) {
    if (!root.children || !root.children.length) {
      _byValue.set(root.name, { root, child: null });
    } else {
      for (const child of root.children) _byValue.set(child.value, { root, child });
    }
  }
  _selected = _byValue.has(value) ? value : '';
  _syncHidden();
  _syncTrigger();
  if (_open) _renderPanel();
}

/** Current backing value (the shim / base-format name). */
export function getSelectedValue() { return _selected; }

/** Programmatic selection; fires the onTypeSelect callback. */
export function selectValue(value) {
  if (!_byValue.has(value)) return;
  if (value === _selected) { _syncTrigger(); if (_open) _renderPanel(); return; }
  _commit(value);
}

// ── Internals ───────────────────────────────────────────────────────

function _setView(view) {
  _view = view === 'tree' ? 'tree' : 'top';
  localStorage.setItem(_VIEW_KEY, _view);
  _renderPanel();
  // Re-rendering replaces every row, so put focus back on (or next to) the
  // selected one — otherwise a keyboard user is stranded on the toggle.
  if (_open) _focusItem();
}

function _openPanel() {
  _open = true;
  document.getElementById('typePickerDropdown')?.classList.add('open');
  _renderPanel();
  _positionPanel();
  _focusItem();
}

/** Keep the panel inside the modal's scroll box: the modal has
 *  `overflow-y:auto`, so a panel anchored below a low trigger would be
 *  clipped.  Flip above the trigger when there is more room there, and cap
 *  the height to the space actually available. */
function _positionPanel() {
  const trigger = document.getElementById('typePickerTrigger');
  const panel = document.getElementById('typePickerDropdown');
  if (!trigger || !panel) return;
  const scroller = trigger.closest('.modal');
  const tRect = trigger.getBoundingClientRect();
  const box = (scroller || document.documentElement).getBoundingClientRect();
  const GAP = 8, MAX = 340, MIN = 120;
  const below = box.bottom - tRect.bottom - GAP;
  const above = tRect.top - box.top - GAP;
  const flip = below < 200 && above > below;
  panel.classList.toggle('tp-flip-up', flip);
  panel.style.maxHeight = `${Math.max(MIN, Math.min(MAX, flip ? above : below))}px`;
}

function _closePanel() {
  _open = false;
  document.getElementById('typePickerDropdown')?.classList.remove('open');
}

function _commit(value) {
  _selected = value;
  _syncHidden();
  _syncTrigger();
  _closePanel();
  document.getElementById('typePickerTrigger')?.focus();
  if (_onSelect) _onSelect(value);
}

function _syncHidden() {
  const input = document.getElementById('provType');
  if (input) input.value = _selected;
}

function _syncTrigger() {
  const logoEl = document.getElementById('provTypeLogo');
  const labelEl = document.getElementById('typePickerLabel');
  const entry = _byValue.get(_selected);
  if (labelEl) {
    labelEl.textContent = entry ? entry.root.label
      : t('label.selectOne', '— select —');
    labelEl.classList.toggle('tp-placeholder', !entry);
  }
  if (logoEl) {
    const logo = entry && entry.root.logo;
    if (logo) { logoEl.src = logo; logoEl.style.display = ''; }
    else { logoEl.src = ''; logoEl.style.display = 'none'; }
  }
}

function _renderPanel() {
  const host = document.getElementById('typePickerList');
  if (!host) return;
  // Reflect the active view on the segmented control.
  const toggle = document.getElementById('typePickerDropdown')?.querySelector('.tp-view-toggle');
  for (const btn of toggle?.querySelectorAll('button[data-view]') || []) {
    const on = btn.dataset.view === _view;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', on ? 'true' : 'false');
  }

  const selectedRoot = _byValue.get(_selected)?.root;
  let html = '';
  for (const root of _roots) {
    // A root row stands for the child it currently owns; if the selection is
    // elsewhere, clicking falls back to the root's default child.  This keeps
    // the highlighted row meaning the same thing as a click on it.
    const ownsSelection = !!selectedRoot && selectedRoot === root;
    const rootValue = ownsSelection ? _selected : root.defaultValue;
    html += `<div class="tp-item tp-root${ownsSelection ? ' tp-active' : ''}" role="option" tabindex="-1" aria-selected="${ownsSelection}" data-value="${_esc(rootValue)}">`
      + _logoHtml(root.logo)
      + `<span class="tp-name">${_esc(root.label)}</span></div>`;
    if (_view === 'tree' && root.children && root.children.length) {
      const last = root.children.length - 1;
      root.children.forEach((child, i) => {
        const guide = i === last ? '└─' : '├─';
        const active = child.value === _selected;
        const suffix = child.label !== child.base && child.base
          ? `<span class="tp-base">${_esc(child.base)}</span>` : '';
        html += `<div class="tp-item tp-child${active ? ' tp-active' : ''}" role="option" tabindex="-1" aria-selected="${active}" data-value="${_esc(child.value)}">`
          + `<span class="tp-guide">${guide}</span>`
          + `<span class="tp-name">${_esc(child.label)}</span>${suffix}</div>`;
      });
    }
  }
  host.innerHTML = html || `<div class="tp-empty">${_esc(t('label.selectOne', '— select —'))}</div>`;

  for (const el of host.querySelectorAll('.tp-item'))
    el.addEventListener('click', () => _commit(el.dataset.value));
}

/** Focus the selected row, else the first row. */
function _focusItem() {
  const items = _items();
  if (!items.length) return;
  const target = items.find(el => el.classList.contains('tp-active')) || items[0];
  target.focus();
}

function _items() {
  return [...(document.getElementById('typePickerList')?.querySelectorAll('.tp-item') || [])];
}

function _onListKeydown(e) {
  const items = _items();
  if (!items.length) return;
  const cur = items.indexOf(document.activeElement);
  if (e.key === 'ArrowDown') { e.preventDefault(); items[Math.min(cur + 1, items.length - 1)].focus(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); items[Math.max(cur - 1, 0)].focus(); }
  else if (e.key === 'Home') { e.preventDefault(); items[0].focus(); }
  else if (e.key === 'End') { e.preventDefault(); items[items.length - 1].focus(); }
  else if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    if (cur >= 0) _commit(items[cur].dataset.value);
  }
}

function _logoHtml(logo) {
  // On load failure (e.g. a cross-origin logo blocked by CORP) fall back to the
  // same dashed placeholder used for logo-less roots.
  return logo
    ? `<img class="tp-logo" src="${_esc(logo)}" alt="" loading="lazy" onerror="this.classList.add('tp-logo-empty');this.removeAttribute('src')">`
    : '<span class="tp-logo tp-logo-empty" aria-hidden="true"></span>';
}

function _esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

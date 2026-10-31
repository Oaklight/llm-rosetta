/**
 * type-picker.js — Rich provider-type dropdown for the provider modal.
 *
 * The modal splits "Provider Type" into two controls: a rich **Top level**
 * dropdown (this module) and a plain **Base type** `<select>` (owned by
 * providers.js).  The top-level dropdown shows each vendor's logo and, in
 * tree view, expands the base/variant children under each root the way the
 * `tree` command does — logos appear on the root row only.
 *
 * The same widget drives the non-LLM endpoint sections: a typed endpoint has
 * one dropdown that names the vendor (or a bare format), because a vendor
 * normally serves one dialect per model type — there is no second level to
 * pick.  Those instances hide the view toggle.
 *
 * The chosen value is the shim (or bare base-format) name, written to the
 * hidden input the owning module names, so the save path is unchanged.
 *
 * Keyboard: the trigger opens on click / Enter / Space / ArrowDown; once the
 * panel is open the rows take roving focus (Up/Down/Home/End) and Enter or
 * Space commits the focused row.
 */

import { t } from '../core/i18n.js';

const _VIEW_KEY = 'provider-type-view';

/**
 * One picker instance over a given set of elements.
 *
 * The module used to keep a single set of module-level variables, which was
 * enough while only the LLM endpoint had one.  A typed endpoint has its own
 * picker per model type, so the state lives on the instance and the module
 * keeps exactly one default instance for the LLM endpoint.
 *
 * @param {object} cfg
 * @param {HTMLElement} cfg.wrapper   `.type-picker` container
 * @param {HTMLElement} cfg.trigger   the button that opens the panel
 * @param {HTMLElement} cfg.dropdown  the panel
 * @param {HTMLElement} cfg.list      the rows host
 * @param {HTMLElement} cfg.label     the trigger's text span
 * @param {HTMLImageElement} cfg.logo the trigger's logo (may be null)
 * @param {HTMLElement} [cfg.scroller] scroll container for flip positioning
 * @param {HTMLInputElement} [cfg.hidden] input that mirrors the value
 * @param {boolean} [cfg.viewToggle]  draw the top/tree toggle
 * @param {(value: string) => void} [cfg.onChange]
 */
export function createTypePicker(cfg) {
  const state = {
    roots: [],
    byValue: new Map(),
    selected: '',
    open: false,
    view: localStorage.getItem(_VIEW_KEY) === 'tree' ? 'tree' : 'top',
    onChange: cfg.onChange || null,
  };

  // ── public surface ────────────────────────────────────────────────

  /** Attach the trigger and document-level handlers.  Idempotent. */
  function init() {
    if (cfg.trigger.dataset.bound) return;
    cfg.trigger.dataset.bound = '1';

    cfg.trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      if (state.open) _closePanel(); else _openPanel();
    });
    cfg.trigger.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' && !state.open) { e.preventDefault(); _openPanel(); }
    });

    if (cfg.viewToggle) {
      cfg.dropdown.querySelector('.tp-view-toggle')?.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-view]');
        if (!btn) return;
        e.stopPropagation();
        _setView(btn.dataset.view);
      });
    }

    cfg.list.addEventListener('keydown', _onListKeydown);

    // Close when focus leaves the picker entirely (e.g. Tab out of the rows).
    cfg.wrapper.addEventListener('focusout', (e) => {
      if (!state.open) return;
      if (e.relatedTarget && cfg.wrapper.contains(e.relatedTarget)) return;
      _closePanel();
    });

    // The panel is anchored to the trigger; if the modal scrolls or the window
    // resizes while it is open, close it rather than let it drift.
    (cfg.scroller || cfg.trigger.closest('.modal'))?.addEventListener(
      'scroll', () => { if (state.open) _closePanel(); }, { passive: true });
    window.addEventListener('resize', () => { if (state.open) _closePanel(); });

    document.addEventListener('click', (e) => {
      if (!cfg.wrapper.contains(e.target)) _closePanel();
    }, true);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && state.open) {
        e.preventDefault();
        _closePanel();
        cfg.trigger.focus();
      }
    });
  }

  /** Populate the tree and select *value* ('' leaves the picker unselected).
   *
   *  A non-empty value the tree does not know is kept, not dropped: it can be
   *  a dialect this build does not list (a config written by a newer release,
   *  or an external shim that did not load), and reopening the form must not
   *  rewrite it to the type default.  The trigger shows it verbatim.
   */
  function setTree(roots, value) {
    state.roots = roots || [];
    state.byValue = new Map();
    for (const root of state.roots) {
      if (!root.children || !root.children.length) {
        state.byValue.set(root.name, { root, child: null });
      } else {
        for (const child of root.children) state.byValue.set(child.value, { root, child });
      }
    }
    state.selected = value || '';
    _syncHidden();
    _syncTrigger();
    if (state.open) _renderPanel();
  }

  /** Current backing value (the shim / base-format name). */
  function getValue() { return state.selected; }

  /** Whether the current tree offers *value* as a row. */
  function has(value) { return state.byValue.has(value); }

  /** Programmatic selection; fires the onChange callback. */
  function select(value) {
    if (!state.byValue.has(value)) return;
    if (value === state.selected) { _syncTrigger(); if (state.open) _renderPanel(); return; }
    _commit(value);
  }

  // ── internals ─────────────────────────────────────────────────────

  function _setView(view) {
    state.view = view === 'tree' ? 'tree' : 'top';
    localStorage.setItem(_VIEW_KEY, state.view);
    _renderPanel();
    // Re-rendering replaces every row, so put focus back on (or next to) the
    // selected one — otherwise a keyboard user is stranded on the toggle.
    if (state.open) _focusItem();
  }

  function _openPanel() {
    state.open = true;
    cfg.dropdown.classList.add('open');
    _renderPanel();
    _positionPanel();
    _focusItem();
  }

  /** Keep the panel inside the modal's scroll box: the modal has
   *  `overflow-y:auto`, so a panel anchored below a low trigger would be
   *  clipped.  Flip above the trigger when there is more room there, and cap
   *  the height to the space actually available. */
  function _positionPanel() {
    const scroller = cfg.scroller || cfg.trigger.closest('.modal');
    const tRect = cfg.trigger.getBoundingClientRect();
    const box = (scroller || document.documentElement).getBoundingClientRect();
    const GAP = 8, MAX = 340, MIN = 120;
    const below = box.bottom - tRect.bottom - GAP;
    const above = tRect.top - box.top - GAP;
    const flip = below < 200 && above > below;
    cfg.dropdown.classList.toggle('tp-flip-up', flip);
    cfg.dropdown.style.maxHeight = `${Math.max(MIN, Math.min(MAX, flip ? above : below))}px`;
  }

  function _closePanel() {
    state.open = false;
    cfg.dropdown.classList.remove('open');
  }

  function _commit(value) {
    state.selected = value;
    _syncHidden();
    _syncTrigger();
    _closePanel();
    cfg.trigger.focus();
    if (state.onChange) state.onChange(value);
  }

  function _syncHidden() {
    if (cfg.hidden) cfg.hidden.value = state.selected;
  }

  function _syncTrigger() {
    const entry = state.byValue.get(state.selected);
    if (cfg.label) {
      // An unknown non-empty value (see setTree) is shown as it is, so the
      // operator can see what the config holds rather than "— select —".
      cfg.label.textContent = entry ? entry.root.label
        : (state.selected || t('label.selectOne', '— select —'));
      cfg.label.classList.toggle('tp-placeholder', !entry && !state.selected);
    }
    if (cfg.logo) {
      const logo = entry && entry.root.logo;
      if (logo) { cfg.logo.src = logo; cfg.logo.style.display = ''; }
      else { cfg.logo.src = ''; cfg.logo.style.display = 'none'; }
    }
  }

  /** One root row and, in tree view, its children. */
  function _rootHtml(root, selectedRoot) {
    // A root row stands for the child it currently owns; if the selection is
    // elsewhere, clicking falls back to the root's default child.  This keeps
    // the highlighted row meaning the same thing as a click on it.
    const ownsSelection = !!selectedRoot && selectedRoot === root;
    const rootValue = ownsSelection ? state.selected : root.defaultValue;
    let html = `<div class="tp-item tp-root${ownsSelection ? ' tp-active' : ''}" role="option" tabindex="-1" aria-selected="${ownsSelection}" data-value="${_esc(rootValue)}">`
      + _logoHtml(root.logo)
      + `<span class="tp-name">${_esc(root.label)}</span></div>`;
    if (state.view === 'tree' && root.children && root.children.length) {
      const last = root.children.length - 1;
      root.children.forEach((child, i) => {
        const guide = i === last ? '└─' : '├─';
        const active = child.value === state.selected;
        const suffix = child.label !== child.base && child.base
          ? `<span class="tp-base">${_esc(child.base)}</span>` : '';
        html += `<div class="tp-item tp-child${active ? ' tp-active' : ''}" role="option" tabindex="-1" aria-selected="${active}" data-value="${_esc(child.value)}">`
          + `<span class="tp-guide">${guide}</span>`
          + `<span class="tp-name">${_esc(child.label)}</span>${suffix}</div>`;
      });
    }
    return html;
  }

  /** A labelled section of roots; an empty section renders nothing. */
  function _sectionHtml(label, roots, selectedRoot) {
    if (!roots.length) return '';
    return `<div class="tp-group-label" role="presentation">${_esc(label)}</div>`
      + roots.map((root) => _rootHtml(root, selectedRoot)).join('');
  }

  function _renderPanel() {
    // Reflect the active view on the segmented control.
    if (cfg.viewToggle) {
      for (const btn of cfg.dropdown.querySelectorAll('.tp-view-toggle button[data-view]')) {
        const on = btn.dataset.view === state.view;
        btn.classList.toggle('active', on);
        btn.setAttribute('aria-selected', on ? 'true' : 'false');
      }
    }

    const selectedRoot = state.byValue.get(state.selected)?.root;
    // Bare base formats and vendor groups are different kinds of choice; the
    // builder marks them, and the dropdown draws them as two sections.
    const formatRoots = state.roots.filter((root) => root.kind === 'format');
    const shimRoots = state.roots.filter((root) => root.kind !== 'format');
    const html = _sectionHtml(t('label.baseFormats', 'Base formats'), formatRoots, selectedRoot)
      + _sectionHtml(t('label.providerShims', 'Provider shims'), shimRoots, selectedRoot);
    cfg.list.innerHTML = html || `<div class="tp-empty">${_esc(t('label.selectOne', '— select —'))}</div>`;

    for (const el of cfg.list.querySelectorAll('.tp-item'))
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
    return [...cfg.list.querySelectorAll('.tp-item')];
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
    // On load failure (e.g. a cross-origin logo blocked by CORP) fall back to
    // the same dashed placeholder used for logo-less roots.
    return logo
      ? `<img class="tp-logo" src="${_esc(logo)}" alt="" loading="lazy" onerror="this.classList.add('tp-logo-empty');this.removeAttribute('src')">`
      : '<span class="tp-logo tp-logo-empty" aria-hidden="true"></span>';
  }

  return { init, setTree, getValue, has, select };
}

function _esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── The LLM endpoint's picker ───────────────────────────────────────
// One instance over the markup in admin.html; the exported names below keep
// providers.js's existing calls unchanged.

let _llm = null;
let _llmOnSelect = null;

function _llmPicker() {
  if (_llm) return _llm;
  const wrapper = document.getElementById('typePickerWrapper');
  const trigger = document.getElementById('typePickerTrigger');
  const dropdown = document.getElementById('typePickerDropdown');
  const list = document.getElementById('typePickerList');
  if (!wrapper || !trigger || !dropdown || !list) return null;
  _llm = createTypePicker({
    wrapper,
    trigger,
    dropdown,
    list,
    label: document.getElementById('typePickerLabel'),
    logo: document.getElementById('provTypeLogo'),
    hidden: document.getElementById('provType'),
    viewToggle: true,
    onChange: (value) => { if (_llmOnSelect) _llmOnSelect(value); },
  });
  return _llm;
}

/** Register the selection callback (fired on user and programmatic changes). */
export function onTypeSelect(cb) {
  _llmOnSelect = cb;
  _llmPicker()?.init();
}

/** Attach trigger / toggle / outside-click handlers.  Idempotent. */
export function initTypePicker() {
  _llmPicker()?.init();
}

/** Populate the tree and select *value* ('' leaves the picker unselected). */
export function setTypeTree(roots, value) {
  _llmPicker()?.setTree(roots, value);
}

/** Current backing value (the shim / base-format name). */
export function getSelectedValue() { return _llm ? _llm.getValue() : ''; }

/** Programmatic selection; fires the onTypeSelect callback. */
export function selectValue(value) { _llm?.select(value); }

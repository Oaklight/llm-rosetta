/**
 * shim-sources-editor.js — edit `shims.sources` and `shims.install` (#1660 S9).
 *
 * The Sources view's "Manifest sources" group: one block per source the
 * gateway reads, with a checkbox per vendor, per tag and for every vendor,
 * plus a field to add a source.  The edits stay in a draft until Save, which
 * writes both keys through the same `PUT /admin/api/config/shims` the rest of
 * the view uses (config lock included) and then reloads the sources, so the
 * install happens before the toast reports it.
 *
 * Only entries qualified with a source (`contrib:opendesign`) are rewritten;
 * an unqualified entry reads every source and is shown as written.  The raw
 * `remotes` / `paths` fields below this group are untouched.
 */

import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, esc, showToast, withBusyButton } from '../core/utils.js';
import {
  DEFAULT_SOURCES, addSource, parseSelector, sameList, sharedSelectors,
  shownSources, sourceEntryKey, sourceSelection, sourceTags, withSourceSelection,
} from '../core/shim-sources.js';

/** The last source catalogue read, and the unsaved edit. */
let _cat = null;
let _draft = null;

function _section() {
  const sec = S.configData && S.configData.shims;
  return sec && typeof sec === 'object' ? sec : {};
}

/** The saved state, in the shape the draft holds. */
function _saved() {
  const section = _section();
  return { sources: shownSources(section), install: [...(section.install || [])] };
}

/** The catalogue source a `shims.sources` entry names, if it was read. */
function _catalogueFor(entry) {
  const key = sourceEntryKey(entry);
  return ((_cat && _cat.sources) || []).find((src) => key === src.key || key === src.url
    || (src.entry != null && key === sourceEntryKey(src.entry))) || null;
}

/** The id selectors use for an entry: the manifest's, or a catalogue id. */
function _idFor(entry) {
  const src = _catalogueFor(entry);
  if (src) return src.id;
  const key = sourceEntryKey(entry);
  return key.includes('://') ? null : key;
}

function _check(id, kind, value, on, label, extra = '', disabled = false) {
  return `<label class="shim-src-check${on ? ' is-on' : ''}">`
    + `<input type="checkbox" data-src="${esc(id)}" data-kind="${esc(kind)}" data-value="${esc(value)}"`
    + `${on ? ' checked' : ''}${disabled ? ' disabled' : ''} onchange="shimSourceDraftToggle(this)">`
    + `<span class="shim-src-check-name">${esc(label)}</span>`
    + `${extra ? `<span class="shim-src-check-extra">${esc(extra)}</span>` : ''}</label>`;
}

/** One source's block: its head, then what it offers to select. */
function _blockHtml(entry) {
  const key = sourceEntryKey(entry);
  const src = _catalogueFor(entry);
  const id = _idFor(entry);
  const label = src ? (src.label || src.id) : key;
  const state = !src
    ? t('shimSrc.saveToRead')
    : (!src.ok ? t('shimSrc.sourceError', {error: src.error || ''})
      : t('shimSrc.vendorsAt', {n: (src.vendors || []).length, url: src.url}));
  const head = `<div class="shim-src-head">`
    + `<span class="shim-src-name">${esc(label)}</span>`
    + `${id && id !== label ? `<code class="shim-src-id">${esc(id)}</code>` : ''}`
    + `<span class="shim-src-state${src && !src.ok ? ' is-bad' : ''}" title="${esc(state)}">${esc(state)}</span>`
    + `<button type="button" class="btn btn-sm" data-key="${esc(key)}" data-id="${esc(id || '')}"`
    + ` onclick="shimSourceDraftRemove(this)">${esc(t('shimSrc.removeSource'))}</button></div>`;
  if (!src || !src.ok || !id) return `<div class="shim-src-block">${head}</div>`;
  const sel = sourceSelection(_draft.install, id);
  const every = _check(id, 'all', '', sel.all, t('shimSrc.every', {selector: `${id}:*`}));
  const tags = sourceTags(src).map((tag) => _check(id, 'tag', tag, sel.all || sel.tags.includes(tag),
    `tag:${tag}`, '', sel.all)).join('');
  const vendors = (src.vendors || []).map((v) => {
    const preset = (v.selected_origin || '').startsWith('presets:')
      ? t('shimSrc.viaPreset', {preset: v.selected_origin.slice('presets:'.length)}) : '';
    const extra = [v.maturity, v.installable ? `v${v.installable}` : t('shimSrc.noInstallable'), preset]
      .filter(Boolean).join(' · ');
    return _check(id, 'vendor', v.vendor, sel.all || sel.vendors.includes(v.vendor), v.vendor, extra, sel.all);
  }).join('');
  return `<div class="shim-src-block">${head}`
    + `<div class="shim-src-row">${every}</div>`
    + (tags ? `<div class="shim-src-row"><span class="shim-src-label">${esc(t('shimSrc.tags'))}</span>${tags}</div>` : '')
    + `<div class="shim-src-row shim-src-vendors"><span class="shim-src-label">${esc(t('shimSrc.vendors'))}</span>${vendors}</div>`
    + `</div>`;
}

/** Repaint the group from the draft. */
function _paint() {
  const host = document.getElementById('shimManifestEditor');
  if (!host) return;
  const shared = sharedSelectors(_draft.install);
  const core = _cat && _cat.core;
  const known = DEFAULT_SOURCES.map((k) => `<option value="${esc(k)}"></option>`).join('');
  const blocks = _draft.sources.map(_blockHtml).join('');
  const sharedHtml = shared.length
    ? `<p class="field-hint">${esc(t('shimSrc.shared', {list: shared.join(', ')}))}</p>` : '';
  const coreHtml = core
    ? `<p class="field-hint">${esc(t('shimSrc.core', {surface: core.surface, ir: core.ir}))}</p>` : '';
  const addHtml = `<div class="shim-src-add">`
    + `<input type="text" id="shimSourceAddInput" class="input is-sm" list="shimSourceKnown" spellcheck="false"`
    + ` autocomplete="off" placeholder="${esc(t('shimSrc.addPlaceholder'))}" aria-label="${esc(t('shimSrc.addSource'))}"`
    + ` onkeydown="if(event.key==='Enter'){event.preventDefault();shimSourceDraftAdd();}">`
    + `<datalist id="shimSourceKnown">${known}</datalist>`
    + `<button type="button" class="btn btn-sm" onclick="shimSourceDraftAdd()">${esc(t('shimSrc.addSource'))}</button></div>`;
  // A save goes through the config writer, which rewrites the whole file.
  const noteHtml = `<p class="field-hint shim-src-rewrite">${esc(t('shimSrc.rewriteNote'))}</p>`;
  const html = blocks + sharedHtml + addHtml + coreHtml + noteHtml;
  host.innerHTML = html;
  _paintDirty();
}

function _paintDirty() {
  const saved = _saved();
  const dirty = !sameList(saved.sources, _draft.sources) || !sameList(saved.install, _draft.install);
  const note = document.getElementById('shimManifestDirty');
  if (note) note.hidden = !dirty;
  const btn = document.getElementById('shimManifestSave');
  if (btn) btn.disabled = !dirty;
}

/**
 * Paint the editor, reading the source catalogue when it is not loaded yet.
 *
 * @param {boolean} [refresh] - Re-read every manifest (the view's Refetch).
 */
export async function renderShimManifestEditor(refresh = false) {
  if (!document.getElementById('shimManifestEditor')) return;
  if (!_draft) _draft = _saved();
  if (!_cat || refresh) {
    try {
      const r = await api.get('/admin/api/config/shims/source-catalog' + (refresh ? '?refresh=1' : ''));
      _cat = r && Array.isArray(r.sources) ? r : null;
    } catch {
      _cat = null;
    }
  }
  _paint();
}

/**
 * Drop the cached catalogue, and the draft unless *keepDraft*: the saved
 * config changed (a save), or only the manifests did (a refetch).
 */
export function resetShimManifestEditor({ keepDraft = false } = {}) {
  if (!keepDraft) _draft = null;
  _cat = null;
}

/** A checkbox in a source block changed. */
export function shimSourceDraftToggle(input) {
  const id = input.dataset.src;
  const kind = input.dataset.kind;
  const value = input.dataset.value;
  const sel = sourceSelection(_draft.install, id);
  if (kind === 'all') {
    sel.all = input.checked;
  } else {
    const list = kind === 'tag' ? sel.tags : sel.vendors;
    const at = list.indexOf(value);
    if (input.checked && at < 0) list.push(value);
    if (!input.checked && at >= 0) list.splice(at, 1);
  }
  _draft.install = withSourceSelection(_draft.install, id, sel);
  _paint();
}

/** Add the source typed in the field: an id or a manifest URL. */
export function shimSourceDraftAdd() {
  const input = document.getElementById('shimSourceAddInput');
  const text = (input && input.value || '').trim();
  if (!text) return;
  if (!text.includes('://') && !/^[a-z0-9][a-z0-9_-]*$/.test(text)) {
    showToast(t('shimSrc.badSource'), 'error');
    return;
  }
  const { sources, added } = addSource({ sources: _draft.sources }, text);
  if (!added) {
    showToast(t('shimSrc.duplicate'), 'error');
    return;
  }
  _draft.sources = sources;
  _paint();
}

/** Remove one source, and the entries qualified with its id. */
export function shimSourceDraftRemove(btn) {
  const key = btn.dataset.key;
  const id = btn.dataset.id;
  _draft.sources = _draft.sources.filter((s) => sourceEntryKey(s) !== key);
  if (id) {
    _draft.install = _draft.install.filter((raw) => {
      const sel = parseSelector(raw);
      return !(sel && sel.source === id);
    });
  }
  _paint();
}

/**
 * Save the draft: `shims.sources` and `shims.install` through the config
 * save path, then reload so the selection installs.
 *
 * The source list is written whenever it differs from the DECLARED one, so
 * a config that only implied the defaults gets them written out: the saved
 * file names every source it reads and does not depend on the built-in list.
 */
export async function saveShimManifestSources(btn) {
  const section = _section();
  const declared = Array.isArray(section.sources) ? section.sources : [];
  const body = { install: _draft.install };
  if (!sameList(declared, _draft.sources)) body.sources = _draft.sources;
  await withBusyButton(btn, async () => {
    try {
      // `api.put` resolves with the body on a 400 too: a refused write must
      // fail here, or the toast would report a selection the gateway refused.
      const w = await api.put('/admin/api/config/shims', body);
      if (w && (w.ok === false || w.error)) throw new Error(w.error || t('toast.error'));
      const r = await api.post('/admin/api/config/shims/reload', {});
      if (r && (r.ok === false || r.error)) throw new Error(r.error || t('toast.error'));
      await window.loadConfig();
      resetShimManifestEditor();
      // Repaints the page, and with it this group from the saved config.
      await window.loadShimsPage?.();
      const failures = (r && r.status && r.status.failure_count) || 0;
      showToast(failures
        ? t('toast.shimsReloadFailed', {count: failures})
        : t('shimSrc.saved'), failures ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

Object.assign(window, {
  renderShimManifestEditor, resetShimManifestEditor, shimSourceDraftToggle,
  shimSourceDraftAdd, shimSourceDraftRemove, saveShimManifestSources,
});

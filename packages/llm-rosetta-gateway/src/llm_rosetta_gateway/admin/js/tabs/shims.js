// shims.js — External shim sources (Settings tab).
//
// Lets the operator declare extra shim directories, preload a catalogue
// package by name, and add remote shim archives from the admin UI.  The
// fields map 1:1 onto the top-level `shims` config section; the status table
// reports what the gateway loaded; the inventory below is grouped by it.
import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, esc, copyText, openModal, closeModal } from '../core/utils.js';

function _section() {
  const sec = S.configData && S.configData.shims;
  return sec && typeof sec === 'object' ? sec : {};
}

function _status() {
  return (S.configData && S.configData.shim_sources) || { sources: [] };
}

/** Providers still on a built-in shim with a removal scheduled. */
function _removals() {
  return (S.configData && S.configData.shim_removals) || [];
}

/** The catalogue of standalone shim packages, shown as presets. */
function _presets() {
  return (S.configData && S.configData.shim_source_presets) || [];
}

/** Fill the form from the loaded config and render the status table. */
function renderShimSources() {
  const pathsEl = document.getElementById('shimPathsInput');
  if (!pathsEl) return;
  const sec = _section();
  pathsEl.value = Array.isArray(sec.paths) ? sec.paths.join('\n') : '';
  const presetsEl = document.getElementById('shimPresetsInput');
  if (presetsEl) {
    presetsEl.value = Array.isArray(sec.presets) ? sec.presets.join(', ') : '';
  }
  const remEl = document.getElementById('shimRemotesInput');
  if (remEl) {
    remEl.value = Array.isArray(sec.remotes) && sec.remotes.length
      ? JSON.stringify(sec.remotes, null, 2)
      : '';
  }
  const cacheEl = document.getElementById('shimCacheDirInput');
  if (cacheEl) cacheEl.value = sec.cache_dir || '';
  const ttlEl = document.getElementById('shimTtlInput');
  if (ttlEl) ttlEl.value = sec.remote_ttl != null ? sec.remote_ttl : '';
  const strictEl = document.getElementById('shimStrictInput');
  if (strictEl) strictEl.checked = sec.strict === true;
  const err = document.getElementById('shimSourcesError');
  if (err) err.textContent = '';
  _renderPresets();
  _renderAdvisories();
  loadLoadedShims();
}

/** Persist the shim sources and reload the config. */
async function saveShimSources() {
  const err = document.getElementById('shimSourcesError');
  if (err) err.textContent = '';

  const paths = (document.getElementById('shimPathsInput').value || '')
    .split('\n').map(s => s.trim()).filter(Boolean);

  const presets = (document.getElementById('shimPresetsInput').value || '')
    .split(/[\s,]+/).map(s => s.trim()).filter(Boolean);

  let remotes = [];
  const rawRemotes = (document.getElementById('shimRemotesInput').value || '').trim();
  if (rawRemotes) {
    try {
      remotes = JSON.parse(rawRemotes);
      if (!Array.isArray(remotes)) throw new Error('not an array');
    } catch {
      if (err) err.textContent = t('shim.error.remotesJson');
      return;
    }
  }

  try {
    const body = { paths, remotes, presets };
    const cacheDir = (document.getElementById('shimCacheDirInput').value || '').trim();
    body.cache_dir = cacheDir;
    const ttlRaw = (document.getElementById('shimTtlInput').value || '').trim();
    body.remote_ttl = ttlRaw === '' ? null : Number(ttlRaw);
    body.strict = document.getElementById('shimStrictInput').checked;
    await api.put('/admin/api/config/shims', body);
    await window.loadConfig();
    renderShimSources();
    showToast(t('toast.shimsSaved'));
  } catch (e) {
    const msg = String(e.message || e);
    if (err) err.textContent = msg;
    else showToast(msg, 'error');
  }
}

/** Re-apply the declared sources; force a remote refetch on demand. */
async function reloadShimSources(forceRefresh) {
  try {
    const r = await api.post('/admin/api/config/shims/reload',
      forceRefresh ? { refresh: 'always' } : {});
    await window.loadConfig();
    renderShimSources();
    const failures = (r && r.status && r.status.failure_count) || 0;
    const providerWarnings = (r && r.provider_warnings) || [];
    if (providerWarnings.length) {
      showToast(t('toast.providersAutoDisabled', {
        providers: providerWarnings.map(x => x.provider).join(', ')
      }), 'error');
    } else if (failures) {
      showToast(t('toast.shimsReloadFailed', { count: failures }), 'error');
    } else {
      showToast(t('toast.shimsReloaded'));
    }
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/** Offer the standalone shim packages the gateway knows about. */
function _renderPresets() {
  const sel = document.getElementById('shimPresetSelect');
  if (!sel) return;
  const presets = _presets();
  if (!presets.length) {
    sel.innerHTML = `<option value="">${esc(t('shim.preset.select'))}</option>`;
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  sel.innerHTML = presets.map((p, i) => {
    const label = t('shim.preset.option', {
      label: p.label || p.source_repo,
      count: Array.isArray(p.shims) ? p.shims.length : 0,
    });
    return `<option value="${i}">${esc(label)}</option>`;
  }).join('');
}

/** List the providers that are still on a shim with a removal scheduled. */
function _renderAdvisories() {
  const host = document.getElementById('shimRemovalAdvisories');
  if (!host) return;
  const removals = _removals();
  if (!removals.length) {
    host.innerHTML = '';
    return;
  }
  const title = `<div style="color:var(--orange,#e67e22);margin-bottom:4px">${esc(t('shim.removal.title'))}</div>`;
  host.innerHTML = title + removals.map(a => `<div style="display:flex;gap:8px;align-items:center;padding:3px 0;border-top:1px solid var(--border)">
      <span style="flex:1">${esc(t('shim.removal.line', { provider: a.provider, shim: a.shim, version: a.removed_in }))}</span>
      <button class="btn btn-sm" data-url="${esc(a.remote.url)}" onclick="addShimPresetFromRow(this)">${esc(t('shim.removal.fix'))}</button>
    </div>`).join('');
}

/**
 * Append a catalogue source to the remotes textarea.
 *
 * Deliberately only edits the textarea: the operator sees exactly what would be
 * written, and the existing Save stays the one thing that persists (and
 * validates) anything.
 */
function appendShimPreset(entry) {
  const remEl = document.getElementById('shimRemotesInput');
  if (!remEl || !entry || !entry.remote) return;
  const err = document.getElementById('shimSourcesError');
  let remotes = [];
  const raw = (remEl.value || '').trim();
  if (raw) {
    try {
      remotes = JSON.parse(raw);
      if (!Array.isArray(remotes)) throw new Error('not an array');
    } catch {
      if (err) err.textContent = t('shim.error.remotesJson');
      return;
    }
  }
  if (remotes.some(r => r && r.url === entry.remote.url)) {
    showToast(t('shim.preset.duplicate'));
    return;
  }
  remotes.push({ ...entry.remote });
  remEl.value = JSON.stringify(remotes, null, 2);
  if (err) err.textContent = '';
  showToast(t('shim.preset.added', { name: entry.label || entry.remote.url }));
}

/** "Add source" on the preset picker. */
function addShimPresetFromSelect() {
  const sel = document.getElementById('shimPresetSelect');
  const entry = _presets()[Number(sel && sel.value)];
  if (entry) appendShimPreset(entry);
}

/** "Add the replacement source" on one advisory row. */
function addShimPresetByUrl(url) {
  const entry = _presets().find(p => p.remote && p.remote.url === url);
  if (entry) appendShimPreset(entry);
}

/**
 * The same, for a button that carries its URL in a `data-url` attribute.
 *
 * Reading the attribute keeps the URL out of the `onclick` string, so `esc()`
 * only has to satisfy the HTML parser: a value containing a quote cannot escape
 * into the JavaScript that runs.  The picker passes an index for the same
 * reason, and this is the one place a config-derived string reaches a handler.
 */
function addShimPresetFromRow(button) {
  const url = button && button.dataset ? button.dataset.url : null;
  if (url) addShimPresetByUrl(url);
}


// ── Loaded shims ──────────────────────────────────────────────────────
//
// The sources above say where shims come from; this list says what actually
// loaded and where each definition lives.  A shim in a declared local
// directory is editable here — the operator already vouched for that
// directory by listing it.  A built-in ships in the package and a remote one
// is a fetched artifact, so both are shown read-only.

let _shimRows = [];
let _shimDetail = null;

/** Fetch the inventory and repaint the list. */
async function loadLoadedShims() {
  if (!document.getElementById('shimInventoryList')) return;
  try {
    const d = await api.get('/admin/api/config/shims');
    _shimRows = Array.isArray(d.shims) ? d.shims : [];
  } catch {
    _shimRows = [];
  }
  renderLoadedShims();
}

/**
 * Group the inventory by the source each shim came from.
 *
 * The source report and the shim list answer one question — what loaded, and
 * where from — so they are one list: a group per declared source, carrying
 * that source's state, resolved ref or error, and its shims.  Built-ins, which
 * no source declares, form the last group.
 */
function _sourceGroups() {
  const bySource = new Map();
  for (const r of _shimRows) {
    const key = r.source || '';
    if (!bySource.has(key)) bySource.set(key, []);
    bySource.get(key).push(r);
  }
  const groups = [];
  for (const st of _status().sources || []) {
    groups.push({
      origin: st.kind, source: st.location, status: st,
      rows: bySource.get(st.location) || [],
    });
    bySource.delete(st.location);
  }
  for (const [source, rows] of bySource) {
    groups.push({ origin: rows[0].origin, source, status: null, rows });
  }
  const rank = { path: 0, remote: 1, builtin: 2 };
  groups.sort((a, b) =>
    (rank[a.origin] - rank[b.origin]) || String(a.source).localeCompare(String(b.source)));
  return groups;
}

/** Paint the list for the current filter. */
function renderLoadedShims() {
  const host = document.getElementById('shimInventoryList');
  if (!host) return;
  const q = (document.getElementById('shimFilterInput')?.value || '').trim().toLowerCase();
  const showBuiltin = document.getElementById('shimShowBuiltin')?.checked !== false;
  const match = r => {
    if (!showBuiltin && r.origin === 'builtin') return false;
    if (!q) return true;
    if (r.name.toLowerCase().includes(q)) return true;
    return (r.used_by || []).some(p => p.toLowerCase().includes(q));
  };
  const filtering = !!q;
  const groups = _sourceGroups()
    .map(g => ({ ...g, shown: g.rows.filter(match) }))
    .filter(g => {
      // A source that failed has nothing to show but is exactly what the
      // operator needs to see, so it survives the filter.
      if (g.shown.length || (g.status && !g.status.ok)) return true;
      // With no filter, a declared source that loaded nothing still gets a
      // head: an empty directory is something the operator has to notice.
      const hiddenBuiltin = g.origin === 'builtin' && !showBuiltin;
      return !filtering && !hiddenBuiltin;
    });
  const shownCount = groups.reduce((n, g) => n + g.shown.length, 0);
  const count = document.getElementById('shimInventoryCount');
  if (count) count.textContent = shownCount === _shimRows.length
    ? String(_shimRows.length)
    : `${shownCount} / ${_shimRows.length}`;
  if (!groups.length) {
    host.innerHTML = `<div class="shim-empty">${esc(t('shim.inventory.empty'))}</div>`;
    return;
  }
  host.innerHTML = groups.map(g => {
    const st = g.status;
    const failed = st ? !st.ok : false;
    const state = `${failed ? t('shim.status.failed') : t('shim.status.loaded')}`
      + (st && st.stale ? ` \u00b7 ${t('shim.status.stale')}` : '');
    const note = failed ? (st.error || '') : (st && st.resolved_ref ? String(st.resolved_ref) : '');
    const where = g.source || t('shim.group.builtin');
    return `<div class="shim-group">
      <div class="shim-group-head">
        <span class="shim-group-state ${failed ? 'bad' : 'ok'}">${esc(state)}</span>
        <span class="shim-pill ${esc(g.origin)}">${esc(t('shim.origin.' + g.origin))}</span>
        <span class="shim-group-source" title="${esc(where)}">${esc(where)}</span>
        <span class="shim-group-count">${esc(t('shim.status.count', { count: g.shown.length }))}</span>
      </div>
      ${note ? `<div class="shim-group-note" title="${esc(note)}">${esc(note)}</div>` : ''}
      ${g.shown.map(_shimRowHtml).join('')}
    </div>`;
  }).join('');
}

/** One inventory row. */
function _shimRowHtml(r) {
  const used = (r.used_by || []).length
    ? t('shim.usedBy', { providers: r.used_by.join(', ') })
    : t('shim.unused');
  const meta = `${r.model_type} \u00b7 ${r.format} \u00b7 ${used}`;
  const local = r.origin === 'path';
  const off = r.enabled === false;
  const state = off
    ? `<span class="shim-pill off">${esc(t('shim.state.disabled'))}</span>`
    : '';
  const toggle = `<button class="btn btn-sm" data-shim="${esc(r.name)}" data-enable="${off ? '1' : ''}" onclick="toggleShimEnabled(this.dataset.shim, this.dataset.enable === '1')">${esc(t(off ? 'shim.enable' : 'shim.disable'))}</button>`;
  const del = r.deletable
    ? `<button class="btn btn-sm btn-danger" data-shim="${esc(r.name)}" onclick="deleteLoadedShim(this.dataset.shim)">${esc(t('shim.delete'))}</button>`
    : '';
  return `<div class="shim-row${off ? ' is-disabled' : ''}">
    <div class="shim-row-main">
      <div class="shim-row-id">
        <code class="shim-name">${esc(r.name)}</code>${state}
      </div>
      <div class="shim-row-meta" title="${esc(r.path || r.source || '')}">${esc(meta)}</div>
    </div>
    <div class="shim-row-actions">
      <button class="btn btn-sm" data-shim="${esc(r.name)}" onclick="openShimDetail(this.dataset.shim)">${esc(t(local ? 'shim.edit' : 'shim.open'))}</button>
      ${toggle}${del}
    </div>
  </div>`;
}

/** Open one definition: editable for a local shim, read-only otherwise. */
async function openShimDetail(name) {
  try {
    const d = await api.get('/admin/api/config/shims/' + encodeURIComponent(name));
    _shimDetail = d;
    document.getElementById('shimModalTitle').textContent = d.name;
    document.getElementById('shimModalMeta').textContent =
      [t('shim.origin.' + d.origin), d.path || d.source || ''].filter(Boolean).join(' \u00b7 ');
    const ta = document.getElementById('shimModalYaml');
    ta.value = d.yaml || '';
    ta.readOnly = !d.editable;
    document.getElementById('shimModalSave').style.display = d.editable ? '' : 'none';
    document.getElementById('shimModalNote').textContent = d.editable
      ? t('shim.editHint')
      : (d.read_only_reason || '');
    openModal('shimModal');
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/** Save the edited definition; the server rolls back if it does not load. */
async function saveShimYaml() {
  const name = _shimDetail && _shimDetail.name;
  if (!name) return;
  const yaml = document.getElementById('shimModalYaml').value;
  const btn = document.getElementById('shimModalSave');
  if (btn) btn.disabled = true;
  try {
    const res = await api.put('/admin/api/config/shims/' + encodeURIComponent(name), { yaml });
    if (res && res.error) { showToast(res.error, 'error'); return; }
    closeModal('shimModal');
    showToast(t('toast.saved'));
    await loadLoadedShims();
    await window.loadConfig();
  } catch (e) {
    showToast(String(e.message || e), 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

/** Copy the open definition, as a starting point for a local shim. */
async function copyShimYaml() {
  const ta = document.getElementById('shimModalYaml');
  if (ta) await copyText(ta.value, t('shim.copied'));
}

/** Turn one shim off or back on (a config switch, not a deletion). */
async function toggleShimEnabled(name, enable) {
  try {
    const res = await api.post(
      '/admin/api/config/shims/' + encodeURIComponent(name) + '/enabled',
      { enabled: !!enable });
    if (res && res.error) { showToast(res.error, 'error'); return; }
    showToast(t(enable ? 'shim.enabled' : 'shim.disabled'));
    await loadLoadedShims();
    await window.loadConfig();
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/** Remove a local shim's directory, once nothing references it. */
async function deleteLoadedShim(name) {
  if (!confirm(t('shim.delete.confirm', { name }))) return;
  try {
    const res = await api.del('/admin/api/config/shims/' + encodeURIComponent(name));
    if (res && res.error) { showToast(res.error, 'error'); return; }
    showToast(t('toast.deleted', { count: 1 }));
    await loadLoadedShims();
    await window.loadConfig();
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

Object.assign(window, {
  renderShimSources, saveShimSources, reloadShimSources,
  addShimPresetFromSelect, addShimPresetFromRow, appendShimPreset,
  loadLoadedShims, renderLoadedShims, openShimDetail, saveShimYaml,
  copyShimYaml, deleteLoadedShim, toggleShimEnabled,
});

export { renderShimSources, saveShimSources, reloadShimSources, appendShimPreset, loadLoadedShims };

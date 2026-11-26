// shims.js — External shim sources (Settings tab).
//
// Lets the operator set the extension-index cache settings from the admin UI;
// index sources are edited by shim-sources-editor.js.  The fields map 1:1 onto
// the top-level `shims` config section; the status table reports what the
// gateway loaded; the inventory below is grouped by it.
import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, esc, copyText, openModal } from '../core/utils.js';
import { inheritanceRows, nameList } from '../core/manifest-inherit.js';
import { renderShimManifestEditor, resetShimManifestEditor } from './shim-sources-editor.js';

function _section() {
  const sec = S.configData && S.configData.shims;
  return sec && typeof sec === 'object' ? sec : {};
}

function _status() {
  return (S.configData && S.configData.shim_sources) || { sources: [] };
}

/** Fill the form from the loaded config and render the status table. */
function renderShimSources() {
  const cacheEl = document.getElementById('shimCacheDirInput');
  if (!cacheEl) return;
  const sec = _section();
  cacheEl.value = sec.cache_dir || '';
  checkShimUpdates(false);
  const ttlEl = document.getElementById('shimTtlInput');
  if (ttlEl) ttlEl.value = sec.remote_ttl != null ? sec.remote_ttl : '';
  const strictEl = document.getElementById('shimStrictInput');
  if (strictEl) strictEl.checked = sec.strict === true;
  const err = document.getElementById('shimSourcesError');
  if (err) err.textContent = '';
  // Last: every field above must be filled before the form is compared with
  // the saved section, or a config with a cache/TTL/strict value opens dirty.
  _updateShimDirty();
  loadLoadedShims();
  // The manifest sources and their selection (#1660 S9).
  renderShimManifestEditor();
}

/** Persist the shim sources and reload the config. */
async function saveShimSources() {
  const err = document.getElementById('shimSourcesError');
  if (err) err.textContent = '';

  try {
    const body = {};
    const cacheDir = (document.getElementById('shimCacheDirInput').value || '').trim();
    body.cache_dir = cacheDir;
    const ttlRaw = (document.getElementById('shimTtlInput').value || '').trim();
    body.remote_ttl = ttlRaw === '' ? null : Number(ttlRaw);
    body.strict = document.getElementById('shimStrictInput').checked;
    const r = await api.put('/admin/api/config/shims', body);
    // `api.put` resolves with the body even on 400; without this check the
    // server's refusal (a bad TTL, say) was shown nowhere and the
    // reload below silently reverted the operator's pick.
    if (r && r.error) throw new Error(r.error);
    await window.loadConfig();
    renderShimSources();
    showToast(t('toast.shimsSaved'));
  } catch (e) {
    const msg = String(e.message || e);
    if (err) err.textContent = msg;
    else showToast(msg, 'error');
  }
}

/** Re-apply the declared sources; force a manifest refetch on demand. */
async function reloadShimSources(forceRefresh) {
  try {
    const r = await api.post('/admin/api/config/shims/reload',
      forceRefresh ? { refresh: 'always' } : {});
    // A refetch rewrote the manifest cache; the selection being edited stays.
    resetShimManifestEditor({ keepDraft: true });
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

/**
 * Whether the form holds changes the saved config does not.
 *
 * Save is the only thing that persists, so the button says so: disabled when
 * there is nothing to write, with a "Unsaved changes" note beside it while
 * there is.  Comparing the form against the saved section (rather than
 * counting keystrokes) means editing a value back to what was saved clears
 * the note again.
 */
function _shimFormDirty() {
  const sec = _section();
  const el = id => document.getElementById(id);
  if ((el('shimCacheDirInput').value || '').trim() !== (sec.cache_dir || '')) return true;
  const ttl = (el('shimTtlInput').value || '').trim();
  const savedTtl = sec.remote_ttl != null ? String(sec.remote_ttl) : '';
  if (ttl !== savedTtl) return true;
  if (el('shimStrictInput').checked !== (sec.strict === true)) return true;
  return false;
}

/** Repaint the Save button and the unsaved note. */
function _updateShimDirty() {
  const dirty = _shimFormDirty();
  const note = document.getElementById('shimSourcesDirty');
  if (note) note.hidden = !dirty;
  const btn = document.getElementById('shimSaveBtn');
  if (btn) btn.disabled = !dirty;
}

// ── Loaded shims ──────────────────────────────────────────────────────
//
// The sources above say where shims come from; this list says what actually
// loaded and where each definition lives.  A built-in ships in the package and
// an installed extension is a fetched artifact, so both are shown read-only.

let _shimRows = [];
// Vendor manifests ride in the same payload under their own key: they have no
// shim name, so they are never entries of `_shimRows`.
let _manifests = [];
// location -> {current, head, update, error}, from the last check.
let _shimUpdates = new Map();

/** Fetch the inventory and repaint the list. */
async function loadLoadedShims() {
  if (!document.getElementById('shimInventoryList')) return;
  try {
    const d = await api.get('/admin/api/config/shims');
    _shimRows = Array.isArray(d.shims) ? d.shims : [];
    _manifests = Array.isArray(d.manifests) ? d.manifests : [];
  } catch {
    _shimRows = [];
    _manifests = [];
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
  const rank = { remote: 0, builtin: 1 };
  groups.sort((a, b) =>
    (rank[a.origin] - rank[b.origin]) || String(a.source).localeCompare(String(b.source)));
  return groups;
}

/**
 * The update verdict for an installed vendor's group head.
 *
 * `update` is true when the manifest lists a newer version this core can
 * install.  Nothing is claimed before the check runs, and a failed lookup says
 * so instead of "up to date".
 */
function _updateBadge(group) {
  if (group.origin !== 'remote') return '';
  const u = _shimUpdates.get(group.source);
  if (!u) return '';
  if (u.error) {
    return `<span class="shim-update unknown"
      title="${esc(u.error)}">${esc(t('shim.update.unknown'))}</span>`;
  }
  if (!u.update) return `<span class="shim-update ok">${esc(t('shim.update.current'))}</span>`;
  const from = String(u.current || '');
  const to = String(u.head || '');
  return `<button type="button" class="shim-update new" onclick="reloadShimSources(true)"
    title="${esc(t('shim.update.tip', { from, to }))}"
    >${esc(t('shim.update.available', { from, to }))}</button>`;
}

/**
 * Ask the gateway whether any installed vendor has a newer installable version.
 *
 * Cached for five minutes on the server; *force* is the panel's "Check now".
 * A failure leaves the badges as they were rather than showing a verdict that
 * was never read.
 */
async function checkShimUpdates(force) {
  try {
    const d = await api.post('/admin/api/config/shims/check-updates', force ? { refresh: true } : {});
    // A failed answer carries no `sources`; reading that as "nothing to
    // update" would empty the map and clear every badge — the opposite of
    // what this function promises to do on a failure.
    if (!d || !d.ok) return;
    const rows = (d && d.sources) || [];
    _shimUpdates = new Map(rows.map(r => [r.location, r]));
    renderLoadedShims();
  } catch {
    /* the badges simply stay as they were */
  }
}

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
      if (g.shown.length || (g.status && (!g.status.ok || g.status.skipped))) return true;
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
    // A manifest vendor that was selected but not installed says why (#1660).
    const note = failed ? (st.error || '')
      : (st && st.skipped ? String(st.skipped)
        : (st && st.resolved_ref ? String(st.resolved_ref) : ''));
    const where = g.source || t('shim.group.builtin');
    return `<div class="shim-group">
      <div class="shim-group-head">
        <span class="shim-group-state ${failed ? 'bad' : 'ok'}">${esc(state)}</span>
        <span class="shim-pill ${esc(g.origin)}">${esc(t('shim.origin.' + g.origin))}</span>
        <span class="shim-group-source" title="${esc(where)}">${esc(where)}</span>
        <span class="shim-group-count">${esc(t('shim.status.count', { count: g.shown.length }))}</span>
        ${_updateBadge(g)}
      </div>
      ${note ? `<div class="shim-group-note" title="${esc(note)}">${esc(note)}</div>` : ''}
      ${_groupBodyHtml(g.shown, q, showBuiltin)}
    </div>`;
  }).join('');
}

/**
 * One source group's rows, with each vendor manifest above its leaves.
 *
 * A manifest is not a shim: it has no name to resolve and no enable switch of
 * its own.  It heads the leaves it feeds, so the defaults they share are
 * visible where they apply rather than only inside the editor.
 */
function _groupBodyHtml(rows, query, showBuiltin) {
  const byPath = new Map(_manifests.map(m => [m.path, m]));
  const blocks = [];
  const seen = new Map();
  for (const row of rows) {
    const manifest = row.manifest_path ? byPath.get(row.manifest_path) : null;
    if (!manifest) {
      blocks.push({ manifest: null, rows: [row] });
      continue;
    }
    if (!seen.has(manifest.path)) {
      const block = { manifest, rows: [] };
      seen.set(manifest.path, block);
      blocks.push(block);
    }
    seen.get(manifest.path).rows.push(row);
  }
  // A manifest whose leaves are all filtered out still answers the filter
  // when the filter names the vendor.
  for (const manifest of byPath.values()) {
    if (seen.has(manifest.path)) continue;
    if (!query || !String(manifest.vendor || '').toLowerCase().includes(query)) {
      continue;
    }
    if (manifest.origin === 'builtin' && !showBuiltin) continue;
    blocks.push({ manifest, rows: [] });
  }
  return blocks.map(b => (b.manifest
    ? `<div class="shim-vendor">${_manifestRowHtml(b.manifest)}`
      + `${b.rows.map(_shimRowHtml).join('')}</div>`
    : b.rows.map(_shimRowHtml).join(''))).join('');
}

/** The vendor manifest's row: a heading for the leaves below it. */
function _manifestRowHtml(m) {
  const count = (m.leaves || []).length;
  const meta = `${t('shim.manifest.meta', { count })} \u00b7 ${m.path || ''}`;
  return `<div class="shim-row shim-manifest-row">
    <div class="shim-row-main">
      <div class="shim-row-id">
        <code class="shim-name">${esc(m.vendor || '')}/</code>
        <span class="shim-pill manifest">${esc(t('shim.manifest.pill'))}</span>
      </div>
      <div class="shim-row-meta" title="${esc(meta)}">${esc(meta)}</div>
    </div>
    <div class="shim-row-actions">
      <button class="btn btn-sm" data-manifest="${esc(m.path)}" onclick="openManifestDetail(this.dataset.manifest)">${esc(t('shim.open'))}</button>
    </div>
  </div>`;
}

/** One inventory row. */
function _shimRowHtml(r) {
  const used = (r.used_by || []).length
    ? t('shim.usedBy', { providers: r.used_by.join(', ') })
    : t('shim.unused');
  const meta = `${r.model_type} \u00b7 ${r.format} \u00b7 ${used}`;
  const off = r.enabled === false;
  const state = off
    ? `<span class="shim-pill off">${esc(t('shim.state.disabled'))}</span>`
    : '';
  const toggle = `<button class="btn btn-sm" data-shim="${esc(r.name)}" data-enable="${off ? '1' : ''}" onclick="toggleShimEnabled(this.dataset.shim, this.dataset.enable === '1')">${esc(t(off ? 'shim.enable' : 'shim.disable'))}</button>`;
  return `<div class="shim-row${off ? ' is-disabled' : ''}">
    <div class="shim-row-main">
      <div class="shim-row-id">
        <code class="shim-name">${esc(r.name)}</code>${state}
      </div>
      <div class="shim-row-meta" title="${esc(r.path || r.source || '')}">${esc(meta)}</div>
    </div>
    <div class="shim-row-actions">
      <button class="btn btn-sm" data-shim="${esc(r.name)}" onclick="openShimDetail(this.dataset.shim)">${esc(t('shim.open'))}</button>
      ${toggle}
    </div>
  </div>`;
}

/** Open one definition, read-only. */
async function openShimDetail(name) {
  try {
    const d = await api.get('/admin/api/config/shims/' + encodeURIComponent(name));
    document.getElementById('shimModalInherit').hidden = true;
    document.getElementById('shimModalTitle').textContent = d.name;
    document.getElementById('shimModalMeta').textContent =
      [t('shim.origin.' + d.origin), d.path || d.source || ''].filter(Boolean).join(' \u00b7 ');
    const ta = document.getElementById('shimModalYaml');
    ta.value = d.yaml || '';
    document.getElementById('shimModalNote').textContent = d.read_only_reason || '';
    openModal('shimModal');
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/** Copy the open definition. */
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

// ── Vendor manifests ──────────────────────────────────────────────────
//
// A manifest sets the connection defaults every leaf below it inherits.  It
// opens in the same modal as a shim, plus an inheritance panel: which leaves
// take each field from here and which override it.

/** Open one vendor manifest for viewing. */
async function openManifestDetail(path) {
  try {
    const d = await api.get(
      '/admin/api/config/shim-manifests/detail?path=' + encodeURIComponent(path));
    if (d && d.error) throw new Error(d.error);
    document.getElementById('shimModalTitle').textContent = (d.vendor || '') + '/';
    document.getElementById('shimModalMeta').textContent =
      [t('shim.origin.' + d.origin), d.path || ''].filter(Boolean).join(' \u00b7 ');
    const ta = document.getElementById('shimModalYaml');
    ta.value = d.yaml || '';
    document.getElementById('shimModalNote').textContent = d.read_only_reason || '';
    _paintManifestRows(inheritanceRows(d.fields), (d.leaves || []).length);
    document.getElementById('shimModalInherit').hidden = false;
    openModal('shimModal');
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
}

/**
 * Paint the inheritance rows.
 *
 * Rows are built as nodes rather than markup, so a leaf name is text and never
 * something the parser could read back as HTML.
 */
function _paintManifestRows(rows, total) {
  const list = document.getElementById('shimModalInheritList');
  if (!list) return;
  list.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'shim-inherit-empty';
    empty.textContent = t('shim.manifest.none');
    list.append(empty);
  }
  for (const row of rows) {
    const el = document.createElement('div');
    el.className = 'shim-inherit-row';
    el.dataset.field = row.id;
    const name = document.createElement('code');
    name.className = 'shim-inherit-field';
    name.textContent = row.label;
    const who = document.createElement('span');
    who.className = 'shim-inherit-who';
    const parts = [];
    if (row.inherited.length) {
      parts.push(t('shim.manifest.inheritedBy', { names: nameList(row.inherited) }));
    }
    if (row.overridden.length) {
      parts.push(t('shim.manifest.overriddenBy', { names: nameList(row.overridden) }));
    }
    who.textContent = parts.join(' \u00b7 ');
    el.append(name, who);
    list.append(el);
  }
  const summary = document.getElementById('shimModalInheritSummary');
  if (!summary) return;
  summary.textContent = t('shim.manifest.leafCount', { count: total });
}

Object.assign(window, {
  renderShimSources, saveShimSources, reloadShimSources, checkShimUpdates,
  loadLoadedShims, renderLoadedShims, openShimDetail,
  copyShimYaml, toggleShimEnabled, openManifestDetail,
  // Repainted on a language change: the update badges are translated strings.
  repaintShimUpdateBadges: renderLoadedShims,
});

export { renderShimSources, saveShimSources, reloadShimSources, loadLoadedShims };

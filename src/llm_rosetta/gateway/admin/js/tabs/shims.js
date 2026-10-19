// shims.js — External shim sources (Settings tab).
//
// Lets the operator declare extra shim directories and remote shim archives
// from the admin UI.  The two textareas map 1:1 onto the top-level `shims`
// config section; the status table reports what the gateway loaded.
import { S } from '../core/state.js';
import { t } from '../core/i18n.js';
import { api, showToast, esc } from '../core/utils.js';

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
  _renderStatus();
  _renderPresets();
  _renderAdvisories();
}

function _renderStatus() {
  const host = document.getElementById('shimSourceStatus');
  if (!host) return;
  const sources = _status().sources || [];
  if (!sources.length) {
    host.innerHTML = `<span style="color:var(--text-dim)">${esc(t('shim.status.none'))}</span>`;
    return;
  }
  host.innerHTML = sources.map(s => {
    const badge = s.ok
      ? `<span style="color:var(--green)">${esc(t('shim.status.loaded'))}</span>`
      : `<span style="color:var(--red)">${esc(t('shim.status.failed'))}</span>`;
    const stale = s.stale
      ? ` <span style="color:var(--orange,#e67e22)">${esc(t('shim.status.stale'))}</span>`
      : '';
    const names = Array.isArray(s.shim_names) ? s.shim_names : [];
    const count = `${s.shim_count || 0} shim(s)`;
    const title = names.length ? ` title="${esc(names.join('\n'))}"` : '';
    const detail = s.ok ? `<span${title}>${count}</span>` : esc(s.error || '');
    const ref = s.resolved_ref
      ? `<div style="color:var(--text-dim);font-size:11px">${esc(String(s.resolved_ref))}</div>`
      : '';
    return `<div style="display:flex;gap:8px;padding:3px 0;border-top:1px solid var(--border)">
      <span style="min-width:56px">${badge}${stale}</span>
      <span style="min-width:48px;color:var(--text-dim)">${esc(s.kind)}</span>
      <span style="flex:1;word-break:break-all">${esc(s.location)}
        <div style="color:var(--text-dim);font-size:11px">${detail}</div>${ref}
      </span>
    </div>`;
  }).join('');
}

/** Persist the shim sources and reload the config. */
async function saveShimSources() {
  const err = document.getElementById('shimSourcesError');
  if (err) err.textContent = '';

  const paths = (document.getElementById('shimPathsInput').value || '')
    .split('\n').map(s => s.trim()).filter(Boolean);

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
    const body = { paths, remotes };
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
    if (failures) {
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

Object.assign(window, {
  renderShimSources, saveShimSources, reloadShimSources,
  addShimPresetFromSelect, addShimPresetFromRow, appendShimPreset,
});

export { renderShimSources, saveShimSources, reloadShimSources, appendShimPreset };

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

Object.assign(window, { renderShimSources, saveShimSources, reloadShimSources });

export { renderShimSources, saveShimSources, reloadShimSources };

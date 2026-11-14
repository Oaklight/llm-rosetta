/**
 * shims-page.js — the Shims rail page: installed, available, and sources.
 *
 * One entry point, three views.  Installed is grouped by vendor: a vendor's
 * shims share one logo and one heading, because the logo identifies the
 * vendor; a row stays one per shim, because enable, add-provider and the
 * switch act on a shim.  Available lists the standalone packages this build
 * knows about, and Sources edits where shims are loaded from.
 *
 * The page is the catalogue; adding a provider from a row goes through
 * `startProviderFromShim()` in `tabs/providers.js`, which owns the id rule
 * and the capability tick.
 */

import { S, _CAP_ICONS } from '../core/state.js';
import { t, typeLabel, typeLabelAlt } from '../core/i18n.js';
import { api, esc, showToast, withBusyButton } from '../core/utils.js';
import { jsArg } from '../core/html.js';

let _view = 'installed';
let _shims = [];
let _packages = [];
let _shimMeta = {};
let _loaded = false;
/** Vendors whose rows are open.  Empty means every card starts collapsed; a
 *  filter re-render keeps whatever the operator opened. */
const _openVendors = new Set();

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : '');
const vendorOf = (name) => (name || '').split('--')[0];
/** Vendors whose name is not a capitalised word; a shim's own label wins. */
const VENDOR_NAMES = {
  openai: 'OpenAI', openai_chatgpt: 'OpenAI (ChatGPT plan)', opencode: 'OpenCode',
  openrouter: 'OpenRouter', opendesign: 'OpenDesign', kilocode: 'Kilo Code',
  typesafe: 'TypeSafe', minimax: 'MiniMax', xai: 'xAI', zhipu: 'Zhipu',
  qwen: 'Qwen', moonshot: 'Moonshot', volcengine: 'Volcengine',
  // Manifest-backed vendors: the manifest names the group, and the leaves
  // (`alcf_metis`) are models of it, so their display_name is not the vendor.
  alcf: 'ALCF', argo: 'Argo', asksage: 'AskSage',
};
const vendorName = (vendor) =>
  _vendorMeta(vendor).group_label || VENDOR_NAMES[vendor] || cap(vendor);

// A manifest-backed vendor names its leaves (`alcf_metis`, display "metis")
// and every one of them serves the same format, so the name suffix alone would
// render three identical rows.
const labelOf = (s) => s.display_name || (s.name.includes('--') ? s.name.split('--')[1] : s.name);

/**
 * The capability badge class for a model type.
 *
 * `model_types[].badge_class` is where the colour lives — the plural model
 * type (`embeddings`, `decisions`) and the badge suffix (`embedding`,
 * `decision`) differ, and `cap-badge-embeddings` has no colour rule at all.
 */
function typeBadgeClass(kind) {
  const desc = (S.configData?.model_types || []).find((d) => d.name === kind);
  return (desc && desc.badge_class) || `cap-badge-${kind}`;
}

/**
 * Row order inside one vendor card: `llm` first, the remaining model types
 * alphabetically after it, and the label alphabetically within a type.  A
 * vendor that ships a chat shim and an embedding shim, or two decision
 * shims, then reads the same way everywhere instead of in inventory order.
 */
export function shimRowOrder(a, b) {
  const ta = a.model_type || 'llm';
  const tb = b.model_type || 'llm';
  if (ta !== tb) {
    if (ta === 'llm') return -1;
    if (tb === 'llm') return 1;
    return ta < tb ? -1 : 1;
  }
  return labelOf(a).localeCompare(labelOf(b));
}

/** A count with its singular form, so a one-item card does not read "1 formats". */
const plural = (n, key, oneKey) => t(n === 1 ? oneKey : key, {n});

function _vendorMeta(vendor) {
  return _shimMeta[vendor] || {};
}

async function loadShimsPage() {
  try {
    const cfg = S.configData || (await api.get('/admin/api/config'));
    S.configData = cfg;
    _shimMeta = {};
    for (const s of cfg.registered_shims || []) {
      // A vendor manifest names the group (`alcf`), which is the vendor its
      // `alcf_metis--…` leaves belong to; the name prefix would split them.
      _shimMeta[s.group || vendorOf(s.name)] = s;
    }
    const [inv, cat] = await Promise.all([
      api.get('/admin/api/config/shims'),
      api.get('/admin/api/config/shims/catalog'),
    ]);
    _shims = inv.shims || [];
    _packages = cat.packages || [];
    _loaded = true;
  } catch (e) {
    showToast(String(e.message || e), 'error');
  }
  renderShimsPage();
}

function setShimsView(view) {
  _view = view;
  renderShimsPage();
}

function _filters() {
  return {
    q: (document.getElementById('shimPageSearch')?.value || '').trim().toLowerCase(),
    origin: document.getElementById('shimOriginFilter')?.value || '',
    type: document.getElementById('shimTypeFilter')?.value || '',
    builtin: document.getElementById('shimPageShowBuiltin')?.checked !== false,
  };
}

function _matches(s, f) {
  if (f.origin && s.origin !== f.origin) return false;
  if (f.type && (s.model_type || 'llm') !== f.type) return false;
  if (!f.builtin && s.origin === 'builtin') return false;
  if (!f.q) return true;
  const hay = `${s.name} ${s.display_name || ''} ${s.format || ''} ${s.model_type || ''}`.toLowerCase();
  return hay.includes(f.q);
}

function _originBadge(origin) {
  if (origin === 'builtin') return '';
  const label = origin === 'path' ? t('shimPage.originLocal') : t('shimPage.originExternal');
  return `<span class="shim-origin is-${esc(origin)}">${esc(label)}</span>`;
}

function _markStyle(text) {
  let h = 0;
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return ` style="background:hsl(${h} 45% 90%);color:hsl(${h} 55% 32%)"`;
}

/** The vendor's logo, or a tinted initials tile when it has none. */
function _vendorLogo(vendor) {
  const logo = _vendorMeta(vendor).logo;
  const initials = esc(cap(vendor).slice(0, 2));
  const fallback = `<span class="shim-logo-fallback"${_markStyle(vendor)}>${initials}</span>`;
  if (!logo) return fallback;
  return `<img class="shim-logo" src="${esc(logo)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><span class="shim-logo-fallback" style="display:none"${_markStyle(vendor)}>${initials}</span>`;
}

/** One row: the format this shim serves, with the two actions it allows. */
function _formatRow(s) {
  // The type wears the same capability badge as a provider card: the icon the
  // panel already teaches, its name in the tooltip.  A bare word read as a
  // second format next to the row's own name.
  // Every row names its type, LLM included: a vendor card can mix a chat row
  // with an embedding or decision row, and an unlabelled row read as "the
  // default one" rather than as the type it is.
  const kind = s.model_type || 'llm';
  const icon = kind ? (_CAP_ICONS[kind] || '') : '';
  const cls = esc(typeBadgeClass(kind));
  const type = kind
    ? (icon
      ? `<span class="cap-badge ${cls}" title="${esc(typeLabelAlt(kind))}" aria-label="${esc(typeLabel(kind))}">${icon}</span>`
      : `<span class="cap-badge ${cls}" title="${esc(typeLabelAlt(kind))}">${esc(typeLabel(kind))}</span>`)
    : '';
  return `
    <div class="shim-row${s.enabled === false ? ' is-off' : ''}">
      <span class="shim-row-name">${esc(labelOf(s))}</span>
      ${type}
      <button type="button" class="shim-row-add"
              title="${esc(t(s.enabled === false ? 'shimPage.addBlocked' : 'shimPage.addProvider'))}"
              aria-label="${esc(t('shimPage.addProvider'))}"${s.enabled === false ? ' disabled' : ''}
              onclick="startProviderFromShim(${jsArg(s.name)})">+</button>
      <input type="checkbox" role="switch" class="shim-switch"
             title="${esc(t('shimPage.switchHint'))}" aria-label="${esc(t('shimPage.switchHint'))}"
             ${s.enabled === false ? '' : 'checked'} onchange="shimToggleEnabled(this, ${jsArg(s.name)})">
    </div>`;
}

const isUrl = (s) => /^https?:\/\//i.test(s || '');
const shortRepo = (url) =>
  (url || '').replace(/^https?:\/\/github\.com\//i, '').replace(/\.git$/i, '');

/** A source URL as a link that opens in a new tab; a bare string otherwise. */
function _sourceLink(source, cls) {
  if (!source) return '';
  if (!isUrl(source)) return `<span class="${cls}">${esc(source)}</span>`;
  return `<a class="${cls}" href="${esc(source)}" target="_blank" rel="noopener noreferrer">${esc(shortRepo(source))}<span aria-hidden="true"> ↗</span></a>`;
}

/** A vendor is one card: its logo, its formats, one row each.  Collapsed by
 *  default — 18 vendors × 2 rows is a long scroll to read a list of names, and
 *  a row only does anything once you are working on that vendor. */
function _vendorCard(vendor, rows) {
  const meta = _vendorMeta(vendor);
  const usedBy = new Set();
  for (const s of rows) (s.used_by || []).forEach((p) => usedBy.add(p));
  const origin = rows.some((s) => s.origin !== 'builtin') ? rows[0].origin : 'builtin';
  const external = rows.some((s) => s.origin && s.origin !== 'builtin');
  const source = (rows.find((s) => s.source) || {}).source || '';
  const open = _openVendors.has(vendor);
  const rowsId = esc(`shimRows-${vendor}`);
  return `
    <article class="shim-vendor-card${open ? ' is-open' : ''}" data-vendor="${esc(vendor)}">
      <button type="button" class="shim-vendor-head" aria-expanded="${open}" aria-controls="${rowsId}"
              onclick="toggleShimVendor(${jsArg(vendor)})">
        <span class="shim-vendor-logo">${_vendorLogo(vendor)}</span>
        <span class="shim-vendor-name">${esc(vendorName(vendor))}</span>
        ${_originBadge(origin)}
        <span class="shim-vendor-meta">${esc(plural(rows.length, 'shimPage.formats', 'shimPage.formatOne'))}${usedBy.size ? ' · ' + esc(plural(usedBy.size, 'shimPage.usedBy', 'shimPage.usedByOne')) : ''}</span>
        <svg class="shim-vendor-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="shim-rows" id="${rowsId}"${open ? '' : ' hidden'}>
        ${external ? `<div class="shim-vendor-src">
          <span class="shim-vendor-src-label">${esc(t('shimPage.source'))}</span>
          ${_sourceLink(source, 'shim-repo') || `<span class="shim-vendor-src-path">${esc(t('shimPage.sourceUnknown'))}</span>`}
        </div>` : ''}
        ${rows.map(_formatRow).join('')}
      </div>
    </article>`;
}

/** Open or close one vendor card in place, so the scroll position holds. */
export function toggleShimVendor(vendor) {
  const card = [...document.querySelectorAll('#shimGrid .shim-vendor-card')]
    .find((c) => c.dataset.vendor === vendor);
  const open = !_openVendors.has(vendor);
  if (open) _openVendors.add(vendor); else _openVendors.delete(vendor);
  if (!card) return;
  card.classList.toggle('is-open', open);
  const head = card.querySelector('.shim-vendor-head');
  const rows = card.querySelector('.shim-rows');
  if (head) head.setAttribute('aria-expanded', String(open));
  if (rows) rows.hidden = !open;
}

function _packageCard(p) {
  const formats = (p.formats || []).map((f) => `<span class="shim-chip">${esc(f)}</span>`).join('');
  const status = p.installed
    ? '<span class="shim-origin is-builtin">' + esc(t('shimPage.installedBadge')) + '</span>'
    : '<span class="shim-origin is-remote">' + esc(t('shimPage.availableBadge')) + '</span>';
  const link = _sourceLink(p.source_repo, 'shim-repo');
  return `
    <article class="shim-card" data-package="${esc(p.key)}">
      <div class="shim-card-head">
        <span class="shim-name">${esc(p.label)}</span>
        ${status}
      </div>
      <div class="shim-chips">${formats || '<span class="shim-chip">—</span>'}</div>
      <div class="shim-usage">${esc(plural(p.shims.length, 'shimPage.shims', 'shimPage.shimOne'))}${link ? ' · ' + link : ''}</div>
      ${p.installed ? '' : `<p class="shim-trust" role="note">${esc(t('shimPage.trustWarning'))}</p>`}
      <div class="shim-card-actions">
        ${p.installed
          ? `<button type="button" class="btn btn-sm" onclick="startProviderFromShim(${jsArg(p.installed_shims[0] || '')})">${esc(t('shimPage.addProvider'))}</button>
             <button type="button" class="btn btn-sm" onclick="removeShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.remove'))}</button>`
          : `<button type="button" class="btn btn-sm btn-primary" onclick="installShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.install'))}</button>`}
      </div>
    </article>`;
}

/**
 * A write result is an object, not an exception: `api.put`/`api.post` hand
 * back `{ok, status, error}` and a refused write (a 409 on a shim a provider
 * still names, a bad ref on reload) must fail the call here, or the panel
 * would report success for state the gateway never took.
 */
async function _mustWrite(request) {
  const res = await request();
  if (res && res.ok === false) throw new Error(res.error || t('toast.error'));
  return res;
}

/** The `shims.presets` this build is running with. */
function _presets() {
  const section = (S.configData && S.configData.shims) || {};
  return Array.isArray(section.presets) ? [...section.presets] : [];
}

/**
 * Declare a preset, reload the sources, and repaint.
 *
 * Installing is two calls on purpose: the PUT persists `shims.presets`, and
 * the reload is what actually fetches the repository and registers its shims.
 * Either can fail on its own — a bad ref passes the PUT and fails the fetch —
 * so the toast reports what the reload said, not what the PUT accepted.
 */
export async function installShimPackage(btn, key) {
  const presets = _presets();
  if (presets.includes(key)) return;
  await withBusyButton(btn, async () => {
    try {
      await _mustWrite(() => api.put('/admin/api/config/shims', {presets: [...presets, key]}));
      const r = await _mustWrite(() => api.post('/admin/api/config/shims/reload', {}));
      S.configData = await api.get('/admin/api/config');
      await loadShimsPage();
      const failures = (r && r.status && r.status.failure_count) || 0;
      showToast(failures
        ? t('shimPage.installFailed', {key, count: failures})
        : t('shimPage.installDone', {key}), failures ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/** Drop a preset from `shims.presets` and reload, so its shims unregister. */
export async function removeShimPackage(btn, key) {
  const presets = _presets().filter((k) => k !== key);
  await withBusyButton(btn, async () => {
    try {
      await _mustWrite(() => api.put('/admin/api/config/shims', {presets}));
      const r = await _mustWrite(() => api.post('/admin/api/config/shims/reload', {}));
      S.configData = await api.get('/admin/api/config');
      await loadShimsPage();
      const warnings = (r && r.provider_warnings) || [];
      showToast(warnings.length
        ? t('toast.providersAutoDisabled', {providers: warnings.map((x) => x.provider).join(', ')})
        : t('shimPage.removeDone', {key}), warnings.length ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

function renderShimsPage() {
  const seg = document.getElementById('shimPageSeg');
  if (seg) {
    seg.querySelectorAll('[data-view]').forEach((b) => {
      const on = b.dataset.view === _view;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
  }
  const installed = document.getElementById('shimGrid');
  const catalog = document.getElementById('shimCatalogGrid');
  const sources = document.getElementById('shimPageSources');
  const toolbar = document.getElementById('shimPageToolbar');
  const count = document.getElementById('shimsCount');
  if (!installed || !catalog) return;
  installed.hidden = _view !== 'installed';
  catalog.hidden = _view !== 'available';
  if (sources) sources.hidden = _view !== 'sources';
  if (toolbar) toolbar.hidden = _view !== 'installed';
  if (_view === 'sources') window.renderShimSources?.();

  if (_view === 'installed') {
    const f = _filters();
    const rows = _shims.filter((s) => _matches(s, f));
    const groups = new Map();
    for (const s of rows) {
      const v = s.group || vendorOf(s.name);
      if (!groups.has(v)) groups.set(v, []);
      groups.get(v).push(s);
    }
    for (const rs of groups.values()) rs.sort(shimRowOrder);
    const sorted = [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    if (count) count.textContent = _loaded
      ? t('shimPage.count', {shown: rows.length, total: _shims.length,
                             vendors: plural(sorted.length, 'shimPage.vendors', 'shimPage.vendorOne')})
      : '';
    installed.innerHTML = sorted.length
      ? sorted.map(([v, rs]) => _vendorCard(v, rs)).join('')
      : '<div class="shim-empty">' + esc(t('shim.inventory.empty')) + '</div>';
  } else if (_view === 'available') {
    if (count) count.textContent = _loaded ? t('shimPage.packages', {n: _packages.length}) : '';
    catalog.innerHTML = _packages.length
      ? _packages.map(_packageCard).join('')
      : '<div class="shim-empty">' + esc(t('shimPage.noPackages')) + '</div>';
  } else if (count) {
    count.textContent = '';
  }
}

/**
 * Register or unregister one shim for provider types.
 *
 * The switch writes `shims.disabled`, so an off shim keeps its definition but
 * stops appearing as a provider type.  The server refuses while a configured
 * provider still names it; the switch then goes back to where it was, because
 * the state it shows is the state the gateway runs.
 */
export async function shimToggleEnabled(input, name) {
  const enabled = input.checked;
  input.disabled = true;
  try {
    await _mustWrite(() => api.post(`/admin/api/config/shims/${encodeURIComponent(name)}/enabled`, {enabled}));
    // Repaint the row it belongs to, not the page: a full re-render closes
    // and reopens cards under the cursor and loses the operator's place.
    const row = input.closest('.shim-row');
    if (row) {
      row.classList.toggle('is-off', !enabled);
      const add = row.querySelector('.shim-row-add');
      if (add) {
        add.disabled = !enabled;
        add.title = t(enabled ? 'shimPage.addProvider' : 'shimPage.addBlocked');
      }
    }
    const record = _shims.find((s) => s.name === name);
    if (record) record.enabled = enabled;
    // The provider drawer reads the type list from the config; refresh it
    // without touching the board.
    try { S.configData = await api.get('/admin/api/config'); } catch { /* keep the old copy */ }
    showToast(t(enabled ? 'shimPage.enabledDone' : 'shimPage.disabledDone', {name}));
  } catch (e) {
    input.checked = !enabled;
    showToast(String(e.message || e), 'error');
  } finally {
    input.disabled = false;
  }
}

Object.assign(window, {
  loadShimsPage, renderShimsPage, setShimsView, toggleShimVendor,
  installShimPackage, removeShimPackage, shimToggleEnabled,
});

export { loadShimsPage, renderShimsPage, setShimsView };

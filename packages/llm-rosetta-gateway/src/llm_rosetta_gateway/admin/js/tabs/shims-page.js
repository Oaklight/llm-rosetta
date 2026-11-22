/**
 * shims-page.js — the Shims rail page: installed, available, and sources.
 *
 * One entry point, three views.  Installed is grouped by vendor: a vendor's
 * shims share one logo and one heading, because the logo identifies the
 * vendor; a row stays one per shim, because enable, add-provider and the
 * switch act on a shim.  Available lists every vendor the manifest sources
 * publish (#1660), and Sources edits where shims are loaded from.
 *
 * The page is the catalogue; adding a provider from a row goes through
 * `startProviderFromShim()` in `tabs/providers.js`, which owns the id rule
 * and the capability tick.
 */

import { S, _CAP_ICONS } from '../core/state.js';
import { t, typeLabel, typeLabelAlt } from '../core/i18n.js';
import { api, esc, showToast, withBusyButton } from '../core/utils.js';
import { cssValue, jsArg, logoUrl, safeUrl } from '../core/html.js';
import { toggleCard } from '../core/disclosure.js';
import {
  declareSource, deselectVendor, followLatest, notInstalledVendors, pinVendor, selectVendor,
  vendorFacts, vendorMatches, vendorStateKey, vendorTier,
} from '../core/shim-sources.js';

let _view = 'installed';
let _shims = [];
let _shimMeta = {};
let _loaded = false;
/** `/admin/api/config/shims/source-catalog`: every manifest source and every
 *  vendor it publishes (#1660 S9), or `null` when it could not be read. */
let _sourceCat = null;
/** Vendors whose rows are open.  Empty means every card starts collapsed; a
 *  filter re-render keeps whatever the operator opened. */
const _openVendors = new Set();

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : '');
const vendorOf = (name) => (name || '').split('--')[0];
/** Core vendors whose name is not a capitalised word; a shim's own label
 *  wins.  A source vendor's name comes from the catalogue (`_catalogueOf`). */
const VENDOR_NAMES = { openai: 'OpenAI', typesafe: 'TypeSafe', minimax: 'MiniMax' };
const vendorName = (vendor) => _vendorMeta(vendor).group_label
  || _catalogueOf(vendor).label || VENDOR_NAMES[vendor] || cap(vendor);

/** The first manifest-source record of *vendor*: its catalogue mark and name. */
function _catalogueOf(vendor) {
  for (const src of (_sourceCat && _sourceCat.sources) || []) {
    const v = (src.vendors || []).find((x) => x.vendor === vendor);
    if (v) return v;
  }
  return {};
}

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
    const inv = await api.get('/admin/api/config/shims');
    _shims = inv.shims || [];
    // A shim the operator declined is not registered, so it never appears in
    // `registered_shims`; its card would then paint initials beside a loaded
    // sibling that shows the vendor's mark. The inventory row of that same
    // source carries the mark, so fill the gap from there and let a
    // registered shim win whenever it has an opinion.
    for (const row of _shims) {
      const key = row.group || vendorOf(row.name);
      const known = _shimMeta[key] || {};
      _shimMeta[key] = {
        ...row,
        ...known,
        logo: known.logo || row.logo || null,
        group_label: known.group_label || row.group_label || null,
      };
    }
    try {
      const sc = await api.get('/admin/api/config/shims/source-catalog');
      _sourceCat = sc && Array.isArray(sc.sources) ? sc : null;
    } catch {
      _sourceCat = null;
    }
    // A loaded vendor whose release declares no `logo:` still has no mark in
    // the inventory.  The source catalogue names one, so fill the gap from
    // there and let a loaded shim's own declaration win.
    for (const src of (_sourceCat && _sourceCat.sources) || []) {
      for (const v of src.vendors || []) {
        const known = _shimMeta[v.vendor] || {};
        if (v.logo && !known.logo) _shimMeta[v.vendor] = { ...known, logo: v.logo };
      }
    }
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
    tier: document.getElementById('shimTierFilter')?.value || '',
    builtin: document.getElementById('shimPageShowBuiltin')?.checked !== false,
  };
}

/**
 * The resource tier of one loaded shim.
 *
 * A shim says so itself (`free_source` on a keyless pool); a plan a user pays
 * for elsewhere is a package-level fact today, so a shim inside one inherits
 * nothing here and reads as paid until the per-shim field exists.
 */
const _shimTier = (s) => s.tier || (s.free_source ? 'free' : 'paid');

function _matches(s, f) {
  // An installed package's shim is external too, so the one filter covers
  // both spellings rather than making the operator guess which list to pick.
  if (f.origin === 'remote') {
    if (s.origin !== 'remote' && s.origin !== 'plugin') return false;
  } else if (f.origin && s.origin !== f.origin) return false;
  if (f.type && (s.model_type || 'llm') !== f.type) return false;
  if (f.tier && _shimTier(s) !== f.tier) return false;
  if (!f.builtin && s.origin === 'builtin') return false;
  if (!f.q) return true;
  const hay = `${s.name} ${s.display_name || ''} ${s.format || ''} ${s.model_type || ''}`.toLowerCase();
  return hay.includes(f.q);
}

function _originBadge(origin) {
  if (origin === 'builtin') return '';
  const label = origin === 'path' ? t('shimPage.originLocal') : t('shimPage.originExt');
  return `<span class="tag is-square is-${esc(origin)}">${esc(label)}</span>`;
}

function _markStyle(text) {
  let h = 0;
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return ` style="background:${cssValue(`hsl(${h} 45% 90%)`)};color:${cssValue(`hsl(${h} 55% 32%)`)}"`;
}

/**
 * A mark for *url*, or a tinted initials tile when there is none.
 *
 * *alt* is a second candidate tried when *url* fails to load: a loaded
 * release can still name a file in a repository that has since been archived
 * or renamed, and the catalogue's copy of the mark is the better answer than
 * the vendor's initials.
 */
export function logoHtml(url, vendor, alt = null) {
  const initials = esc(cap(vendor).slice(0, 2));
  const fallback = `<span class="shim-logo-fallback"${_markStyle(vendor)}>${initials}</span>`;
  const first = url || alt;
  if (!first) return fallback;
  const second = alt && alt !== first ? ` data-alt="${esc(logoUrl(alt))}"` : '';
  return `<img class="shim-logo" src="${esc(logoUrl(first))}"${second} alt="" loading="lazy" onerror="if(this.dataset.alt){this.src=this.dataset.alt;this.dataset.alt='';}else{this.style.display='none';this.nextElementSibling.style.display='flex'}"><span class="shim-logo-fallback" style="display:none"${_markStyle(vendor)}>${initials}</span>`;
}

/** The vendor's logo, or a tinted initials tile when it has none. */
function _vendorLogo(vendor) {
  return logoHtml(_vendorMeta(vendor).logo, vendor, _catalogueOf(vendor).logo);
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
      ? `<span class="tag is-square ${cls}" title="${esc(typeLabelAlt(kind))}" aria-label="${esc(typeLabel(kind))}">${icon}</span>`
      : `<span class="tag is-square ${cls}" title="${esc(typeLabelAlt(kind))}">${esc(typeLabel(kind))}</span>`)
    : '';
  const free = s.free_source
    ? `<span class="tag is-square is-free">${esc(t('shimPage.free'))}</span>`
    : '';
  return `
    <div class="shim-row${s.enabled === false ? ' is-off' : ''}">
      <span class="shim-row-name">${esc(labelOf(s))}</span>
      ${type}
      ${free}
      <button type="button" class="shim-row-add"
              title="${esc(t(s.enabled === false ? 'shimPage.addBlocked' : 'shimPage.addProvider'))}"
              aria-label="${esc(t('shimPage.addProvider'))}"${s.enabled === false ? ' disabled' : ''}
              onclick="startProviderFromShim(${jsArg(s.name)})">+</button>
      <label class="toggle is-sm" title="${esc(t('shimPage.switchHint'))}">
        <input type="checkbox" role="switch" aria-label="${esc(t('shimPage.switchHint'))}"
               ${s.enabled === false ? '' : 'checked'} onchange="shimToggleEnabled(this, ${jsArg(s.name)})">
        <span class="slider"></span>
      </label>
    </div>`;
}

const isUrl = (s) => /^https?:\/\//i.test(s || '');
const shortRepo = (url) =>
  (url || '').replace(/^https?:\/\/github\.com\//i, '').replace(/\.git$/i, '');

/** A source URL as a link that opens in a new tab; a bare string otherwise. */
function _sourceLink(source, cls) {
  if (!source) return '';
  if (!isUrl(source)) return `<span class="${cls}">${esc(source)}</span>`;
  return `<a class="${cls}" href="${esc(safeUrl(source))}" target="_blank" rel="noopener noreferrer">${esc(shortRepo(source))}<span aria-hidden="true"> ↗</span></a>`;
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
  // The head is two fixed rows rather than one flex line: the title row keeps
  // the mark, the name and the badges, the second row carries the counts.  A
  // single line made the layout depend on how long the vendor's name is — the
  // name wrapped, the counts wrapped under it, and the card grew taller than
  // its neighbours while the badge slid to wherever the name ended.
  const badges = [
    rows.some((s) => s.free_source)
      ? `<span class="tag is-square is-free">${esc(t('shimPage.free'))}</span>`
      : '',
    _originBadge(origin),
  ].filter(Boolean).join('');
  const name = vendorName(vendor);
  const metaText = `${plural(rows.length, 'shimPage.formats', 'shimPage.formatOne')}`
    + (usedBy.size ? ' · ' + plural(usedBy.size, 'shimPage.usedBy', 'shimPage.usedByOne') : '');
  return `
    <article class="shim-vendor-card${open ? ' is-open' : ''}" data-vendor="${esc(vendor)}">
      <button type="button" class="shim-vendor-head" aria-expanded="${open}" aria-controls="${rowsId}"
              onclick="toggleShimVendor(${jsArg(vendor)})">
        <span class="shim-vendor-logo">${_vendorLogo(vendor)}</span>
        <span class="shim-vendor-title">
          <span class="shim-vendor-name" title="${esc(name)}">${esc(name)}</span>
          ${badges}
        </span>
        <span class="shim-vendor-meta" title="${esc(metaText)}">${esc(metaText)}</span>
        <svg class="shim-vendor-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="shim-rows" id="${rowsId}"${open ? '' : ' hidden'}>
        ${external ? `<div class="shim-vendor-src">
          <span class="shim-vendor-src-label">${esc(t('shimPage.source'))}</span>
          ${_sourceLink(source, 'shim-repo') || `<span class="shim-vendor-src-path">${esc(t('shimPage.sourceUnknown'))}</span>`}
        </div>` : ''}
        ${_installedManifestLine(vendor)}
        ${rows.map(_formatRow).join('')}
      </div>
    </article>`;
}

/** The manifest record a loaded vendor was installed from, with its source. */
function _installedSourceVendor(vendor) {
  for (const src of (_sourceCat && _sourceCat.sources) || []) {
    const v = (src.vendors || []).find((x) => x.vendor === vendor && x.installed);
    if (v) return { src, v };
  }
  return null;
}

/** One line of facts, each a span, joined the way the pin line joins them. */
function _factsLine(facts) {
  if (!facts.length) return '';
  const parts = facts.map((f) => `<span${f.cls ? ` class="${esc(f.cls)}"` : ''}`
    + `${f.title ? ` title="${esc(f.title)}"` : ''}>${esc(t(f.key, f.params || {}))}</span>`);
  return `<p class="shim-card-hash">${parts.join(' · ')}</p>`;
}

/** A loaded manifest vendor says which source and release it runs. */
function _installedManifestLine(vendor) {
  const hit = _installedSourceVendor(vendor);
  return hit ? _factsLine(vendorFacts(hit.v, hit.src)) : '';
}

/** Open or close one vendor card in place, so the scroll position holds. */
export function toggleShimVendor(vendor) {
  toggleCard({
    root: '#shimGrid', card: '.shim-vendor-card', attr: 'vendor', key: vendor,
    open: _openVendors, head: '.shim-vendor-head', body: '.shim-rows',
  });
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

// ── Manifest sources: one card per vendor, grouped by source (#1660 S9) ──
//
// Every vendor a manifest source publishes gets a card: where it comes from,
// what runs, what the source offers, whether the installed release's
// `requires` matches this core, and the entry that selects it.

/** Vendor cards stay closed until they are the one being worked on. */
const _openSrcVendors = new Set();

const _srcVendorKey = (src, v) => `${src.id}:${v.vendor}`;

export function toggleShimSourceVendor(key) {
  toggleCard({
    // Both lists paint this card: the Available view, and the "not installed"
    // section a search adds under the Installed rows.  The hidden list keeps
    // its last paint, so only the showing one is searched.
    root: '.shim-grid:not([hidden])', card: '.shim-card', attr: 'srcvendor', key,
    open: _openSrcVendors, head: '.shim-card-head', body: '.shim-card-body',
  });
}

/** The maturity and conformance line: what the source says about the vendor. */
function _maturityLine(v) {
  const parts = [];
  if (v.maturity && v.since) parts.push(t('shimSrc.since', {maturity: v.maturity, date: v.since}));
  else if (v.maturity) parts.push(v.maturity);
  const c = v.conformance;
  if (c && c.result) {
    parts.push(c.date
      ? t('shimSrc.conformanceOn', {result: c.result, date: c.date})
      : t('shimSrc.conformance', {result: c.result}));
  }
  if (v.promoted_to) parts.push(t('shimSrc.promoted', {target: v.promoted_to}));
  return parts.length ? `<p class="shim-card-hash">${esc(parts.join(' · '))}</p>` : '';
}

/**
 * Whether the gateway selected the vendor itself, because a provider names
 * one of its shims (`moved:<vendor>`, #1660).  No config entry carries it, so
 * there is nothing for the card's Remove to take out.
 */
export const selectedForProvider = (v) => (v.selected_origin || '').startsWith('moved:');

/** The configured providers that use one of the vendor's loaded shims. */
function _providersUsing(v) {
  const names = new Set();
  for (const name of v.installed_shims || []) {
    const row = _shims.find((x) => x.name === name);
    for (const provider of (row && row.used_by) || []) names.add(provider);
  }
  return [...names];
}

/** Which install entry picks the vendor, which one keeps it out, or that none does. */
function _selectedLine(v) {
  if (v.excluded_by) {
    return `<p class="field-hint">${esc(t('shimSrc.excludedBy', {selector: v.excluded_by}))}</p>`;
  }
  if (!v.selected_by) return `<p class="field-hint">${esc(t('shimSrc.notSelected'))}</p>`;
  const users = _providersUsing(v);
  const line = selectedForProvider(v)
    ? (users.length ? t('shimSrc.viaProviders', {providers: users.join(', ')}) : t('shimSrc.viaProvider'))
    : t('shimSrc.selectedBy', {selector: v.selected_by});
  return `<p class="field-hint">${esc(line)}</p>`;
}

/**
 * One shim of a vendor card: its name, whether it is loaded, and — for a
 * loaded one — the switch.  The vendor is the install unit; the shim is the
 * enable unit (`shims.disabled`), the same switch the Installed rows carry.
 */
function _vendorShimRow(v, name) {
  const loaded = new Set(v.installed_shims || []);
  const off = v.installed && !loaded.has(name);
  const mark = v.installed ? t(loaded.has(name) ? 'shimPage.loadedMark' : 'shimPage.notLoadedMark') : '';
  const row = loaded.has(name) ? _shims.find((x) => x.name === name) : null;
  const on = row && row.enabled !== false;
  const toggle = row
    ? `<label class="toggle is-sm" title="${esc(t('shimPage.switchHint'))}">`
      + `<input type="checkbox" role="switch" aria-label="${esc(t('shimSrc.shimSwitch', {name}))}"`
      + `${on ? ' checked' : ''} onchange="shimToggleEnabled(this, ${jsArg(name)})">`
      + `<span class="slider"></span></label>`
    : '';
  return `<div class="shim-preset-shim${off || (row && !on) ? ' off' : ''}" role="listitem">`
    + `<span class="shim-preset-name">${esc(name)}</span>`
    + `<span class="shim-preset-state">${esc(mark)}</span>${toggle}</div>`;
}

/**
 * The card's install actions: Install or Remove, and the version pin.
 *
 * A vendor the gateway installed for a provider has no Remove (the provider
 * needs it); everything else does, whatever selects it.  A pinned vendor
 * offers "Update to" the newest installable release and "Follow latest"; an
 * unpinned installed one offers "Pin this version".
 */
function _vendorActions(src, v) {
  const id = jsArg(src.id);
  const vendor = jsArg(v.vendor);
  const out = [];
  if (v.excluded_by || !v.selected_by) {
    out.push(`<button type="button" class="btn btn-sm btn-primary"${v.installable ? '' : ' disabled'} onclick="selectShimSourceVendor(this, ${id}, ${vendor})">${esc(t('shimSrc.select'))}</button>`);
    return out.join('');
  }
  if (!selectedForProvider(v)) {
    out.push(`<button type="button" class="btn btn-sm" onclick="deselectShimSourceVendor(this, ${id}, ${vendor})">${esc(t('shimSrc.deselect'))}</button>`);
  }
  if (v.pinned) {
    if (v.installable && v.installable !== v.pinned && v.update_available) {
      out.push(`<button type="button" class="btn btn-sm btn-primary" onclick="pinShimSourceVendor(this, ${id}, ${vendor}, ${jsArg(v.installable)})">${esc(t('shimSrc.updatePin', {version: v.installable}))}</button>`);
    }
    out.push(`<button type="button" class="btn btn-sm" onclick="followShimSourceVendor(this, ${id}, ${vendor})">${esc(t('shimSrc.followLatest'))}</button>`);
  } else if (v.installed) {
    out.push(`<button type="button" class="btn btn-sm" onclick="pinShimSourceVendor(this, ${id}, ${vendor}, ${jsArg(v.installed)})">${esc(t('shimSrc.pin'))}</button>`);
  }
  return out.join('');
}

function _sourceVendorCard(src, v) {
  const key = _srcVendorKey(src, v);
  const open = _openSrcVendors.has(key);
  const bodyId = esc(`shimSrcBody-${key}`);
  const name = v.label || vendorName(v.vendor);
  // Free only, as on the Installed view: it is what tells `kilo_free` from
  // `kilo`, and a "Subscription" badge cost the longer names their ending.
  const tierTag = vendorTier(v) === 'free'
    ? `<span class="tag is-square is-free">${esc(t('shimPage.free'))}</span>` : '';
  const maturity = v.maturity
    ? `<span class="tag is-square is-maturity is-${esc(v.maturity)}">${esc(v.maturity)}</span>` : '';
  const state = `<span class="tag is-square is-state is-${esc(v.state)}">${esc(t(vendorStateKey(v)))}</span>`;
  const version = v.installed || v.installable || v.newest || '';
  const meta = [version ? `v${version}` : '', plural((v.shims || []).length, 'shimPage.shims', 'shimPage.shimOne')]
    .filter(Boolean).join(' · ');
  const tags = (v.tags || []).map((tag) => `<span class="tag is-plain">${esc(tag)}</span>`).join('');
  const shims = (v.shims || []).map((name) => _vendorShimRow(v, name)).join('');
  const add = (v.installed_shims || [])[0]
    ? `<button type="button" class="btn btn-sm" onclick="startProviderFromShim(${jsArg(v.installed_shims[0])})">${esc(t('shimPage.addProvider'))}</button>`
    : '';
  const action = _vendorActions(src, v);
  return `
    <article class="shim-card shim-src-card${open ? ' is-open' : ''}" data-srcvendor="${esc(key)}">
      <button type="button" class="shim-card-head" aria-expanded="${open}" aria-controls="${bodyId}"
              onclick="toggleShimSourceVendor(${jsArg(key)})">
        <span class="shim-vendor-logo">${logoHtml(v.logo || _vendorMeta(v.vendor).logo, v.vendor, _vendorMeta(v.vendor).logo)}</span>
        <span class="shim-vendor-title">
          <span class="shim-vendor-name" title="${esc(name)}">${esc(name)}</span>
          ${tierTag}${maturity}${state}
        </span>
        <span class="shim-vendor-meta" title="${esc(meta)}">${esc(meta)}</span>
        <svg class="shim-vendor-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="shim-card-body" id="${bodyId}"${open ? '' : ' hidden'}>
        ${_factsLine(vendorFacts(v, src))}
        ${_maturityLine(v)}
        ${tags ? `<div class="shim-chips">${tags}</div>` : ''}
        <div class="shim-card-shims" role="list" aria-label="${esc(t('shimSrc.shimList'))}">${shims}</div>
        ${_selectedLine(v)}
        ${v.installed ? '' : `<p class="shim-trust" role="note">${esc(t('shimPage.trustWarning'))}</p>`}
        <div class="shim-card-actions"><span class="shim-card-buttons">${action}${add}</span></div>
      </div>
    </article>`;
}

/** One source's section head: its name, where it is read from, and its state. */
function _sourceHead(src, shown) {
  const states = [];
  if (!src.configured) states.push(`<span class="tag is-square is-plain">${esc(t('shimSrc.sourceImplied'))}</span>`);
  if (!src.ok) states.push(`<span class="tag is-square is-warn">${esc(t('shimSrc.sourceUnreachable'))}</span>`);
  else if (src.stale) states.push(`<span class="tag is-square is-warn">${esc(t('shimSrc.sourceStale'))}</span>`);
  const link = _sourceLink(src.repo || src.url, 'shim-repo');
  return `<div class="shim-section-head shim-source-head">`
    + `<span>${esc(src.label || src.id)}</span>${states.join('')}`
    + `<span class="shim-vendor-meta">${esc(plural(shown, 'shimPage.vendors', 'shimPage.vendorOne'))}</span>`
    + `${link ? `<span class="shim-source-link">${link}</span>` : ''}</div>`;
}

/** Every source, each with the vendor cards that pass the filters. */
function _sourceSections(f) {
  const sources = (_sourceCat && _sourceCat.sources) || [];
  let vendors = 0;
  const html = sources.map((src) => {
    const shown = (src.vendors || []).filter((v) => vendorMatches(v, src, f.q)
      && (!f.tier || (v.tags || []).includes(`tier:${f.tier}`)));
    vendors += shown.length;
    if (!shown.length && src.ok && (f.q || f.tier)) return '';
    const body = !src.ok
      ? `<p class="shim-source-error" title="${esc(src.error || '')}">${esc(t('shimSrc.sourceError', {error: src.error || ''}))}</p>`
      : `<div class="shim-grid shim-grid-inline">${shown.map((v) => _sourceVendorCard(src, v)).join('')}</div>`;
    return _sourceHead(src, shown.length) + body;
  }).join('');
  return { html, vendors, sources: sources.length };
}

/** The source record and vendor a card names. */
function _sourceVendor(sourceId, vendor) {
  const src = ((_sourceCat && _sourceCat.sources) || []).find((s) => s.id === sourceId);
  const v = src && (src.vendors || []).find((x) => x.vendor === vendor);
  return { src, v };
}

/** Write a `shims` body, reload the sources, and repaint the page. */
async function _writeSources(body) {
  await _mustWrite(() => api.put('/admin/api/config/shims', body));
  const r = await _mustWrite(() => api.post('/admin/api/config/shims/reload', {}));
  S.configData = await api.get('/admin/api/config');
  window.resetShimManifestEditor?.();
  await loadShimsPage();
  return r;
}

/**
 * Select one vendor of a source: add `<source>:<vendor>` to `shims.install`,
 * and the source to `shims.sources` when the declared list does not name it
 * (a config with no `sources` gets the defaults written out plus the
 * source), then reload, which installs it.
 */
export async function selectShimSourceVendor(btn, sourceId, vendor) {
  const { src, v } = _sourceVendor(sourceId, vendor);
  if (!src || !v) return;
  const section = (S.configData && S.configData.shims) || {};
  const install = selectVendor(section.install || [], sourceId, vendor, v.tags || []);
  const body = { install };
  const declared = declareSource(section, src.entry ?? (src.key || src.url));
  if (declared.changed) body.sources = declared.sources;
  await withBusyButton(btn, async () => {
    try {
      _openSrcVendors.add(_srcVendorKey(src, v));
      const r = await _writeSources(body);
      const failures = (r && r.status && r.status.failure_count) || 0;
      showToast(failures
        ? t('shimPage.installFailed', {key: vendor, count: failures})
        : t('shimSrc.selectDone', {vendor}), failures ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/**
 * Remove one vendor.  Never a refusal (`deselectVendor`): its own entries go,
 * and when a tag, `*` or a shared entry would still install it,
 * `!<source>:<vendor>` is written.
 */
export async function deselectShimSourceVendor(btn, sourceId, vendor) {
  const { src, v } = _sourceVendor(sourceId, vendor);
  if (!src || !v || selectedForProvider(v)) return;
  const section = (S.configData && S.configData.shims) || {};
  const plan = deselectVendor(section.install || [], sourceId, vendor, v.tags || []);
  if (plan.refuse) {
    showToast(t(plan.refuse.key, plan.refuse.params), 'error');
    return;
  }
  await withBusyButton(btn, async () => {
    try {
      const r = await _writeSources({ install: plan.install });
      const warnings = (r && r.provider_warnings) || [];
      const exclusion = `!${sourceId}:${vendor}`;
      const done = (plan.install.includes(exclusion)
        ? t('shimSrc.excludeDone', {vendor, selector: exclusion})
        : t('shimSrc.deselectDone', {vendor}))
        + (plan.droppedPin ? ' ' + t('shimSrc.pinDropped', {version: plan.droppedPin}) : '');
      // A provider the reload disabled is worth the error colour, but what
      // the Remove did (the exclusion, a dropped pin) still shows.
      const warned = warnings.length
        ? ' ' + t('toast.providersAutoDisabled', {providers: warnings.map((x) => x.provider).join(', ')})
        : '';
      showToast(done + warned, warnings.length ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/** Hold one vendor at *version* (also "Update to": a newer pin). */
export async function pinShimSourceVendor(btn, sourceId, vendor, version) {
  const { src, v } = _sourceVendor(sourceId, vendor);
  if (!src || !v || !version) return;
  const section = (S.configData && S.configData.shims) || {};
  const body = { install: pinVendor(section.install || [], sourceId, vendor, version) };
  const declared = declareSource(section, src.entry ?? (src.key || src.url));
  if (declared.changed) body.sources = declared.sources;
  await withBusyButton(btn, async () => {
    try {
      _openSrcVendors.add(_srcVendorKey(src, v));
      const r = await _writeSources(body);
      const failures = (r && r.status && r.status.failure_count) || 0;
      showToast(failures
        ? t('shimPage.installFailed', {key: vendor, count: failures})
        : t('shimSrc.pinDone', {vendor, version}), failures ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/** Drop a pin: the vendor follows the newest installable release again. */
export async function followShimSourceVendor(btn, sourceId, vendor) {
  const { src, v } = _sourceVendor(sourceId, vendor);
  if (!src || !v) return;
  const section = (S.configData && S.configData.shims) || {};
  await withBusyButton(btn, async () => {
    try {
      _openSrcVendors.add(_srcVendorKey(src, v));
      await _writeSources({ install: followLatest(section.install || [], sourceId, vendor) });
      showToast(t('shimSrc.followDone', {vendor}));
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
  // Search and tier read the same way on both lists; origin and model type
  // only mean something for loaded shims.
  if (toolbar) toolbar.hidden = _view === 'sources';
  const originFilter = document.getElementById('shimOriginFilter');
  if (originFilter) originFilter.hidden = _view !== 'installed';
  const typeFilter = document.getElementById('shimTypeFilter');
  if (typeFilter) typeFilter.hidden = _view !== 'installed';
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
    const extra = notInstalledVendors(_sourceCat, f);
    const rowsHtml = sorted.map(([v, rs]) => _vendorCard(v, rs)).join('');
    const extraHtml = extra.length
      ? `<div class="shim-section-head">`
        + `<span>${esc(t('shimPage.notInstalled'))}</span>`
        + `<span class="shim-vendor-meta">${esc(plural(extra.length, 'shimPage.vendors', 'shimPage.vendorOne'))}</span>`
        + `</div>`
        + `<div class="shim-grid shim-grid-inline">${extra.map(({src, v}) => _sourceVendorCard(src, v)).join('')}</div>`
      : '';
    installed.innerHTML = sorted.length || extra.length
      ? rowsHtml + extraHtml
      : '<div class="shim-empty">' + esc(t('shim.inventory.empty')) + '</div>';
    // The count covers both halves: a query that matched only catalogue
    // entries would otherwise read "0 of 0 vendors" above two visible cards.
    if (count) {
      // A query that matched only source vendors has no installed half to
      // report: "0 of 52 shims · 0 vendors" above one visible card is noise.
      const more = plural(extra.length, 'shimPage.availableMore',
        'shimPage.availableMoreOne');
      count.textContent = !_loaded ? ''
        : rows.length || !extra.length
          ? t('shimPage.count', {shown: rows.length, total: _shims.length,
                                 vendors: plural(sorted.length, 'shimPage.vendors', 'shimPage.vendorOne')})
            + (extra.length ? ' · ' + more : '')
          : more;
    }
  } else if (_view === 'available') {
    const f = _filters();
    // Every vendor of every manifest source, grouped by source.
    const sections = _sourceSections(f);
    if (count) {
      count.textContent = !_loaded ? ''
        : plural(sections.vendors, 'shimPage.vendors', 'shimPage.vendorOne');
    }
    catalog.innerHTML = sections.html
      ? sections.html
      : '<div class="shim-empty">' + esc(t('shimSrc.noSources')) + '</div>';
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
    const row = input.closest('.shim-row, .shim-preset-shim');
    if (row && row.classList.contains('shim-preset-shim')) row.classList.toggle('off', !enabled);
    else if (row) {
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
  loadShimsPage, renderShimsPage, setShimsView, toggleShimVendor, shimToggleEnabled,
  toggleShimSourceVendor, selectShimSourceVendor, deselectShimSourceVendor,
  pinShimSourceVendor, followShimSourceVendor,
  // Repainted on a language change: the per-shim rows carry translated marks.
  repaintShimCatalog: () => { if (_view === 'available') renderShimsPage(); },
});

export { loadShimsPage, renderShimsPage, setShimsView };

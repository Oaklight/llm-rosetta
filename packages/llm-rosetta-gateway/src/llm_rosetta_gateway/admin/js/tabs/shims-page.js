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
import { jsArg, logoUrl } from '../core/html.js';
import {
  availableNotInstalled, catalogueMatches, executeRefInstall, hasPattern,
  keptShims, packageLogoUrl, removeSourcesWrite, selectionMaps, selectionOf,
  selectionWrite, shortRef,
} from '../core/shim-selection.js';
import { toggleCard } from '../core/disclosure.js';

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
  // A keyless pool is named for the product, not for the pool marker.
  kilo_free: 'Kilo', opencode_zen_free: 'OpenCode Zen',
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
    const inv = await api.get('/admin/api/config/shims');
    // The pin lives in the load report; the head comes from the check, which
    // the server caches for five minutes.  Asking first means the catalogue
    // below carries both halves as one snapshot instead of a card that shows
    // a pin and no verdict.
    try { await api.post('/admin/api/config/shims/check-updates', {}); } catch { /* newest is unknown */ }
    const cat = await api.get('/admin/api/config/shims/catalog');
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
    _packages = cat.packages || [];
    // A loaded package whose archive predates its vendor's `logo:` still has
    // no mark in the inventory.  The catalogue names the same files, so fill
    // the gap from there and let a loaded shim's own declaration win.
    for (const p of _packages) {
      for (const [vendor, url] of Object.entries(p.logos || {})) {
        const known = _shimMeta[vendor] || {};
        if (!known.logo) _shimMeta[vendor] = { ...known, logo: url };
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
  return ` style="background:hsl(${h} 45% 90%);color:hsl(${h} 55% 32%)"`;
}

/** A mark for *url*, or a tinted initials tile when there is none. */
function _logoHtml(url, vendor) {
  const initials = esc(cap(vendor).slice(0, 2));
  const fallback = `<span class="shim-logo-fallback"${_markStyle(vendor)}>${initials}</span>`;
  if (!url) return fallback;
  return `<img class="shim-logo" src="${esc(logoUrl(url))}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><span class="shim-logo-fallback" style="display:none"${_markStyle(vendor)}>${initials}</span>`;
}

/** The vendor's logo, or a tinted initials tile when it has none. */
function _vendorLogo(vendor) {
  return _logoHtml(_vendorMeta(vendor).logo, vendor);
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
        ${rows.map(_formatRow).join('')}
      </div>
    </article>`;
}

/** Open or close one vendor card in place, so the scroll position holds. */
export function toggleShimVendor(vendor) {
  toggleCard({
    root: '#shimGrid', card: '.shim-vendor-card', attr: 'vendor', key: vendor,
    open: _openVendors, head: '.shim-vendor-head', body: '.shim-rows',
  });
}

/** Whether a catalogue package passes the search and tier filters. */
function _matchesPackage(p, f) {
  if (f.tier && (p.tier || (p.free ? 'free' : 'paid')) !== f.tier) return false;
  return catalogueMatches(p, f.q);
}

/**
 * Catalogue packages a query names that are not installed.
 *
 * The rule — a query is required, the tier and origin filters apply — lives in
 * `shim-selection.js`, where it is testable without a DOM; this is the page's
 * one-line binding to it.
 */
function _availableNotInstalled(f) {
  return availableNotInstalled(_packages, f);
}

/**
 * The card's overflow menu: one action today — load this package at a ref the
 * operator names instead of the one the catalogue pins.
 *
 * It is a menu rather than a fourth button because the action is the rare
 * one: a released tag is right for almost everyone, and an operator pinning a
 * commit is deliberately stepping off it.  The button stays out of the head's
 * click target (`stopPropagation`) so opening the menu never closes the card.
 */
function _packageMenu(p) {
  const id = esc(`shimMenu-${p.key}`);
  const item = p.installed
    ? `<div class="menu-item" role="menuitem" tabindex="0"
              onclick="closeMenus();openShimRefForm(${jsArg(p.key)})">${esc(t('shimPage.pinAtRef'))}</div>`
    : `<div class="menu-item" role="menuitem" tabindex="0"
              onclick="closeMenus();openShimRefForm(${jsArg(p.key)})">${esc(t('shimPage.installAtRef'))}</div>`;
  return `<span class="shim-card-tools">
      <button type="button" class="icon-btn more-menu-toggle" aria-haspopup="true"
              aria-expanded="false" aria-label="${esc(t('shimPage.cardMenu'))} ${esc(p.label)}"
              title="${esc(t('shimPage.cardMenu'))} ${esc(p.label)}"
              onclick="event.stopPropagation();toggleMoreMenu(this)">&#8943;</button>
      <div class="menu more-menu shim-card-menu" role="menu" id="${id}">${item}</div>
    </span>`;
}

/** The ref form a menu action opens, hidden until asked for. */
function _refForm(p) {
  return `<form class="shim-ref-form" hidden onsubmit="event.preventDefault();installShimAtRef(this, ${jsArg(p.key)})">
      <label class="shim-ref-label" for="shimRef-${esc(p.key)}">${esc(t('shimPage.refLabel'))}</label>
      <input type="text" id="shimRef-${esc(p.key)}" name="ref" class="input is-sm"
             placeholder="${esc(t('shimPage.refPlaceholder'))}" autocomplete="off" spellcheck="false">
      <button type="submit" class="btn btn-sm btn-primary">${esc(t('shimPage.refInstall'))}</button>
      <button type="button" class="btn btn-sm" onclick="closeShimRefForm(${jsArg(p.key)})">${esc(t('btn.cancel'))}</button>
    </form>`;
}

/** The saved selection of one package, as declaration lists. */
function _selectionOf(p) {
  const section = (S.configData && S.configData.shims) || {};
  return selectionOf(section, p);
}

/** Packages stay closed until they are the one being worked on, as vendors do. */
const _openPackages = new Set();

export function toggleShimPackage(key) {
  toggleCard({
    // Both lists paint this card: the Available view, and the "not installed"
    // section a search adds under the Installed rows.
    root: '.shim-grid', card: '.shim-card', attr: 'package', key,
    open: _openPackages, head: '.shim-card-head', body: '.shim-card-body',
  });
}

/** The mark of one vendor inside a package, before or after it is fetched. */
function _packageVendorLogo(pkg, vendor) {
  return _logoHtml(packageLogoUrl(pkg, vendor, _vendorMeta(vendor).logo), vendor);
}

/**
 * The tile a package shows on its closed card.
 *
 * A keyless pool is several vendors behind one key, and its own four marks in
 * a row say "which vendors" when the card's job is to say "what kind of
 * resource" — so it wears one FREE wordmark instead.  Any other package shows
 * the marks its vendors declare, at most three.
 */
function _packageMark(pkg) {
  const tier = pkg.tier || (pkg.free ? 'free' : 'paid');
  if (tier === 'free') {
    return `<span class="shim-pkg-mark is-free" aria-hidden="true">`
      + `<svg width="34" height="18" viewBox="0 0 34 18" role="img">`
      + `<rect x="0.5" y="0.5" width="33" height="17" rx="4" fill="none" stroke="currentColor"/>`
      + `<text x="17" y="12.4" text-anchor="middle" font-size="8" font-weight="700"`
      + ` letter-spacing="0.7" fill="currentColor">FREE</text></svg></span>`;
  }
  const vendors = [...new Set((pkg.shims || []).map(vendorOf))];
  const marks = [...new Set(
    vendors.map((v) => packageLogoUrl(pkg, v, _vendorMeta(v).logo)).filter(Boolean),
  )].slice(0, 3);
  if (!marks.length) return _logoHtml(null, pkg.key);
  return marks
    .map((url) => `<img class="shim-logo" src="${esc(logoUrl(url))}" alt="" loading="lazy">`)
    .join('');
}

/**
 * Where the running copy sits, and how far the repository has moved since.
 *
 * The pin is the commit the source was fetched at; the head is what the last
 * check read.  Only a pair that differs is an upgrade, and a head that could
 * not be read says so instead of pretending the pin is current.
 */
function _packagePinLine(p) {
  if (!p.installed) return '';
  const pin = shortRef(p.pin);
  const head = shortRef(p.head);
  if (!pin && !p.check_error) return '';
  const parts = [];
  if (pin) parts.push(`<span title="${esc(p.pin)}">${esc(t('shimPage.pinAt', {ref: pin}))}</span>`);
  if (p.update_available && head) {
    parts.push(`<span class="is-moved" title="${esc(p.head)}">${esc(t('shimPage.headAt', {ref: head}))}</span>`);
  } else if (head) {
    parts.push(`<span>${esc(t('shimPage.upToDate'))}</span>`);
  } else if (p.check_error) {
    parts.push(`<span class="is-unknown">${esc(p.check_error)}</span>`);
  }
  // A manifest install (#1660) states each vendor's maturity; the card shows
  // the distinct stages so an experimental vendor is not mistaken for stable.
  const stages = [...new Set(Object.values(p.maturity || {}).filter(Boolean))];
  if (stages.length) parts.push(`<span>${esc(stages.join(' / '))}</span>`);
  // A selected vendor the source did not install (not in the manifest, core
  // provides it, promoted) is named with the reason instead of vanishing.
  for (const [vendor, why] of Object.entries(p.skipped || {})) {
    parts.push(`<span class="is-unknown">${esc(vendor + ': ' + why)}</span>`);
  }
  return `<p class="shim-card-hash">${parts.join(' · ')}</p>`;
}

/**
 * One package card: its marks, tier and origin up front, its shims behind a
 * disclosure — the same shape the Installed view gives a vendor.
 *
 * The checkboxes are the install, not a preference: the package's shims are
 * read from the catalogue, so the list exists before the repository is ever
 * fetched, and what is ticked is exactly the `include` map that gets written.
 * Everything ticked writes no entry at all — that is the default, and it lets
 * a shim the package adds later switch itself on.
 */
function _packageCard(p) {
  const link = _sourceLink(p.source_repo, 'shim-repo');
  const tier = p.tier || (p.free ? 'free' : 'paid');
  const tierTag = `<span class="tag is-square is-${esc(tier)}">${esc(t('shimPage.' + tier))}</span>`;
  const kept = new Set(keptShims(p.shims, _selectionOf(p)));
  const loaded = new Set(p.installed_shims || []);
  const { include, exclude } = _selectionOf(p);
  const pattern = hasPattern(include) || hasPattern(exclude);
  const rows = pattern
    ? `<p class="field-hint">${esc(t('shimPage.patternNote'))}</p>`
    : (p.shims || []).map((name) => {
      const on = kept.has(name);
      const state = !p.installed ? '' : t(loaded.has(name)
        ? 'shimPage.loadedMark' : 'shimPage.notLoadedMark');
      return `<label class="shim-preset-shim${on ? '' : ' off'}">`
        + `<input type="checkbox" data-shim="${esc(name)}" value="${esc(name)}"${on ? ' checked' : ''}>`
        + `<span class="shim-vendor-logo">${_packageVendorLogo(p, vendorOf(name))}</span>`
        + `<span class="shim-preset-name">${esc(name)}</span>`
        + `<span class="shim-preset-state">${esc(state)}</span></label>`;
    }).join('');
  const open = _openPackages.has(p.key);
  const bodyId = esc(`shimBody-${p.key}`);
  const upgrade = p.installed && p.update_available
    ? `<button type="button" class="btn btn-sm btn-primary" onclick="upgradeShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.upgrade'))}</button>`
    : '';
  const actions = p.installed
    ? `${upgrade}
       <button type="button" class="btn btn-sm${upgrade ? '' : ' btn-primary'}" onclick="updateShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.saveSelection'))}</button>
       <button type="button" class="btn btn-sm" onclick="startProviderFromShim(${jsArg(p.installed_shims[0] || '')})">${esc(t('shimPage.addProvider'))}</button>
       <button type="button" class="btn btn-sm" onclick="removeShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.remove'))}</button>`
    : `<button type="button" class="btn btn-sm btn-primary" onclick="installShimPackage(this, ${jsArg(p.key)})">${esc(t('shimPage.install'))}</button>`;
  return `
    <article class="shim-card${open ? ' is-open' : ''}" data-package="${esc(p.key)}">
      <button type="button" class="shim-card-head" aria-expanded="${open}" aria-controls="${bodyId}"
              onclick="toggleShimPackage(${jsArg(p.key)})">
        <span class="shim-vendor-logo">${_packageMark(p)}</span>
        <span class="shim-vendor-title">
          <span class="shim-vendor-name" title="${esc(p.label)}">${esc(p.label)}</span>
          ${tierTag}
        </span>
        <span class="shim-vendor-meta" title="${esc(plural((p.shims || []).length, 'shimPage.shims', 'shimPage.shimOne'))}">${esc(plural((p.shims || []).length, 'shimPage.shims', 'shimPage.shimOne'))}</span>
        <svg class="shim-vendor-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="shim-card-body" id="${bodyId}"${open ? '' : ' hidden'}>
        ${_refForm(p)}
        <div class="shim-chips">${(p.formats || []).map((f) => `<span class="tag is-plain">${esc(f)}</span>`).join('') || '<span class="tag is-plain">—</span>'}${link ? ' ' + link : ''}</div>
        ${_packagePinLine(p)}
        <div class="shim-card-shims" role="group" aria-label="${esc(t('shimPage.chooseShims'))}">${rows}</div>
        ${p.installed ? '' : `<p class="shim-trust" role="note">${esc(t('shimPage.trustWarning'))}</p>`}
        <div class="shim-card-actions"><span class="shim-card-buttons">${actions}</span>${_packageMenu(p)}</div>
      </div>
    </article>`;
}

/** The ticked shims of one card, and how many it offers. */
function _cardSelection(key) {
  const card = _cardOf(key);
  const boxes = card ? [...card.querySelectorAll('input[data-shim]')] : [];
  // The name rides in `data-shim`: a bare checkbox's value is the string "on".
  return {
    checked: boxes.filter((b) => b.checked).map((b) => b.dataset.shim),
    total: boxes.length,
  };
}

/** The same maps, built from the section this panel is running with. */
function _mapsWith(key, checked, total) {
  const section = (S.configData && S.configData.shims) || {};
  return selectionMaps(section, key, checked, total);
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
 * Declare a preset with the shims its card has ticked, reload, and repaint.
 *
 * Installing is two calls on purpose: the PUT persists `shims.presets` and the
 * selection, and the reload is what actually fetches the repository and
 * registers the shims.  Either can fail on its own — a bad ref passes the PUT
 * and fails the fetch — so the toast reports what the reload said, not what
 * the PUT accepted.
 */
export async function installShimPackage(btn, key) {
  const presets = _presets();
  if (presets.includes(key)) return;
  const { checked, total } = _cardSelection(key);
  if (!checked.length) {
    showToast(t('shimPage.noneSelected'), 'error');
    return;
  }
  const { include, exclude } = _mapsWith(key, checked, total);
  await withBusyButton(btn, async () => {
    try {
      await _mustWrite(() => api.put('/admin/api/config/shims', {
        presets: [...presets, key], include, exclude,
      }));
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

/** Save a new selection for a package that is already installed. */
export async function updateShimPackage(btn, key) {
  const { checked, total } = _cardSelection(key);
  if (!checked.length) {
    showToast(t('shimPage.noneSelected'), 'error');
    return;
  }
  const maps = _mapsWith(key, checked, total);
  // A pinned package (an explicit remote) keeps its selection on the entry;
  // writing only the preset-keyed maps would be a no-op that reports success.
  const section = (S.configData && S.configData.shims) || {};
  const pkg = _packages.find((p) => p.key === key) || { key };
  const body = selectionWrite(section, pkg, maps);
  await withBusyButton(btn, async () => {
    try {
      await _mustWrite(() => api.put('/admin/api/config/shims', body));
      const r = await _mustWrite(() => api.post('/admin/api/config/shims/reload', {}));
      S.configData = await api.get('/admin/api/config');
      await loadShimsPage();
      const failures = (r && r.status && r.status.failure_count) || 0;
      showToast(failures
        ? t('shimPage.installFailed', {key, count: failures})
        : t('shimPage.updateDone', {key}), failures ? 'error' : undefined);
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/**
 * Move one package's source to its repository head, if it has moved.
 *
 * The server recomputes the verdict, so this button is a request, not the
 * decision: a call that arrives while the pin already equals the head comes
 * back `changed: false` and nothing is written or refetched.
 */
export async function upgradeShimPackage(btn, key) {
  await withBusyButton(btn, async () => {
    try {
      const r = await _mustWrite(() => api.post(
        `/admin/api/config/shims/packages/${encodeURIComponent(key)}/upgrade`, {}));
      S.configData = await api.get('/admin/api/config');
      await loadShimsPage();
      showToast(r && r.changed
        ? t('shimPage.upgradeDone', {key})
        : t('shimPage.upgradeCurrent', {key}), r && r.changed ? undefined : 'error');
    } catch (e) {
      showToast(String(e.message || e), 'error');
    }
  });
}

/** The card element of one package, in whichever list is showing it. */
function _cardOf(key) {
  return [...document.querySelectorAll('#shimCatalogGrid .shim-card, #shimGrid .shim-card')]
    .find((c) => c.dataset.package === key);
}

/** Open one card's ref form, and put the cursor in it. */
export function openShimRefForm(key) {
  const card = _cardOf(key);
  if (!card) return;
  // Opening the form inside a closed card is a no-op the operator cannot see,
  // so the body opens first — the same disclosure the head controls.
  if (card.querySelector('.shim-card-body')?.hidden) {
    toggleShimPackage(key);
  }
  const form = card.querySelector('.shim-ref-form');
  if (!form) return;
  form.hidden = false;
  form.querySelector('input[name="ref"]')?.focus();
}

/** Hide one card's ref form, and forget what was typed in it. */
export function closeShimRefForm(key) {
  const form = _cardOf(key)?.querySelector('.shim-ref-form');
  if (!form) return;
  form.hidden = true;
  const input = form.querySelector('input[name="ref"]');
  if (input) input.value = '';
}

/**
 * Load a package at the ref the operator typed.
 *
 * A preset pins a released tag and its archive digest, so another ref cannot
 * ride on it: `shims.remotes` merges over a preset with the same URL, and the
 * preset's `sha256` was computed for bytes this ref does not have.  The key
 * therefore leaves `presets` and an explicit remote takes its place — same URL
 * and path, the chosen ref, no mirror.
 *
 * The write lands before anything is fetched, and a ref the repository does
 * not have fails the fetch.  That is the one failure that must not stay on
 * disk, so the previous section is restored and reloaded: a typo costs a
 * toast, not the next startup.
 */
export async function installShimAtRef(form, key) {
  const pkg = _packages.find((p) => p.key === key);
  if (!pkg) return;
  const input = form.querySelector('input[name="ref"]');
  const ref = (input?.value || '').trim();
  if (!ref) {
    showToast(t('shimPage.refRequired'), 'error');
    input?.focus();
    return;
  }
  const section = (S.configData && S.configData.shims) || {};
  const wasInstalled = !!pkg.installed;
  await withBusyButton(form.querySelector('button[type="submit"]'), async () => {
    try {
      // Write, reload, undo-if-broken is one tested sequence
      // (`executeRefInstall`); the page only supplies the calls.
      const outcome = await executeRefInstall({
        section,
        pkg,
        ref,
        io: {
          writeShims: (body) => _mustWrite(() => api.put('/admin/api/config/shims', body)),
          reload: () => _mustWrite(() => api.post('/admin/api/config/shims/reload', {})),
          readConfig: async () => {
            S.configData = await api.get('/admin/api/config');
            _openPackages.add(key);
            await loadShimsPage();
          },
        },
      });
      if (!outcome.ok) {
        showToast(t('shimPage.refFailed',
          {ref, error: outcome.error || t('toast.error')}), 'error');
        return;
      }
      showToast(t(wasInstalled ? 'shimPage.refPinned' : 'shimPage.refInstalled',
        {key, ref}));
    } catch (e) {
      // A refused write (a 400 from validation) leaves the previous section
      // on disk already; only a reload failure after a successful write
      // needed the restore, and that path is inside the helper.
      S.configData = await api.get('/admin/api/config');
      await loadShimsPage();
      showToast(t('shimPage.refFailed', {ref, error: String(e.message || e)}), 'error');
    }
  });
}

/**
 * Remove a package, whichever shape installed it, and reload.
 *
 * A preset-installed package leaves `shims.presets`; a ref install has no
 * preset at all and leaves an explicit remote. Rewriting `presets` alone left
 * the pinned package loading while the toast said it was removed, so the
 * write is built from the live section and drops both carriers.
 */
export async function removeShimPackage(btn, key) {
  const pkg = _packages.find((p) => p.key === key) || { key };
  const section = (S.configData && S.configData.shims) || {};
  const next = removeSourcesWrite(section, pkg);
  await withBusyButton(btn, async () => {
    try {
      await _mustWrite(() => api.put('/admin/api/config/shims', next));
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
    const extra = _availableNotInstalled(f);
    const rowsHtml = sorted.map(([v, rs]) => _vendorCard(v, rs)).join('');
    const extraHtml = extra.length
      ? `<div class="shim-section-head">`
        + `<span>${esc(t('shimPage.notInstalled'))}</span>`
        + `<span class="shim-vendor-meta">${esc(plural(extra.length, 'shimPage.packages', 'shimPage.packageOne'))}</span>`
        + `</div>`
        + `<div class="shim-grid shim-grid-inline">${extra.map(_packageCard).join('')}</div>`
      : '';
    installed.innerHTML = sorted.length || extra.length
      ? rowsHtml + extraHtml
      : '<div class="shim-empty">' + esc(t('shim.inventory.empty')) + '</div>';
    // The count covers both halves: a query that matched only catalogue
    // entries would otherwise read "0 of 0 vendors" above two visible cards.
    if (count) {
      // A query that matched only catalogue entries has no installed half to
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
    const packages = _packages.filter((p) => _matchesPackage(p, f));
    if (count) count.textContent = _loaded ? t('shimPage.packages', {n: packages.length}) : '';
    catalog.innerHTML = packages.length
      ? packages.map(_packageCard).join('')
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
  toggleShimPackage, upgradeShimPackage,
  installShimPackage, updateShimPackage, removeShimPackage, shimToggleEnabled,
  openShimRefForm, closeShimRefForm, installShimAtRef,
  // Repainted on a language change: the per-shim rows carry translated marks.
  repaintShimCatalog: () => { if (_view === 'available') renderShimsPage(); },
});

export { loadShimsPage, renderShimsPage, setShimsView };

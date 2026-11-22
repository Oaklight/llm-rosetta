/**
 * shim-sources.js — the `shims.sources` / `shims.install` rules the panel
 * edits, and the facts a manifest vendor card shows (#1660 S9).
 *
 * The selector grammar mirrors the gateway's (`shims/manifest_sources.py`):
 * `<vendor>`, `tag:<full tag>`, `*`, each optionally behind `<source>:`; a
 * vendor entry may be an exclusion (`!<vendor>`, `!<source>:<vendor>`) or a
 * pin (`<source>:<vendor>@<version>`).  An exclusion always wins, and an
 * explicit vendor entry beats a tag or `*` entry.  A selector without a
 * source reads every source; the editor leaves those as written and only
 * rewrites the selectors qualified with the source it is editing, so a
 * hand-written entry is never lost by a save.
 *
 * No DOM, no state, no imports: the page binds these to its markup, and the
 * tests drive them directly.
 */

/** The sources `shims.install` reads when `shims.sources` is not declared. */
export const DEFAULT_SOURCES = Object.freeze(['contrib', 'argonne']);

/**
 * Parse one `shims.install` entry, or return `null` when it is malformed.
 *
 * @param {string} raw
 * @returns {{raw: string, source: string|null, kind: 'all'|'tag'|'vendor',
 *            value: string, negate: boolean, version: string|null}|null}
 */
export function parseSelector(raw) {
  if (typeof raw !== 'string' || !raw.trim()) return null;
  const text = raw.trim();
  const negate = text.startsWith('!');
  let rest = negate ? text.slice(1).trim() : text;
  let version = null;
  if (rest.includes('@')) {
    const at = rest.indexOf('@');
    version = rest.slice(at + 1);
    rest = rest.slice(0, at);
    if (negate || !/^[0-9][0-9A-Za-z.+-]*$/.test(version)) return null;
  }
  let source = null;
  if (!rest.startsWith('tag:') && rest.includes(':')) {
    const at = rest.indexOf(':');
    source = rest.slice(0, at);
    rest = rest.slice(at + 1);
    if (!/^[a-z0-9][a-z0-9_-]*$/.test(source)) return null;
  }
  const base = { raw: text, source, negate: false, version: null };
  if ((negate || version) && (rest === '*' || rest.startsWith('tag:'))) return null;
  if (rest === '*') return { ...base, kind: 'all', value: '' };
  if (rest.startsWith('tag:')) {
    const tag = rest.slice(4);
    if (!/^[a-z0-9][a-z0-9_-]*(:[a-z0-9][a-z0-9_.-]*)?$/.test(tag)) return null;
    return { ...base, kind: 'tag', value: tag };
  }
  if (!/^[a-z0-9][a-z0-9_]*$/.test(rest)) return null;
  return { ...base, kind: 'vendor', value: rest, negate, version };
}

/** Whether *sel*'s pattern names *vendor* (whose installable record has *tags*) in *sourceId*. */
function selectorMatches(sel, sourceId, vendor, tags = []) {
  if (!sel) return false;
  if (sel.source !== null && sel.source !== sourceId) return false;
  if (sel.kind === 'all') return true;
  if (sel.kind === 'tag') return tags.includes(sel.value);
  return sel.value === vendor;
}

/** Whether *sel* installs *vendor* in *sourceId*: an exclusion never does. */
export function selectorPicks(sel, sourceId, vendor, tags = []) {
  return !!sel && !sel.negate && selectorMatches(sel, sourceId, vendor, tags);
}

/**
 * The install entry that installs *vendor*, by the gateway's rule
 * (`pick_selector`): an exclusion wins, then an explicit vendor entry, then
 * the first entry that matches.
 */
export function pickerFor(install, sourceId, vendor, tags = []) {
  const parsed = (install || []).map(parseSelector);
  if (parsed.some((sel) => sel && sel.negate && selectorMatches(sel, sourceId, vendor, tags))) return null;
  const hits = parsed.filter((sel) => selectorPicks(sel, sourceId, vendor, tags));
  const explicit = hits.find((sel) => sel.kind === 'vendor');
  return (explicit || hits[0] || {}).raw || null;
}

/** The exclusion entry that keeps *vendor* of *sourceId* out, or `null`. */
export function excludedBy(install, sourceId, vendor) {
  for (const raw of install || []) {
    const sel = parseSelector(raw);
    if (sel && sel.negate && selectorMatches(sel, sourceId, vendor)) return sel.raw;
  }
  return null;
}

/** Whether *raw* is this source's own positive entry for *vendor* (pinned or not). */
function ownEntry(raw, sourceId, vendor) {
  const sel = parseSelector(raw);
  return !!sel && !sel.negate && sel.kind === 'vendor' && sel.source === sourceId && sel.value === vendor;
}

/** Whether *raw* pins *vendor* for *sourceId*: qualified with it, or unqualified. */
function pinEntry(raw, sourceId, vendor) {
  const sel = parseSelector(raw);
  return !!sel && !sel.negate && sel.kind === 'vendor' && !!sel.version
    && sel.value === vendor && (sel.source === sourceId || sel.source === null);
}

/** *install* without the exclusions that keep *vendor* of *sourceId* out. */
function withoutExclusions(install, sourceId, vendor) {
  return (install || []).filter((raw) => {
    const sel = parseSelector(raw);
    return !(sel && sel.negate && selectorMatches(sel, sourceId, vendor));
  });
}

/** The text a `shims.sources` entry is known by: its id or its URL. */
export function sourceEntryKey(entry) {
  if (typeof entry === 'string') return entry;
  if (entry && typeof entry === 'object' && typeof entry.url === 'string') return entry.url;
  return '';
}

/**
 * The selection the editor shows for one source: only entries qualified with
 * that source.  Unqualified entries are listed separately (`sharedSelectors`).
 *
 * @returns {{all: boolean, tags: string[], vendors: string[]}}
 */
export function sourceSelection(install, sourceId) {
  const out = { all: false, tags: [], vendors: [] };
  for (const raw of install || []) {
    const sel = parseSelector(raw);
    if (!sel || sel.source !== sourceId || sel.negate) continue;
    if (sel.kind === 'all') out.all = true;
    else if (sel.kind === 'tag') out.tags.push(sel.value);
    else out.vendors.push(sel.value);
  }
  return out;
}

/** The install entries that name no source, so they read every source. */
export function sharedSelectors(install) {
  return (install || []).filter((raw) => {
    const sel = parseSelector(raw);
    return !sel || sel.source === null;
  });
}

/**
 * Replace one source's qualified entries with *selection*, keeping every
 * other entry where it was.  The new entries go where the first old one sat,
 * so a save does not reorder a hand-edited list more than it has to.
 */
export function withSourceSelection(install, sourceId, selection) {
  // A pin and an exclusion are not checkboxes: a ticked vendor keeps its
  // pin, and every exclusion of this source stays as written.
  const pins = {};
  const exclusions = [];
  for (const raw of install || []) {
    const sel = parseSelector(raw);
    if (!sel || sel.source !== sourceId) continue;
    if (sel.negate) exclusions.push(sel.raw);
    else if (sel.kind === 'vendor' && sel.version) pins[sel.value] = sel.version;
  }
  const wanted = [];
  if (selection.all) wanted.push(`${sourceId}:*`);
  else {
    for (const tag of selection.tags || []) wanted.push(`${sourceId}:tag:${tag}`);
    for (const vendor of selection.vendors || []) {
      wanted.push(pins[vendor] ? `${sourceId}:${vendor}@${pins[vendor]}` : `${sourceId}:${vendor}`);
    }
  }
  wanted.push(...exclusions);
  // Only the edited source's entries are rebuilt (and only they are
  // de-duplicated); every other entry stays exactly as written, duplicates
  // included, because the loader accepts them and the operator wrote them.
  const fresh = [...new Set(wanted)];
  const out = [];
  let placed = false;
  for (const raw of install || []) {
    const sel = parseSelector(raw);
    if (sel && sel.source === sourceId) {
      if (!placed) { out.push(...fresh); placed = true; }
      continue;
    }
    out.push(raw);
  }
  if (!placed) out.push(...fresh);
  return out;
}

/**
 * Install one vendor from its card.
 *
 * The exclusions that keep it out (`!<source>:<vendor>`, `!<vendor>`) go
 * first; `<source>:<vendor>` is added only when nothing else then installs
 * it, so removing a tag-selected vendor and installing it back returns the
 * list to what it was.  A pin is the exception: Remove deletes the vendor's
 * pinned entry, and Install does not bring the version back — the vendor
 * follows the newest release until it is pinned again (the Remove toast
 * says "pin dropped").
 *
 * @returns {string[]} the new `shims.install`
 */
export function selectVendor(install, sourceId, vendor, tags = []) {
  const list = withoutExclusions(install, sourceId, vendor);
  if (pickerFor(list, sourceId, vendor, tags)) return list;
  return [...list, `${sourceId}:${vendor}`];
}

/**
 * Remove one vendor from its card.  Never a refusal: the vendor is the
 * install unit, so any one of them can go on its own.
 *
 * This source's own entries for it (`<source>:<vendor>`, pinned or not) are
 * deleted.  When anything else would still install it — a tag, `*`, an
 * unqualified entry — an exclusion `!<source>:<vendor>` is added instead.
 *
 * @param {string[]} install
 * @param {string} sourceId
 * @param {string} vendor
 * @param {string[]} [tags] the vendor's installable tags
 * @returns {{install: string[], droppedPin: string|null,
 *            refuse: {key: string, params: object}|null}}
 */
export function deselectVendor(install, sourceId, vendor, tags = []) {
  const list = [...(install || [])];
  const kept = list.filter((raw) => !ownEntry(raw, sourceId, vendor));
  const pinned = list.map(parseSelector)
    .find((sel) => sel && ownEntry(sel.raw, sourceId, vendor) && sel.version);
  const droppedPin = pinned ? pinned.version : null;
  if (pickerFor(kept, sourceId, vendor, tags)) {
    const exclusion = `!${sourceId}:${vendor}`;
    return {
      install: kept.includes(exclusion) ? kept : [...kept, exclusion],
      droppedPin,
      refuse: null,
    };
  }
  if (kept.length === list.length) {
    return { install: list, droppedPin: null,
      refuse: { key: 'shimSrc.nothingToRemove', params: { vendor } } };
  }
  return { install: kept, droppedPin, refuse: null };
}

/**
 * Pin one vendor to *version*: this source's entries for it, and any
 * unqualified pin of it, become one `<source>:<vendor>@<version>` where the
 * first of them sat, and an exclusion of it goes.  A pin replaces a pin:
 * two pins of one vendor are refused by the gateway, never order-decided.
 * An explicit entry beats a tag, so the pin holds even when a tag also
 * selects the vendor.
 *
 * @returns {string[]} the new `shims.install`
 */
export function pinVendor(install, sourceId, vendor, version) {
  const pin = `${sourceId}:${vendor}@${version}`;
  const out = [];
  let placed = false;
  for (const raw of withoutExclusions(install, sourceId, vendor)) {
    if (ownEntry(raw, sourceId, vendor) || pinEntry(raw, sourceId, vendor)) {
      if (!placed) { out.push(pin); placed = true; }
      continue;
    }
    out.push(raw);
  }
  if (!placed) out.push(pin);
  return out;
}

/** Drop a pin: `<source>:<vendor>@<version>` becomes `<source>:<vendor>`. */
export function followLatest(install, sourceId, vendor) {
  const out = [];
  for (const raw of install || []) {
    if (!ownEntry(raw, sourceId, vendor)) { out.push(raw); continue; }
    const plain = `${sourceId}:${vendor}`;
    if (!out.includes(plain)) out.push(plain);
  }
  return out;
}

/**
 * The `shims.sources` list that declares *entry*, judged against what the
 * config DECLARES rather than what the panel shows.
 *
 * A config with no `sources` key reads the built-in defaults; selecting a
 * vendor from the card then writes the defaults out plus the entry, so the
 * saved file names where the vendor comes from and does not depend on the
 * built-in list.
 *
 * @returns {{sources: Array, changed: boolean}}
 */
export function declareSource(section, entry) {
  const declared = section && Array.isArray(section.sources) ? [...section.sources] : [];
  const key = sourceEntryKey(entry).trim();
  const base = declared.length ? declared : [...DEFAULT_SOURCES];
  const has = base.some((s) => sourceEntryKey(s) === key);
  const sources = has || !key ? base : [...base, typeof entry === 'string' ? key : entry];
  return { sources, changed: !sameList(declared, sources) };
}

/**
 * The `shims.sources` list the panel shows: as declared, or the defaults
 * when only `install` (or nothing) is declared.
 */
export function shownSources(section) {
  const declared = section && Array.isArray(section.sources) ? section.sources : [];
  return declared.length ? [...declared] : [...DEFAULT_SOURCES];
}

/**
 * Add a source (an id, a manifest URL, or `{url, …}`).
 *
 * Starts from the shown list, so adding a URL to a config that relied on the
 * defaults keeps reading the defaults.
 *
 * @returns {{sources: Array, added: boolean}}
 */
export function addSource(section, entry) {
  const sources = shownSources(section);
  const key = sourceEntryKey(entry).trim();
  if (!key) return { sources, added: false };
  if (sources.some((s) => sourceEntryKey(s) === key)) return { sources, added: false };
  return { sources: [...sources, typeof entry === 'string' ? key : entry], added: true };
}

/**
 * Remove a source and the install entries qualified with its id.
 *
 * *sourceId* is the id selectors use (a URL source's manifest names it);
 * *entryKey* is how the `sources` list spells the entry.
 *
 * @returns {{sources: Array, install: string[]}}
 */
export function removeSource(section, entryKey, sourceId) {
  const sources = shownSources(section).filter((s) => sourceEntryKey(s) !== entryKey);
  const install = ((section && section.install) || []).filter((raw) => {
    const sel = parseSelector(raw);
    return !(sel && sel.source !== null && sel.source === sourceId);
  });
  return { sources, install };
}

/** Whether two lists hold the same entries in the same order. */
export function sameList(a, b) {
  return JSON.stringify(a || []) === JSON.stringify(b || []);
}

/**
 * The facts one manifest vendor card shows, in order, as i18n keys with
 * parameters.  `cls` marks the ones the card paints as a warning.
 *
 * @param {object} v   one vendor of `/admin/api/config/shims/source-catalog`
 * @param {object} src its source
 * @returns {Array<{key: string, params?: object, cls?: string, title?: string}>}
 */
export function vendorFacts(v, src) {
  const facts = [{ key: 'shimSrc.fromSource', params: { source: src.label || src.id } }];
  if (v.installed) {
    facts.push({ key: 'shimSrc.installedVersion', params: { version: v.installed } });
  }
  if (v.pinned) {
    facts.push({ key: 'shimSrc.pinnedAt', params: { version: v.pinned } });
    if (v.pin_problem) {
      facts.push({ key: 'shimSrc.pinProblem', params: { problem: v.pin_problem }, cls: 'is-unknown' });
    }
  }
  if (v.update_available && v.installable) {
    facts.push({ key: 'shimSrc.updateTo', params: { version: v.installable }, cls: 'is-moved' });
  }
  if (v.newest && v.newest !== v.installed && v.newest !== v.installable) {
    facts.push({ key: 'shimSrc.newestVersion', params: { version: v.newest },
      cls: v.newest_problem ? 'is-unknown' : '', title: v.newest_problem || '' });
  } else if (!v.installed && v.installable) {
    facts.push({ key: 'shimSrc.newestVersion', params: { version: v.installable } });
  }
  if (v.installed && v.requires_ok === true) {
    facts.push({ key: 'shimSrc.requiresOk' });
  } else if (v.installed && v.requires_ok === false) {
    facts.push({ key: 'shimSrc.requiresBad', cls: 'is-unknown', title: v.installed_problem || '' });
  } else if (!v.installed && !v.installable && v.newest) {
    facts.push({ key: 'shimSrc.noInstallable', cls: 'is-unknown', title: v.newest_problem || '' });
  }
  if (v.skipped) facts.push({ key: 'shimSrc.skipped', params: { reason: v.skipped }, cls: 'is-unknown' });
  if (v.error) facts.push({ key: 'shimSrc.failed', params: { error: v.error }, cls: 'is-unknown' });
  if (v.claimed_by) facts.push({ key: 'shimSrc.claimedBy', params: { source: v.claimed_by } });
  if ((v.core_provides || []).length) {
    facts.push({ key: 'shimSrc.coreProvides', params: { names: v.core_provides.join(', ') } });
  }
  return facts;
}

/** The status word a vendor card head shows, as an i18n key. */
export function vendorStateKey(v) {
  return {
    installed: 'shimSrc.state.installed',
    selected: 'shimSrc.state.selected',
    skipped: 'shimSrc.state.skipped',
    failed: 'shimSrc.state.failed',
  }[v.state] || 'shimSrc.state.available';
}

/** The distinct tags across a source's vendors, sorted. */
export function sourceTags(src) {
  const tags = new Set();
  for (const v of (src && src.vendors) || []) for (const tag of v.tags || []) tags.add(tag);
  return [...tags].sort();
}

/** Whether a vendor passes the page's search box. */
export function vendorMatches(v, src, query) {
  if (!query) return true;
  const hay = [v.vendor, v.label, src.id, src.label, v.maturity, ...(v.tags || []), ...(v.shims || [])]
    .filter(Boolean).join(' ').toLowerCase();
  return hay.includes(query);
}

/**
 * The tier a source record states in its tags, when it is not the default.
 *
 * `kilo` and `kilo_free` are one product, so they share a name and a mark;
 * the tier is what tells the two cards apart, as the Installed view's free
 * badge already does.
 *
 * @returns {'free'|'subscription'|''} '' for a paid (or untagged) vendor
 */
export function vendorTier(v) {
  const tags = (v && v.tags) || [];
  if (tags.includes('tier:free')) return 'free';
  if (tags.includes('tier:subscription')) return 'subscription';
  return '';
}

/**
 * The source vendors a search on the Installed view adds below its rows.
 *
 * The Installed view lists what is registered, so a vendor a source offers
 * but this gateway has not installed was invisible to its search box.  A
 * query is required — without one this would be the whole Available view —
 * and the filters read the way they do on the loaded rows: the tier is the
 * record's `tier:` tag (untagged reads as paid), and every source vendor is
 * external, so a built-in or local origin filter matches none.  A vendor two
 * sources publish is listed once, from the first.
 *
 * @param {{sources?: object[]}|null} catalogue `/shims/source-catalog`
 * @param {{q?: string, tier?: string, origin?: string}} filters
 * @returns {{src: object, v: object}[]}
 */
export function notInstalledVendors(catalogue, filters) {
  const q = ((filters && filters.q) || '').trim().toLowerCase();
  if (!q) return [];
  const origin = (filters && filters.origin) || '';
  if (origin && origin !== 'remote') return [];
  const tier = (filters && filters.tier) || '';
  const seen = new Set();
  const out = [];
  for (const src of (catalogue && catalogue.sources) || []) {
    for (const v of src.vendors || []) {
      if (seen.has(v.vendor)) continue;
      seen.add(v.vendor);
      if (v.installed) continue;
      if (tier && (vendorTier(v) || 'paid') !== tier) continue;
      if (!vendorMatches(v, src, q)) continue;
      out.push({ src, v });
    }
  }
  return out;
}

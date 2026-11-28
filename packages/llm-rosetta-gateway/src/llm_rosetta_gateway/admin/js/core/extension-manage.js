// extension-manage.js — what the panel says and asks about an installed extension (#1829).
//
// An installed extension is one row, whatever put it on the host (the panel, `ext
// install`, `shims.install`, a hand-placed folder, or a pip package in the image).
// This module turns the listing (`GET /admin/api/extensions`), the update check and
// the removal plan into the facts a row shows and the requests its buttons send.  It
// is DOM-free: the panel builds the nodes, and the tests call these directly.

const base = (extId) => `/admin/api/extensions/${encodeURIComponent(extId)}`;

/** The extension whose id is *vendor*, or the one that provides a shim of it. */
export function extensionForVendor(listing, vendor) {
  const rows = Array.isArray(listing) ? listing : [];
  return rows.find((e) => e && e.id === vendor)
    || rows.find((e) => e && (e.shims || []).some((s) => String(s).split('--')[0] === vendor))
    || null;
}

/**
 * The programs of an extension that a shim (or the whole vendor) needs.
 *
 * @param {object|null} ext one record of the listing
 * @param {string|null} shim a shim name, or `null` for every shim of the extension
 * @returns {Array<object>} program records of the listing, each with `used_by`
 */
export function neededPrograms(ext, shim = null) {
  const programs = (ext && Array.isArray(ext.external) ? ext.external : []);
  return programs.filter((p) => {
    const by = Array.isArray(p.used_by) ? p.used_by : [];
    return shim === null ? by.length > 0 : by.includes(shim);
  });
}

/** The programs of *ext* that a shim needs and that are not ready. */
export function unreadyPrograms(ext, shim) {
  return neededPrograms(ext, shim).filter((p) => p.status !== 'ready');
}

/**
 * What the row of an extension offers.
 *
 * An extension that lies outside the extensions directory (a pip package in the
 * image) cannot be removed, replaced or updated by the panel: the row says so and
 * names the command instead of offering a button that would fail.
 *
 * @param {object} ext one record of the listing
 * @param {object|null} update its record of the last update check, or `null`
 * @param {object|null} declared the declared source that selects it (a catalogue
 *   vendor row), or `null`
 */
export function rowActions(ext, update, declared) {
  const image = ext.origin === 'image';
  const newer = !image && update && update.status === 'update';
  return {
    origin: ext.origin === 'image' ? 'image' : 'directory',
    canRemove: !image,
    removeHint: image ? (ext.remove_hint || null) : null,
    canCheck: true,
    canUpdate: Boolean(newer),
    updateTo: newer ? update.available : null,
    // "Use declared source" when a declared source selects this extension;
    // otherwise a chosen published version.
    replace: image ? null : (declared ? 'declared' : 'version'),
    versions: !image && update && Array.isArray(update.versions) ? update.versions : [],
  };
}

/**
 * The line an update check leaves on a row, as an i18n key with parameters.
 * A failure carries its reason; a failed check is never shown as "up to date".
 */
export function updateLine(ext, update) {
  if (!update) return null;
  const p = { current: update.current || ext.version || '?', available: update.available || '' };
  switch (update.status) {
    case 'update': return { key: 'extMgr.updateAvail', params: p };
    case 'current': return { key: 'extMgr.upToDate', params: p };
    case 'not_published': return { key: 'extMgr.notPublished', params: p };
    case 'no_compatible': return { key: 'extMgr.noCompatible', params: { ...p, reason: update.error || '' }, cls: 'is-warn' };
    default: return { key: 'extMgr.checkFailed', params: { reason: update.error || '' }, cls: 'is-bad' };
  }
}

/** The declared-source vendor row that selects *extId* but was skipped, if any. */
export function skippedDeclaration(catalogue, extId) {
  for (const src of (catalogue && catalogue.sources) || []) {
    const hit = (src.vendors || []).find((v) => v.vendor === extId && v.skipped);
    if (hit) return { source: src.id, reason: hit.skipped, selectedBy: hit.selected_by || null };
  }
  return null;
}

/** Whether a declared source selects *extId* (so "use declared source" can take over). */
export function declaredFor(catalogue, extId) {
  for (const src of (catalogue && catalogue.sources) || []) {
    const hit = (src.vendors || []).find((v) => v.vendor === extId && v.selected_by);
    if (hit) return { source: src.id, selectedBy: hit.selected_by, version: hit.installable || null };
  }
  return null;
}

export const checkUpdatesRequest = (ids) => ({
  url: '/admin/api/extensions/check-updates',
  body: ids ? { ids } : {},
});
export const updateRequest = (extId, version) => ({
  url: `${base(extId)}/update`,
  body: version ? { version } : {},
});
export const removeRequest = (extId, replace) => ({
  url: `${base(extId)}/remove`,
  body: replace ? { replace: true } : {},
});
export const planRequest = (extId) => ({ url: `${base(extId)}/removal-plan` });

/** Bytes as a short size ("214 MB"); `0` is "0 B". */
export function humanSize(bytes) {
  const n = Number(bytes) || 0;
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${i === 0 ? value : value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`;
}

/**
 * The confirmation a removal shows: what goes and what stays.
 *
 * @param {object} plan the body of `GET .../removal-plan`
 * @returns {{del: Array<{path, size}>, keep: Array<{path, note}>, total: number}}
 */
export function removalSummary(plan) {
  const del = (plan.delete || []).map((e) => ({ path: e.path, size: humanSize(e.size) }));
  const keep = (plan.keep || []).map((e) => ({ path: e.path, note: e.note || '' }));
  return { del, keep, total: del.length };
}

/**
 * What a write answered.  `api.post` resolves for every status (a refusal is
 * `{ok: false, error}`, see write-result.js), so success is `ok === true` and
 * nothing else: a missing field is not a pass.
 */
export function writeOutcome(result) {
  const r = result && typeof result === 'object' ? result : {};
  if (r.ok === true) return { ok: true, result: r };
  return { ok: false, error: String(r.error || 'Request failed') };
}

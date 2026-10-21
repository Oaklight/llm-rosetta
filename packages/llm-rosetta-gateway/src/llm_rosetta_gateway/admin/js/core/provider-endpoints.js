/**
 * provider-endpoints.js — the endpoints a provider failed to parse.
 *
 * A provider can serve several model types.  When one endpoint's shim is
 * unknown, or serves another model type, the config keeps the endpoints it did
 * parse and records the failure on the provider entry as
 * `_auto_disabled_endpoints: {<model type>: {reason, category}}` (#1106,
 * #1150).  The admin payload carries the same shape — reason and category
 * only, never credentials — and the provider card renders it.
 *
 * DOM-free, so `node --test tests/js/provider-endpoints.test.mjs` can pin the
 * extraction; the caller owns the markup.
 */

/** Categories the gateway classifies an endpoint failure into. */
const CATEGORIES = new Set([
  'unknown_shim',
  'wrong_model_type',
  'missing_base_url',
  'invalid_endpoint_config',
]);

/** i18n key naming a category, or '' when it is not one we know. */
export function categoryKey(category) {
  return CATEGORIES.has(category) ? `provider.endpointCategory.${category}` : '';
}

/** Normalise one raw marker into `{type, reason, category}`, or null. */
function entry(type, marker) {
  if (!marker || typeof marker !== 'object' || Array.isArray(marker)) return null;
  const reason = typeof marker.reason === 'string' ? marker.reason : '';
  const raw = typeof marker.category === 'string' ? marker.category : '';
  // A marker with neither a reason nor a known category says nothing the card
  // can show; dropping it keeps an empty chip off the card.
  if (!reason && !categoryKey(raw)) return null;
  return { type: String(type), reason, category: raw };
}

/**
 * Degraded endpoints of a provider, sorted by model type so the card does not
 * reshuffle between renders.
 */
export function degradedEndpoints(cfg) {
  const raw = cfg && typeof cfg === 'object' ? cfg._auto_disabled_endpoints : null;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return [];
  return Object.entries(raw)
    .map(([type, marker]) => entry(type, marker))
    .filter(Boolean)
    .sort((a, b) => a.type.localeCompare(b.type));
}

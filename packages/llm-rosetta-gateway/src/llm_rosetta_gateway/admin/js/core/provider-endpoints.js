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

/** Categories the gateway classifies a failure into.
 *
 *  The first four come from the typed endpoint parser
 *  (`config/typed.py`); the last three from whole-provider validation
 *  (`ops/provider_validation.py`), which is what `_auto_disabled` carries.
 */
const CATEGORIES = new Set([
  'unknown_shim',
  'wrong_model_type',
  'missing_base_url',
  'invalid_endpoint_config',
  'missing_external_shim',
  'invalid_provider_info',
  'invalid_provider_config',
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

/**
 * The marker behind a degraded capability, or null.
 *
 * Two shapes reach the panel: a per-endpoint marker
 * (`_auto_disabled_endpoints[<type>]`) when one endpoint failed to parse, and
 * a whole-provider marker (`_auto_disabled`) when the provider's own shim did
 * — which disables every capability the card shows.
 */
export function degradedMarker(cfg, type) {
  if (!cfg || typeof cfg !== 'object' || !type) return null;
  const whole = cfg._auto_disabled;
  if (whole && typeof whole === 'object' && !Array.isArray(whole)) {
    return {
      type,
      reason: typeof whole.reason === 'string' ? whole.reason : '',
      category: typeof whole.category === 'string' ? whole.category : '',
    };
  }
  return degradedEndpoints(cfg).find((e) => e.type === type) || null;
}

/** The reason a degraded endpoint carries, or '' for any other state. */
export function degradedReason(cfg, type) {
  const marker = degradedMarker(cfg, type);
  return marker ? marker.reason || '' : '';
}

/**
 * How one capability of a provider stands.
 *
 * - `degraded` — the endpoint failed to parse and was dropped automatically
 *   (or the provider's own shim failed, which drops all of them); the reason
 *   is in the marker.
 * - `disabled` — the operator turned this endpoint off with
 *   `<type>_enabled: false`, keeping its configuration.
 * - `ok` — declared and routable.
 *
 * A degraded endpoint wins over a disabled one: the failure is the thing a
 * reader needs explained, and the two never coexist in practice — a disabled
 * endpoint is never parsed, so it cannot fail.
 */
export function endpointState(cfg, type) {
  if (!cfg || typeof cfg !== 'object' || !type) return 'ok';
  if (degradedMarker(cfg, type)) return 'degraded';
  if (cfg[`${type}_enabled`] === false) return 'disabled';
  return 'ok';
}

/**
 * How a whole provider stands, for the providers page's status facet.
 *
 * - `problem` — something needs attention: an endpoint was dropped by the
 *   parser, or the provider itself failed to load (`_auto_disabled`).
 * - `off` — nothing it serves routes: the operator switched the provider off
 *   (`enabled: false`), or turned off every capability it declares.
 * - `ok` — serving.
 *
 * A provider that failed to load is `problem`, not `off`: it is disabled, but
 * the reason is an error the operator has to fix, and it belongs at the top of
 * the list rather than at the bottom with the ones deliberately switched off.
 *
 * `caps` is the capability list the card shows; without it only the
 * provider-level switch is known, and a provider whose endpoints are all
 * turned off still reads `ok` to a caller that did not supply them.
 */
export function providerStatus(cfg, caps = null) {
  if (!cfg || typeof cfg !== 'object') return 'ok';
  const failed = cfg._auto_disabled && typeof cfg._auto_disabled === 'object';
  if (failed || degradedEndpoints(cfg).length > 0) return 'problem';
  if (cfg.enabled === false) return 'off';
  if (
    Array.isArray(caps) &&
    caps.length > 0 &&
    caps.every((cap) => endpointState(cfg, cap) === 'disabled')
  ) {
    return 'off';
  }
  return 'ok';
}

/** Sort key: problems first, healthy providers next, switched-off last. */
export function providerOrder(cfg, caps = null) {
  const status = providerStatus(cfg, caps);
  if (status === 'problem') return 0;
  return status === 'off' ? 2 : 1;
}

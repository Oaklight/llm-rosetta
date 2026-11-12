/**
 * posture.js — what the panel says about its own configuration.
 *
 * DOM-free so it runs under node, like `pagination.js` and `write-result.js`.
 *
 * Two statements the panel used to make by accident, or not at all (#1502 §2):
 *
 * - an unprotected admin panel said nothing about having no password, so the
 *   first screen an operator saw looked the same as a secured one;
 * - the API Keys page printed "Endpoints are open without authentication."
 *   whatever `server.open_on_no_keys` said — and its default is `false`, so
 *   the page claimed an open gateway while the gateway was refusing every
 *   request with 403.
 */

/**
 * Which sentence the API Keys page shows when the store is empty.
 *
 * The two outcomes are opposite, so the page has to read the switch the
 * gateway resolved (`server.open_on_no_keys`, sent by the config payload)
 * rather than assume one.
 *
 * @param {boolean} openOnNoKeys
 * @returns {'keys.noKeysOpen' | 'keys.noKeysBlocked'}
 */
export function keysEmptyMessageKey(openOnNoKeys) {
  return openOnNoKeys ? 'keys.noKeysOpen' : 'keys.noKeysBlocked';
}

/**
 * Whether to show the "no admin password" notice.
 *
 * `undefined` means `/auth-check` has not answered yet: showing the notice
 * before the server has spoken would flash it on every page load of a
 * protected panel.
 *
 * @param {boolean|undefined} requiresAuth
 * @returns {boolean}
 */
export function showsNoPasswordBanner(requiresAuth) {
  return requiresAuth === false;
}

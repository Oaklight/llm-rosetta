// Pure helpers for the shim-credential section of the provider editor.
//
// Kept out of `tabs/providers.js` so they can be tested without a DOM: both
// are places where a string from outside the panel becomes something the
// browser acts on.

/**
 * A URL safe to put in an href, or null.
 *
 * The usage page is declared by a shim, and a shim is trusted code — but this
 * link lives in the admin origin, where the session cookie is, so a
 * `javascript:` value must not reach an `href` at all.  ``https`` only: these
 * are vendor pages, so plain ``http`` buys nothing and the admin page itself
 * blocks it as mixed content anyway.
 */
export function safeLinkUrl(raw) {
  if (typeof raw !== 'string' || raw === '') return null;
  let url;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (url.protocol !== 'https:') return null;
  return url.href;
}

/**
 * Parse a pasted credential record, or null.
 *
 * The exchange owns the record's shape, so this only insists on the JSON
 * object the endpoint expects; anything else is refused before a request
 * leaves the page.
 */
export function parseRecord(text) {
  if (typeof text !== 'string') return null;
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value;
}

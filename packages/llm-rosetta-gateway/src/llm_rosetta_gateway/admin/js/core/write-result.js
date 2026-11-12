/**
 * write-result.js — the one shape a mutating admin request answers with.
 *
 * DOM-free so it runs under node, like `pagination.js` and `seg.js`.
 * `js/core/utils.js` wraps `fetch` with it; every tab reads the result.
 *
 * The three mutating helpers used to hand back `r.json()` for every status and
 * reject only when the body was not JSON.  A refusal that *was* JSON reached
 * the operator; a reverse proxy's 502 HTML page and a dropped connection
 * reached nobody — the promise rejected, no toast ran, and the dialog sat
 * unchanged (issue #1502 §1, measured for both cases).
 *
 * Every outcome now resolves:
 *
 *   success    the server's body, carrying `ok: true`
 *   refusal    the server's body, carrying `ok: false` and a non-empty `error`
 *   unreadable `{ok: false, status, error: 'HTTP 502 Bad Gateway'}`
 *   transport  see `networkResult`
 *
 * A refusal keeps the server's other fields (`merge_possible`, `installed`,
 * `changed_leaves`, …): several call sites read them off answers that are also
 * failures.
 */

/**
 * The message for a status that carried no JSON body of its own.
 *
 * @param {number} status
 * @param {string} [statusText] - the HTTP reason phrase, when the response has one
 * @returns {string}
 */
export function httpError(status, statusText) {
  if (status === 504) return 'Request timed out — the upstream service may be unreachable';
  const text = String(statusText || '').trim();
  return text ? `HTTP ${status} ${text}` : `HTTP ${status}`;
}

/** A response that never reached the server (connection refused, DNS, offline). */
export function networkResult(message) {
  return { ok: false, status: 0, error: message || 'Network error' };
}

const isObject = (value) => value !== null && typeof value === 'object';

/**
 * Read one `Response` into the shape above.  Never throws.
 *
 * @param {{ok: boolean, status: number, statusText?: string,
 *          json: () => Promise<unknown>}} response
 * @returns {Promise<object>}
 */
export async function writeResult(response) {
  const status = response.status;
  const body = await response.json().catch(() => null);
  if (response.ok) {
    if (!isObject(body)) return { ok: true };
    // The server normally says `ok` itself; the few endpoints that answer with
    // a bare object get it added so one `if (res.ok)` works everywhere.
    if (!('ok' in body)) body.ok = true;
    return body;
  }
  const error = (isObject(body) && body.error) || httpError(status, response.statusText);
  // Arrays and primitives carry no fields worth keeping.
  const fields = isObject(body) && !Array.isArray(body) ? body : {};
  return { ...fields, ok: false, status, error };
}

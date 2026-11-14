/**
 * What the address bar says about the page, and how the page says it back.
 *
 * One grammar for the whole console (#1521): `#<tab>`, `#<tab>/<id>` for the
 * screen inside a tab (a model's drawer, an observability view), and
 * `?<key>=<value>` for the search and filter that produced what is on screen.
 * Parsing and building live here, DOM-free, so the parser can be pinned by
 * `node --test` and no tab invents its own spelling.
 *
 * `#logs` is the pre-rename spelling of the observability tab and stays an
 * alias, the way the rail already treats it.
 */

/** The tab a bare `/admin` opens. */
export const DEFAULT_TAB = 'dashboard';

/** Tab ids an old bookmark may carry. */
const TAB_ALIASES = { logs: 'observability' };

/** The params a hash may carry.  Unknown keys are dropped, not round-tripped:
 *  a link from an older panel must not resurrect a filter this one removed. */
export const ROUTE_PARAMS = ['q', 'provider', 'type', 'status'];

/**
 * Read a hash into the page it names.
 *
 * @param {string} hash `location.hash`, with or without the `#`.
 * @returns {{tab: string, id: string, params: Record<string, string>}}
 */
export function parseHash(hash) {
  const raw = String(hash || '').replace(/^#/, '');
  const [path, query] = raw.split('?');
  const [tabRaw = '', ...rest] = path.split('/');
  const tab = TAB_ALIASES[tabRaw] || tabRaw || DEFAULT_TAB;
  const id = decodeURIComponent(rest.join('/'));
  const params = {};
  for (const [key, value] of new URLSearchParams(query || '')) {
    if (ROUTE_PARAMS.includes(key) && value !== '') params[key] = value;
  }
  return { tab, id, params };
}

/**
 * Write a page back into a hash.
 *
 * Empty parts are omitted, so `{tab: 'models'}` is `#models`, not
 * `#models/?`.
 *
 * @param {{tab: string, id?: string, params?: Record<string, string>}} route
 * @returns {string} A hash beginning with `#`.
 */
export function buildHash(route) {
  const tab = TAB_ALIASES[(route && route.tab) || ''] || (route && route.tab) || DEFAULT_TAB;
  const id = (route && route.id) ? '/' + encodeURIComponent(route.id) : '';
  const params = new URLSearchParams();
  for (const key of ROUTE_PARAMS) {
    const value = route && route.params ? route.params[key] : '';
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return '#' + tab + id + (query ? '?' + query : '');
}

/**
 * Whether two hashes name the same page.
 *
 * Used to decide whether a filter update is a new history step (it is not) or
 * a navigation (it is).
 *
 * @param {string} a
 * @param {string} b
 * @returns {boolean}
 */
export function sameRoute(a, b) {
  return buildHash(parseHash(a)) === buildHash(parseHash(b));
}

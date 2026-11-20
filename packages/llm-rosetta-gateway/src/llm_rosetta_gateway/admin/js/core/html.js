// html.js — text → markup, with no DOM.
//
// The admin escaped through a detached <div>: `textContent` in, `innerHTML`
// out.  That is correct, and it made every module that renders markup
// untestable outside a browser — the lane controls among them, so their
// escaping could only be read, never run.  The escaping is small and fixed;
// it is written out here once and `utils.js` re-exports it, so the browser
// behaviour does not change.
//
// `esc` escapes the five characters that end a text node or a quoted
// attribute — & < > " — and then the apostrophe, because the admin also puts
// these strings inside single-quoted JS in an attribute
// (`onclick="fn('…')"`).  The DOM version escaped & < > only, so a name with a
// double quote left `aria-label="…"` early; the comment there claimed
// otherwise.  That single-quote case is still the wrong escape for the
// position, which is what `jsArg` is for.

/** Escape a value for HTML text or a quoted attribute. */
export function esc(s) {
  if (s === null || s === undefined) return '';
  // `&` first, or the `&` of the entities below is encoded again.
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** A JS string literal, escaped to sit inside a double-quoted HTML attribute.
 *
 *  `esc` is the wrong tool for handler arguments: it turns `'` into `&#39;`,
 *  which the HTML parser decodes back to `'` before the handler is compiled,
 *  so a name like `a'-alert(1)-'b` closes the literal and runs as code.
 *  `jsArg` builds the literal with JSON and escapes the double quotes the
 *  attribute carries, so the parsed attribute is always valid JS. */
export function jsArg(v) {
  const literal = JSON.stringify(v === undefined ? null : v);
  return literal.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}


// ---- Auto-escaping templates ------------------------------------------------
//
// `html` is the escaped-by-default way to build markup: every `${...}` is run
// through `esc` unless it is already a SafeHtml (a nested `html` result, a
// `raw()` string, or an array of those).  The result is a branded SafeHtml,
// and `render` refuses anything else, so a plain string can never reach
// innerHTML through this path.  Handler arguments still use `jsArg` — wrap the
// call in `raw()` since its output is already attribute-safe.
//
// Quote every attribute value that holds an interpolation: `title="${v}"`,
// never `title=${v}`.  `esc` leaves whitespace and `=` alone, so an unquoted
// value such as `" onmouseover="window.__pwned=1` adds attributes of its own.
// tests/js/innerhtml-lint.test.mjs fails on an unquoted position.  `html`
// protects HTML quoting only: `style="…${v}"` and `<script>` context are not
// covered (see the lint's scope note).  URL-valued attributes (`href`, `src`,
// `action`, `formaction`) also need a scheme allow-list, which `html` does not
// provide: `href="${v}"` still accepts `javascript:` because `esc` only stops
// the attribute ending early (follow-up #1628).

const SAFE = Symbol('SafeHtml');

class SafeHtml {
  constructor(value) {
    this.value = value;
    this[SAFE] = true;
  }
  toString() {
    return this.value;
  }
}

/** True for a value built by `html` or `raw`. */
export function isSafeHtml(v) {
  return v instanceof SafeHtml;
}

function _part(v) {
  if (v === null || v === undefined || v === false) return '';
  if (v instanceof SafeHtml) return v.value;
  if (Array.isArray(v)) return v.map(_part).join('');
  return esc(v);
}

/** Mark a string as trusted markup (static HTML, or `jsArg` output). */
export function raw(str) {
  return new SafeHtml(String(str === null || str === undefined ? '' : str));
}

/** Tagged template: interpolations are escaped unless already SafeHtml. */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += _part(values[i]) + strings[i + 1];
  return new SafeHtml(out);
}

/** Set `el.innerHTML` from a SafeHtml; a plain string is a programming error. */
export function render(el, safeHtml) {
  if (!(safeHtml instanceof SafeHtml)) {
    throw new TypeError('render() takes SafeHtml (html`` or raw()), not ' + typeof safeHtml);
  }
  el.innerHTML = safeHtml.value;
  return el;
}


/**
 * Point a logo at the gateway rather than at the host that holds it.
 *
 * The panel used to load provider logos from `cdn.jsdelivr.net` and from
 * whatever host a deployment's entries name, so a page load talked to third
 * parties and an air-gapped install showed broken images.  The gateway now
 * fetches them once and serves them from its own origin (#1522); a relative
 * path — a logo shipped inside the panel — is left alone.
 *
 * @param {string} src
 * @returns {string}
 */
export function logoUrl(src) {
  const value = String(src || '');
  if (!/^https?:\/\//i.test(value)) return value;
  return '/admin/api/logo?src=' + encodeURIComponent(value);
}
